"""Tests for rfdetr.engine module."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from rfdetr.engine import (
    get_autocast_args,
    update_dropout_schedules,
    compute_losses,
    process_gradient_accumulation_batch,
    process_evaluation_outputs,
)
from rfdetr.util.misc import NestedTensor


class TestEngineHelpers(unittest.TestCase):
    """Test helper functions in engine module."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_get_autocast_args_amp_enabled(self):
        """Test autocast args when AMP is enabled."""
        args = MagicMock(amp=True)
        result = get_autocast_args(args)

        self.assertIn("enabled", result)
        self.assertTrue(result["enabled"])
        self.assertIn("dtype", result)

    def test_get_autocast_args_amp_disabled(self):
        """Test autocast args when AMP is disabled."""
        args = MagicMock(amp=False)
        result = get_autocast_args(args)

        self.assertIn("enabled", result)
        self.assertFalse(result["enabled"])

    def test_update_dropout_schedules_distributed(self):
        """Test dropout schedule updates in distributed mode."""
        # Create mock model with module attribute
        model = MagicMock()
        model.module = MagicMock()

        schedules = {"dp": [0.1, 0.2, 0.3], "do": [0.05, 0.1, 0.15]}

        update_dropout_schedules(
            model=model,
            schedules=schedules,
            iteration=1,
            is_distributed=True,
            vit_encoder_num_layers=12,
        )

        # Check that distributed model methods were called
        model.module.update_drop_path.assert_called_once_with(0.2, 12)
        model.module.update_dropout.assert_called_once_with(0.1)

    def test_update_dropout_schedules_single_gpu(self):
        """Test dropout schedule updates in single GPU mode."""
        model = MagicMock()

        schedules = {"dp": [0.1, 0.2, 0.3], "do": [0.05, 0.1, 0.15]}

        update_dropout_schedules(
            model=model,
            schedules=schedules,
            iteration=0,
            is_distributed=False,
            vit_encoder_num_layers=12,
        )

        # Check that model methods were called directly
        model.update_drop_path.assert_called_once_with(0.1, 12)
        model.update_dropout.assert_called_once_with(0.05)

    def test_update_dropout_schedules_empty(self):
        """Test dropout schedule updates with empty schedules."""
        model = MagicMock()
        schedules = {}

        # Should not raise any errors
        update_dropout_schedules(
            model=model,
            schedules=schedules,
            iteration=0,
            is_distributed=False,
            vit_encoder_num_layers=12,
        )

        # No methods should be called
        model.update_drop_path.assert_not_called()
        model.update_dropout.assert_not_called()

    @patch("rfdetr.engine.autocast")
    def test_compute_losses(self, mock_autocast):
        """Test loss computation."""
        # Set up mocks
        mock_autocast.__enter__ = MagicMock()
        mock_autocast.__exit__ = MagicMock()

        model = MagicMock()
        criterion = MagicMock()

        # Mock model outputs and criterion results
        model.return_value = {"pred": torch.randn(2, 10)}
        loss_dict = {
            "loss_ce": torch.tensor(1.0),
            "loss_bbox": torch.tensor(2.0),
            "loss_giou": torch.tensor(0.5),
        }
        criterion.return_value = loss_dict
        criterion.weight_dict = {"loss_ce": 1.0, "loss_bbox": 2.5, "loss_giou": 2.0}

        # Create sample inputs
        samples = NestedTensor(
            tensors=torch.randn(2, 3, 224, 224),
            mask=torch.zeros(2, 224, 224, dtype=torch.bool),
        )
        targets = [
            {"labels": torch.tensor([1, 2]), "boxes": torch.randn(2, 4)},
            {"labels": torch.tensor([3]), "boxes": torch.randn(1, 4)},
        ]

        args = MagicMock(amp=False)

        # Compute losses
        total_loss, returned_loss_dict = compute_losses(
            model, criterion, samples, targets, self.device, args
        )

        # Check outputs
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(returned_loss_dict, loss_dict)

        # Expected total: 1.0*1.0 + 2.0*2.5 + 0.5*2.0 = 7.0
        self.assertAlmostEqual(total_loss.item(), 7.0, places=5)

    def test_process_evaluation_outputs_fp16(self):
        """Test FP16 output processing."""
        outputs = {
            "pred_logits": torch.randn(2, 100, 10, dtype=torch.float16),
            "pred_boxes": torch.randn(2, 100, 4, dtype=torch.float16),
            "enc_outputs": {
                "pred_logits": torch.randn(2, 50, 10, dtype=torch.float16),
                "pred_boxes": torch.randn(2, 50, 4, dtype=torch.float16),
            },
            "aux_outputs": [
                {
                    "pred_logits": torch.randn(2, 100, 10, dtype=torch.float16),
                    "pred_boxes": torch.randn(2, 100, 4, dtype=torch.float16),
                }
            ],
        }

        processed = process_evaluation_outputs(outputs, fp16_eval=True)

        # Check all tensors are converted to float32
        self.assertEqual(processed["pred_logits"].dtype, torch.float32)
        self.assertEqual(processed["pred_boxes"].dtype, torch.float32)
        self.assertEqual(processed["enc_outputs"]["pred_logits"].dtype, torch.float32)
        self.assertEqual(processed["enc_outputs"]["pred_boxes"].dtype, torch.float32)
        self.assertEqual(
            processed["aux_outputs"][0]["pred_logits"].dtype, torch.float32
        )
        self.assertEqual(processed["aux_outputs"][0]["pred_boxes"].dtype, torch.float32)

    def test_process_evaluation_outputs_no_fp16(self):
        """Test output processing when FP16 is disabled."""
        outputs = {
            "pred_logits": torch.randn(2, 100, 10),
            "pred_boxes": torch.randn(2, 100, 4),
        }

        processed = process_evaluation_outputs(outputs, fp16_eval=False)

        # Should return the same object
        self.assertIs(processed, outputs)

    @patch("rfdetr.engine.GradScaler")
    def test_process_gradient_accumulation_batch(self, mock_scaler_class):
        """Test gradient accumulation batch processing."""
        # Set up mocks
        mock_scaler = MagicMock()
        mock_scaler_class.return_value = mock_scaler

        model = MagicMock()
        criterion = MagicMock()

        # Mock model outputs and criterion results
        model.return_value = {"pred": torch.randn(1, 10)}
        loss_dict = {"loss": torch.tensor(1.0)}
        criterion.return_value = loss_dict
        criterion.weight_dict = {"loss": 1.0}

        # Create sample inputs for batch size 4, accumulation steps 2
        samples = NestedTensor(
            tensors=torch.randn(4, 3, 224, 224),
            mask=torch.zeros(4, 224, 224, dtype=torch.bool),
        )
        targets = [
            {"labels": torch.tensor([1]), "boxes": torch.randn(1, 4)},
            {"labels": torch.tensor([2]), "boxes": torch.randn(1, 4)},
            {"labels": torch.tensor([3]), "boxes": torch.randn(1, 4)},
            {"labels": torch.tensor([4]), "boxes": torch.randn(1, 4)},
        ]

        args = MagicMock(amp=False)

        # Process batch
        returned_loss_dict = process_gradient_accumulation_batch(
            model=model,
            criterion=criterion,
            samples=samples,
            targets=targets,
            device=self.device,
            args=args,
            scaler=mock_scaler,
            sub_batch_size=2,
            grad_accum_steps=2,
        )

        # Check that backward was called twice (once per accumulation step)
        self.assertEqual(mock_scaler.scale.return_value.backward.call_count, 2)
        self.assertEqual(returned_loss_dict, loss_dict)


if __name__ == "__main__":
    unittest.main()
