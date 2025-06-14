"""
Tests for training utilities and helper functions.
"""

import unittest
import torch
from unittest.mock import Mock

from rfdetr.engine import (
    get_autocast_args,
    update_dropout_schedules,
    compute_losses,
)
from rfdetr.util.misc import NestedTensor


class TestTrainingUtils(unittest.TestCase):
    """Test training utility functions."""

    def test_get_autocast_args_with_amp(self):
        """Test autocast args with AMP enabled."""
        args = Mock()
        args.amp = True

        autocast_args = get_autocast_args(args)

        self.assertIn("enabled", autocast_args)
        self.assertTrue(autocast_args["enabled"])
        self.assertIn("dtype", autocast_args)
        # Should use bfloat16 if available, else float16
        self.assertIn(autocast_args["dtype"], [torch.bfloat16, torch.float16])

    def test_get_autocast_args_without_amp(self):
        """Test autocast args with AMP disabled."""
        args = Mock()
        args.amp = False

        autocast_args = get_autocast_args(args)

        self.assertIn("enabled", autocast_args)
        self.assertFalse(autocast_args["enabled"])

    def test_update_dropout_schedules(self):
        """Test dropout schedule updates."""
        # Create mock model
        model = Mock()
        model.module = Mock()
        model.module.update_drop_path = Mock()
        model.module.update_dropout = Mock()

        schedules = {"dp": [0.1, 0.2, 0.3], "do": [0.05, 0.1, 0.15]}

        # Test distributed mode
        update_dropout_schedules(
            model,
            schedules,
            iteration=1,
            is_distributed=True,
            vit_encoder_num_layers=12,
        )

        model.module.update_drop_path.assert_called_with(0.2, 12)
        model.module.update_dropout.assert_called_with(0.1)

    def test_compute_losses(self):
        """Test loss computation."""
        # Create mock model
        model = Mock()
        model.return_value = {"pred_logits": torch.randn(2, 100, 91)}

        # Create mock criterion
        criterion = Mock()
        criterion.weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 2.0,
            "loss_giou": 1.0,
        }
        criterion.return_value = {
            "loss_ce": torch.tensor(0.5),
            "loss_bbox": torch.tensor(0.3),
            "loss_giou": torch.tensor(0.2),
        }

        # Create mock inputs
        batch_size = 2
        samples = NestedTensor(
            torch.randn(batch_size, 3, 224, 224),
            torch.ones(batch_size, 224, 224, dtype=torch.bool),
        )

        targets = [
            {
                "labels": torch.randint(0, 91, (10,)),
                "boxes": torch.rand(10, 4),
            }
            for _ in range(batch_size)
        ]

        device = torch.device("cpu")
        args = Mock()
        args.amp = False

        # Compute losses
        total_loss, loss_dict = compute_losses(
            model, criterion, samples, targets, device, args
        )

        # Check outputs
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)
        # Total loss should be weighted sum
        expected_total = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 1.0
        self.assertAlmostEqual(total_loss.item(), expected_total, places=5)


class TestNestedTensor(unittest.TestCase):
    """Test NestedTensor functionality."""

    def test_nested_tensor_creation(self):
        """Test creating a NestedTensor."""
        tensors = torch.randn(2, 3, 224, 224)
        mask = torch.ones(2, 224, 224, dtype=torch.bool)

        nested = NestedTensor(tensors, mask)

        self.assertEqual(nested.tensors.shape, (2, 3, 224, 224))
        self.assertEqual(nested.mask.shape, (2, 224, 224))

    def test_nested_tensor_to_device(self):
        """Test moving NestedTensor to device."""
        tensors = torch.randn(2, 3, 224, 224)
        mask = torch.ones(2, 224, 224, dtype=torch.bool)

        nested = NestedTensor(tensors, mask)
        device = torch.device("cpu")  # Would be 'cuda' in real usage

        nested_on_device = nested.to(device)

        self.assertEqual(nested_on_device.tensors.device, device)
        self.assertEqual(nested_on_device.mask.device, device)


if __name__ == "__main__":
    unittest.main()
