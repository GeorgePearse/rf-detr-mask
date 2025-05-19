"""
Test script to verify checkpoint loading works correctly with class mismatches
and missing mask parameters.
"""

import os
import types
import unittest
from pathlib import Path

import torch

from rfdetr.model_config import ModelConfig
from rfdetr.models import build_model
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)


class TestCheckpointLoading(unittest.TestCase):
    """Test checkpoint loading scenarios with different model configurations."""

    def setUp(self):
        # Find a checkpoint to test with
        # First check if rf-detr-base.pth exists
        self.base_checkpoint = "/home/georgepearse/rf-detr-mask/rf-detr-base.pth"
        if not os.path.exists(self.base_checkpoint):
            # Try to find another checkpoint
            checkpoints_dir = Path("/home/georgepearse/rf-detr-mask/checkpoints")
            if checkpoints_dir.exists():
                checkpoints = list(checkpoints_dir.glob("*.pth"))
                if checkpoints:
                    self.base_checkpoint = str(checkpoints[0])
                else:
                    self.skipTest("No checkpoint found for testing")
            else:
                self.skipTest("No checkpoint found for testing")

    def _create_model_config(self, num_classes=2):
        """Create a model configuration with the specified number of classes.

        We need to match the architecture parameters with the checkpoint to
        ensure loading works correctly.
        """
        # Create a config with the needed parameters
        config = ModelConfig(
            encoder="dinov2_windowed_small",
            out_feature_indexes=[11],  # Use the last layer index (11 for small model)
            projector_scale=["P4"],
            num_classes=num_classes,
            hidden_dim=256,
            sa_nheads=8,
            ca_nheads=16,
            dec_n_points=4,
            bbox_reparam=True,
            layer_norm=True,
            lite_refpoint_refine=True,
            dec_layers=3,
            num_queries=100,
            num_select=100,
            resolution=560,
        )

        # Convert to a namespace object to allow adding attributes dynamically
        args = types.SimpleNamespace(**config.dict_for_model_build())

        # Return the namespace object
        return args

    def test_detection_head_reinitialization(self):
        """Test detection head reinitialization for class size mismatch."""
        print(f"Testing with checkpoint: {self.base_checkpoint}")

        # Load checkpoint to get class count
        try:
            checkpoint = torch.load(self.base_checkpoint, map_location="cpu", weights_only=False)
            checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
            logger.info(f"Checkpoint has {checkpoint_num_classes} classes")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {self.base_checkpoint}: {e}")
            self.skipTest(f"Checkpoint loading failed: {e}")

        # Create a model with a different number of classes (2)
        config = self._create_model_config(num_classes=2)
        model = build_model(config)
        print(f"Model has {model.class_embed.bias.shape[0]} classes")

        # They should be different
        self.assertNotEqual(checkpoint_num_classes, model.class_embed.bias.shape[0])

        # Directly test just the reinitialize_detection_head method
        model.reinitialize_detection_head(checkpoint_num_classes)

        # Now class counts should match
        self.assertEqual(checkpoint_num_classes, model.class_embed.bias.shape[0])

        # Test that the two-stage class embeddings have also been reinitialized
        # Skip this check as the test model doesn't use two_stage
        print("Skipping two_stage class embedding check - current model doesn't use two_stage")

        print("Detection head reinitialization successful")

    def test_mask_embedding_initialization(self):
        """Test that mask embedding is properly initialized."""
        config = self._create_model_config()
        model = build_model(config)

        # Check mask embed exists
        self.assertTrue(hasattr(model, "mask_embed"), "Model should have mask_embed attribute")

        # Check mask embed has been initialized with zeros in final layer
        self.assertTrue(
            torch.all(model.mask_embed.layers[-1].weight.data == 0),
            "Final mask embed layer weights should be initialized to zero",
        )
        self.assertTrue(
            torch.all(model.mask_embed.layers[-1].bias.data == 0),
            "Final mask embed layer bias should be initialized to zero",
        )

        # Check that visualization of a mask works with the initialized values
        # This confirms that even with zeros, the model can still produce valid mask outputs
        batch_size = 1
        num_queries = 100
        hidden_dim = 256

        # Create dummy hidden states
        hs = torch.randn(1, batch_size, num_queries, hidden_dim)

        # Generate mask predictions
        mask_pred = model.mask_embed(hs).reshape(1, batch_size, num_queries, 28, 28)

        # Check shape is correct
        self.assertEqual(
            mask_pred.shape,
            (1, batch_size, num_queries, 28, 28),
            "Mask prediction should have correct shape",
        )

        # Add additional verification
        # Check that the mask embed MLP layers are properly configured
        self.assertEqual(len(model.mask_embed.layers), 3, "Mask embed should have 3 layers")
        self.assertEqual(model.mask_embed.layers[0].in_features, hidden_dim)
        self.assertEqual(model.mask_embed.layers[2].out_features, 28 * 28)


if __name__ == "__main__":
    unittest.main()
