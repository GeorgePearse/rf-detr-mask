#!/usr/bin/env python
"""Test script to verify model construction and forward pass"""

import argparse
import os
import unittest

import torch

from rfdetr.config_utils import load_config
from rfdetr.models import build_model
from rfdetr.util.misc import NestedTensor


class TestModelConstruction(unittest.TestCase):
    """Test basic model construction and forward pass"""

    def setUp(self):
        """Set up the test environment"""
        # Load default configuration
        config_path = os.path.join("configs", "default.yaml")
        self.config = load_config(config_path)

        # Create args directly without going through to_args() which causes duplicates
        self.args = argparse.Namespace(
            device="cpu",
            num_classes=self.config.model.num_classes,
            # Use training_width and training_height from the config
            # For backward compatibility, also set resolution to the same value
            resolution=448,  # Default resolution
            training_width=self.config.model.training_width,
            training_height=self.config.model.training_height,
            encoder=self.config.model.encoder,
            out_feature_indexes=self.config.model.out_feature_indexes,
            hidden_dim=self.config.model.hidden_dim,
            projector_scale=self.config.model.projector_scale,
            dec_layers=self.config.model.dec_layers,
            dec_n_points=self.config.model.dec_n_points,
            group_detr=self.config.model.group_detr,
            num_queries=self.config.model.num_queries,
        )

        # Adjust some parameters for simple test
        self.args.batch_size = 1
        self.args.amp = False

    def test_model_construction(self):
        """Test that the model can be constructed properly"""
        # Print args for debug
        print("Using args:")
        print(f"  encoder: {self.args.encoder}")
        print(f"  resolution: {self.args.resolution}")
        print(f"  amp: {self.args.amp}")
        print(f"  num_classes: {self.args.num_classes}")

        # Build the model
        print("Building model...")
        model = build_model(self.args)
        model.to(self.args.device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model built with {num_params} parameters")

        # Assert model was built correctly
        self.assertIsNotNone(model)
        self.assertGreater(num_params, 0)

        # Return model for use in other tests
        return model

    def test_forward_pass(self):
        """Test that a forward pass through the model succeeds"""
        # Build the model first
        model = self.test_model_construction()

        # Create a dummy input
        dummy_input = torch.randn(1, 3, self.args.resolution, self.args.resolution).to(
            self.args.device
        )
        dummy_targets = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200]], device=self.args.device),
                "labels": torch.tensor([1], device=self.args.device),
                "area": torch.tensor([10000.0], device=self.args.device),
                "iscrowd": torch.tensor([0], device=self.args.device),
                "masks": torch.ones(
                    (1, self.args.resolution, self.args.resolution), device=self.args.device
                ),
            }
        ]

        # Try a forward pass
        print("Testing forward pass...")
        try:
            samples = NestedTensor(
                dummy_input, torch.ones_like(dummy_input[:, 0, :, :], dtype=torch.bool)
            )
            outputs = model(samples, dummy_targets)
            print("Forward pass successful!")
            print(f"Output keys: {outputs.keys()}")

            # Assert the outputs contain the expected keys
            self.assertIn("pred_logits", outputs)
            self.assertIn("pred_boxes", outputs)
            if "pred_masks" in outputs:
                print("Model includes segmentation head")

        except Exception as e:
            print(f"Forward pass failed with error: {e}")
            import traceback

            traceback.print_exc()
            self.fail(f"Forward pass failed: {e}")

        print("Test complete")


if __name__ == "__main__":
    unittest.main()
