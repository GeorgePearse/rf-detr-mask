#!/usr/bin/env python
"""Test script that uses fixed-size inputs for testing model training"""

import argparse
import os
import time
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from rfdetr.config_utils import load_config
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.logging_config import get_logger
from rfdetr.util.misc import NestedTensor

logger = get_logger(__name__)


class FixedSizeDataset(Dataset):
    """Dummy dataset with fixed size images for testing"""

    def __init__(self, num_samples=10, resolution=560, num_classes=69):
        self.num_samples = num_samples
        self.resolution = resolution
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a fixed-size tensor
        image = torch.randn(3, self.resolution, self.resolution)
        mask = torch.ones(self.resolution, self.resolution, dtype=torch.bool)

        # Create a target with a box and mask - use float32 for boxes
        target = {
            "boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]], dtype=torch.float32),
            "labels": torch.tensor([1]),
            "area": torch.tensor([10000.0]),
            "iscrowd": torch.tensor([0]),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([self.resolution, self.resolution]),
            "size": torch.tensor([self.resolution, self.resolution]),
            "masks": torch.ones((1, self.resolution, self.resolution)),
        }

        return image, mask, target


def collate_fn(batch):
    """Collate function for dataloader"""
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    targets = [item[2] for item in batch]

    # Create the NestedTensor directly
    batched_imgs = torch.stack(images)
    batched_masks = torch.stack(masks)

    return NestedTensor(batched_imgs, batched_masks), targets


class TestFixedSize(unittest.TestCase):
    """Test training with fixed-size inputs"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Load default configuration
        cls.config_path = os.path.join("configs", "default.yaml")
        cls.config = load_config(cls.config_path)

        # Create args directly without going through to_args() which causes duplicates
        cls.args = argparse.Namespace(
            device="cpu",
            num_classes=cls.config.model.num_classes,
            # Use training_width and training_height from the config
            # For backward compatibility, also set resolution to the same value
            resolution=448,  # Default resolution
            training_width=cls.config.model.training_width,
            training_height=cls.config.model.training_height,
            encoder=cls.config.model.encoder,
            out_feature_indexes=cls.config.model.out_feature_indexes,
            hidden_dim=cls.config.model.hidden_dim,
            projector_scale=cls.config.model.projector_scale,
            dec_layers=cls.config.model.dec_layers,
            dec_n_points=cls.config.model.dec_n_points,
            group_detr=cls.config.model.group_detr,
            num_queries=cls.config.model.num_queries,
        )

        # Adjust some parameters for the test
        cls.args.batch_size = 1
        cls.args.amp = False
        cls.args.epochs = 1
        cls.args.lr = 5e-5
        cls.args.lr_encoder = 5e-6  # Add lr_encoder for backbone learning rate
        cls.args.weight_decay = 1e-4

        # Set device
        cls.device = torch.device(cls.args.device)

    def test_model_construction(self):
        """Test model construction with fixed-size inputs"""
        # Build the model
        logger.info("Building model...")
        model = build_model(self.args)
        model.to(self.device)

        # Get parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {param_count} trainable parameters")

        self.assertIsNotNone(model, "Model should be created")
        self.assertGreater(param_count, 0, "Model should have trainable parameters")

        return model

    def test_fixed_size_dataset(self):
        """Test fixed size dataset functionality"""
        # Create dataset
        dataset = FixedSizeDataset(
            num_samples=5, resolution=self.args.resolution, num_classes=self.args.num_classes
        )
        self.assertEqual(len(dataset), 5, "Dataset should have 5 samples")

        # Test getting a sample
        image, mask, target = dataset[0]
        self.assertEqual(
            image.shape,
            (3, self.args.resolution, self.args.resolution),
            "Image should have correct shape",
        )
        self.assertEqual(
            mask.shape,
            (self.args.resolution, self.args.resolution),
            "Mask should have correct shape",
        )
        self.assertIn("boxes", target, "Target should have boxes key")
        self.assertIn("masks", target, "Target should have masks key")

        return dataset

    def test_training_loop(self):
        """Test full training loop with fixed-size dataset"""
        # Build model and dataset
        model = self.test_model_construction()
        dataset = self.test_fixed_size_dataset()

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=collate_fn)

        # Build criterion and postprocessors
        criterion, postprocessors = build_criterion_and_postprocessors(self.args)
        criterion.to(self.device)

        # Build optimizer
        param_dicts = get_param_dict(self.args, model)
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # Training loop
        logger.info("Starting mini-test training...")
        model.train()

        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            start_time = time.time()

            for i, (samples, targets) in enumerate(dataloader):
                # Move to device
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                try:
                    outputs = model(samples, targets)
                    self.assertIn("pred_logits", outputs, "Output should contain pred_logits")
                    self.assertIn("pred_boxes", outputs, "Output should contain pred_boxes")

                    # Compute loss
                    loss_dict = criterion(outputs, targets)
                    weight_dict = criterion.weight_dict
                    losses = sum(
                        loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict
                    )

                    # Backward pass
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    epoch_loss += losses.item()

                    logger.info(f"Batch {i + 1}/{len(dataloader)}, Loss: {losses.item():.4f}")

                    # Print some output properties
                    if i == 0:
                        logger.info(f"Output keys: {outputs.keys()}")
                        logger.info(f"Loss keys: {loss_dict.keys()}")

                        self.assertGreater(losses.item(), 0, "Loss should be positive")
                except Exception as e:
                    logger.error(f"Error in batch {i}: {e}")
                    import traceback

                    traceback.print_exc()
                    self.fail(f"Training failed with error: {e}")

            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}, Time: {epoch_time:.2f}s"
            )

        logger.info("Test training complete!")


def main():
    """Run the tests when executed as a script"""
    unittest.main()


if __name__ == "__main__":
    main()
