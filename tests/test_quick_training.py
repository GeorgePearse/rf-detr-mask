#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Quick Test Script
# ------------------------------------------------------------------------
"""
Test script to run a quick test of the RF-DETR-Mask training pipeline.
This script limits the dataset size to a few samples for both train and validation
to ensure the entire pipeline can run quickly for testing.
"""

import argparse
import unittest

import torch
from torch.utils.data import Subset

from rfdetr.config import load_config
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.logging_config import get_logger
from rfdetr.util.misc import collate_fn
from rfdetr.util.utils import ModelEma

logger = get_logger(__name__)


class FloatOnlyAdamW(torch.optim.AdamW):
    """AdamW optimizer that ensures everything happens in float32, regardless of input."""

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # Convert gradients to float if they're half
                    if p.grad.dtype == torch.float16:
                        p.grad = p.grad.float()
        return super().step(closure)


class TestQuickTraining(unittest.TestCase):
    """Test the quick training pipeline for RF-DETR-Mask"""

    @classmethod
    def setUpClass(cls):
        """Set up test configuration"""
        # Define default arguments
        cls.config_path = "configs/default.yaml"
        cls.train_limit = 2
        cls.val_limit = 2
        cls.epochs = 1
        cls.steps = 2
        cls.validate = True

    def test_quick_training(self):
        """Test the quick training pipeline"""
        # Load configuration
        config = load_config(self.config_path)

        # Create args directly without going through to_args() which causes duplicates
        config_args = argparse.Namespace(
            device="cpu",
            num_classes=config.model.num_classes,
            # Use training_width and training_height from the config
            # For backward compatibility, also set resolution to the same value
            resolution=448,  # Default resolution
            training_width=config.model.training_width,
            training_height=config.model.training_height,
            encoder=config.model.encoder,
            out_feature_indexes=config.model.out_feature_indexes,
            hidden_dim=config.model.hidden_dim,
            projector_scale=config.model.projector_scale,
            dec_layers=config.model.dec_layers,
            dec_n_points=config.model.dec_n_points,
            group_detr=config.model.group_detr,
            num_queries=config.model.num_queries,
        )

        # Override with test-specific settings
        config_args.batch_size = 1
        config_args.num_workers = 0  # No workers for testing
        config_args.amp = False  # Disable AMP for stability
        config_args.square_resize_div_64 = True  # Enable square resize
        config_args.lr = 5e-5
        config_args.lr_encoder = 5e-6  # Add lr_encoder for backbone learning rate
        config_args.weight_decay = 1e-4

        # Set device
        device = torch.device(config_args.device)

        # Build model
        logger.info("Building model...")
        model = build_model(config_args)
        model.to(device)

        # Print parameters
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {n_parameters}")
        self.assertGreater(n_parameters, 0, "Model should have trainable parameters")

        # Build criterion and postprocessors
        criterion, postprocessors = build_criterion_and_postprocessors(config_args)
        criterion.to(device)

        # Build optimizer
        param_dicts = get_param_dict(config_args, model)

        optimizer = FloatOnlyAdamW(
            param_dicts,
            lr=config_args.lr,
            weight_decay=config_args.weight_decay,
            fused=False,
            eps=1e-4,
        )

        # Build learning rate scheduler
        _ = torch.optim.lr_scheduler.MultiStepLR(optimizer, [config_args.lr_drop], gamma=0.1)

        # Build datasets - only use a small subset for quick testing
        logger.info("Building datasets...")

        dataset_train = build_dataset(
            image_set="train", args=config_args, resolution=config_args.resolution
        )
        self.assertIsNotNone(dataset_train, "Training dataset should be created")

        # Create a subset of the dataset for quick testing
        train_indices = list(range(min(self.train_limit, len(dataset_train))))
        dataset_train_subset = Subset(dataset_train, train_indices)

        # Create dataloader
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train_subset,
            batch_size=config_args.batch_size,
            collate_fn=collate_fn,
            num_workers=config_args.num_workers,
            shuffle=True,
        )

        # Create EMA model if needed
        ema_m = getattr(config_args, "ema_decay", None)
        ema = ModelEma(model, ema_m) if ema_m and getattr(config_args, "use_ema", False) else None

        # Set up validation dataset if needed
        if self.validate:
            dataset_val = build_dataset(
                image_set="val", args=config_args, resolution=config_args.resolution
            )
            self.assertIsNotNone(dataset_val, "Validation dataset should be created")

            val_indices = list(range(min(self.val_limit, len(dataset_val))))
            dataset_val_subset = Subset(dataset_val, val_indices)

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val_subset,
                batch_size=config_args.batch_size,
                collate_fn=collate_fn,
                num_workers=config_args.num_workers,
                shuffle=False,
            )

            base_ds = get_coco_api_from_dataset(dataset_val)

        # Run a few training steps
        logger.info("Starting training...")
        num_steps = min(self.steps, len(data_loader_train))

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch}")

            # Training step counter
            step_counter = 0

            # Wrap in try-except to catch and display any errors
            try:
                for samples, targets in data_loader_train:
                    if step_counter >= num_steps:
                        break

                    # Move to device
                    samples = samples.to(device)
                    targets = [
                        {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()
                        }
                        for t in targets
                    ]

                    # Forward pass
                    logger.info(f"Running step {step_counter + 1}/{num_steps}")
                    logger.info(f"Input shape: {samples.tensors.shape}")

                    # Print sample properties for debugging in first step
                    if step_counter == 0:
                        for k, v in targets[0].items():
                            if isinstance(v, torch.Tensor):
                                logger.info(f"Target {k} shape: {v.shape}, dtype: {v.dtype}")
                            else:
                                logger.info(f"Target {k} type: {type(v)}")

                    # Forward pass with model
                    outputs = model(samples, targets)
                    self.assertIn("pred_logits", outputs, "Model output should include pred_logits")
                    self.assertIn("pred_boxes", outputs, "Model output should include pred_boxes")

                    # Compute loss
                    loss_dict = criterion(outputs, targets)
                    weight_dict = criterion.weight_dict
                    losses = sum(
                        loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict
                    )

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    # Update EMA model if enabled
                    if ema:
                        ema.update(model)

                    logger.info(f"Step {step_counter + 1} completed with loss: {losses.item():.4f}")
                    self.assertGreater(losses.item(), 0, "Loss should be positive")

                    if step_counter == 0:
                        # Print output keys
                        logger.info(f"Output keys: {list(outputs.keys())}")

                        # Print loss keys
                        logger.info(f"Loss keys: {list(loss_dict.keys())}")

                    step_counter += 1

            except Exception as e:
                logger.error(f"Error during training: {e}")
                import traceback

                traceback.print_exc()
                self.fail(f"Training failed with error: {e}")
                break

        # Run validation if requested
        if self.validate:
            logger.info("Running validation...")
            try:
                # Use EMA model's module if available
                model_to_evaluate = ema.module if ema else model
                test_stats, coco_evaluator = evaluate(
                    model_to_evaluate,
                    criterion,
                    postprocessors,
                    data_loader_val,
                    base_ds,
                    device,
                    config_args,
                )

                # Print validation results
                for k, v in test_stats.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        logger.info(f"{k}: {v[0]:.4f}")
                    else:
                        logger.info(f"{k}: {v}")

                self.assertIsNotNone(test_stats, "Validation should produce stats")
            except Exception as e:
                logger.error(f"Error during validation: {e}")
                import traceback

                traceback.print_exc()
                self.fail(f"Validation failed with error: {e}")

        logger.info("Test complete!")


def parse_args():
    """Parse command line arguments when run as a script"""
    parser = argparse.ArgumentParser("Quick test for RF-DETR-Mask", add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--train_limit", type=int, default=5, help="Limit training dataset to N samples"
    )
    parser.add_argument(
        "--val_limit", type=int, default=5, help="Limit validation dataset to N samples"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps to run")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
    return parser.parse_args()


def main():
    """Run script from command line"""
    args = parse_args()

    # Create test instance and set class variables
    test = TestQuickTraining()
    TestQuickTraining.config_path = args.config
    TestQuickTraining.train_limit = args.train_limit
    TestQuickTraining.val_limit = args.val_limit
    TestQuickTraining.epochs = args.epochs
    TestQuickTraining.steps = args.steps
    TestQuickTraining.validate = args.validate

    # Run the test
    TestQuickTraining.setUpClass()
    test.test_quick_training()


if __name__ == "__main__":
    main()
