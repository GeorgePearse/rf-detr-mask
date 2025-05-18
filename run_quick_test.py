#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Quick Test Script
# ------------------------------------------------------------------------

"""
Script to run a quick test of the RF-DETR-Mask training pipeline.
This script limits the dataset size to a few samples for both train and validation
to ensure the entire pipeline can run quickly for testing.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from rfdetr.config_utils import load_config
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.misc import collate_fn
from rfdetr.util.logging_config import get_logger
from rfdetr.engine import train_one_epoch, evaluate
from rfdetr.util.utils import ModelEma

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Quick test for RF-DETR-Mask", add_help=True)
    parser.add_argument(
        "--config", type=str, default="configs/fixed_test_config.yaml", 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--train_limit", type=int, default=5,
        help="Limit training dataset to N samples"
    )
    parser.add_argument(
        "--val_limit", type=int, default=5,
        help="Limit validation dataset to N samples"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--steps", type=int, default=5,
        help="Number of training steps to run"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation after training"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Convert to args for backward compatibility
    config_args = config.to_args()
    
    # Override with test-specific settings
    config_args.batch_size = 1
    config_args.num_workers = 0  # No workers for testing
    config_args.amp = False      # Disable AMP for stability
    config_args.square_resize_div_64 = True  # Enable square resize
    
    # Set device
    device = torch.device(config_args.device)
    
    # Build model
    logger.info("Building model...")
    model = build_model(config_args)
    model.to(device)
    
    # Print parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {n_parameters}")
    
    # Build criterion and postprocessors
    criterion, postprocessors = build_criterion_and_postprocessors(config_args)
    criterion.to(device)
    
    # Build optimizer
    param_dicts = get_param_dict(config_args, model)
    
    # Use FloatOnlyAdamW optimizer for stability
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

    optimizer = FloatOnlyAdamW(
        param_dicts, lr=config_args.lr, weight_decay=config_args.weight_decay, fused=False, eps=1e-4
    )
    
    # Build learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [config_args.lr_drop], gamma=0.1)
    
    # Build datasets - only use a small subset for quick testing
    logger.info("Building datasets...")
    
    dataset_train = build_dataset(image_set="train", args=config_args, resolution=config_args.resolution)
    
    # Create a subset of the dataset for quick testing
    from torch.utils.data import Subset
    train_indices = list(range(min(args.train_limit, len(dataset_train))))
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
    ema = ModelEma(model, ema_m) if ema_m and config_args.use_ema else None
    
    # Set up validation dataset if needed
    if args.validate:
        dataset_val = build_dataset(image_set="val", args=config_args, resolution=config_args.resolution)
        val_indices = list(range(min(args.val_limit, len(dataset_val))))
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
    num_steps = min(args.steps, len(data_loader_train))
    
    for epoch in range(args.epochs):
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
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                
                # Forward pass
                logger.info(f"Running step {step_counter+1}/{num_steps}")
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
                
                # Compute loss
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                # Update EMA model if enabled
                if ema:
                    ema.update(model)
                
                logger.info(f"Step {step_counter+1} completed with loss: {losses.item():.4f}")
                
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
            break
    
    # Run validation if requested
    if args.validate:
        logger.info("Running validation...")
        try:
            # Use EMA model's module if available
            model_to_evaluate = ema.module if ema else model
            test_stats, coco_evaluator = evaluate(
                model_to_evaluate, criterion, postprocessors, 
                data_loader_val, base_ds, device, config_args
            )
            
            # Print validation results
            for k, v in test_stats.items():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    logger.info(f"{k}: {v[0]:.4f}")
                else:
                    logger.info(f"{k}: {v}")
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Test complete!")

if __name__ == "__main__":
    main()