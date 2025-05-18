#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Training Script Using YAML Configuration (TEST VERSION)
# ------------------------------------------------------------------------

"""
Test script to train RF-DETR with mask head using YAML configuration.
This is a simplified version for quick testing with limited number of samples.
"""

import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

import rfdetr.util.misc as utils
from rfdetr.config_utils import load_config
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.logging_config import get_logger
from rfdetr.util.utils import BestMetricHolder, ModelEma

logger = get_logger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Test train RF-DETR-Mask using YAML config", add_help=True)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Override output directory from config"
    )
    parser.add_argument(
        "--pretrain_weights", type=str, help="Override pretrained weights path from config"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Override batch size from config"
    )
    parser.add_argument(
        "--epochs", type=int, help="Override number of epochs from config"
    )
    parser.add_argument(
        "--resume", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Only run evaluation"
    )
    parser.add_argument(
        "--seed", type=int, help="Override random seed from config"
    )
    parser.add_argument(
        "--max_samples", type=int, default=50, help="Max number of samples to use for quick test"
    )
    return parser


def main(args):
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    logger.info(f"git:\n  {utils.get_sha()}\n")
    logger.info(f"Arguments: {args}")

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build the model
    model = build_model(args)

    # Build criterion and postprocessors
    criterion, postprocessors = build_criterion_and_postprocessors(args)
    model.to(device)

    # Wrap with Distributed Data Parallel if needed
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {n_parameters}")

    # Build optimizer
    param_dicts = get_param_dict(args, model_without_ddp)
    
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
        param_dicts, lr=args.lr, weight_decay=args.weight_decay, fused=False, eps=1e-4
    )

    # Build learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop], gamma=0.1)

    # Build datasets
    dataset_train = build_dataset(image_set="train", args=args, resolution=args.resolution)
    dataset_val = build_dataset(image_set="val", args=args, resolution=args.resolution)

    # For quick testing, use only a subset of the data
    max_samples = getattr(args, "max_samples", 50)
    logger.info(f"Using only {max_samples} samples for quick testing")
    
    # Create subsets
    indices_train = list(range(min(max_samples, len(dataset_train))))
    indices_val = list(range(min(max_samples // 2, len(dataset_val))))
    
    dataset_train = Subset(dataset_train, indices_train)
    dataset_val = Subset(dataset_val, indices_val)

    # Build data samplers
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Build data loaders
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    # Create model EMA if needed
    ema_m = getattr(args, "ema_decay", None)
    ema = ModelEma(model_without_ddp, ema_m) if ema_m and args.use_ema else None

    # Get COCO API for evaluation
    base_ds = get_coco_api_from_dataset(dataset_val)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and "lr_scheduler" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
        if ema and "ema" in checkpoint:
            ema.ema.load_state_dict(checkpoint["ema"])

    # Evaluation only mode
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        logger.info("\nEvaluation Results:")
        for k, v in test_stats.items():
            if isinstance(v, list):
                logger.info(f"{k}: {v}")
            else:
                logger.info(f"{k}: {v}")
        return

    # Create metrics holders
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)

    # Training loop
    logger.info("Starting training")
    start_time = time.time()
    
    # Limit epochs for quick test
    args.epochs = min(args.epochs, 2)
    
    for epoch in range(getattr(args, "start_epoch", 0), args.epochs):
        # Set epoch for sampler
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # Calculate number of training steps per epoch
        num_training_steps_per_epoch = len(data_loader_train)

        # Create empty callbacks dictionary
        from collections import defaultdict
        callbacks = defaultdict(list)

        # Add validation callback if steps_per_validation is specified
        if args.steps_per_validation > 0:
            total_steps = 0

            def validation_callback(callback_dict):
                nonlocal total_steps
                total_steps += 1

                if total_steps % args.steps_per_validation == 0:
                    logger.info(f"Running validation at step {total_steps}")
                    model = callback_dict["model"]

                    # Run evaluation
                    test_stats, coco_evaluator = evaluate(
                        model, criterion, postprocessors, data_loader_val, base_ds, device, args
                    )

                    # Log validation results
                    logger.info(f"Validation at step {total_steps}:")
                    for k, v in test_stats.items():
                        if isinstance(v, (list, tuple)) and len(v) > 0:
                            logger.info(f"  {k}: {v[0]:.4f}")
                        else:
                            logger.info(f"  {k}: {v}")

                    # Put model back to training mode
                    model.train()

            callbacks["on_train_batch_start"].append(validation_callback)

        # Train for one epoch
        train_stats = train_one_epoch(
            model,
            criterion,
            lr_scheduler,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.batch_size,
            args.clip_max_norm,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            vit_encoder_num_layers=getattr(args, "vit_encoder_num_layers", 12),
            args=args,
            ema_m=ema,
            callbacks=callbacks,
        )

        # Update LR scheduler
        lr_scheduler.step()

        # Save checkpoint periodically
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = output_dir / f"checkpoint_{epoch:04d}.pth"
            checkpoint_dict = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if ema:
                checkpoint_dict["ema"] = ema.ema.state_dict()
            utils.save_on_master(checkpoint_dict, checkpoint_path)

        # Evaluate
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args
        )

        # Check if this is the best model so far
        map_regular = test_stats["coco_eval_bbox"][0]
        map_mask = test_stats.get("coco_eval_masks", [0])[0] if args.masks else 0

        # Update best metrics
        _is_best = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _is_best:
            checkpoint_path = output_dir / "checkpoint_best.pth"
            checkpoint_dict = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if ema:
                checkpoint_dict["ema"] = ema.ema.state_dict()
            utils.save_on_master(checkpoint_dict, checkpoint_path)

        # Log statistics
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training completed in {total_time_str}")


if __name__ == "__main__":
    parser = get_args_parser()
    cmd_args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(cmd_args.config)
    
    # Convert to args for backward compatibility
    args = config.to_args()
    
    # Override configuration with command line arguments if provided
    if cmd_args.output_dir:
        args.output_dir = cmd_args.output_dir
    if cmd_args.pretrain_weights:
        args.pretrain_weights = cmd_args.pretrain_weights
    if cmd_args.batch_size:
        args.batch_size = cmd_args.batch_size
    if cmd_args.epochs:
        args.epochs = cmd_args.epochs
    if cmd_args.resume:
        args.resume = cmd_args.resume
    if cmd_args.eval:
        args.eval = True
    if cmd_args.seed:
        args.seed = cmd_args.seed
    
    # Add max_samples for quick testing
    args.max_samples = cmd_args.max_samples
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the configuration to the output directory
    config_path = output_dir / "config_used.yaml"
    with open(config_path, "w") as f:
        import yaml
        # Convert the original config to yaml and save
        config_dict = {
            "model": config.model.model_dump(),
            "training": config.training.model_dump(),
            "dataset": config.dataset.model_dump(),
            "mask": config.mask.model_dump(),
            "other": config.other.model_dump(),
        }
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Call the main training function with the args
    main(args)