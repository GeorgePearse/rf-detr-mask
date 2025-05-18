#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Training Script Using YAML Configuration (TEST VERSION)
# This version uses a small subset of the data for quicker testing
# ------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser("Train RF-DETR-Mask on small subset", add_help=True)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Override output directory from config"
    )
    parser.add_argument(
        "--max_samples", type=int, default=20, help="Max number of samples to use for training"
    )
    parser.add_argument(
        "--max_val_samples", type=int, default=10, help="Max number of samples to use for validation"
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=10, help="Max steps per epoch"
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

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {n_parameters}")

    # Build optimizer
    param_dicts = get_param_dict(args, model_without_ddp)
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    # Build learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.lr_drop], gamma=0.1)

    # Build datasets
    dataset_train = build_dataset(image_set="train", args=args, resolution=args.resolution)
    dataset_val = build_dataset(image_set="val", args=args, resolution=args.resolution)
    
    # Use only a small subset for quick testing
    max_samples = getattr(args, "max_samples", 20)
    max_val_samples = getattr(args, "max_val_samples", 10)
    
    logger.info(f"Using only {max_samples} training samples and {max_val_samples} validation samples")
    
    # Create subsets
    indices_train = list(range(min(max_samples, len(dataset_train))))
    indices_val = list(range(min(max_val_samples, len(dataset_val))))
    
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

    # Create model EMA if needed - simplified for testing
    use_ema = getattr(args, "use_ema", False)
    ema = None
    
    # Get COCO API for evaluation
    base_ds = get_coco_api_from_dataset(dataset_val)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metrics holders
    best_map_holder = BestMetricHolder(use_ema=use_ema)

    # Training loop
    logger.info("Starting training")
    start_time = time.time()
    
    # Limit steps per epoch for quicker testing
    steps_per_epoch = getattr(args, "steps_per_epoch", 10)
    
    for epoch in range(getattr(args, "start_epoch", 0), args.epochs):
        # Set epoch for sampler
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # Train for one epoch or specified number of steps
        train_stats = train_for_steps(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.batch_size,
            args.clip_max_norm,
            steps_per_epoch=steps_per_epoch,
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
            utils.save_on_master(checkpoint_dict, checkpoint_path)

        # Evaluate
        logger.info("Starting validation...")
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


def train_for_steps(
    model, criterion, data_loader, optimizer, device, epoch, batch_size, clip_max_norm, 
    steps_per_epoch=10
):
    model.train()
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    header = f"Epoch: [{epoch}]"
    print_freq = 10
    
    step_count = 0
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Stop after specified number of steps
        step_count += 1
        if step_count > steps_per_epoch:
            break
        
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        loss_value = losses_reduced_scaled.item()
        
        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training")
            logger.info(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            
        optimizer.step()
            
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:" + str(metric_logger))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    # Handle missing import
    import math
    
    parser = get_args_parser()
    cmd_args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(cmd_args.config)
    
    # Convert to args for backward compatibility
    args = config.to_args()
    
    # Override configuration with command line arguments if provided
    if cmd_args.output_dir:
        args.output_dir = cmd_args.output_dir
    
    # Add our custom parameters for quick testing
    args.max_samples = cmd_args.max_samples
    args.max_val_samples = cmd_args.max_val_samples
    args.steps_per_epoch = cmd_args.steps_per_epoch
    
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