#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Lightning Training Script Using YAML Configuration
# ------------------------------------------------------------------------

"""
Script to train RF-DETR with mask head using PyTorch Lightning and YAML configuration.
This script replaces the traditional training loop with PyTorch Lightning's Trainer.
"""

import argparse
import random
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

sys.path.insert(0, str(Path(__file__).parent.parent))

from rfdetr.config_utils import load_config
from rfdetr.lightning_module import RFDETRDataModule, RFDETRLightningModule
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Train RF-DETR-Mask using Lightning", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--output_dir", type=str, help="Override output directory from config")
    parser.add_argument(
        "--pretrain_weights", type=str, help="Override pretrained weights path from config"
    )
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    parser.add_argument("--epochs", type=int, help="Override number of epochs from config")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main(args):
    # Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration to the output directory
    config_path = output_dir / "config_used.yaml"
    with open(config_path, "w") as f:
        import yaml

        # Convert the original config to yaml and save
        config_dict = {
            "model": args.model_config.model_dump(),
            "training": args.training_config.model_dump(),
            "dataset": args.dataset_config.model_dump(),
            "mask": args.mask_config.model_dump(),
            "other": args.other_config.model_dump(),
        }
        yaml.dump(config_dict, f, default_flow_style=False)

    # Copy common attributes from config to match lightning module expected values
    args.batch_size = getattr(args, "train_batch_size", getattr(args, "batch_size", 2))
    args.grad_accum_steps = getattr(
        args, "gradient_accumulation_steps", getattr(args, "grad_accum_steps", 1)
    )

    # Create Lightning Module and Data Module
    model = RFDETRLightningModule(args)
    data_module = RFDETRDataModule(args)

    # Setup logging
    loggers = []
    if getattr(args, "tensorboard", True):
        tb_logger = TensorBoardLogger(save_dir=args.output_dir, name="lightning_logs")
        loggers.append(tb_logger)

    if getattr(args, "wandb", False):
        wandb_logger = WandbLogger(
            project=getattr(args, "project", "rfdetr-mask"),
            name=getattr(args, "run", None),
            save_dir=args.output_dir,
            log_model="all",
        )
        loggers.append(wandb_logger)

    # Always add CSV logger
    csv_logger = CSVLogger(save_dir=args.output_dir, name="csv_logs")
    loggers.append(csv_logger)

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="checkpoint_{epoch:04d}-{val/mAP:.4f}",
        monitor="val/mAP",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=getattr(args, "checkpoint_interval", 10),
    )
    callbacks.append(checkpoint_callback)

    # Also save best model checkpoint
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="checkpoint_best",
        monitor="val/mAP",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    callbacks.append(best_checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Progress bar
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks.append(progress_bar)

    # Early stopping if enabled
    if getattr(args, "early_stopping", False):
        from lightning.pytorch.callbacks import EarlyStopping

        early_stopping = EarlyStopping(
            monitor="val/mAP",
            mode="max",
            patience=getattr(args, "early_stopping_patience", 10),
            min_delta=getattr(args, "early_stopping_min_delta", 0.001),
        )
        callbacks.append(early_stopping)

    # Setup strategy for distributed training
    strategy = (
        DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
        if torch.cuda.device_count() > 1
        else None
    )

    # Create Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        precision="16-mixed" if getattr(args, "amp", True) else "32-true",
        gradient_clip_val=getattr(args, "clip_max_norm", 0.0)
        if getattr(args, "clip_max_norm", 0.0) > 0
        else None,
        accumulate_grad_batches=getattr(args, "grad_accum_steps", 1),
        log_every_n_steps=10,
        deterministic=False,  # For performance reasons
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=args.output_dir,
        check_val_every_n_epoch=1,
    )

    # Resume from checkpoint if provided
    if args.resume:
        # Just evaluate if eval flag is set
        if getattr(args, "eval", False):
            logger.info(f"Evaluating model from checkpoint: {args.resume}")
            trainer.validate(model, datamodule=data_module, ckpt_path=args.resume)
            return

        # Otherwise resume training
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    else:
        # Just evaluate if eval flag is set (using untrained model, probably not useful)
        if getattr(args, "eval", False):
            logger.info("Evaluating untrained model")
            trainer.validate(model, datamodule=data_module)
            return

        # Start training from scratch
        logger.info("Starting training from scratch")
        trainer.fit(model, datamodule=data_module)

    # Log best model and save final checkpoint
    if checkpoint_callback.best_model_path:
        logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
        logger.info(f"Best mAP: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    parser = get_args_parser()
    cmd_args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(cmd_args.config)

    # Convert to args for backward compatibility
    args = config.to_args()

    # Save original configuration for later usage
    args.model_config = config.model
    args.training_config = config.training
    args.dataset_config = config.dataset
    args.mask_config = config.mask
    args.other_config = config.other

    # Override configuration with command line arguments if provided
    if cmd_args.output_dir:
        args.output_dir = cmd_args.output_dir
    if cmd_args.pretrain_weights:
        args.pretrain_weights = cmd_args.pretrain_weights
    if cmd_args.batch_size:
        args.batch_size = cmd_args.batch_size
        args.train_batch_size = cmd_args.batch_size
    if cmd_args.epochs:
        args.epochs = cmd_args.epochs
    if cmd_args.resume:
        args.resume = cmd_args.resume
    if cmd_args.eval:
        args.eval = True

    args.seed = cmd_args.seed

    # Add attributes needed for Lightning compatibility
    args.masks = args.mask_config.enabled
    args.loss_mask_coef = args.mask_config.loss_mask_coef
    args.loss_dice_coef = args.mask_config.loss_dice_coef

    # Copy more attributes for consistency
    args.device = args.model_config.device
    args.clip_max_norm = args.other_config.clip_max_norm
    args.resolution = args.model_config.resolution

    # Copy training parameters
    args.use_ema = args.training_config.use_ema
    args.ema_decay = args.training_config.ema_decay
    args.tensorboard = args.training_config.tensorboard
    args.wandb = args.training_config.wandb
    args.early_stopping = args.training_config.early_stopping
    args.early_stopping_patience = args.training_config.early_stopping_patience
    args.early_stopping_min_delta = args.training_config.early_stopping_min_delta

    # Set up dataset paths
    args.coco_path = args.dataset_config.coco_path
    args.coco_train = args.dataset_config.coco_train
    args.coco_val = args.dataset_config.coco_val
    args.coco_img_path = args.dataset_config.coco_img_path

    # Print training configuration
    logger.info("Training RF-DETR-Mask with Lightning")
    logger.info(f"Config file: {cmd_args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Grad accumulation steps: {args.grad_accum_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Using masks: {args.masks}")

    # Call the main training function with the args
    main(args)
