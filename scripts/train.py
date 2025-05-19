#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Training Script with Iteration-based Training
# ------------------------------------------------------------------------

"""
Updated training script for RF-DETR using iteration-based training with pydantic config.
"""

import argparse
import datetime
import os
import random
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

import rfdetr.util.misc as utils
from rfdetr.hooks import ONNXCheckpointHook
from rfdetr.lightning_module import RFDETRDataModule, RFDETRLightningModule
from rfdetr.training_config import TrainingConfig
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RF-DETR with iteration-based approach")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    args = parser.parse_args()

    # Load configuration
    config = TrainingConfig.from_yaml(args.config)

    # Apply command line overrides
    if args.output_dir:
        config.output_dir = args.output_dir

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for reference
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = output_dir / f"checkpoints_{timestamp}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_file = checkpoints_dir / "config.yaml"
    config.to_yaml(config_file)
    logger.info(f"Configuration saved to {config_file}")

    # Fix the seed for reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Convert config to args dict for compatibility with existing code
    args_dict = config.to_args_dict()

    # For test_limit, adjust parameters for faster evaluation
    if config.test_limit is not None and config.test_limit > 0:
        config.num_queries = min(100, config.num_queries)
        config.hidden_dim = min(128, config.hidden_dim)
        config.dec_layers = min(3, config.dec_layers)
        config.batch_size = 1

    # Create Lightning Module and Data Module
    model = RFDETRLightningModule(args_dict)
    data_module = RFDETRDataModule(args_dict)

    # Setup logging
    loggers = []
    if config.tensorboard:
        try:
            tb_logger = TensorBoardLogger(save_dir=config.output_dir, name="lightning_logs")
            loggers.append(tb_logger)
        except ModuleNotFoundError:
            logger.warning("TensorBoard not installed. Skipping TensorBoard logging.")
            config.tensorboard = False

    if config.wandb:
        try:
            wandb_logger = WandbLogger(
                project=config.project or "rfdetr-mask",
                name=config.run,
                save_dir=config.output_dir,
                log_model="all",
            )
            loggers.append(wandb_logger)
        except ModuleNotFoundError:
            logger.warning("Weights & Biases not installed. Skipping W&B logging.")
            config.wandb = False

    # Always add CSV logger
    csv_logger = CSVLogger(save_dir=config.output_dir, name="csv_logs")
    loggers.append(csv_logger)

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="checkpoint_step_{step:06d}-{val/mAP:.4f}",
        monitor="val/mAP",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=getattr(
            config,
            "eval_save_frequency",
            getattr(config, "checkpoint_frequency", getattr(config, "val_frequency", 200)),
        ),
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

    # ONNX and torch checkpoint hook
    if config.export_onnx or config.export_torch:
        # Create export directory
        export_dir = Path(config.output_dir) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        onnx_hook = ONNXCheckpointHook(
            export_dir=export_dir,
            export_onnx=config.export_onnx,
            export_torch=config.export_torch,
            simplify_onnx=config.simplify_onnx,
            export_frequency=getattr(
                config,
                "eval_save_frequency",
                getattr(config, "checkpoint_frequency", getattr(config, "val_frequency", 200)),
            ),
            input_shape=(config.resolution, config.resolution),
            opset_version=config.opset_version,
        )
        callbacks.append(onnx_hook)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Progress bar
    progress_bar = TQDMProgressBar(refresh_rate=1)
    callbacks.append(progress_bar)

    # Early stopping if enabled
    if config.early_stopping:
        from lightning.pytorch.callbacks import EarlyStopping

        early_stopping_cb = EarlyStopping(
            monitor="val/mAP",
            mode="max",
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )
        callbacks.append(early_stopping_cb)

    # Setup strategy for distributed training
    strategy = "auto"  # Default to auto strategy
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Trainer
    accelerator = "cpu" if config.device == "cpu" else "auto"

    trainer = pl.Trainer(
        max_steps=config.max_steps,
        max_epochs=None,  # No epoch limit, only step limit
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        precision="32-true",  # Always use full precision (32-bit) to avoid cdist_cuda issues
        gradient_clip_val=config.clip_max_norm if config.clip_max_norm > 0 else None,
        accumulate_grad_batches=config.grad_accum_steps,
        log_every_n_steps=1,
        default_root_dir=config.output_dir,
        val_check_interval=getattr(
            config,
            "eval_save_frequency",
            getattr(config, "val_frequency", getattr(config, "checkpoint_frequency", 200)),
        ),
        accelerator=accelerator,
        devices=1,
    )

    # Resume from checkpoint if specified
    resume_path = os.environ.get("RESUME_CHECKPOINT", None)
    if resume_path:
        # Just evaluate if evaluation mode is requested
        if os.environ.get("EVAL_ONLY", "0").lower() in ["1", "true"]:
            logger.info(f"Evaluating model from checkpoint: {resume_path}")
            trainer.validate(model, datamodule=data_module, ckpt_path=resume_path)
            return

        # Otherwise resume training
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)
    else:
        # Just evaluate if evaluation mode is requested
        if os.environ.get("EVAL_ONLY", "0").lower() in ["1", "true"]:
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

    # Calculate and log total training time
    if hasattr(trainer, "callback_metrics") and "val/mAP" in trainer.callback_metrics:
        logger.info(f"Final mAP: {trainer.callback_metrics['val/mAP']:.4f}")

    # Log total parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {n_parameters}")

    total_batches = (
        trainer.fit_loop.epoch_loop.total_batch_idx
        if hasattr(trainer.fit_loop.epoch_loop, "total_batch_idx")
        else 0
    )
    logger.info(f"Training completed in {total_batches} batches")
    return trainer.callback_metrics.get("val/mAP", 0.0)


if __name__ == "__main__":
    # Initialize logging
    logger.info(f"git:\n  {utils.get_sha()}\n")
    logger.info("Starting training with PyTorch Lightning - Iteration-based approach")

    # Make sure GPU memory is properly cleaned up before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run main training function
    main()
