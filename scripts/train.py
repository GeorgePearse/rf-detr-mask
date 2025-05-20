#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Training Script with Iteration-based Training
# ------------------------------------------------------------------------

"""
Updated training script for RF-DETR using iteration-based training with pydantic config.
"""
import json
import datetime
import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
    EarlyStopping
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from rfdetr.config import load_config
from rfdetr.rfdetr_lightning import RFDETRLightningModule
from rfdetr.data_module import RFDETRDataModule
from rfdetr.util.logging_config import get_logger
from rfdetr.config import ModelConfig

logger = get_logger(__name__)


def get_number_of_classes(config: ModelConfig) -> int:
    coco_path = Path(config.dataset.coco_path)
    annotation_file = (
        coco_path / config.dataset.coco_train
        if not Path(config.dataset.coco_train).is_absolute()
        else Path(config.dataset.coco_train)
    )

    # Get number of classes from annotation file - fail if can't determine
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with open(annotation_file) as f:
        annotations = json.load(f)
        categories = annotations.get("categories", [])
        if not categories:
            raise ValueError(f"No categories found in annotation file: {annotation_file}")

        num_classes = len(categories)
        logger.info(f"Detected {num_classes} classes from annotation file")
    return num_classes


def main(config_path: str = "configs/default.yaml"):
    """Main training function."""
    # Load configuration from YAML file
    config = load_config(config_path)

    # Determine the number of classes from annotation file
    num_classes = get_number_of_classes(config)

    # Set the num_classes in config
    config.set_num_classes(num_classes)

    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for reference
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = output_dir / f"checkpoints_{timestamp}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_file = checkpoints_dir / "config.yaml"
    config.to_yaml(config_file)
    logger.info(f"Configuration saved to {config_file}")

    # Fix the seed for reproducibility
    seed = config.other.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set a shorter run
    if config.dataset.test_mode:
        # Setting these as variables to use in the Trainer, not in config
        max_steps = 10
        val_frequency = 5
        checkpoint_frequency = 5

    # Create Lightning Module and Data Module
    model = RFDETRLightningModule(config)
    data_module = RFDETRDataModule(config)

    # Setup logging
    loggers = [
        TensorBoardLogger(
            save_dir=config.training.output_dir, name="lightning_logs"
        )
    ]

    # Always add CSV logger
    csv_logger = CSVLogger(save_dir=config.training.output_dir, name="csv_logs")
    loggers.append(csv_logger)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="checkpoint_step_{step:06d}-{val_mAP:.4f}",
            monitor="val_mAP",  # Changed from val/mAP to val_mAP
            mode="max",
            save_top_k=3,
            save_last=True,
            every_n_train_steps=checkpoint_frequency,
        ),
        # Also save best model checkpoint
        ModelCheckpoint(
            dirpath=output_dir,
            filename="checkpoint_best",
            monitor="val_mAP",  # Changed from val/mAP to val_mAP
            mode="max",
            save_top_k=1,
            save_last=False,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="step"),
        # Progress bar
        TQDMProgressBar(refresh_rate=1),
        EarlyStopping(
            monitor="val_mAP",
            mode="max",
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
        )
    ]

    # Setup strategy for distributed training
    strategy = "auto"  # Default to auto strategy
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    # Set default values for training parameters
    # Initialize max_steps and val_frequency with default values
    max_steps = 1000
    val_frequency = (
        config.other.steps_per_validation if config.other.steps_per_validation > 0 else 200
    )
    # Get checkpoint_frequency from config or use default value of 10
    checkpoint_frequency = getattr(config.training, "checkpoint_interval", 10)

    # Override if in test mode
    if config.dataset.test_mode:
        max_steps = 10
        val_frequency = 5
        checkpoint_frequency = 5

    trainer = pl.Trainer(
        max_steps=max_steps,
        max_epochs=None,  # No epoch limit, only step limit
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        precision="32-true",  # Always use full precision (32-bit) to avoid cdist_cuda issues
        gradient_clip_val=config.other.clip_max_norm if config.other.clip_max_norm > 0 else None,
        accumulate_grad_batches=config.training.grad_accum_steps,
        log_every_n_steps=1,
        default_root_dir=config.training.output_dir,
        val_check_interval=val_frequency,
        accelerator="cuda",
        devices=torch.cuda.device_count(),
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
    main()
