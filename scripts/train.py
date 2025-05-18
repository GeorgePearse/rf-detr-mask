#!/usr/bin/env python
# ------------------------------------------------------------------------
# RF-DETR-Mask Training Script for CMR Dataset using PyTorch Lightning
# ------------------------------------------------------------------------

"""
Script to train RF-DETR with mask head on CMR segmentation data using PyTorch Lightning.
Adapted from the original RF-DETR training script to work with CMR instance segmentation dataset.
"""

import argparse
import datetime
import json
import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import rfdetr.util.misc as utils
from rfdetr.lightning_module import RFDETRDataModule, RFDETRLightningModule
from rfdetr.util.logging_config import get_logger

logger = get_logger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Train RF-DETR-Mask on CMR segmentation", add_help=True)

    # Dataset parameters - using CMR dataset paths
    parser.add_argument("--dataset", default="coco", type=str, help="Dataset name")
    parser.add_argument("--dataset_file", default="coco", type=str, help="Dataset file name")
    parser.add_argument(
        "--coco_path",
        type=str,
        default="/home/georgepearse/data/cmr/annotations",
        help="Path to the annotations directory",
    )
    parser.add_argument(
        "--coco_train",
        type=str,
        default="2025-05-15_12:38:23.077836_train_ordered.json",
        help="Training annotation file name",
    )
    parser.add_argument(
        "--coco_val",
        type=str,
        default="2025-05-15_12:38:38.270134_val_ordered.json",
        help="Validation annotation file name",
    )
    parser.add_argument(
        "--coco_img_path",
        type=str,
        default="/home/georgepearse/data/images",
        help="Path to the images directory",
    )
    parser.add_argument(
        "--output_dir", default="output_cmr_segmentation", help="Path to save logs and checkpoints"
    )

    # Training parameters
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument(
        "--lr_encoder", default=5e-6, type=float, help="Learning rate of the encoder"
    )
    parser.add_argument(
        "--lr_projector", default=5e-6, type=float, help="Learning rate of the projector"
    )
    parser.add_argument(
        "--lr_vit_layer_decay",
        default=1.0,
        type=float,
        help="Layer-wise learning rate decay for ViT",
    )
    parser.add_argument(
        "--lr_component_decay", default=0.9, type=float, help="Component-wise learning rate decay"
    )
    parser.add_argument("--lr_drop", default=50, type=int, help="lr_drop")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument(
        "--train_batch_size", default=1, type=int, help="Training batch size per device"
    )
    parser.add_argument(
        "--val_batch_size", default=1, type=int, help="Validation batch size per device"
    )
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train for")
    parser.add_argument(
        "--clip_max_norm", default=0.5, type=float, help="Gradient clipping max norm"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of gradient accumulation steps",
    )

    # Model parameters
    parser.add_argument(
        "--encoder", default="dinov2_small", type=str, help="Name of the transformer backbone"
    )
    parser.add_argument(
        "--pretrain_weights", type=str, default=None, help="Path to pretrained weights"
    )
    parser.add_argument(
        "--resolution",
        default=336,
        type=int,
        help="Input resolution to the encoder (must be divisible by 14 for DINOv2)",
    )
    parser.add_argument(
        "--set_loss", default="lw_detr", type=str, help="Type of loss for object detection matching"
    )
    parser.add_argument(
        "--set_cost_class", default=5, type=float, help="Class coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=2,
        type=float,
        help="Bounding box L1 coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou", default=1, type=float, help="giou coefficient in the matching cost"
    )
    parser.add_argument(
        "--loss_class_coef", default=4.5, type=float, help="coefficient for loss on classification"
    )
    parser.add_argument(
        "--loss_bbox_coef",
        default=2.0,
        type=float,
        help="coefficient for loss on bounding box regression",
    )
    parser.add_argument(
        "--loss_giou_coef", default=1, type=float, help="coefficient for loss on bounding box giou"
    )
    parser.add_argument(
        "--num_classes",
        default=69,
        type=int,  # CMR has 69 classes
        help="Number of classes",
    )
    parser.add_argument(
        "--masks",
        action="store_true",
        default=False,
        help="Train segmentation head for panoptic segmentation (disabled for testing)",
    )

    # Loss parameters for masks
    parser.add_argument(
        "--loss_mask_coef", default=1.0, type=float, help="coefficient for loss on mask prediction"
    )
    parser.add_argument(
        "--loss_dice_coef", default=1.0, type=float, help="coefficient for loss on dice loss"
    )

    # Other parameters
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of workers for data loading"
    )
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync_bn", action="store_true", help="Enable NVIDIA Apex or Torch native sync batchnorm."
    )
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument(
        "--steps_per_validation",
        default=0,
        type=int,
        help="Run validation every N steps during training. 0 means validate only at epoch end",
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="dropout rate that applies to transformer backbone",
    )
    parser.add_argument(
        "--bbox_reparam", default=True, type=bool, help="reparameterize bbox loss (CWH)"
    )
    parser.add_argument("--group_detr", default=1, type=int, help="Number of groups for group DETR")
    parser.add_argument(
        "--two_stage", default=True, type=bool, help="Use two-stage variant of DETR"
    )
    parser.add_argument(
        "--no_intermittent_layers",
        default=False,
        type=bool,
        help="Avoid computing intermediate decodings",
    )
    parser.add_argument("--use_fp16", default=True, type=bool, help="Use FP16 models (half)")
    parser.add_argument("--amp", action="store_true", help="use mixed precision")
    parser.add_argument("--square_resize", action="store_true", help="use square resize for images")
    parser.add_argument(
        "--square_resize_div_64",
        action="store_true",
        help="use square resize with dimensions divisible by 64",
    )
    parser.add_argument(
        "--test_limit",
        default=None,
        type=int,
        help="Limit dataset to first N samples for faster testing. If not specified, the full dataset is used.",
    )

    return parser


def main(args):
    # Fix the seed for reproducibility
    seed = (
        args.seed + utils.get_rank()
        if hasattr(args, "distributed") and args.distributed
        else args.seed
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration to JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = output_dir / f"checkpoints_{timestamp}"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_file = checkpoints_dir / "config.json"
    with open(config_file, "w") as f:
        config_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        json.dump(config_dict, f, indent=2)

    # For test_limit, adjust parameters for faster evaluation
    if getattr(args, "test_limit", None) is not None and args.test_limit > 0:
        # Use a smaller model for testing
        args.num_queries = min(100, args.num_queries)  # Reduce number of queries
        args.hidden_dim = min(128, args.hidden_dim)  # Smaller hidden dimension
        args.num_decoder_layers = min(3, args.num_decoder_layers)  # Fewer decoder layers
        args.dec_layers = min(3, args.dec_layers)

        # Don't disable masks - we need to test the mask functionality
        # args.masks = True  # Keep masks enabled

        # Use smaller batch size
        args.train_batch_size = 1
        args.val_batch_size = 1

    # Create Lightning Module and Data Module
    model = RFDETRLightningModule(args)
    data_module = RFDETRDataModule(args)

    # No need to override data loader as it's handled by the data module

    # Setup logging
    loggers = []
    use_tensorboard = getattr(args, "tensorboard", True)  # Default to True if not specified
    if use_tensorboard:
        try:
            tb_logger = TensorBoardLogger(save_dir=args.output_dir, name="lightning_logs")
            loggers.append(tb_logger)
        except ModuleNotFoundError:
            logger.warning("TensorBoard not installed. Skipping TensorBoard logging.")
            # Set tensorboard to False so future code doesn't try to use it
            args.tensorboard = False

    use_wandb = getattr(args, "wandb", False)  # Default to False if not specified
    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=getattr(args, "project", "rfdetr-mask"),
                name=getattr(args, "run", None),
                save_dir=args.output_dir,
                log_model="all",
            )
            loggers.append(wandb_logger)
        except ModuleNotFoundError:
            logger.warning("Weights & Biases not installed. Skipping W&B logging.")
            # Set wandb to False so future code doesn't try to use it
            args.wandb = False

    # Always add CSV logger
    csv_logger = CSVLogger(save_dir=args.output_dir, name="csv_logs")
    loggers.append(csv_logger)

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
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
    early_stopping = getattr(args, "early_stopping", False)
    if early_stopping:
        from lightning.pytorch.callbacks import EarlyStopping

        early_stopping_cb = EarlyStopping(
            monitor="val/mAP",
            mode="max",
            patience=getattr(args, "early_stopping_patience", 10),
            min_delta=getattr(args, "early_stopping_min_delta", 0.001),
        )
        callbacks.append(early_stopping_cb)

    # Setup strategy for distributed training
    strategy = "auto"  # Default to auto strategy
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    # Log steps per validation if applicable
    steps_per_val = getattr(args, "steps_per_validation", 0)
    val_check_interval = steps_per_val if steps_per_val > 0 else 1.0

    # Create Trainer
    accelerator = "cpu" if args.device == "cpu" else "auto"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        precision="16-mixed" if getattr(args, "amp", True) and accelerator != "cpu" else "32-true",
        gradient_clip_val=getattr(args, "clip_max_norm", 0.0)
        if getattr(args, "clip_max_norm", 0.0) > 0
        else None,
        accumulate_grad_batches=getattr(args, "gradient_accumulation_steps", 1),
        log_every_n_steps=10,
        default_root_dir=args.output_dir,
        val_check_interval=val_check_interval,
        accelerator=accelerator,
        devices=1,
    )

    # Resume from checkpoint if specified
    resume_path = getattr(args, "resume", None)
    if resume_path:
        # Just evaluate if eval flag is set
        if getattr(args, "eval", False):
            logger.info(f"Evaluating model from checkpoint: {resume_path}")
            trainer.validate(model, datamodule=data_module, ckpt_path=resume_path)
            return

        # Otherwise resume training
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)
    else:
        # Just evaluate if eval flag is set (using untrained model)
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
    parser = get_args_parser()
    args = parser.parse_args()

    # Process CLI arguments and add defaults

    # Set some defaults for compatibility
    args.focal_loss = True
    args.focal_alpha = 0.25
    args.focal_gamma = 2.0
    args.num_queries = 100  # Reduced from 900 for memory
    args.hidden_dim = 128  # Reduced from 256 for memory
    args.position_embedding_scale = None
    args.backbone_feature_layers = ["res2", "res3", "res4", "res5"]
    args.vit_encoder_num_layers = 12
    args.num_decoder_layers = 3  # Reduced from 6 for memory
    args.num_decoder_points = 4
    args.dec_layers = 3  # Reduced from 6 for memory

    # Add missing attributes for build_model
    args.pretrained_encoder = True  # Use pretrained encoder by default
    args.window_block_indexes = []  # Empty list for window block indexes
    args.drop_path = 0.1  # Default drop path rate
    args.out_feature_indexes = [2, 5, 8]  # Smaller output features (reduced from [3, 7, 11])
    args.projector_scale = [
        "P3",
        "P4",
        "P5",
    ]  # Default projector scale levels, let's try without P6
    args.use_cls_token = True  # Use CLS token
    args.position_embedding = "sine"  # Use sine position embedding
    args.freeze_encoder = False  # Don't freeze encoder by default
    args.layer_norm = True  # Use layer normalization
    args.rms_norm = False  # Don't use RMS normalization
    args.backbone_lora = False  # No LoRA for backbone
    args.force_no_pretrain = False  # Use pretrained weights
    args.gradient_checkpointing = False  # No gradient checkpointing by default
    args.encoder_only = False  # Use full DETR model, not just encoder
    args.backbone_only = False  # Use full model, not just backbone

    # Transformer parameters
    args.sa_nheads = 4  # Self-attention heads (reduced from 8)
    args.ca_nheads = 4  # Cross-attention heads (reduced from 8)
    args.dim_feedforward = 512  # Feedforward dimension (reduced from 2048)
    args.num_feature_levels = (
        3  # Number of feature levels for multi-scale (matching projector scale)
    )
    args.dec_n_points = 4  # Number of attention points for decoder
    args.lite_refpoint_refine = True  # Use lightweight reference point refinement for speed
    args.decoder_norm = "LN"  # Type of normalization in decoder (LN or Identity)

    # Additional model parameters
    args.aux_loss = True  # Use auxiliary loss in decoder layers

    # Map loss coefficient names to match build_criterion_and_postprocessors
    args.cls_loss_coef = args.loss_class_coef
    args.bbox_loss_coef = args.loss_bbox_coef
    args.giou_loss_coef = args.loss_giou_coef

    # Additional loss configuration
    args.use_varifocal_loss = False  # Not using varifocal loss
    args.mask_loss_coef = args.loss_mask_coef
    args.dice_loss_coef = args.loss_dice_coef
    args.use_position_supervised_loss = False  # Not using position supervised loss
    args.ia_bce_loss = False  # Not using instance-aware BCE loss
    args.sum_group_losses = False  # Don't sum group losses
    args.num_select = 300  # Number of top predictions to select in postprocessing

    # Data augmentation parameters
    args.multi_scale = False  # Don't use multi-scale training by default
    args.expanded_scales = [
        480,
        512,
        544,
        576,
        608,
        640,
        672,
        704,
        736,
        768,
        800,
    ]  # Multi-scale options
    args.square_resize = True  # Use square resize to ensure compatibility with Dinov2
    args.square_resize_div_64 = False  # Don't force div 64, we'll handle Dinov2 div 14

    # Lightning-specific parameters
    args.tensorboard = getattr(args, "tensorboard", True)  # Enable TensorBoard by default
    args.wandb = getattr(args, "wandb", False)  # Disable WandB by default
    args.early_stopping = getattr(
        args, "early_stopping", False
    )  # Disable early stopping by default
    args.batch_size = getattr(args, "train_batch_size", 2)  # Use train_batch_size for batch_size

    # Additional parameter mapping
    args.grad_accum_steps = args.gradient_accumulation_steps
    args.fp16_eval = args.use_fp16 and args.device != "cpu"  # Use FP16 for evaluation only on GPU

    # Logging
    logger.info(f"git:\n  {utils.get_sha()}\n")
    logger.info(f"Arguments: {args}")
    logger.info("Starting training with PyTorch Lightning")

    # Call the main training function
    main(args)
