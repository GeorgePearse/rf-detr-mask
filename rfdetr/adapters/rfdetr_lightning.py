# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
PyTorch Lightning modules for RF-DETR-Mask training.
"""

from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.amp

import rfdetr.util.misc as utils
from rfdetr.adapters.config import TrainingConfig, ModelConfig
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.logging_config import get_logger
from rfdetr.util.utils import ModelEma

logger = get_logger(__name__)


class RFDETRLightningModule(pl.LightningModule):
    """Lightning module for RF-DETR training using iteration-based approach."""

    def __init__(
        self,
        num_classes: int,
        training_config: TrainingConfig,
        model_config: ModelConfig,
    ):
        """Initialize the RF-DETR Lightning Module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
        super().__init__()
        self.training_config = training_config
        self.model_config = model_config

        self.save_hyperparameters()
        self.ema_decay = training_config.ema_decay
        self.use_ema = training_config.use_ema
        # Build model, criterion, and postprocessors
        self.model = build_model(model_config)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(training_config)
        self.ema = ModelEma(self.model, self.ema_decay) if self.ema_decay and self.use_ema else None

        # Track metrics
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []

        # Track best metrics
        self.best_map = 0.0

        # Use automatic optimization to work with gradient clipping
        self.automatic_optimization = True

        self.output_dir = training_config.output_dir
        self.export_onnx = training_config.export_onnx
        self.export_torch = training_config.export_torch
        self.max_steps = training_config.max_steps
        self.val_frequency = training_config.val_frequency

        # Setup export directories
        self.export_dir = Path(self.output_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Setup autocast for mixed precision training
        self._setup_autocast_args()

    def _setup_autocast_args(self):
        """Set up arguments for autocast (mixed precision training)."""
        # Use full precision (float32) by default - avoid cdist_cuda issues
        self.autocast_args = {
            "device_type": "cuda" if torch.cuda.is_available() else "cpu",
            "enabled": True,
            "dtype": torch.float32,
        }

    def forward(self, samples, targets=None):
        """Forward pass through the model."""
        return self.model(samples, targets)

    def training_step(self, batch, batch_idx):
        """Training step logic."""
        samples, targets = batch

        # Move batch to device
        samples = samples.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass with autocast
        with torch.autocast(**self.autocast_args):
            outputs = self.model(samples, targets)
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        # Log metrics
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        # Log metrics - iteration-based logging
        self.log(
            "train/loss",
            losses_reduced_scaled,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in loss_dict_reduced_scaled.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            "train/class_error",
            loss_dict_reduced["class_error"],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Get optimizer for LR logging
        optimizer = self.optimizers()
        self.log(
            "train/lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Update EMA model if available
        if self.ema is not None:
            self.ema.update(self.model)

        # Store metrics for epoch-end callback
        train_metrics = {
            "loss": losses_reduced_scaled.item(),
            "class_error": loss_dict_reduced["class_error"].item(),
            "lr": optimizer.param_groups[0]["lr"],
            **{k: v.item() for k, v in loss_dict_reduced_scaled.items()},
            **{k: v.item() for k, v in loss_dict_reduced_unscaled.items()},
        }
        self.train_metrics.append(train_metrics)

        return losses

    def validation_step(self, batch, batch_idx):
        """Validation step logic."""
        try:
            samples, targets = batch

            # Move to device
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Determine which model to evaluate (EMA or regular)
            model_to_eval = self.model
            # Forward pass
            with torch.autocast(**self.autocast_args):
                outputs = model_to_eval(samples)

            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict

            # Process results for COCO evaluation
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = self.postprocessors["bbox"](outputs, orig_target_sizes)
            res = {target["image_id"].item(): output for target, output in zip(targets, results)}

            # Log reduced loss
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
            }
            loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            # Log metrics - iteration-based validation
            self.log(
                "val/loss",
                losses_reduced_scaled,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )
            for k, v in loss_dict_reduced_scaled.items():
                self.log(f"val/{k}", v, on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                "val/class_error",
                loss_dict_reduced["class_error"],
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

            # Store metrics and results for validation_epoch_end
            val_metrics = {
                "loss": losses_reduced_scaled.item(),
                "class_error": loss_dict_reduced["class_error"].item(),
                **{k: v.item() for k, v in loss_dict_reduced_scaled.items()},
                **{k: v.item() for k, v in loss_dict_reduced_unscaled.items()},
            }
            self.val_metrics.append(val_metrics)

            return {"metrics": val_metrics, "results": res, "targets": targets}
        except Exception as e:
            print(f"Error in validation step: {e}")
            # Return minimal output to keep the validation process from crashing
            return None

    def on_validation_epoch_start(self,):
        """Set up validation epoch and export model."""
        # Reset metrics
        pass

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Process validation batch results."""
        pass

    def on_validation_epoch_end(self):
        """Process validation epoch results."""
        # Default metric values
        pass

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler for iteration-based training."""
        # Get parameters from config for optimizer

        return {
            "optimizer": torch.optim.AdamW(lr=self.model_config.lr, weight_decay=0.0001),
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(),
                "interval": "step",
                "frequency": 1,
            },
        }
