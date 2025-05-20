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
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.logging_config import get_logger

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
        self.training_config: TrainingConfig = training_config
        self.model_config: ModelConfig = model_config

        self.save_hyperparameters()
        # Build model, criterion, and postprocessors
        self.model = build_model(model_config)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(training_config)

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
        outputs = self.model(samples, targets)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        # Log metrics
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        return losses_reduced_scaled

    def validation_step(self, batch, batch_idx):
        pass
            # 

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.0001)
        return optimizer
