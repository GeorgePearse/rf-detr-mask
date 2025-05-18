# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
PyTorch Lightning modules for RF-DETR-Mask training.
"""

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import ModelEma


class RFDETRLightningModule(pl.LightningModule):
    """Lightning module for RF-DETR training."""

    def __init__(self, args):
        """Initialize the RF-DETR Lightning Module.

        Args:
            args: Configuration arguments (can be from args namespace or config)
        """
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # Build model, criterion, and postprocessors
        self.model = build_model(args)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(args)

        # Setup EMA if enabled
        self.ema_decay = getattr(args, "ema_decay", None)
        self.ema = (
            ModelEma(self.model, self.ema_decay)
            if self.ema_decay and getattr(args, "use_ema", True)
            else None
        )

        # Track metrics
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []

        # Setup autocast args for mixed precision training
        self._setup_autocast_args()

        # Track best metrics
        self.best_map = 0.0

        # Setup automatic optimization
        self.automatic_optimization = (
            False  # We'll handle optimization manually for gradient accumulation
        )

    def _setup_autocast_args(self):
        """Set up arguments for autocast (mixed precision training)."""
        # Prefer bfloat16 if available, otherwise use float16
        import torch
        self._dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        try:
            # Check if torch.amp is available
            import torch.amp

            self.amp_backend = "torch"
        except ImportError:
            # Fall back to cuda.amp if needed
            self.amp_backend = "cuda"

        if self.amp_backend == "torch":
            self.autocast_args = {
                "device_type": "cuda" if torch.cuda.is_available() else "cpu",
                "enabled": getattr(self.args, "amp", True),
                "dtype": self._dtype,
            }
        else:
            self.autocast_args = {
                "enabled": getattr(self.args, "amp", True),
                "dtype": self._dtype,
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

        # Get optimizers
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        # Zero gradients for every batch_size // grad_accum_steps
        grad_accum_steps = getattr(self.args, "grad_accum_steps", 1)
        batch_size = len(samples.tensors)
        sub_batch_size = batch_size // grad_accum_steps

        # Initialize loss for logging
        total_loss = 0.0
        loss_dict_for_logging = None

        # Process each sub-batch with gradient accumulation
        for i in range(grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size

            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = utils.NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_targets = targets[start_idx:final_idx]

            # Forward pass with autocast
            with torch.autocast(**self.autocast_args):
                outputs = self.model(new_samples, new_targets)
                loss_dict = self.criterion(outputs, new_targets)
                weight_dict = self.criterion.weight_dict
                losses = sum(
                    (1 / grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict
                    if k in weight_dict
                )

            # Backward pass
            self.manual_backward(losses)
            total_loss += losses

            # Save loss dict for the last sub-batch for logging
            if i == grad_accum_steps - 1:
                loss_dict_for_logging = loss_dict

        # Clip gradients if needed
        max_norm = getattr(self.args, "clip_max_norm", 0)
        if max_norm > 0:
            self.clip_gradients(optimizer, gradient_clip_val=max_norm)

        # Step optimizer and scheduler
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # Update EMA model if available
        if self.ema is not None:
            self.ema.update(self.model)

        # Log metrics
        loss_dict_reduced = utils.reduce_dict(loss_dict_for_logging)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        # Log metrics
        self.log(
            "train/loss",
            losses_reduced_scaled,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in loss_dict_reduced_scaled.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "train/class_error",
            loss_dict_reduced["class_error"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Store metrics for epoch-end callback
        train_metrics = {
            "loss": losses_reduced_scaled.item(),
            "class_error": loss_dict_reduced["class_error"].item(),
            "lr": optimizer.param_groups[0]["lr"],
            **{k: v.item() for k, v in loss_dict_reduced_scaled.items()},
            **{k: v.item() for k, v in loss_dict_reduced_unscaled.items()},
        }
        self.train_metrics.append(train_metrics)

        return {"loss": losses_reduced_scaled}

    def validation_step(self, batch, batch_idx):
        """Validation step logic."""
        samples, targets = batch

        # Move to device
        samples = samples.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Determine which model to evaluate (EMA or regular)
        model_to_eval = self.model

        # Half precision for evaluation if specified
        fp16_eval = getattr(self.args, "fp16_eval", False)
        if fp16_eval:
            model_to_eval = model_to_eval.half()
            samples.tensors = samples.tensors.half()

        # Forward pass
        with torch.autocast(**self.autocast_args):
            outputs = model_to_eval(samples)

        # Convert back to float if using fp16 eval
        if fp16_eval:
            for key in outputs:
                if key == "enc_outputs":
                    for sub_key in outputs[key]:
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx]:
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

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

        # Log metrics
        self.log(
            "val/loss",
            losses_reduced_scaled,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in loss_dict_reduced_scaled.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val/class_error",
            loss_dict_reduced["class_error"],
            on_step=False,
            on_epoch=True,
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

    def on_validation_epoch_start(self):
        """Set up validation epoch."""
        # Reset metrics
        self.val_metrics = []

        # Create COCO evaluator
        dataset_val = self.trainer.datamodule.dataset_val
        base_ds = get_coco_api_from_dataset(dataset_val)
        iou_types = tuple(k for k in ("segm", "bbox") if k in self.postprocessors)
        self.coco_evaluator = CocoEvaluator(base_ds, iou_types)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Process validation batch results."""
        if self.coco_evaluator is not None:
            self.coco_evaluator.update(outputs["results"])

    def on_validation_epoch_end(self):
        """Process validation epoch results."""
        if self.coco_evaluator is not None:
            # Synchronize if distributed
            if self.trainer.world_size > 1:
                self.coco_evaluator.synchronize_between_processes()

            # Accumulate and summarize
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()

            # Extract stats
            coco_stats = {}
            if "bbox" in self.postprocessors:
                coco_stats["coco_eval_bbox"] = self.coco_evaluator.coco_eval["bbox"].stats.tolist()

                # Log mAP
                map_value = self.coco_evaluator.coco_eval["bbox"].stats[0]
                self.log("val/mAP", map_value, on_epoch=True, sync_dist=True)

                # Track best model
                if map_value > self.best_map:
                    self.best_map = map_value
                    self.log("val/best_mAP", self.best_map, on_epoch=True, sync_dist=True)

            if "segm" in self.postprocessors:
                coco_stats["coco_eval_masks"] = self.coco_evaluator.coco_eval["segm"].stats.tolist()

                # Log mask mAP
                mask_map_value = self.coco_evaluator.coco_eval["segm"].stats[0]
                self.log("val/mask_mAP", mask_map_value, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler."""

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

        param_dicts = get_param_dict(self.args, self.model)
        optimizer = FloatOnlyAdamW(
            param_dicts,
            lr=getattr(self.args, "lr", 1e-4),
            weight_decay=getattr(self.args, "weight_decay", 1e-4),
            fused=False,
            eps=1e-4,
        )

        # Build learning rate scheduler
        lr_drop = getattr(self.args, "lr_drop", 100)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [lr_drop], gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class RFDETRDataModule(pl.LightningDataModule):
    """Lightning data module for RF-DETR-Mask."""

    def __init__(self, args):
        """Initialize the RF-DETR data module.

        Args:
            args: Configuration arguments
        """
        super().__init__()
        self.args = args
        self.batch_size = getattr(args, "batch_size", 4)
        self.num_workers = getattr(args, "num_workers", 2)
        self.resolution = getattr(args, "resolution", 560)

    def setup(self, stage=None):
        """Set up datasets for training and validation."""
        # We need to set up datasets for all stages
        self.dataset_train = build_dataset(
            image_set="train", args=self.args, resolution=self.resolution
        )
        self.dataset_val = build_dataset(
            image_set="val", args=self.args, resolution=self.resolution
        )

    def train_dataloader(self):
        """Create training data loader."""
        if self.trainer.world_size > 1:
            sampler_train = DistributedSampler(self.dataset_train)
        else:
            sampler_train = RandomSampler(self.dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, self.batch_size, drop_last=True
        )

        dataloader_train = DataLoader(
            self.dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
        )

        return dataloader_train

    def val_dataloader(self):
        """Create validation data loader."""
        if self.trainer.world_size > 1:
            sampler_val = DistributedSampler(self.dataset_val, shuffle=False)
        else:
            sampler_val = SequentialSampler(self.dataset_val)

        dataloader_val = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=self.num_workers,
        )

        return dataloader_val
