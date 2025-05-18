# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
PyTorch Lightning modules for RF-DETR-Mask training.
"""

import datetime
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

import rfdetr.datasets.transforms as T
import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator
# Temporarily disable ONNX imports for testing
# from rfdetr.deploy.export import export_onnx, onnx_simplify
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

        # Use automatic optimization to work with gradient clipping
        self.automatic_optimization = True
        
        # Setup export directories
        self.export_dir = Path(getattr(args, "output_dir", "exports"))
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Flags for exports
        self.export_onnx = getattr(args, "export_onnx", True)
        self.export_torch = getattr(args, "export_torch", True)
        self.simplify_onnx = getattr(args, "simplify_onnx", True)
        self.export_on_validation = getattr(args, "export_on_validation", True)

    def _setup_autocast_args(self):
        """Set up arguments for autocast (mixed precision training)."""
        # Use full precision (float32) by default
        import torch

        # Always use float32 for main training to avoid cdist_cuda issues
        self._dtype = torch.float32

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
                # Disable autocast/mixed precision by default
                "enabled": getattr(self.args, "amp", False),
                "dtype": self._dtype,
            }
        else:
            self.autocast_args = {
                # Disable autocast/mixed precision by default
                "enabled": getattr(self.args, "amp", False),
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

    def _make_dummy_input(self, batch_size=1):
        """Generate a dummy input for ONNX export.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            A dummy input tensor with the correct shape for ONNX export
        """
        # Get resolution from args
        resolution = getattr(self.args, "resolution", 640)
        
        # Create dummy input
        dummy = np.random.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
        image = torch.from_numpy(dummy).permute(2, 0, 1).float() / 255.0
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        image = (image - mean) / std
        
        # Repeat for batch size
        images = torch.stack([image for _ in range(batch_size)])
        
        # Create nested tensor
        mask = torch.zeros((batch_size, resolution, resolution), dtype=torch.bool)
        nested_tensor = utils.NestedTensor(images, mask)
        
        return nested_tensor
    
    def export_model(self, epoch):
        """Export model to ONNX and save PyTorch weights.
        
        Args:
            epoch: Current epoch number
        """
        if not (self.export_onnx or self.export_torch):
            return
        
        # Use CPU for exports to avoid CUDA errors
        device = torch.device("cpu")
        
        # Create timestamped directory for this export
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f"epoch_{epoch:04d}_{timestamp}"
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Save to logs
        print(f"Exporting model to {export_path}")
        
        # Get model to export (use EMA if available)
        model_to_export = self.ema.ema if self.ema is not None else self.model
        model_to_export = model_to_export.to(device)
        model_to_export.eval()
        
        try:
            # Export PyTorch weights
            if self.export_torch:
                torch_path = export_path / "model.pth"
                torch.save({
                    "model": model_to_export.state_dict(),
                    "args": self.args,
                    "epoch": epoch,
                    "map": self.best_map
                }, torch_path)
                print(f"Saved PyTorch weights to {torch_path}")
            
            # Placeholder for ONNX export (actual export disabled for testing)
            if self.export_onnx:
                # Placeholder for ONNX export
                onnx_path = export_path / "inference_model.onnx"
                print(f"ONNX export would save to: {onnx_path} (disabled for testing)")
                
                # Placeholder for ONNX simplification
                if self.simplify_onnx:
                    sim_onnx_path = export_path / "inference_model.sim.onnx"
                    print(f"ONNX simplification would save to: {sim_onnx_path} (disabled for testing)")
        except Exception as e:
            print(f"Error during model export: {e}")
        finally:
            # Move model back to original device
            model_to_export.to(self.device)
    
    def on_validation_epoch_start(self):
        """Set up validation epoch and export model."""
        # Reset metrics
        self.val_metrics = []

        # Create COCO evaluator
        try:
            dataset_val = self.trainer.datamodule.dataset_val
            base_ds = get_coco_api_from_dataset(dataset_val)
            iou_types = tuple(k for k in ("segm", "bbox") if k in self.postprocessors)
            self.coco_evaluator = CocoEvaluator(base_ds, iou_types)
        except Exception as e:
            print(
                f"Error initializing COCO evaluator: {e}. This can happen with small validation sets."
            )
            self.coco_evaluator = None
            
        # Export model before validation if enabled
        if self.export_on_validation:
            current_epoch = self.trainer.current_epoch if self.trainer else 0
            self.export_model(current_epoch)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Process validation batch results."""
        if self.coco_evaluator is not None and "results" in outputs:
            try:
                self.coco_evaluator.update(outputs["results"])
            except Exception as e:
                print(f"Error updating COCO evaluator: {e}")

    def on_validation_epoch_end(self):
        """Process validation epoch results."""
        # Default metric values
        map_value = 0.0
        mask_map_value = 0.0

        # Try to use COCO evaluator if available
        if self.coco_evaluator is not None:
            try:
                # Check if we have enough data to evaluate
                if (
                    hasattr(self.coco_evaluator, "eval_imgs")
                    and len(self.coco_evaluator.eval_imgs) > 0
                ):
                    # Synchronize if distributed
                    if self.trainer.world_size > 1:
                        self.coco_evaluator.synchronize_between_processes()

                    # Accumulate and summarize
                    self.coco_evaluator.accumulate()
                    self.coco_evaluator.summarize()

                    # Extract stats
                    if (
                        "bbox" in self.postprocessors
                        and hasattr(self.coco_evaluator, "coco_eval")
                        and "bbox" in self.coco_evaluator.coco_eval
                    ):
                        if hasattr(self.coco_evaluator.coco_eval["bbox"], "stats"):
                            stats = self.coco_evaluator.coco_eval["bbox"].stats
                            if hasattr(stats, "tolist"):
                                # Log mAP
                                map_value = stats[0]

                                # Track best model
                                if map_value > self.best_map:
                                    self.best_map = map_value

                    if (
                        "segm" in self.postprocessors
                        and hasattr(self.coco_evaluator, "coco_eval")
                        and "segm" in self.coco_evaluator.coco_eval
                    ):
                        if hasattr(self.coco_evaluator.coco_eval["segm"], "stats"):
                            stats = self.coco_evaluator.coco_eval["segm"].stats
                            if hasattr(stats, "tolist"):
                                # Log mask mAP
                                mask_map_value = stats[0]
            except Exception as e:
                print(
                    f"Error during COCO evaluation: {e}. This can happen with small validation sets."
                )

        # Always log the metrics
        self.log("val/mAP", map_value, on_epoch=True, sync_dist=True)
        self.log("val/best_mAP", self.best_map, on_epoch=True, sync_dist=True)

        if "segm" in self.postprocessors:
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

                            # Convert gradients to float if they're half, ensuring types match
                            if p.grad.dtype != p.dtype:
                                if p.dtype == torch.float32:
                                    p.grad = p.grad.to(torch.float32)
                                # If parameter is half precision, keep grad as half too
                                elif p.dtype == torch.float16 and p.grad.dtype == torch.float32:
                                    p.grad = p.grad.to(torch.float16)

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

        # Configure for Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


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
