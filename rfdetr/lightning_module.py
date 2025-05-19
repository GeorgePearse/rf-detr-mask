# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
PyTorch Lightning modules for RF-DETR-Mask training.
"""

import datetime
import math
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator

# Import ONNX export functionality
try:
    # Import validation only - not directly used
    from rfdetr.deploy.export import export_onnx as _export_onnx  # noqa: F401

    ONNX_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ONNX_AVAILABLE = False
    print("ONNX libraries not fully available - will only export PyTorch weights")
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import ModelEma


class RFDETRLightningModule(pl.LightningModule):
    """Lightning module for RF-DETR training using iteration-based approach."""

    def __init__(self, config):
        """Initialize the RF-DETR Lightning Module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
        super().__init__()
        self.save_hyperparameters()

        # Import here to avoid circular imports
        from rfdetr.model_config import ModelConfig

        # Handle different types of input for backward compatibility
        if isinstance(config, dict):
            # For dict input, try to convert to ModelConfig
            try:
                self.config = ModelConfig(**config)
            except Exception:
                # If conversion fails, keep original dict
                self.config = config
        elif hasattr(config, "model_dump") and callable(config.model_dump):
            # It's already a Pydantic model
            self.config = config
        else:
            # Other object with attributes, keep as is
            self.config = config

        # Build model, criterion, and postprocessors
        self.model = build_model(self.config)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(self.config)

        # Setup EMA if enabled
        if isinstance(self.config, ModelConfig):
            # Direct attribute access for ModelConfig
            self.ema_decay = self.config.ema_decay if hasattr(self.config, "ema_decay") else None
            use_ema = self.config.use_ema if hasattr(self.config, "use_ema") else True
        elif isinstance(self.config, dict):
            # Dictionary access
            self.ema_decay = self.config.get("ema_decay", None)
            use_ema = self.config.get("use_ema", True)
        else:
            # Object attribute access
            self.ema_decay = getattr(self.config, "ema_decay", None)
            use_ema = getattr(self.config, "use_ema", True)

        self.ema = ModelEma(self.model, self.ema_decay) if self.ema_decay and use_ema else None

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

        # Get configuration values based on type
        if isinstance(self.config, dict):
            output_dir = self.config.get("output_dir", "exports")
            self.export_onnx = self.config.get("export_onnx", True)
            self.export_torch = self.config.get("export_torch", True)
            self.simplify_onnx = self.config.get("simplify_onnx", True)
            self.export_on_validation = self.config.get("export_on_validation", True)
            self.max_steps = self.config.get("max_steps", 2000)
            self.val_frequency = self.config.get(
                "eval_save_frequency", self.config.get("val_frequency", 200)
            )
        else:
            output_dir = getattr(self.config, "output_dir", "exports")
            self.export_onnx = getattr(self.config, "export_onnx", True)
            self.export_torch = getattr(self.config, "export_torch", True)
            self.simplify_onnx = getattr(self.config, "simplify_onnx", True)
            self.export_on_validation = getattr(self.config, "export_on_validation", True)
            self.max_steps = getattr(self.config, "max_steps", 2000)
            self.val_frequency = getattr(
                self.config, "eval_save_frequency", getattr(self.config, "val_frequency", 200)
            )

        # Setup export directories
        self.export_dir = Path(output_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

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

        # Get amp setting from config
        if isinstance(self.config, dict):
            amp_enabled = self.config.get("amp", False)
        else:
            amp_enabled = getattr(self.config, "amp", False)

        if self.amp_backend == "torch":
            self.autocast_args = {
                "device_type": "cuda" if torch.cuda.is_available() else "cpu",
                "enabled": amp_enabled,
                "dtype": self._dtype,
            }
        else:
            self.autocast_args = {
                "enabled": amp_enabled,
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

            # Half precision for evaluation if specified
            if isinstance(self.config, dict):
                fp16_eval = self.config.get("fp16_eval", False)
            else:
                fp16_eval = getattr(self.config, "fp16_eval", False)

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

    def _make_dummy_input(self, batch_size=1):
        """Generate a dummy input for ONNX export.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            A dummy input tensor with the correct shape for ONNX export
        """
        # Get resolution from config
        if isinstance(self.config, dict):
            resolution = self.config.get("resolution", 640)
        else:
            resolution = getattr(self.config, "resolution", 640)

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
        if not hasattr(self, "export_onnx") or not hasattr(self, "export_torch"):
            return

        if not (self.export_onnx or self.export_torch):
            return

        # Use CPU for exports to avoid CUDA errors
        device = torch.device("cpu")

        # Create timestamped directory for this export
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Make sure we have an export directory
        if not hasattr(self, "export_dir") or self.export_dir is None:
            self.export_dir = Path("exports")

        export_path = self.export_dir / f"epoch_{epoch:04d}_{timestamp}"
        export_path.mkdir(parents=True, exist_ok=True)

        # Save to logs
        print(f"Exporting model to {export_path}")

        # Get model to export (use EMA if available)
        model_to_export = self.ema.module if self.ema is not None else self.model
        model_to_export = model_to_export.to(device)
        model_to_export.eval()

        try:
            # Export PyTorch weights
            if self.export_torch:
                torch_path = export_path / "model.pth"

                # Get config data properly
                if hasattr(self.config, "model_dump"):
                    config_data = self.config.model_dump()
                elif isinstance(self.config, dict):
                    config_data = self.config
                else:
                    # Convert object to dict if needed
                    config_data = {
                        attr: getattr(self.config, attr)
                        for attr in dir(self.config)
                        if not attr.startswith("_") and not callable(getattr(self.config, attr))
                    }

                # Save the model weights and config
                torch.save(
                    {
                        "model": model_to_export.state_dict(),
                        "config": config_data,
                        "epoch": epoch,
                        "map": self.best_map,
                    },
                    torch_path,
                )
                print(f"Saved PyTorch weights to {torch_path}")

            # Placeholder for ONNX export (actual export disabled for testing)
            if self.export_onnx:
                # Placeholder for ONNX export
                onnx_path = export_path / "inference_model.onnx"
                print(f"ONNX export would save to: {onnx_path} (disabled for testing)")

                # Placeholder for ONNX simplification
                if hasattr(self, "simplify_onnx") and self.simplify_onnx:
                    sim_onnx_path = export_path / "inference_model.sim.onnx"
                    print(
                        f"ONNX simplification would save to: {sim_onnx_path} (disabled for testing)"
                    )
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
            if hasattr(self.trainer, "datamodule") and hasattr(
                self.trainer.datamodule, "dataset_val"
            ):
                dataset_val = self.trainer.datamodule.dataset_val
                base_ds = get_coco_api_from_dataset(dataset_val)
                if base_ds and hasattr(base_ds, "anns") and base_ds.anns:
                    iou_types = tuple(k for k in ("segm", "bbox") if k in self.postprocessors)
                    self.coco_evaluator = CocoEvaluator(base_ds, iou_types)
                else:
                    print("COCO dataset has no annotations, skipping evaluator initialization")
                    self.coco_evaluator = None
            else:
                print("DataModule not properly initialized, skipping evaluator initialization")
                self.coco_evaluator = None
        except Exception as e:
            print(
                f"Error initializing COCO evaluator: {e}. This can happen with small validation sets."
            )
            self.coco_evaluator = None

        # Export model before validation if enabled
        try:
            if hasattr(self, "export_on_validation") and self.export_on_validation:
                current_epoch = self.trainer.current_epoch if self.trainer else 0
                self.export_model(current_epoch)
        except Exception as e:
            print(f"Error during model export in on_validation_epoch_start: {e}")
            # Continue with validation even if export fails

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Process validation batch results."""
        if outputs is None:
            return

        if self.coco_evaluator is not None and isinstance(outputs, dict) and "results" in outputs:
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
                    and self.coco_evaluator.eval_imgs
                    and any(
                        len(imgs) > 0
                        for imgs in self.coco_evaluator.eval_imgs.values()
                        if isinstance(imgs, list)
                    )
                ):
                    # Synchronize if distributed
                    if self.trainer.world_size > 1:
                        self.coco_evaluator.synchronize_between_processes()

                    # Accumulate and summarize
                    try:
                        self.coco_evaluator.accumulate()
                        self.coco_evaluator.summarize()
                    except Exception as e:
                        print(
                            f"Error in COCO accumulate/summarize: {e}. Continuing with validation."
                        )

                    # Extract stats for bounding boxes
                    bbox_valid = (
                        "bbox" in self.postprocessors
                        and hasattr(self.coco_evaluator, "coco_eval")
                        and "bbox" in self.coco_evaluator.coco_eval
                        and hasattr(self.coco_evaluator.coco_eval["bbox"], "stats")
                    )
                    
                    if bbox_valid:
                        stats = self.coco_evaluator.coco_eval["bbox"].stats
                        if hasattr(stats, "tolist"):
                            # Log mAP
                            map_value = float(stats[0]) if stats[0] is not None else 0.0

                            # Track best model
                            if map_value > self.best_map:
                                self.best_map = map_value

                    # Extract stats for segmentation masks
                    segm_valid = (
                        "segm" in self.postprocessors
                        and hasattr(self.coco_evaluator, "coco_eval")
                        and "segm" in self.coco_evaluator.coco_eval
                        and hasattr(self.coco_evaluator.coco_eval["segm"], "stats")
                    )
                    
                    if segm_valid:
                        stats = self.coco_evaluator.coco_eval["segm"].stats
                        if hasattr(stats, "tolist"):
                            # Log mask mAP
                            mask_map_value = float(stats[0]) if stats[0] is not None else 0.0
                else:
                    print("Skipping COCO evaluation - not enough evaluation images")
            except Exception as e:
                print(
                    f"Error during COCO evaluation: {e}. This can happen with small validation sets."
                )

        # Always log the metrics at epoch end
        self.log("val/mAP", map_value, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/best_mAP", self.best_map, on_step=False, on_epoch=True, sync_dist=True)

        if "segm" in self.postprocessors:
            self.log("val/mask_mAP", mask_map_value, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler for iteration-based training."""

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

        # Get parameters from config for optimizer
        if isinstance(self.config, dict):
            lr = self.config.get("lr", 1e-4)
            weight_decay = self.config.get("weight_decay", 1e-4)
        else:
            lr = getattr(self.config, "lr", 1e-4)
            weight_decay = getattr(self.config, "weight_decay", 1e-4)

        param_dicts = get_param_dict(self.config, self.model)
        optimizer = FloatOnlyAdamW(
            param_dicts,
            lr=lr,
            weight_decay=weight_decay,
            fused=False,
            eps=1e-4,
        )

        # Get total training steps and warmup steps
        max_steps = self.max_steps

        # Get scheduling parameters from config
        if isinstance(self.config, dict):
            warmup_ratio = self.config.get("warmup_ratio", 0.1)
            lr_scheduler_type = self.config.get("lr_scheduler", "cosine")
            lr_min_factor = self.config.get("lr_min_factor", 0.0)
        else:
            warmup_ratio = getattr(self.config, "warmup_ratio", 0.1)
            lr_scheduler_type = getattr(self.config, "lr_scheduler", "cosine")
            lr_min_factor = getattr(self.config, "lr_min_factor", 0.0)

        warmup_steps = int(max_steps * warmup_ratio)

        # Define lambda function for scheduler
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing from multiplier 1.0 down to lr_min_factor
                if lr_scheduler_type == "cosine":
                    progress = float(current_step - warmup_steps) / float(
                        max(1, max_steps - warmup_steps)
                    )
                    return lr_min_factor + (1 - lr_min_factor) * 0.5 * (
                        1 + math.cos(math.pi * progress)
                    )
                elif lr_scheduler_type == "step":
                    # Default step schedule if not using cosine
                    return 0.1 if current_step > (max_steps * 0.8) else 1.0
                else:
                    return 1.0

        # Create LambdaLR scheduler
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # Configure for Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # Step-based scheduling
                "frequency": 1,
            },
        }


class RFDETRDataModule(pl.LightningDataModule):
    """Lightning data module for RF-DETR-Mask."""

    def __init__(self, config):
        """Initialize the RF-DETR data module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
        super().__init__()
        # Import here to avoid circular imports
        from rfdetr.model_config import ModelConfig

        # Handle different types of input for backward compatibility
        if isinstance(config, dict):
            # For dict input, try to convert to ModelConfig
            try:
                self.config = ModelConfig(**config)
            except Exception:
                # If conversion fails, keep original dict
                self.config = config
        elif hasattr(config, "model_dump") and callable(config.model_dump):
            # It's already a Pydantic model
            self.config = config
        else:
            # Other object with attributes, keep as is
            self.config = config

        # Get configuration values based on type
        if isinstance(self.config, dict):
            self.batch_size = self.config.get("batch_size", 4)
            self.num_workers = self.config.get("num_workers", 2)
            self.resolution = self.config.get("resolution", 560)
        else:
            self.batch_size = getattr(self.config, "batch_size", 4)
            self.num_workers = getattr(self.config, "num_workers", 2)
            self.resolution = getattr(self.config, "resolution", 560)

    def setup(self, stage=None):
        """Set up datasets for training and validation."""
        # We need to set up datasets for all stages
        self.dataset_train = build_dataset(
            image_set="train", args=self.config, resolution=self.resolution
        )
        self.dataset_val = build_dataset(
            image_set="val", args=self.config, resolution=self.resolution
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
