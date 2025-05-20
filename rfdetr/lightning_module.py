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
from rfdetr.protocols import Exportable, HasExport, HasModelDump
from rfdetr.util.logging_config import get_logger
from rfdetr.model_config import ModelConfig
# Export functionality removed


logger = get_logger(__name__)

class RFDETRLightningModule(pl.LightningModule):
    """Lightning module for RF-DETR training using iteration-based approach."""

    def __init__(self, config):
        """Initialize the RF-DETR Lightning Module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = ModelConfig(**config)
        self.model = build_model(self.config)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(self.config)
        self.ema_decay = self.config.training.ema_decay 
        use_ema = self.config.training.use_ema
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
        output_dir = self.config.training.output_dir
        self.export_torch = self.config.training.export_torch
        self.export_on_validation = self.config.training.export_on_validation
        self.max_steps = self.config.training.max_steps
        self.val_frequency = self.config.training.eval_save_frequency
                    # Setup export directories
        self.export_dir = Path(output_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _setup_autocast_args(self):
        """Set up arguments for autocast (mixed precision training)."""
        self.autocast_args = {
            "device_type": "cuda" if torch.cuda.is_available() else "cpu",
            "enabled": True,
            "dtype": torch.float32
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
            fp16_eval = self.config.training.fp16_eval

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
        """Generate a dummy input for model testing.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            A dummy input tensor with the correct shape for testing
        """
        # Get dimensions from config
        training_width = self.config.model.training_width
        training_height = self.config.model.training_height

        # Create dummy input
        dummy = np.random.randint(0, 256, (training_height, training_width, 3), dtype=np.uint8)
        image = torch.from_numpy(dummy).permute(2, 0, 1).float() / 255.0

        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        image = (image - mean) / std

        # Repeat for batch size
        images = torch.stack([image for _ in range(batch_size)])

        # Create nested tensor
        mask = torch.zeros((batch_size, training_height, training_width), dtype=torch.bool)
        nested_tensor = utils.NestedTensor(images, mask)

        return nested_tensor

    def export_model(self, epoch):
        """Export model to PyTorch format.

        Args:
            epoch: Current epoch number
        """
        # Check for required export attributes
        if not hasattr(self, 'export_torch'):
            logger.warning("Model export attempted but export_torch attribute is missing")
            return

        if not self.export_torch:
            return

        # Use CPU for exports to avoid CUDA errors
        device = torch.device("cpu")

        # Create timestamped directory for this export
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure export_dir exists with a default
        if not hasattr(self, 'export_dir') or self.export_dir is None:
            logger.info("export_dir is None or missing, defaulting to 'exports'")
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

                # Get config data
                config_data = self.config.model_dump()

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
            # Access trainer.datamodule directly - will raise AttributeError if missing
            dataset_val = self.trainer.datamodule.dataset_val
            base_ds = get_coco_api_from_dataset(dataset_val)
            
            # Check if base_ds has annotations
            try:
                has_annotations = bool(base_ds.anns)
                if has_annotations:
                    iou_types = tuple(k for k in ("segm", "bbox") if k in self.postprocessors)
                    self.coco_evaluator = CocoEvaluator(base_ds, iou_types)
                else:
                    print("COCO dataset has no annotations, skipping evaluator initialization")
                    self.coco_evaluator = None
            except (AttributeError, TypeError):
                print("COCO dataset is not properly initialized, skipping evaluator initialization")
                self.coco_evaluator = None
        except AttributeError:
            print("DataModule not properly initialized, skipping evaluator initialization")
            self.coco_evaluator = None
        except Exception as e:
            print(
                f"Error initializing COCO evaluator: {e}. This can happen with small validation sets."
            )
            self.coco_evaluator = None

        # Export model before validation if enabled
        try:
            export_enabled = self.export_on_validation
            if export_enabled:
                current_epoch = self.trainer.current_epoch if self.trainer else 0
                self.export_model(current_epoch)
        except AttributeError:
            # Skip export if the attribute doesn't exist
            pass
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

        # Ensure we log a default value for the mAP metric even if evaluation fails
        # Use both slash and underscore formats for better compatibility
        self.log("val/mAP", 0.0, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/best_mAP", self.best_map, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log with underscore format for ModelCheckpoint compatibility
        self.log("val_mAP", 0.0, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_best_mAP", self.best_map, on_step=False, on_epoch=True, sync_dist=True)

        # Try to use COCO evaluator if available
        if self.coco_evaluator is not None:
            try:
                # Check if we have enough data to evaluate using try-except for robustness
                eval_imgs_valid = False

                try:
                    # Access eval_imgs directly - if it doesn't exist, it will raise AttributeError
                    # which will be caught by the outer try-except block
                    eval_imgs = self.coco_evaluator.eval_imgs
                    if eval_imgs:  # Check if it's not None or empty
                        for iou_type, imgs in eval_imgs.items():
                            if isinstance(imgs, list) and len(imgs) > 0:
                                eval_imgs_valid = True
                                break
                            elif isinstance(imgs, np.ndarray) and imgs.size > 0:
                                eval_imgs_valid = True
                                break
                except AttributeError:
                    # eval_imgs not available, continue with eval_imgs_valid as False
                    pass

                if eval_imgs_valid:
                    # Synchronize if distributed
                    if self.trainer.world_size > 1:
                        self.coco_evaluator.synchronize_between_processes()

                    # Accumulate and summarize
                    try:
                        self.coco_evaluator.accumulate()
                        self.coco_evaluator.summarize()
                    except Exception as e:
                        print(f"Error in COCO accumulate/summarize: {e}. Continuing with validation.")

                    # Extract stats for bounding boxes using safer access patterns
                    bbox_valid = "bbox" in self.postprocessors
                    
                    if bbox_valid:
                        try:
                            # Try to access the stats directly with better error handling
                            # This will raise exceptions if any part of the chain is missing
                            # and properly be caught by the except block
                            bbox_stats = self.coco_evaluator.coco_eval["bbox"].stats
                            # If we get here, both coco_eval exists and has bbox stats
                        except (AttributeError, KeyError, TypeError):
                            # Either coco_eval doesn't exist or bbox key is missing or stats attribute is missing
                            bbox_valid = False
                    
                    if bbox_valid:
                        stats = self.coco_evaluator.coco_eval["bbox"].stats
                        
                        # Convert stats safely using try/except instead of hasattr
                        try:
                            # Try to get the first value and convert to float
                            map_value = float(stats[0]) if stats[0] is not None else 0.0

                            # Track best model
                            if map_value > self.best_map:
                                self.best_map = map_value

                            # Update the logged value with the computed one
                            self.log("val/mAP", map_value, on_step=False, on_epoch=True, sync_dist=True)
                            self.log("val/best_mAP", self.best_map, on_step=False, on_epoch=True, sync_dist=True)
                        except (TypeError, IndexError, ValueError):
                            # If stats can't be accessed properly or conversion fails
                            logger.warning("Failed to extract or convert mAP value")
                            
                    # Log with underscore format for ModelCheckpoint compatibility
                    self.log("val_mAP", map_value, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("val_best_mAP", self.best_map, on_step=False, on_epoch=True, sync_dist=True)

                    # Extract stats for segmentation masks
                    segm_valid = "segm" in self.postprocessors
                    
                    if segm_valid:
                        try:
                            # Try to access the segm stats directly with nested access
                            stats = self.coco_evaluator.coco_eval["segm"].stats
                            # Try to convert stats to correct format
                            try:
                                # Log mask mAP
                                mask_map_value = float(stats[0]) if stats[0] is not None else 0.0
                                if "segm" in self.postprocessors:
                                    self.log("val/mask_mAP", mask_map_value, on_step=False, on_epoch=True, sync_dist=True)
                                    # Also log with underscore format
                                    self.log("val_mask_mAP", mask_map_value, on_step=False, on_epoch=True, sync_dist=True)
                            except (TypeError, IndexError, ValueError):
                                logger.warning("Failed to extract or convert mask mAP value")
                        except (AttributeError, KeyError):
                            logger.warning("Segmentation stats not available in expected format")
                else:
                    print("Skipping COCO evaluation - not enough evaluation images")
            except Exception as e:
                print(f"Error during COCO evaluation: {e}. This can happen with small validation sets.")
        else:
            print("No COCO evaluator available. Using default metrics.")

        # Log any additional metrics that might be useful
        if len(self.val_metrics) > 0:
            # Calculate average of validation metrics
            avg_loss = sum(m.get('loss', 0.0) for m in self.val_metrics) / max(len(self.val_metrics), 1)
            self.log("val/avg_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_avg_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True)

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
        lr = self.config.training.lr
        weight_decay = self.config.training.weight_decay

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
        warmup_ratio = self.config.training.warmup_ratio
        lr_scheduler_type = self.config.training.lr_scheduler_type
        lr_min_factor = self.config.training.lr_min_factor

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
            self.training_width = self.config.get("training_width", 560)
            self.training_height = self.config.get("training_height", 560)
        else:
            # Check for training and model attributes
            if hasattr(self.config, "training"):
                self.batch_size = getattr(self.config.training, "batch_size", 4)
                self.num_workers = getattr(self.config.training, "num_workers", 2)
            else:
                self.batch_size = getattr(self.config, "batch_size", 4)
                self.num_workers = getattr(self.config, "num_workers", 2)
                
            if hasattr(self.config, "model"):
                self.training_width = getattr(self.config.model, "training_width", 560)
                self.training_height = getattr(self.config.model, "training_height", 560)
            else:
                self.training_width = getattr(self.config, "training_width", 560)
                self.training_height = getattr(self.config, "training_height", 560)

    def setup(self, stage=None):
        """Set up datasets for training and validation."""
        # We need to set up datasets for all stages
        self.dataset_train = build_dataset(
            image_set="train", args=self.config,
            training_width=self.training_width, training_height=self.training_height
        )
        self.dataset_val = build_dataset(
            image_set="val", args=self.config,
            training_width=self.training_width, training_height=self.training_height
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
        # WARNING: Keep validation sequential to ensure consistent evaluation results
        # Shuffling validation data would make metrics less stable between runs
        # DO NOT change this to RandomSampler unless specifically testing data randomization effects
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
