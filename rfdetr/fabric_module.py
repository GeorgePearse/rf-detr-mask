# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Lightning Fabric implementation for RF-DETR-Mask training with iteration-based approach.
Fabric provides a lightweight alternative to PyTorch Lightning while retaining
key distributed training and mixed precision capabilities.
"""

import datetime
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

# Import autocast for mixed precision training
try:
    from torch.amp import GradScaler, autocast
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    DEPRECATED_AMP = True

import rfdetr.datasets.transforms as T
import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.models import build_criterion_and_postprocessors, build_model
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import ModelEma


def get_autocast_args(config):
    """Get autocast arguments for mixed precision training.
    
    Args:
        config: Configuration object or dict
        
    Returns:
        dict: Autocast arguments
    """
    # Prefer bfloat16 if available, otherwise use float16
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    
    # Check if config is a dict or object
    if isinstance(config, dict):
        amp_enabled = config.get("amp", False)
    else:
        amp_enabled = getattr(config, "amp", False)
    
    if DEPRECATED_AMP:
        return {"enabled": amp_enabled, "dtype": dtype}
    else:
        return {"device_type": "cuda", "enabled": amp_enabled, "dtype": dtype}


class RFDETRFabricModule:
    """Fabric-based module for RF-DETR training using iteration-based approach."""

    def __init__(self, config):
        """Initialize the RF-DETR Fabric Module.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
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
        if isinstance(self.config, dict):
            # Dictionary access
            self.ema_decay = self.config.get("ema_decay", None)
            use_ema = self.config.get("use_ema", True)
        else:
            # Object attribute access
            self.ema_decay = getattr(self.config, "ema_decay", None)
            use_ema = getattr(self.config, "use_ema", True)
            
        self.ema = None  # Will be initialized after model is setup with fabric

        # Track metrics
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        
        # Setup autocast args for mixed precision training
        self._setup_autocast_args()

        # Track best metrics
        self.best_map = 0.0
        
        # Get configuration values based on type
        if isinstance(self.config, dict):
            output_dir = self.config.get("output_dir", "exports")
            self.export_onnx = self.config.get("export_onnx", True)
            self.export_torch = self.config.get("export_torch", True)
            self.simplify_onnx = self.config.get("simplify_onnx", True)
            self.export_on_validation = self.config.get("export_on_validation", True)
            self.max_steps = self.config.get("max_steps", 2000)
            self.eval_save_frequency = self.config.get("eval_save_frequency", 
                                                     self.config.get("val_frequency", 200))
        else:
            output_dir = getattr(self.config, "output_dir", "exports")
            self.export_onnx = getattr(self.config, "export_onnx", True)
            self.export_torch = getattr(self.config, "export_torch", True) 
            self.simplify_onnx = getattr(self.config, "simplify_onnx", True)
            self.export_on_validation = getattr(self.config, "export_on_validation", True)
            self.max_steps = getattr(self.config, "max_steps", 2000)
            self.eval_save_frequency = getattr(self.config, "eval_save_frequency", 
                                             getattr(self.config, "val_frequency", 200))
        
        # Setup export directories
        self.export_dir = Path(output_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _setup_autocast_args(self):
        """Set up arguments for autocast (mixed precision training)."""
        # Get autocast arguments from helper function
        self.autocast_args = get_autocast_args(self.config)
            
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
        dummy = torch.randint(0, 256, (batch_size, 3, resolution, resolution), dtype=torch.uint8)
        image = dummy.float() / 255.0
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def setup(self, fabric):
        """Setup the module with Fabric.
        
        Args:
            fabric: Lightning Fabric instance
        """
        self.fabric = fabric
        self.model = fabric.setup_module(self.model)
        self.criterion = fabric.setup_module(self.criterion)
        
        # Initialize EMA after model has been set up with fabric
        if self.ema_decay and getattr(self.config, "use_ema", True):
            self.ema = ModelEma(self.model, self.ema_decay)
        
        # Get optimizer parameters
        if isinstance(self.config, dict):
            lr = self.config.get("lr", 1e-4)
            weight_decay = self.config.get("weight_decay", 1e-4)
        else:
            lr = getattr(self.config, "lr", 1e-4)
            weight_decay = getattr(self.config, "weight_decay", 1e-4)

        # Configure optimizer
        param_dicts = get_param_dict(self.config, self.model)
        self.optimizer = torch.optim.AdamW(
            param_dicts,
            lr=lr,
            weight_decay=weight_decay,
            fused=False,
            eps=1e-4,
        )
        self.optimizer = fabric.setup_optimizers(self.optimizer)
        
        # Configure learning rate scheduler
        self._setup_lr_scheduler()
        
    def _setup_lr_scheduler(self):
        """Setup learning rate scheduler."""
        # Get scheduling parameters from config
        if isinstance(self.config, dict):
            warmup_ratio = self.config.get("warmup_ratio", 0.1)
            lr_scheduler_type = self.config.get("lr_scheduler", "cosine")
            lr_min_factor = self.config.get("lr_min_factor", 0.0)
        else:
            warmup_ratio = getattr(self.config, "warmup_ratio", 0.1)
            lr_scheduler_type = getattr(self.config, "lr_scheduler", "cosine")
            lr_min_factor = getattr(self.config, "lr_min_factor", 0.0)
        
        # Calculate warmup steps
        warmup_steps = int(self.max_steps * warmup_ratio)
        
        # Define lambda function for scheduler
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing from multiplier 1.0 down to lr_min_factor
                if lr_scheduler_type == "cosine":
                    progress = float(current_step - warmup_steps) / float(
                        max(1, self.max_steps - warmup_steps)
                    )
                    return lr_min_factor + (1 - lr_min_factor) * 0.5 * (
                        1 + math.cos(math.pi * progress)
                    )
                elif lr_scheduler_type == "step":
                    # Default step schedule if not using cosine
                    return 0.1 if current_step > (self.max_steps * 0.8) else 1.0
                else:
                    return 1.0

        # Create LambdaLR scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train_step(self, batch):
        """Execute a single training step.
        
        Args:
            batch: Tuple of (samples, targets)
            
        Returns:
            dict: Dictionary containing loss values and metrics
        """
        samples, targets = batch
        self.model.train()
        self.criterion.train()
        
        # Forward pass with autocast
        with torch.autocast(**self.autocast_args):
            outputs = self.model(samples, targets)
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        # Backward and optimize
        self.fabric.backward(losses)
        
        # Clip gradients if needed
        if hasattr(self.config, "clip_max_norm") and self.config.clip_max_norm > 0:
            self.fabric.clip_gradients(self.model, optimizer=self.optimizer, max_norm=self.config.clip_max_norm)
        
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # Update EMA model if available
        if self.ema is not None:
            self.ema.update(self.model)
            
        # Log metrics
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        metrics = {
            "loss": losses_reduced_scaled.item(),
            "class_error": loss_dict_reduced["class_error"].item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            **{k: v.item() for k, v in loss_dict_reduced_scaled.items()},
            **{k: v.item() for k, v in loss_dict_reduced_unscaled.items()},
        }
        
        self.train_metrics.append(metrics)
        return metrics
    
    def validation_step(self, batch):
        """Execute a single validation step.
        
        Args:
            batch: Tuple of (samples, targets)
            
        Returns:
            tuple: (metrics, results, targets) for COCO evaluation
        """
        samples, targets = batch
        self.model.eval()
        self.criterion.eval()
        
        # Determine which model to evaluate (EMA or regular)
        model_to_eval = self.ema.ema if self.ema is not None else self.model

        # Half precision for evaluation if specified
        if isinstance(self.config, dict):
            fp16_eval = self.config.get("fp16_eval", False)
        else:
            fp16_eval = getattr(self.config, "fp16_eval", False)
            
        # Store original precision
        orig_dtype = None
        if fp16_eval:
            orig_dtype = next(model_to_eval.parameters()).dtype
            model_to_eval = model_to_eval.half()
            samples = samples.half()

        # Forward pass
        with torch.no_grad():
            with torch.autocast(**self.autocast_args):
                outputs = model_to_eval(samples)

        # Convert back to float if using fp16 eval
        if fp16_eval:
            # Convert model back to original precision
            for p in model_to_eval.parameters():
                p.data = p.data.to(orig_dtype)
                
            # Convert outputs back to float
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

        # Store metrics for validation_epoch_end
        val_metrics = {
            "loss": losses_reduced_scaled.item(),
            "class_error": loss_dict_reduced["class_error"].item(),
            **{k: v.item() for k, v in loss_dict_reduced_scaled.items()},
            **{k: v.item() for k, v in loss_dict_reduced_unscaled.items()},
        }
        self.val_metrics.append(val_metrics)

        return {"metrics": val_metrics, "results": res, "targets": targets}
        
    def export_model(self, epoch):
        """Export model to ONNX and save PyTorch weights.
        
        Args:
            epoch: Current epoch number
        """
        if not hasattr(self, 'export_onnx') or not hasattr(self, 'export_torch'):
            return
            
        if not (self.export_onnx or self.export_torch):
            return
        
        # Use CPU for exports to avoid CUDA errors
        device = torch.device("cpu")
        
        # Create timestamped directory for this export
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Make sure we have an export directory
        if not hasattr(self, 'export_dir') or self.export_dir is None:
            self.export_dir = Path("exports")
            
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
                config_data = self.config.model_dump() if hasattr(self.config, "model_dump") else self.config
                torch.save({
                    "model": model_to_export.state_dict(),
                    "config": config_data,
                    "epoch": epoch,
                    "map": self.best_map
                }, torch_path)
                print(f"Saved PyTorch weights to {torch_path}")
            
            # ONNX export
            if self.export_onnx:
                try:
                    # Import ONNX export modules
                    from rfdetr.deploy.export import export_onnx, onnx_simplify
                    
                    # Export to ONNX format
                    dummy_input = self._make_dummy_input()
                    onnx_path = export_path / "inference_model.onnx"
                    
                    # Perform export
                    export_onnx(
                        output_dir=export_path,
                        model=model_to_export,
                        input_tensors=dummy_input,
                        verbose=False,
                        opset_version=getattr(self.config, "opset_version", 17)
                    )
                    
                    # Simplify if requested
                    if self.simplify_onnx:
                        onnx_simplify(
                            onnx_dir=onnx_path,
                            input_tensors=dummy_input
                        )
                except ImportError:
                    print("ONNX export dependencies not found. Install with: pip install \".[onnxexport]\"")
                except Exception as e:
                    print(f"Error during ONNX export: {e}")
        except Exception as e:
            print(f"Error during model export: {e}")
        finally:
            # Move model back to original device
            if hasattr(self, 'fabric'):
                device = getattr(self.fabric, 'device', None)
                if device:
                    model_to_export.to(device)


class RFDETRFabricData:
    """Data manager for RF-DETR using Fabric."""

    def __init__(self, config):
        """Initialize the RF-DETR data manager.

        Args:
            config: Configuration as a Pydantic model or compatible dict/object
        """
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

        # Datasets will be initialized in setup()
        self.dataset_train = None
        self.dataset_val = None

    def setup(self, fabric=None):
        """Set up datasets for training and validation.
        
        Args:
            fabric: Optional Fabric instance for distributed setup
        """
        # We need to set up datasets for all stages
        self.dataset_train = build_dataset(
            image_set="train", args=self.config, resolution=self.resolution
        )
        self.dataset_val = build_dataset(
            image_set="val", args=self.config, resolution=self.resolution
        )
        
        # Store fabric reference
        self.fabric = fabric

    def train_dataloader(self):
        """Create training data loader."""
        if self.fabric and self.fabric.world_size > 1:
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
        
        if self.fabric:
            return self.fabric.setup_dataloaders(dataloader_train)
        return dataloader_train

    def val_dataloader(self):
        """Create validation data loader."""
        if self.fabric and self.fabric.world_size > 1:
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
        
        if self.fabric:
            return self.fabric.setup_dataloaders(dataloader_val)
        return dataloader_val


def train_with_fabric(
    config, 
    output_dir=None, 
    callbacks=None, 
    precision="32-true", 
    devices=1, 
    accelerator="auto"
):
    """Train RF-DETR model using Lightning Fabric.
    
    Args:
        config: The model configuration
        output_dir: Directory to save outputs (overrides config value)
        callbacks: Optional callbacks to run during training
        precision: Precision setting for Fabric
        devices: Number of devices to use
        accelerator: Hardware accelerator to use
        
    Returns:
        The trained model and final metrics
    """
    # Setup output directory 
    if output_dir:
        if isinstance(config, dict):
            config["output_dir"] = output_dir
        else:
            config.output_dir = output_dir
            
    output_dir = output_dir or (config.output_dir if hasattr(config, "output_dir") else "output_fabric")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize callbacks if not provided
    if callbacks is None:
        callbacks = {
            "on_fit_epoch_end": [],
            "on_train_batch_start": [],
            "on_train_end": []
        }
    
    # Setup Fabric
    strategy = "auto"
    if devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
        
    fabric = Fabric(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        strategy=strategy,
    )
    fabric.launch()
    
    # Create model and data module
    model = RFDETRFabricModule(config)
    data_module = RFDETRFabricData(config)
    
    # Setup data module
    data_module.setup(fabric)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Setup model
    model.setup(fabric)
    
    # Add COCO evaluation capabilities
    base_ds = get_coco_api_from_dataset(data_module.dataset_val)
    iou_types = tuple(k for k in ("segm", "bbox") if k in model.postprocessors)
    
    # Get training parameters
    max_steps = model.max_steps
    eval_save_frequency = model.eval_save_frequency
    
    # Initialize step counter
    global_step = 0
    
    # Start training
    print(f"Starting training with Fabric for {max_steps} steps")
    print(f"Validation and checkpointing every {eval_save_frequency} steps")
    
    # Training loop
    model.model.train()
    train_iter = iter(train_loader)
    while global_step < max_steps:
        # Get next batch, recreate iterator if needed
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        # Call callbacks
        callback_dict = {
            "step": global_step,
            "model": model.model,
            "epoch": global_step // len(train_loader),
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        
        # Execute training step
        metrics = model.train_step(batch)
        
        # Log every 10 steps
        if global_step % 10 == 0:
            fabric.print(f"Step {global_step}/{max_steps}, "
                       f"Loss: {metrics['loss']:.4f}, "
                       f"LR: {metrics['lr']:.6f}")
        
        # Run validation and checkpointing if needed
        if global_step > 0 and global_step % eval_save_frequency == 0:
            # Initialize COCO evaluator
            coco_evaluator = CocoEvaluator(base_ds, iou_types)
            
            # Run validation
            model.model.eval()
            model.val_metrics = []  # Reset metrics
            
            with torch.no_grad():
                for val_batch in val_loader:
                    val_output = model.validation_step(val_batch)
                    # Update COCO evaluator
                    coco_evaluator.update(val_output["results"])
            
            # Process COCO results
            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            
            # Extract metrics
            map_regular = coco_evaluator.coco_eval["bbox"].stats[0] if "bbox" in coco_evaluator.coco_eval else 0
            
            # Update best mAP
            is_best = False
            if map_regular > model.best_map:
                model.best_map = map_regular
                is_best = True
                
            # Print validation results
            fabric.print(f"Step {global_step}: mAP: {map_regular:.4f}, Best mAP: {model.best_map:.4f}")
            
            # Save checkpoints
            if fabric.is_global_zero:
                # Regular checkpoint
                checkpoint_path = Path(output_dir) / f"checkpoint_step_{global_step:07d}.pth"
                torch.save({
                    "model": model.model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                    "lr_scheduler": model.lr_scheduler.state_dict(),
                    "step": global_step,
                    "config": config,
                }, checkpoint_path)
                
                # Best checkpoint if applicable
                if is_best:
                    best_path = Path(output_dir) / "checkpoint_best.pth"
                    torch.save({
                        "model": model.model.state_dict(),
                        "optimizer": model.optimizer.state_dict(),
                        "lr_scheduler": model.lr_scheduler.state_dict(),
                        "step": global_step,
                        "config": config,
                    }, best_path)
                
                # EMA checkpoint if available
                if model.ema is not None:
                    ema_path = Path(output_dir) / f"checkpoint_ema_step_{global_step:07d}.pth"
                    torch.save({
                        "model": model.ema.ema.state_dict(),
                        "optimizer": model.optimizer.state_dict(),
                        "lr_scheduler": model.lr_scheduler.state_dict(),
                        "step": global_step,
                        "config": config,
                    }, ema_path)
                    
                    # Best EMA checkpoint if applicable
                    if is_best:
                        best_ema_path = Path(output_dir) / "checkpoint_best_ema.pth"
                        torch.save({
                            "model": model.ema.ema.state_dict(),
                            "optimizer": model.optimizer.state_dict(),
                            "lr_scheduler": model.lr_scheduler.state_dict(),
                            "step": global_step,
                            "config": config,
                        }, best_ema_path)
            
                # Export model if configured
                if model.export_on_validation:
                    model.export_model(global_step)
            
            # Prepare log stats for callbacks
            log_stats = {
                "train_loss": metrics["loss"],
                "test_coco_eval_bbox": [map_regular],
                "step": global_step,
                "epoch": global_step // len(train_loader),
                "n_parameters": sum(p.numel() for p in model.model.parameters() if p.requires_grad),
                "best_map": model.best_map,
            }
            
            # Run epoch end callbacks
            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)
                
            # Resume training
            model.model.train()
            
        # Increment step counter
        global_step += 1
    
    # Final validation
    model.model.eval()
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    
    with torch.no_grad():
        for val_batch in val_loader:
            val_output = model.validation_step(val_batch)
            coco_evaluator.update(val_output["results"])
    
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    # Get final metrics
    final_map = coco_evaluator.coco_eval["bbox"].stats[0] if "bbox" in coco_evaluator.coco_eval else 0
    
    # Run train end callbacks
    for callback in callbacks["on_train_end"]:
        callback()
    
    fabric.print(f"Training completed. Final mAP: {final_map:.4f}, Best mAP: {model.best_map:.4f}")
    
    # Return model and metrics
    return model, {"final_map": final_map, "best_map": model.best_map}