"""Model factory for creating RF-DETR models.

This module provides a clean interface for model creation,
separating model instantiation from training logic.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.core.config import ModelConfig, Config


class ModelFactory:
    """Factory for creating RF-DETR models and associated components."""
    
    @staticmethod
    def create_model(
        config: Union[ModelConfig, Config],
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        """Create a model from configuration.
        
        Args:
            config: Model or full configuration
            device: Device to place model on
            
        Returns:
            Initialized model
        """
        # Handle both ModelConfig and full Config
        if isinstance(config, Config):
            model_config = config.model
            # Convert to namespace for compatibility with build_model
            args = config.to_namespace()
        else:
            model_config = config
            # Create minimal args for build_model
            from argparse import Namespace
            args = Namespace(**model_config.__dict__)
            
            # Add required attributes that might be missing
            if not hasattr(args, "pretrained_encoder"):
                args.pretrained_encoder = True
            if not hasattr(args, "window_block_indexes"):
                args.window_block_indexes = []
            if not hasattr(args, "use_cls_token"):
                args.use_cls_token = True
            if not hasattr(args, "freeze_encoder"):
                args.freeze_encoder = False
            if not hasattr(args, "layer_norm"):
                args.layer_norm = True
            if not hasattr(args, "rms_norm"):
                args.rms_norm = False
            if not hasattr(args, "backbone_lora"):
                args.backbone_lora = False
            if not hasattr(args, "force_no_pretrain"):
                args.force_no_pretrain = False
            if not hasattr(args, "gradient_checkpointing"):
                args.gradient_checkpointing = False
            if not hasattr(args, "encoder_only"):
                args.encoder_only = False
            if not hasattr(args, "backbone_only"):
                args.backbone_only = False
            if not hasattr(args, "load_dinov2_weights"):
                args.load_dinov2_weights = True
            if not hasattr(args, "num_feature_levels"):
                args.num_feature_levels = len(model_config.projector_scale)
            if not hasattr(args, "lite_refpoint_refine"):
                args.lite_refpoint_refine = False
            if not hasattr(args, "decoder_norm"):
                args.decoder_norm = "LN"
            if not hasattr(args, "vit_encoder_num_layers"):
                args.vit_encoder_num_layers = model_config.num_encoder_layers
            if not hasattr(args, "dec_layers"):
                args.dec_layers = model_config.num_decoder_layers
            if not hasattr(args, "num_decoder_points"):
                args.num_decoder_points = model_config.dec_n_points
            if not hasattr(args, "bbox_reparam"):
                args.bbox_reparam = True
            if not hasattr(args, "group_detr"):
                args.group_detr = 1
            if not hasattr(args, "no_intermittent_layers"):
                args.no_intermittent_layers = False
            if not hasattr(args, "set_loss"):
                args.set_loss = "lw_detr"
        
        # Build model using existing function
        model = build_model(args)
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
        
        return model
    
    @staticmethod
    def create_criterion(
        config: Union[ModelConfig, Config],
        device: Optional[torch.device] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Create loss criterion and postprocessors.
        
        Args:
            config: Model or full configuration
            device: Device to place criterion on
            
        Returns:
            Tuple of (criterion, postprocessors)
        """
        # Handle both ModelConfig and full Config
        if isinstance(config, Config):
            # Convert to namespace for compatibility
            args = config.to_namespace()
        else:
            # Create minimal args for build_criterion_and_postprocessors
            from argparse import Namespace
            args = Namespace(**config.__dict__)
            
            # Add required attributes
            if not hasattr(args, "cls_loss_coef"):
                args.cls_loss_coef = 4.5
            if not hasattr(args, "bbox_loss_coef"):
                args.bbox_loss_coef = 2.0
            if not hasattr(args, "giou_loss_coef"):
                args.giou_loss_coef = 1.0
            if not hasattr(args, "mask_loss_coef"):
                args.mask_loss_coef = 1.0
            if not hasattr(args, "dice_loss_coef"):
                args.dice_loss_coef = 1.0
            if not hasattr(args, "set_cost_class"):
                args.set_cost_class = 5.0
            if not hasattr(args, "set_cost_bbox"):
                args.set_cost_bbox = 2.0
            if not hasattr(args, "set_cost_giou"):
                args.set_cost_giou = 1.0
            if not hasattr(args, "focal_loss"):
                args.focal_loss = True
            if not hasattr(args, "focal_alpha"):
                args.focal_alpha = 0.25
            if not hasattr(args, "focal_gamma"):
                args.focal_gamma = 2.0
            if not hasattr(args, "use_varifocal_loss"):
                args.use_varifocal_loss = False
            if not hasattr(args, "use_position_supervised_loss"):
                args.use_position_supervised_loss = False
            if not hasattr(args, "ia_bce_loss"):
                args.ia_bce_loss = False
            if not hasattr(args, "sum_group_losses"):
                args.sum_group_losses = False
            if not hasattr(args, "num_select"):
                args.num_select = 300
            if not hasattr(args, "set_loss"):
                args.set_loss = "lw_detr"
        
        # Build criterion and postprocessors
        criterion, postprocessors = build_criterion_and_postprocessors(args)
        
        # Move to device if specified
        if device is not None:
            criterion = criterion.to(device)
        
        return criterion, postprocessors
    
    @staticmethod
    def load_pretrained_weights(
        model: nn.Module,
        weights_path: Union[str, Path],
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> nn.Module:
        """Load pretrained weights into model.
        
        Args:
            model: Model to load weights into
            weights_path: Path to weights file
            device: Device for loading
            strict: Whether to strictly enforce matching keys
            
        Returns:
            Model with loaded weights
        """
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=device or "cpu")
        
        # Extract model state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Handle DataParallel/DistributedDataParallel wrapped models
        # Remove "module." prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(
            new_state_dict, strict=strict
        )
        
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        print(f"Loaded pretrained weights from: {weights_path}")
        
        return model
    
    @staticmethod
    def create_model_with_criterion(
        config: Union[ModelConfig, Config],
        device: Optional[torch.device] = None,
        pretrained_weights: Optional[Union[str, Path]] = None,
    ) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
        """Create model, criterion, and postprocessors together.
        
        Args:
            config: Configuration
            device: Device to use
            pretrained_weights: Optional pretrained weights to load
            
        Returns:
            Tuple of (model, criterion, postprocessors)
        """
        # Create model
        model = ModelFactory.create_model(config, device)
        
        # Load pretrained weights if provided
        if pretrained_weights:
            model = ModelFactory.load_pretrained_weights(
                model, pretrained_weights, device
            )
        
        # Create criterion and postprocessors
        criterion, postprocessors = ModelFactory.create_criterion(config, device)
        
        return model, criterion, postprocessors
    
    @staticmethod
    def get_parameter_groups(
        model: nn.Module,
        config: Union[ModelConfig, Config],
    ) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer.
        
        Args:
            model: Model to get parameters from
            config: Configuration
            
        Returns:
            List of parameter groups with learning rates
        """
        # Import the existing function
        from rfdetr.util.get_param_dicts import get_param_dict
        
        # Convert config to args for compatibility
        if isinstance(config, Config):
            args = config.to_namespace()
        else:
            from argparse import Namespace
            args = Namespace(**config.__dict__)
            
            # Add required learning rate attributes
            if not hasattr(args, "lr"):
                args.lr = 1e-4
            if not hasattr(args, "lr_encoder"):
                args.lr_encoder = 1e-5
            if not hasattr(args, "lr_projector"):
                args.lr_projector = 1e-5
            if not hasattr(args, "lr_vit_layer_decay"):
                args.lr_vit_layer_decay = 1.0
            if not hasattr(args, "lr_component_decay"):
                args.lr_component_decay = 0.9
        
        # Get parameter groups
        param_dicts = get_param_dict(args, model)
        
        return param_dicts