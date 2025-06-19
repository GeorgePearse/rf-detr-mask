"""Model factory for creating RF-DETR models with dependency injection.

This module provides a factory pattern implementation for creating models
with proper dependency injection and configuration management.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from rfdetr.config import ModelConfig
from rfdetr.core.models.registry import model_registry, ModelRegistry
from rfdetr.models import build_model as legacy_build_model
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util.files import download_file
from rfdetr.model_utils import HOSTED_MODELS, populate_args


class ModelFactory:
    """Factory for creating models with proper dependency injection.
    
    This class provides a clean interface for creating models with various
    configurations and handling pretrained weights loading.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        """Initialize the model factory.
        
        Args:
            registry: Optional model registry. Uses global registry if not provided.
        """
        self.registry = registry or model_registry
        self._weight_cache_dir = Path.home() / ".cache" / "rfdetr" / "weights"
        self._weight_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def create_model(
        self,
        config: ModelConfig,
        device: Optional[Union[str, torch.device]] = None
    ) -> nn.Module:
        """Create a model instance from configuration.
        
        Args:
            config: Model configuration
            device: Optional device to load model on
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        # Determine device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Check if we have a registered builder for this model
        model_name = getattr(config, "model_name", None)
        if model_name and self.registry.has_model(model_name):
            # Use registered builder
            builder = self.registry.get_builder(model_name)
            model = builder(config)
        else:
            # Fall back to legacy builder
            args = self._config_to_args(config)
            model = legacy_build_model(args)
        
        # Load pretrained weights if specified
        if config.pretrain_weights:
            self.load_pretrained_weights(model, config.pretrain_weights, device)
        
        # Move model to device
        model = model.to(device)
        
        return model
    
    def create_criterion_and_postprocessors(
        self,
        config: ModelConfig
    ) -> tuple[nn.Module, Dict[str, nn.Module]]:
        """Create criterion and postprocessors for the model.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (criterion, postprocessors dict)
        """
        args = self._config_to_args(config)
        return build_criterion_and_postprocessors(args)
    
    def load_pretrained_weights(
        self,
        model: nn.Module,
        weights_path: str,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> None:
        """Load pretrained weights into a model.
        
        Args:
            model: Model to load weights into
            weights_path: Path to weights file or name of hosted weights
            device: Device to load weights on
            strict: Whether to strictly enforce that the keys in state_dict
                   match the keys returned by model.state_dict()
                   
        Raises:
            FileNotFoundError: If weights file not found
            RuntimeError: If weights loading fails
        """
        # Download hosted weights if necessary
        if weights_path in HOSTED_MODELS:
            weights_path = self._download_hosted_weights(weights_path)
        
        # Check if weights file exists
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load checkpoint
        device = device or torch.device("cpu")
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        except Exception as e:
            # Try re-downloading if corrupted
            if weights_path in HOSTED_MODELS:
                print(f"Failed to load weights, re-downloading: {e}")
                weights_path = self._download_hosted_weights(weights_path, force=True)
                checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            else:
                raise RuntimeError(f"Failed to load weights from {weights_path}: {e}")
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume checkpoint is the state dict
                state_dict = checkpoint
        else:
            raise RuntimeError("Invalid checkpoint format")
        
        # Load state dict into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            print(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in state dict: {unexpected_keys}")
        
        # Set model metadata from checkpoint if available
        if isinstance(checkpoint, dict):
            if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
                model.class_names = checkpoint["args"].class_names
            if "epoch" in checkpoint:
                model.loaded_epoch = checkpoint["epoch"]
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        path: Union[str, Path],
        config: Optional[ModelConfig] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Save model checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer state to save
            epoch: Current epoch number
            path: Path to save checkpoint
            config: Optional model configuration to save
            metrics: Optional metrics to save with checkpoint
            **kwargs: Additional metadata to save
        """
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
        }
        
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        
        if config is not None:
            checkpoint["config"] = config.dict()
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        # Add any additional metadata
        checkpoint.update(kwargs)
        
        # Save checkpoint
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def _config_to_args(self, config: ModelConfig) -> Any:
        """Convert ModelConfig to args format for legacy code compatibility.
        
        Args:
            config: Model configuration
            
        Returns:
            Args object for legacy model builder
        """
        # Use the legacy populate_args function
        config_dict = config.dict()
        return populate_args(**config_dict)
    
    def _download_hosted_weights(self, name: str, force: bool = False) -> str:
        """Download hosted weights if not already cached.
        
        Args:
            name: Name of hosted weights
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded weights file
        """
        if name not in HOSTED_MODELS:
            raise ValueError(f"Unknown hosted model: {name}")
        
        cache_path = self._weight_cache_dir / name
        
        if force or not cache_path.exists():
            print(f"Downloading pretrained weights: {name}")
            download_file(HOSTED_MODELS[name], str(cache_path))
        
        return str(cache_path)