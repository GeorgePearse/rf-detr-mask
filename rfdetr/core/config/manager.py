"""Unified configuration management system for RF-DETR.

This module provides a centralized configuration management system that replaces
the scattered argparse, Pydantic, and dictionary-based configurations.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ValidationError, Field

from rfdetr.config import (
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    TrainConfig,
)


T = TypeVar("T", bound=BaseModel)


class ConfigurationManager:
    """Unified configuration management using Pydantic models.
    
    This class provides a single point of entry for all configuration needs,
    supporting loading from files, environment variables, and runtime overrides.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Optional directory containing configuration files.
                       Defaults to configs/ in the project root.
        """
        self.config_dir = config_dir or self._get_default_config_dir()
        self._config_cache: Dict[str, BaseModel] = {}
        
    def _get_default_config_dir(self) -> Path:
        """Get the default configuration directory."""
        # Find project root by looking for pyproject.toml
        current = Path(__file__).resolve()
        while current.parent != current:
            if (current / "pyproject.toml").exists():
                return current / "configs"
            current = current.parent
        raise RuntimeError("Could not find project root")
    
    def load_model_config(
        self,
        model_name: str = "base",
        config_path: Optional[Union[str, Path]] = None,
        **overrides: Any
    ) -> ModelConfig:
        """Load model configuration with support for overrides.
        
        Args:
            model_name: Name of the model variant ("base" or "large")
            config_path: Optional path to custom configuration file
            **overrides: Keyword arguments to override configuration values
            
        Returns:
            ModelConfig instance with applied overrides
            
        Raises:
            ValueError: If model_name is not recognized
            ValidationError: If configuration is invalid
        """
        if config_path:
            config_dict = self._load_config_file(config_path)
            config_class = ModelConfig
        else:
            config_class = self._get_model_config_class(model_name)
            config_dict = {}
        
        # Apply environment variable overrides
        config_dict.update(self._get_env_overrides("MODEL_"))
        
        # Apply runtime overrides
        config_dict.update(overrides)
        
        try:
            config = config_class(**config_dict)
            self._config_cache[f"model_{model_name}"] = config
            return config
        except ValidationError as e:
            raise ValidationError(f"Invalid model configuration: {e}")
    
    def load_training_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        **overrides: Any
    ) -> TrainConfig:
        """Load training configuration with support for overrides.
        
        Args:
            config_path: Optional path to custom configuration file
            **overrides: Keyword arguments to override configuration values
            
        Returns:
            TrainConfig instance with applied overrides
            
        Raises:
            ValidationError: If configuration is invalid
        """
        config_dict = {}
        
        if config_path:
            config_dict = self._load_config_file(config_path)
        
        # Apply environment variable overrides
        config_dict.update(self._get_env_overrides("TRAIN_"))
        
        # Apply runtime overrides
        config_dict.update(overrides)
        
        try:
            config = TrainConfig(**config_dict)
            self._config_cache["training"] = config
            return config
        except ValidationError as e:
            raise ValidationError(f"Invalid training configuration: {e}")
    
    def validate_config_compatibility(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig
    ) -> None:
        """Validate that model and training configurations are compatible.
        
        Args:
            model_config: Model configuration to validate
            train_config: Training configuration to validate
            
        Raises:
            ValueError: If configurations are incompatible
        """
        # Check resolution compatibility
        if hasattr(model_config, "resolution") and hasattr(train_config, "resolution"):
            if model_config.resolution != train_config.resolution:
                raise ValueError(
                    f"Model resolution ({model_config.resolution}) does not match "
                    f"training resolution ({train_config.resolution})"
                )
        
        # Check batch size compatibility with gradient accumulation
        if hasattr(train_config, "batch_size") and hasattr(train_config, "grad_accum_steps"):
            if train_config.batch_size % train_config.grad_accum_steps != 0:
                raise ValueError(
                    f"Batch size ({train_config.batch_size}) must be divisible by "
                    f"gradient accumulation steps ({train_config.grad_accum_steps})"
                )
        
        # Add more validation rules as needed
    
    def save_config(
        self,
        config: BaseModel,
        path: Union[str, Path],
        format: str = "yaml"
    ) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration object to save
            path: Path to save configuration
            format: File format ("yaml" or "json")
            
        Raises:
            ValueError: If format is not supported
        """
        path = Path(path)
        config_dict = config.dict()
        
        if format == "yaml":
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def merge_configs(self, *configs: BaseModel) -> Dict[str, Any]:
        """Merge multiple configurations into a single dictionary.
        
        Args:
            *configs: Configuration objects to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        for config in configs:
            merged.update(config.dict())
        return merged
    
    def _load_config_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "r") as f:
                return yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def _get_env_overrides(self, prefix: str) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Try to parse as JSON first (for lists, dicts)
                try:
                    overrides[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    # Fall back to string value
                    overrides[config_key] = value
        return overrides
    
    def _get_model_config_class(self, model_name: str) -> Type[ModelConfig]:
        """Get the appropriate model config class based on model name."""
        model_configs = {
            "base": RFDETRBaseConfig,
            "large": RFDETRLargeConfig,
        }
        
        if model_name not in model_configs:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Available models: {list(model_configs.keys())}"
            )
        
        return model_configs[model_name]


class ExperimentConfig(BaseModel):
    """Configuration for experiment tracking and management."""
    
    name: str = Field(..., description="Experiment name")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Experiment tags")
    output_dir: Path = Field(..., description="Output directory for experiment artifacts")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    track_gradients: bool = Field(default=False, description="Whether to track gradients")
    log_frequency: int = Field(default=100, description="Logging frequency in steps")
    checkpoint_frequency: int = Field(default=1000, description="Checkpoint save frequency")
    keep_last_n_checkpoints: int = Field(default=5, description="Number of checkpoints to keep")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True