"""Model registry for RF-DETR model variants.

This module implements a registry pattern for managing different model variants
and their configurations.
"""

from typing import Callable, Dict, Optional, Type

import torch.nn as nn

from rfdetr.config import ModelConfig


class ModelRegistry:
    """Registry pattern for model variants.
    
    This class provides a centralized registry for different model architectures
    and their builders, enabling easy extension and management of model variants.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, Type[nn.Module]] = {}
        self._builders: Dict[str, Callable[[ModelConfig], nn.Module]] = {}
        self._configs: Dict[str, Type[ModelConfig]] = {}
        
    def register(
        self,
        name: str,
        model_class: Optional[Type[nn.Module]] = None,
        builder: Optional[Callable[[ModelConfig], nn.Module]] = None,
        config_class: Optional[Type[ModelConfig]] = None
    ) -> None:
        """Register a model variant with the registry.
        
        Args:
            name: Unique name for the model variant
            model_class: Optional model class (nn.Module subclass)
            builder: Optional builder function that creates the model
            config_class: Optional configuration class for the model
            
        Raises:
            ValueError: If neither model_class nor builder is provided
        """
        if model_class is None and builder is None:
            raise ValueError("Either model_class or builder must be provided")
        
        if model_class is not None:
            self._models[name] = model_class
            
        if builder is not None:
            self._builders[name] = builder
            
        if config_class is not None:
            self._configs[name] = config_class
    
    def get_model_class(self, name: str) -> Type[nn.Module]:
        """Get a registered model class by name.
        
        Args:
            name: Name of the model variant
            
        Returns:
            Model class
            
        Raises:
            KeyError: If model is not registered
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self._models[name]
    
    def get_builder(self, name: str) -> Callable[[ModelConfig], nn.Module]:
        """Get a registered model builder by name.
        
        Args:
            name: Name of the model variant
            
        Returns:
            Model builder function
            
        Raises:
            KeyError: If builder is not registered
        """
        if name not in self._builders:
            raise KeyError(f"Builder for model '{name}' not found in registry")
        return self._builders[name]
    
    def get_config_class(self, name: str) -> Type[ModelConfig]:
        """Get a registered configuration class by name.
        
        Args:
            name: Name of the model variant
            
        Returns:
            Configuration class
            
        Raises:
            KeyError: If config class is not registered
        """
        if name not in self._configs:
            raise KeyError(f"Config class for model '{name}' not found in registry")
        return self._configs[name]
    
    def list_models(self) -> list[str]:
        """List all registered model names.
        
        Returns:
            List of registered model names
        """
        # Return union of all registered names
        all_names = set(self._models.keys()) | set(self._builders.keys()) | set(self._configs.keys())
        return sorted(list(all_names))
    
    def has_model(self, name: str) -> bool:
        """Check if a model is registered.
        
        Args:
            name: Name of the model variant
            
        Returns:
            True if model is registered, False otherwise
        """
        return name in self._models or name in self._builders
    
    def unregister(self, name: str) -> None:
        """Unregister a model from the registry.
        
        Args:
            name: Name of the model variant to unregister
        """
        self._models.pop(name, None)
        self._builders.pop(name, None)
        self._configs.pop(name, None)


# Global registry instance
model_registry = ModelRegistry()


def register_model(
    name: str,
    model_class: Optional[Type[nn.Module]] = None,
    builder: Optional[Callable[[ModelConfig], nn.Module]] = None,
    config_class: Optional[Type[ModelConfig]] = None
) -> Callable:
    """Decorator for registering models with the global registry.
    
    Args:
        name: Unique name for the model variant
        model_class: Optional model class (nn.Module subclass)
        builder: Optional builder function that creates the model
        config_class: Optional configuration class for the model
        
    Returns:
        Decorator function
    """
    def decorator(cls_or_func):
        # If decorating a class, use it as the model_class
        if isinstance(cls_or_func, type) and issubclass(cls_or_func, nn.Module):
            model_registry.register(name, model_class=cls_or_func, config_class=config_class)
        # If decorating a function, use it as the builder
        elif callable(cls_or_func):
            model_registry.register(name, builder=cls_or_func, config_class=config_class)
        else:
            raise TypeError("Decorator must be applied to a model class or builder function")
        return cls_or_func
    
    return decorator