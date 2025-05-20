# Better Configuration Management

This document outlines better approaches to configuration management, focusing on replacing anti-patterns like excessive use of `getattr()` with default values.

## Problems with Current Approach

The current configuration approach has several issues:

1. **Scattered defaults**: Default values are defined throughout the codebase with `getattr()`
2. **Implicit options**: It's difficult to discover all available configuration options
3. **Inconsistent validation**: Some options are validated, others aren't
4. **Type inconsistency**: Default values might have different types than user-provided values

## Better Configuration Approaches

### 1. Use Pydantic for Configuration Classes

As mentioned in AGENT.md, Pydantic provides a robust way to define configuration with validation:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class OptimizerConfig(BaseModel):
    lr: float = Field(default=1e-4, description="Learning rate")
    weight_decay: float = Field(default=1e-4, description="Weight decay")
    
class DataConfig(BaseModel):
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    num_workers: int = Field(default=2, ge=0, description="Number of data loader workers")
    training_width: int = Field(..., description="Training image width")  # Required field
    training_height: int = Field(..., description="Training image height")  # Required field
    
class TrainingConfig(BaseModel):
    epochs: int = Field(default=300, ge=1, description="Number of training epochs")
    warmup_ratio: float = Field(default=0.1, ge=0, le=1, description="Warmup ratio")
    lr_scheduler: str = Field(default="cosine", description="LR scheduler type")
    lr_min_factor: float = Field(default=0.0, ge=0, le=1, description="Minimum LR factor")
    clip_max_norm: float = Field(default=0.0, ge=0, description="Gradient clipping max norm")
    use_ema: bool = Field(default=True, description="Use EMA model averaging")
    ema_decay: Optional[float] = Field(default=None, description="EMA decay rate")
    
class RFDETRConfig(BaseModel):
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output_dir: str = Field(default="output", description="Output directory")
    # Other sections as needed
```

### 2. Load Configuration from YAML

Use YAML for configuration files:

```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> RFDETRConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    try:
        return RFDETRConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration in {config_path}: {e}")
```

### 3. Access Configuration Directly

With properly defined configuration classes, you can access values directly:

```python
# Instead of:
# lr = getattr(self.config, "lr", 1e-4)
# weight_decay = getattr(self.config, "weight_decay", 1e-4)

# Access directly with proper typing:
lr = self.config.optimizer.lr
weight_decay = self.config.optimizer.weight_decay
```

### 4. Handle Legacy Configurations

For backward compatibility with existing code:

```python
def normalize_config(config):
    """Convert various config formats to standard Pydantic model."""
    if isinstance(config, RFDETRConfig):
        return config
        
    if hasattr(config, "model_dump") and callable(config.model_dump):
        # Already a Pydantic model, but not our specific type
        return RFDETRConfig(**config.model_dump())
        
    if isinstance(config, dict):
        # Plain dictionary configuration
        return RFDETRConfig(**config)
        
    # Try to convert from legacy object-based config
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith("_") and not callable(getattr(config, attr)):
            config_dict[attr] = getattr(config, attr)
    
    return RFDETRConfig(**config_dict)
```

### 5. Define Nested Configurations

Organize related configuration options together:

```python
# Instead of flat configuration with prefixes:
# export_onnx: bool
# export_torch: bool
# export_dir: str

# Use nested configuration:
class ExportConfig(BaseModel):
    onnx: bool = Field(default=False, description="Export ONNX model")
    torch: bool = Field(default=False, description="Export TorchScript model")
    dir: str = Field(default="exports", description="Export directory")

# Then in main config:
class RFDETRConfig(BaseModel):
    # Other fields
    export: ExportConfig = Field(default_factory=ExportConfig)
    
# Access as:
if self.config.export.onnx:
    # Export ONNX model
```

## Implementation Plan

1. Create Pydantic models for all configuration sections
2. Create a default YAML configuration file
3. Update code to use direct attribute access instead of getattr
4. Add conversion utilities for legacy configuration formats
5. Update documentation to describe all configuration options