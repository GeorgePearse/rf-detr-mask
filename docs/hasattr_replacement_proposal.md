# Proposal: Replacing `hasattr` Anti-Pattern

This document proposes specific replacements for common `hasattr` patterns in the codebase.

## 1. Test Mode Checks

### Current Implementation

```python
if hasattr(args, "test_mode") and args.test_mode:
    # Do test mode specific things
```

### Proposed Replacement

```python
# In configuration class
class EvalConfig(BaseModel):
    test_mode: bool = Field(default=False, description="Run in test mode")
    val_limit: Optional[int] = Field(default=None, description="Limit validation samples")

# In code
if config.eval.test_mode:
    # Do test mode specific things
```

## 2. EMA Decay Checks

### Current Implementation

```python
self.ema_decay = self.config.ema_decay if hasattr(self.config, "ema_decay") else None
```

### Proposed Replacement

```python
# Configuration class already has default value
self.ema_decay = self.config.training.ema_decay  # Default is set in TrainConfig
```

## 3. Export Method Checks

### Current Implementation

```python
if hasattr(model, "export"):
    model.export(**kwargs)
```

### Proposed Replacement

```python
# Use Protocol class from rfdetr/protocols.py
from rfdetr.protocols import HasExport

if isinstance(model, HasExport):
    model.export(**kwargs)
```

## 4. Configuration Validation

### Current Implementation

```python
# Scattered hasattr checks:
if hasattr(config, "training_width") and hasattr(config, "training_height"):
    # Use width and height
```

### Proposed Replacement

```python
def validate_config(config):
    required_attrs = ["training_width", "training_height", "batch_size"]
    missing = [attr for attr in required_attrs if not hasattr(config, attr)]
    if missing:
        raise ValueError(f"Config missing required attributes: {missing}")
    return config
    
# Use early in process:
config = validate_config(config)
# Now use directly without checks
width, height = config.training_width, config.training_height
```

## 5. Protocol Implementation Example

The `rfdetr/protocols.py` file already has a good start with Protocol classes. This can be expanded.

```python
from typing import Protocol, runtime_checkable, Any, Optional

@runtime_checkable
class HasStats(Protocol):
    """Protocol for objects that have statistics."""
    def tolist(self) -> list[float]:
        """Convert statistics to a list."""
        ...
        
@runtime_checkable
class HasEvaluation(Protocol):
    """Protocol for evaluation objects."""
    eval_imgs: list[Any]
    coco_eval: dict[str, Any]
```

These protocol classes can be used with `isinstance` checks instead of `hasattr`:

```python
# Instead of:
# if hasattr(stats, "tolist"):
#     stats = stats.tolist()

# Use:
if isinstance(stats, HasStats):
    stats = stats.tolist()
```

## Implementation Plan

1. Extend Pydantic models in `config.py` to include all configuration options with proper defaults
2. Add additional Protocol classes to `rfdetr/protocols.py`
3. Update code to use Protocol classes and direct attribute access
4. Add validation functions for early configuration checking