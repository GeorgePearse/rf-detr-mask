# Implementation Guide for Replacing `hasattr` Anti-Patterns

Based on the code analysis, here are specific implementations for replacing the most common `hasattr` patterns in the codebase.

## 1. Config Attribute Checks

Many `hasattr` checks are used to check for optional configuration attributes. These should be replaced with proper default values in configuration classes.

### Before:
```python
self.ema_decay = self.config.ema_decay if hasattr(self.config, "ema_decay") else None
```

### After (Pydantic approach):
```python
class TrainingConfig(BaseModel):
    ema_decay: Optional[float] = None
    use_ema: bool = True
    # Other config options
```

## 2. Export Method Checks

Many checks verify if a model has export capabilities.

### Before:
```python
if hasattr(model, "export") and callable(model.export) and hasattr(model, "_export"):
    # Use export functionality
```

### After (Protocol approach):
```python
from typing import Protocol, Any, Callable

class HasExport(Protocol):
    export: Callable
    _export: Any

if isinstance(model, HasExport):
    # Use export functionality
```

## 3. Optional Result Property Checks

Checks for optional properties on results/outputs.

### Before:
```python
if hasattr(stats, "tolist"):
    stats = stats.tolist()
```

### After (Try/Except approach):
```python
try:
    stats = stats.tolist()
except AttributeError:
    # Handle case where tolist isn't available
    pass
```

## 4. Conditional Feature Checks

Checks determining if specific features should be enabled.

### Before:
```python
if hasattr(self.config, "clip_max_norm") and self.config.clip_max_norm > 0:
    # Apply gradient clipping
```

### After (Default value approach):
```python
clip_max_norm = getattr(self.config, "clip_max_norm", 0.0)
if clip_max_norm > 0:
    # Apply gradient clipping
```

## 5. Implementation Plan

1. Create Protocol classes for common interfaces (HasExport, HasModelDump, etc.)
2. Refactor config classes to use Pydantic with proper default values
3. Replace feature flag checks with explicit default values
4. Use try/except for runtime attribute checks where appropriate
5. Update all functions that work with mixed types to normalize inputs early