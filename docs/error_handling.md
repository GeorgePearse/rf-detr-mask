# Error Handling Best Practices

This document outlines better approaches to error handling in the codebase, particularly focusing on replacing anti-patterns like excessive `hasattr()`, `getattr()`, and `isinstance()` checks.

## Problems with Current Approach

The current error handling in the codebase has several issues:

1. **Silent failures**: Using `hasattr()` without proper error handling can lead to silent failures
2. **Scattered defaults**: Using `getattr()` with defaults spreads configuration values throughout the code
3. **Type-based branching**: Excessive `isinstance()` checks create complex, hard-to-maintain code paths
4. **Implicit interfaces**: The expected structure of objects is implied rather than explicitly defined

## Better Error Handling Approaches

### 1. Use Explicit Validation

Instead of checking for attributes at the point of use, validate objects early:

```python
def validate_config(config):
    """Validate configuration has all required attributes."""
    required_fields = ["training_width", "training_height", "batch_size"]
    missing = [field for field in required_fields if not hasattr(config, field)]
    if missing:
        raise ValueError(f"Configuration missing required fields: {missing}")
    return config

# Use early in the program flow:
config = validate_config(config)
# Now you can use config attributes without checks
```

### 2. Use Duck Typing with Try/Except

For runtime behavior where you're not sure if an object has a specific capability:

```python
# Instead of:
# if hasattr(stats, "tolist"):
#     stats = stats.tolist()

# Use:
try:
    stats = stats.tolist()
except AttributeError:
    # Handle case where tolist doesn't exist
    pass
```

### 3. Use Proper Type Hints and Protocol Classes

Make expected interfaces explicit with Protocol classes:

```python
from typing import Protocol, Callable, Any

class ModelWithExport(Protocol):
    export: Callable[[str], Any]
    
# Functions can specify they expect this interface
def export_model(model: ModelWithExport, path: str) -> None:
    """Export a model to the specified path."""
    model.export(path)
```

### 4. Normalize Data Early

Transform data to a consistent type early in the processing pipeline:

```python
def ensure_tensor(data):
    """Convert various input types to tensor."""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to tensor")
        
# Use early in the function:
data = ensure_tensor(data)
# Now all code can assume data is a tensor
```

### 5. Use Standard Python Exceptions

Raise appropriate exceptions for different error conditions:

- `ValueError`: For invalid values (e.g., negative batch size)
- `TypeError`: For invalid types (e.g., string where tensor expected)
- `AttributeError`: When a required attribute is missing
- `NotImplementedError`: For methods that should be implemented by subclasses
- `FileNotFoundError`: When expected files don't exist

### 6. Provide Context in Error Messages

Make error messages informative and actionable:

```python
if batch_size <= 0:
    raise ValueError(f"Batch size must be positive, got {batch_size}")
```

## Implementation Plan

1. Create validation functions for config objects
2. Define Protocol classes for common interfaces 
3. Add early data normalization to functions with type-based branches
4. Replace hasattr/getattr with try/except where appropriate
5. Add explicit error messages to all raised exceptions