# Addressing Anti-Patterns in the Codebase

This document outlines strategies for refactoring three anti-patterns identified in the AGENT.md file:

1. `hasattr()` usage
2. `getattr()` usage 
3. `isinstance()` checks

## 1. Refactoring `hasattr()` Usage

### Problem

The `hasattr()` function is often overused to check for the existence of attributes, which can lead to:
- Silent failures when attributes exist but have `None` values
- Less explicit code that doesn't declare expected interfaces
- Poor editor/IDE support for static analysis

### Solution Approaches

#### A. Use Protocol Classes and Type Checking

```python
from typing import Protocol

class HasExport(Protocol):
    export: callable
    _export: any

# Instead of:
# if hasattr(model, "export") and callable(model.export) and hasattr(model, "_export"):  

# Use:
if isinstance(model, HasExport):
    # Code that uses model.export and model._export
```

#### B. Use Default Values with Direct Attribute Access

```python
# Instead of:
# if hasattr(self.config, "ema_decay") and self.config.ema_decay > 0:

# Use:
if getattr(self.config, "ema_decay", 0) > 0:
    # Code that uses self.config.ema_decay
```

#### C. Use Try/Except for Expected Attributes

```python
# Instead of:
# if hasattr(samples, "mask") and samples.mask is not None:

# Use:
try:
    if samples.mask is not None:
        # Code that uses samples.mask
except AttributeError:
    # Handle the case where mask doesn't exist
```

#### D. Use Configuration Validation

```python
# Instead of scattered hasattr checks throughout code:
# if hasattr(config, "attribute1") and hasattr(config, "attribute2"):

# Use a validation function:
def validate_config(config):
    required_attrs = ["attribute1", "attribute2"]
    missing = [attr for attr in required_attrs if not hasattr(config, attr)]
    if missing:
        raise ValueError(f"Config missing required attributes: {missing}")
    return config
```

## 2. Refactoring `getattr()` Usage

### Problem

Overuse of `getattr()` with default values can hide design issues:
- It makes optional configuration too implicit
- It spreads default values throughout the codebase instead of centralizing them
- It makes it harder to understand what configuration options are available

### Solution Approaches

#### A. Use Data Classes with Default Values

```python
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    lr: float = Field(default=1e-4)
    weight_decay: float = Field(default=1e-4)
    batch_size: int = Field(default=4) 
    use_ema: bool = Field(default=True)

# Then access directly:
self.lr = self.config.lr  # Default is already defined in the class
```

#### B. Centralize Configuration Defaults

```python
# Instead of scattered getattr calls:
# lr = getattr(self.config, "lr", 1e-4)
# weight_decay = getattr(self.config, "weight_decay", 1e-4)

# Create a defaults dictionary:
DEFAULT_CONFIG = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 4,
    "use_ema": True,
}

# And use it to populate missing values:
def get_config_with_defaults(config):
    result = DEFAULT_CONFIG.copy()
    if hasattr(config, "model_dump") and callable(config.model_dump):
        result.update(config.model_dump())
    elif isinstance(config, dict):
        result.update(config)
    else:
        # Handle other config types
        pass
    return result
```

## 3. Refactoring `isinstance()` Checks

### Problem

Excessive `isinstance()` checks can indicate poor design choices:
- They create unnecessary type-based branching
- They can indicate functions that should be split or polymorphic interfaces
- They often handle edge cases that should be normalized earlier

### Solution Approaches

#### A. Use Method Overloading or Multi-Dispatch

```python
from functools import singledispatch

@singledispatch
def process_input(input_data):
    raise TypeError(f"Unsupported input type: {type(input_data)}")

@process_input.register
def _(input_data: list):
    # Process list input
    pass

@process_input.register
def _(input_data: torch.Tensor):
    # Process tensor input
    pass
```

#### B. Normalize Data Early

```python
# Instead of multiple isinstance checks throughout code:
# if isinstance(image, torch.Tensor): ...
# elif isinstance(image, PIL.Image.Image): ...

# Normalize the data early:
def normalize_to_tensor(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        return torch.from_numpy(np.array(image))
    elif isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

# Then work with normalized data:
image_tensor = normalize_to_tensor(image)
# No more isinstance checks needed
```

#### C. Use Duck Typing Where Appropriate

```python
# Instead of:
# if isinstance(obj, list) and len(obj) > 0:

# Use duck typing:
try:
    if len(obj) > 0:  # Works for any sequence type
        # Process non-empty sequence
    else:
        # Handle empty sequence
except (TypeError, AttributeError):
    # Handle non-sequence
```

## Implementation Plan

1. Start with configuration classes using Pydantic to address both `hasattr` and `getattr` issues
2. Implement Protocol classes for commonly checked interfaces
3. Create normalization functions for data that currently requires type checking
4. Update tests to ensure refactored code maintains the same behavior