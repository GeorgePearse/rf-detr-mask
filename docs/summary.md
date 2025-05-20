# Anti-Pattern Replacement Summary

This document provides a short summary of the anti-patterns and replacement strategies identified in the codebase.

## 1. `hasattr` Replacement

The codebase contains many `hasattr()` checks. These should be refactored using the following approaches:

### Key Examples Found

```python
# In detr.py
if hasattr(self.model, "class_names") and self.model.class_names:
    return {i + 1: name for i, name in enumerate(self.model.class_names)}

# In lightning_module.py
if hasattr(self, "simplify_onnx") and self.simplify_onnx:
    # Simplify ONNX model...
```

### Implementation Strategy

1. Use Protocol classes (already defined in `rfdetr/protocols.py`)
2. Use direct attribute access with defaults via Pydantic models
3. Use early validation of configuration objects

## 2. `getattr` Replacement

The codebase uses `getattr()` with defaults in many places, often for configuration values.

### Key Examples Found

```python
# In lightning_module.py
lr = getattr(self.config, "lr", 1e-4)
weight_decay = getattr(self.config, "weight_decay", 1e-4)
```

### Implementation Strategy

1. Move all default values to Pydantic models (as seen in `config.py`)
2. Use direct attribute access with proper typing
3. Implement hierarchical configuration with nested models

## 3. `isinstance` Replacement

The codebase contains many type checks that could be refactored.

### Key Examples Found

```python
# In detr.py
if not isinstance(images, list):
    images = [images]
    
if isinstance(img, str):
    img = Image.open(img)
```

### Implementation Strategy

1. Normalize data early and use Protocol interfaces
2. Use method overloading or singledispatch for multiple input types
3. Reduce conditional branching with cleaner abstractions

## Implementation Plan

1. Extend existing Protocol classes in `rfdetr/protocols.py`
2. Enhance Pydantic models in `config.py` 
3. Create input normalization functions for common data types
4. Add explicit validation functions for configuration
5. Update code to use direct attribute access

See the detailed implementation guides in:
- `docs/anti_patterns_refactoring.md`
- `docs/hasattr_replacement_implementation.md`
- `docs/error_handling.md`
- `docs/configuration.md`