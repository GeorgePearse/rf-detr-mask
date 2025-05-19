# RF-DETR Error Handling Guidelines

This document outlines the standardized approach to error handling in the RF-DETR codebase.

## Core Principles

1. **Consistent Logging**: Use the logging system for all errors, not print statements
2. **Provide Context**: Always include context information when logging errors
3. **Appropriate Error Types**: Use specific exception types for different categories of errors
4. **Recovery Strategy**: Have a clear strategy for whether to recover or propagate errors
5. **Comprehensive Error Messages**: Include all necessary information for debugging

## Using the Error Handling Utilities

### 1. Getting a Logger

Always use the project's standard logger instead of print statements:

```python
from rfdetr.util.logging_config import get_logger

# Get a logger named after your module
logger = get_logger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General information about program execution")
logger.warning("Warning about a potential issue")
logger.error("Error that prevented an operation from completing")
logger.critical("Critical error that might lead to program termination")
```

### 2. Using Custom Exception Types

The project defines several domain-specific exception types:

```python
from rfdetr.util.error_handling import ConfigurationError, DataError, ModelError, TrainingError, ExportError

# Raise appropriate exception types
if config_value is None:
    raise ConfigurationError("Missing required configuration value 'batch_size'")

if not os.path.exists(dataset_path):
    raise DataError(f"Dataset not found at path: {dataset_path}")

if model_output.isnan().any():
    raise ModelError("Model produced NaN values during forward pass")
```

### 3. Exception Decorator Pattern

For functions that need standardized error handling:

```python
from rfdetr.util.error_handling import handle_exception

@handle_exception(DataError, message="Failed to load dataset", reraise=False, fallback_return=None)
def load_dataset(path):
    # Function implementation...
    pass
```

### 4. Try-Except Context Manager

For inline error handling:

```python
from rfdetr.util.error_handling import try_except

# With automatic re-raising
with try_except("parsing configuration file", logger=logger) as result:
    result.value = json.loads(config_text)
    
# Get the result value
config = result.value

# With fallback value and no re-raising
with try_except("loading model weights", logger=logger, reraise=False, fallback={}) as result:
    result.value = torch.load(checkpoint_path)
    
# Check for success and handle accordingly
if result.success:
    weights = result.value
else:
    # Use default weights or take alternative action
    weights = {}
```

### 5. Logging Exceptions

When you need to catch exceptions but don't want to use the decorator or context manager:

```python
from rfdetr.util.error_handling import log_exception

try:
    model = create_model(config)
except Exception as e:
    log_exception(logger, e, "Failed to create model with config")
    # Handle the error appropriately
    raise  # Or recover
```

## Error Handling Patterns

### Import Error Pattern

```python
try:
    import optional_dependency
    OPTIONAL_AVAILABLE = True
except ImportError:
    OPTIONAL_AVAILABLE = False
    
# Later in code
if OPTIONAL_AVAILABLE:
    # Use the optional dependency
else:
    # Fallback approach
```

### Configuration Validation Pattern

```python
def validate_config(config):
    errors = []
    
    if "batch_size" not in config:
        errors.append("Missing required 'batch_size' parameter")
    
    if "learning_rate" not in config:
        errors.append("Missing required 'learning_rate' parameter")
    
    if errors:
        raise ConfigurationError("\n".join(errors))
```

### Recoverable Error Pattern

```python
def process_batch(batch, model):
    try:
        outputs = model(batch)
        return outputs
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("CUDA out of memory, attempting with smaller batch")
            # Split the batch and try again with smaller batches
            return process_with_smaller_batches(batch, model)
        else:
            # For other runtime errors, log and propagate
            log_exception(logger, e, "Error processing batch")
            raise
```

## Best Practices

1. **Never Use Bare Except**: Always specify the exception types you expect to catch
   ```python
   # Good
   try:
       result = some_operation()
   except (ValueError, KeyError) as e:
       logger.error(f"Invalid data: {e}")
   
   # Bad
   try:
       result = some_operation()
   except:  # Catches EVERYTHING including KeyboardInterrupt, SystemExit, etc.
       logger.error("Error occurred")
   ```

2. **Log First, Then Handle**: Log the error before attempting recovery
   ```python
   try:
       data = load_data(path)
   except FileNotFoundError as e:
       logger.error(f"Data file not found: {e}")
       data = load_default_data()  # Recovery action after logging
   ```

3. **Include Contextual Information**: Always provide context when logging errors
   ```python
   # Good
   logger.error(f"Failed to load model checkpoint from {checkpoint_path}: {e}")
   
   # Bad
   logger.error(f"Failed: {e}")
   ```

4. **Use Appropriate Log Levels**: Not everything is an error
   ```python
   try:
       cache_result = load_from_cache()
   except CacheMissError:
       # This is expected sometimes, so it's a debug or info message
       logger.debug("Cache miss, loading from source")
       cache_result = load_from_source()
   ```

5. **Consider User-Facing vs. Developer-Facing Errors**: Sometimes you need both
   ```python
   try:
       config = load_config(path)
   except Exception as e:
       # Developer-facing detailed error
       log_exception(logger, e, f"Failed to load config from {path}")
       
       # User-facing simplified error
       user_message = f"Could not load configuration file. Please check that {path} exists and contains valid JSON."
       print(user_message)
       raise
   ```