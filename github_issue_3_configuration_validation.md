# Issue: Improve configuration validation for removed or deprecated parameters

## Description
The codebase has evolved with parameters being removed (like `two_stage`), but the configuration validation system doesn't properly handle these changes. This leads to confusion when users or developers use configuration files with deprecated or removed parameters.

## Impact
Users may configure parameters that no longer have any effect, leading to unexpected behavior. This makes the system less transparent and harder to debug, especially for new contributors.

## Steps to reproduce
1. Use an older configuration file that includes removed parameters like `two_stage`
2. Note that the parameter is silently ignored rather than triggering a warning or error

## Proposed solution

1. Enhance the configuration validation in `config_utils.py` to detect and warn about deprecated parameters
2. Add a validation step that compares configuration keys against the actual model parameters
3. Implement a proper deprecation mechanism that:
   - Logs warnings when deprecated parameters are used
   - Provides clear information about when parameters were removed and what (if anything) replaces them
   - Eventually removes support for long-deprecated parameters

## Implementation suggestion

```python
class ModelConfig(BaseModel):
    # ... existing fields ...
    
    class Config:
        extra = "forbid"  # This will reject unknown fields
        
        @root_validator
        def check_deprecated_params(cls, values):
            deprecated_params = {
                "two_stage": "No longer needed as this behavior is now the default",
                # Add other deprecated parameters here
            }
            
            for param, message in deprecated_params.items():
                if param in values:
                    warnings.warn(f"Parameter '{param}' is deprecated: {message}")
            
            return values
```

## Additional context
Maintaining backward compatibility is important, but we should also provide clear guidance on changes to the API. A robust validation and deprecation system would improve the development experience.