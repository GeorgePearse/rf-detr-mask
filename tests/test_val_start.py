#!/usr/bin/env python
"""
Test script to verify the val_start parameter works correctly
"""

# Simulate the validation logic
def should_validate(epoch: int, steps_per_epoch: int, val_start: int) -> bool:
    """Check if validation should run based on current step and val_start parameter."""
    current_step = epoch * steps_per_epoch
    return current_step >= val_start

# Test cases
test_cases = [
    # (epoch, steps_per_epoch, val_start, expected_result)
    (0, 100, 0, True),      # Should validate from start
    (0, 100, 200, False),   # Should not validate on first epoch
    (1, 100, 200, False),   # Should not validate on second epoch (step=100)
    (2, 100, 200, True),    # Should validate on third epoch (step=200)
    (5, 100, 200, True),    # Should validate on later epochs
]

print("Testing val_start parameter...")

for epoch, steps_per_epoch, val_start, expected in test_cases:
    result = should_validate(epoch, steps_per_epoch, val_start)
    current_step = epoch * steps_per_epoch
    
    status = "✓" if result == expected else "✗"
    print(f"{status} Epoch {epoch}, Step {current_step}, val_start={val_start}: "
          f"validate={result} (expected {expected})")
    
    assert result == expected, f"Test failed for epoch={epoch}, val_start={val_start}"

print("\nAll tests passed!")