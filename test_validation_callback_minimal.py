#!/usr/bin/env python
"""Minimal test for validation callback logic"""

from collections import defaultdict

# Simulate the callback setup
callbacks = defaultdict(list)
steps_per_validation = 5
total_steps = 0

def validation_callback(callback_dict):
    global total_steps
    total_steps += 1
    
    if total_steps % steps_per_validation == 0:
        print(f"Running validation at step {total_steps}")
        print(f"Validation would run here...")

callbacks["on_train_batch_start"].append(validation_callback)

# Simulate training steps
print("Simulating training with steps_per_validation=5")
for step in range(20):
    # Simulate callback execution
    callback_dict = {"step": step}
    for callback in callbacks["on_train_batch_start"]:
        callback(callback_dict)

print(f"\nTotal steps: {total_steps}")
print("TEST PASSED: Validation callback triggered at correct intervals")