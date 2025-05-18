#!/usr/bin/env python
"""Test script to verify the steps_per_validation functionality"""

import sys
import subprocess

# Test command with steps_per_validation
cmd = [
    "python", "scripts/train.py",
    "--dataset", "coco",
    "--dataset_file", "coco",
    "--coco_path", "/home/georgepearse/data/images",
    "--train_ann", "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json",
    "--val_ann", "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json",
    "--steps_per_validation", "10",
    "--epochs", "1",
    "--batch_size", "2",
    "--num_classes", "4",
    "--masks",
    "--output_dir", "test_validation_steps"
]

print(f"Testing steps_per_validation functionality...")
print(f"Command: {' '.join(cmd)}")

result = subprocess.run(cmd, capture_output=True, text=True)
print("\n--- STDOUT ---")
print(result.stdout)
if result.stderr:
    print("\n--- STDERR ---")
    print(result.stderr)

# Check if validation was triggered
if "Running validation at step" in result.stdout:
    print("\nSUCCESS: steps_per_validation is working correctly!")
else:
    print("\nERROR: steps_per_validation did not trigger validation")

print(f"\nReturn code: {result.returncode}")