#!/usr/bin/env python3
"""Test script to verify dtype fix with minimal training."""

import subprocess
import sys
import os

# Create output directory
os.makedirs("test_dtype_output", exist_ok=True)

# Run training with very short loop to test the fix
cmd = [
    sys.executable,
    "scripts/train.py",
    "--batch_size",
    "2",
    "--epochs",
    "1",  # Just 1 epoch
    "--amp",  # Enable mixed precision
    "--dropout",
    "0.0",
    "--lr",
    "5e-5",
    "--weight_decay",
    "1e-4",
    "--output_dir",
    "test_dtype_output",
    "--masks",  # Enable segmentation
    "--loss_mask_coef",
    "1",
    "--loss_dice_coef",
    "1",
    "--num_classes",
    "8",  # Based on CMR dataset
    "--eval",  # Enable evaluation to trigger the error
    "--print_per_class_metrics",  # This triggers the evaluation callback
]

print("Running minimal training to test dtype fix...")
print("Command:", " ".join(cmd))
print("\nThis will train for 1 epoch with evaluation enabled...")

result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n✓ Training completed successfully! Dtype fix is working.")
else:
    print(f"\n✗ Training failed with return code: {result.returncode}")
    sys.exit(1)
