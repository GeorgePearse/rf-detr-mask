#!/usr/bin/env python3
"""Minimal training test to verify dtype fix."""

import subprocess
import sys
import os
import shutil

# Clean up any previous test output
if os.path.exists("test_minimal_output"):
    shutil.rmtree("test_minimal_output")

# Create a minimal config file for faster testing
config_content = """
TRAIN_SIZE: 2
VAL_SIZE: 2
BATCH_SIZE: 1
EPOCHS: 1
"""

os.makedirs("test_minimal_output", exist_ok=True)
with open("test_minimal_output/config.yaml", "w") as f:
    f.write(config_content)

# Run training with minimal settings
cmd = [
    sys.executable,
    "scripts/train.py",
    "--batch_size",
    "1",
    "--epochs",
    "1",
    "--amp",  # Enable mixed precision to trigger the dtype issue
    "--output_dir",
    "test_minimal_output",
    "--masks",  # Enable segmentation
    "--num_classes",
    "8",
    "--eval",  # Enable evaluation
]

print("Running minimal training loop to verify dtype fix...")
print("This should complete without dtype errors.\n")

result = subprocess.run(cmd, capture_output=True, text=True)

# Check for the specific error
if "mat1 and mat2 must have the same dtype" in result.stderr:
    print("✗ DTYPE ERROR STILL PRESENT!")
    print("\nError output:")
    print(result.stderr)
    sys.exit(1)
elif result.returncode != 0:
    print(f"✗ Training failed with return code: {result.returncode}")
    print("\nSTDOUT:")
    print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
    print("\nSTDERR:")
    print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
    sys.exit(1)
else:
    print("✓ Training completed successfully! No dtype errors.")
    print("\nTraining output (last 20 lines):")
    lines = result.stdout.strip().split("\n")
    for line in lines[-20:]:
        print(line)
