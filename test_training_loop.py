#!/usr/bin/env python3
"""Test script to verify dtype fix with short training loop."""

import subprocess
import sys

# Run training with very short loop to test the fix
cmd = [
    sys.executable,
    "scripts/train.py",
    "--num_queries",
    "10",
    "--batch_size",
    "2",
    "--num_steps",
    "10",
    "--checkpoint_interval",
    "5",
    "--amp",
    "--backbone",
    "dinov2_small",
    "--d_model",
    "256",
    "--nheads",
    "8",
    "--hidden_dim",
    "1024",
    "--dropout",
    "0.0",
    "--no_aux_loss",
    "--lr",
    "5e-5",
    "--lr_backbone",
    "5e-6",
    "--weight_decay",
    "1e-4",
    "--num_select",
    "10",
    "--output_dir",
    "test_dtype_output",
    "--train_annotations_path",
    "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json",
    "--val_annotations_path",
    "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json",
    "--images_dir",
    "/home/georgepearse/data/images",
    "--return_masks",
]

print("Running short training loop to test dtype fix...")
print("Command:", " ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True)

print("\n=== STDOUT ===")
print(result.stdout)
print("\n=== STDERR ===")
print(result.stderr)

if result.returncode == 0:
    print("\n✓ Training completed successfully!")
else:
    print(f"\n✗ Training failed with return code: {result.returncode}")
    sys.exit(1)
