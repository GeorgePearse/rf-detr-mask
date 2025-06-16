#!/usr/bin/env python
"""Test training with pretrained weights to get meaningful metrics faster."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def test_with_pretrained():
    # Create logs directory
    logs_dir = Path("training_logs")
    logs_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pretrained_test_{timestamp}.txt"

    print(
        f"Starting test with pretrained weights. Output will be logged to: {log_file}"
    )
    print(f"{'='*80}")

    # Training command with pretrained weights
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--batch_size",
        "4",  # Increased batch size for faster evaluation
        "--epochs",
        "1",
        "--steps_per_validation",
        "20",  # Less frequent validation
        "--print_per_class_metrics",
        "--output_dir",
        "test_pretrained",
        "--num_workers",
        "4",  # More workers for faster data loading
        "--lr",
        "1e-5",  # Lower learning rate for fine-tuning
        "--lr_encoder",
        "1e-6",
        "--gradient_accumulation_steps",
        "1",
        "--pretrain_weights",
        "rf-detr-base.pth",  # Use pretrained weights
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("Note: Pretrained weights will be downloaded if not already available")

    # Run training and capture output
    with open(log_file, "w") as log:
        log.write(f"Pretrained test started at: {datetime.now()}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        # Run subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Stream output
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        # Wait for completion
        process.wait()

        log.write(f"\n{'='*80}\n")
        log.write(f"Pretrained test completed at: {datetime.now()}\n")
        log.write(f"Exit code: {process.returncode}\n")

    print(f"\n{'='*80}")
    print(f"Pretrained test complete. Full log saved to: {log_file}")

    # Check results
    with open(log_file, "r") as f:
        content = f.read()

        # Check for per-class metrics
        if "Per-Class" in content:
            print("\n✅ SUCCESS: Per-class metrics were printed!")

            # Extract and show AP@0.5 metrics
            lines = content.split("\n")
            found_ap50 = False
            for i, line in enumerate(lines):
                if "Per-Class AP @ IoU=0.5" in line:
                    found_ap50 = True
                    print("\nFound per-class AP@0.5 metrics:")
                    # Print the header and next 20 lines
                    for j in range(i, min(i + 25, len(lines))):
                        print(lines[j])
                    break

            if not found_ap50:
                print("\n⚠️ WARNING: AP@0.5 metrics not found specifically")
        else:
            print("\n❌ WARNING: Per-class metrics were NOT found in the output!")

        # Check for download of pretrained weights
        if "Downloading pretrained weights" in content:
            print("\n📥 Pretrained weights were downloaded")

        # Check for validation completion
        if "Averaged stats:" in content:
            print("\n✅ Validation completed successfully")

    return process.returncode, log_file


if __name__ == "__main__":
    exit_code, log_file = test_with_pretrained()
    sys.exit(exit_code)
