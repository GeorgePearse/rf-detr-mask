#!/usr/bin/env python
"""Test training by loading DINOv2 weights from hub."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def test_dinov2_hub():
    # Create logs directory
    logs_dir = Path("training_logs")
    logs_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"dinov2_hub_test_{timestamp}.txt"

    print(
        f"Starting test with DINOv2 hub weights. Output will be logged to: {log_file}"
    )
    print(f"{'='*80}")

    # Training command without pretrain_weights (will load DINOv2 from hub)
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--batch_size",
        "2",
        "--epochs",
        "1",
        "--steps_per_validation",
        "10",
        "--print_per_class_metrics",
        "--output_dir",
        "test_dinov2",
        "--num_workers",
        "2",
        "--lr",
        "1e-5",
        "--lr_encoder",
        "1e-6",
        "--gradient_accumulation_steps",
        "1",
        # Note: NOT specifying pretrain_weights, so it will load DINOv2 from hub
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("Note: DINOv2 weights will be loaded from Hugging Face hub")

    # Run training and capture output
    with open(log_file, "w") as log:
        log.write(f"DINOv2 hub test started at: {datetime.now()}\n")
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
        has_per_class = False
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

            # Check for per-class metrics
            if "Per-Class" in line:
                has_per_class = True

        # Wait for completion
        process.wait()

        log.write(f"\n{'='*80}\n")
        log.write(f"DINOv2 hub test completed at: {datetime.now()}\n")
        log.write(f"Exit code: {process.returncode}\n")

    print(f"\n{'='*80}")
    print(f"DINOv2 hub test complete. Full log saved to: {log_file}")

    # Summary
    if has_per_class:
        print("\n✅ SUCCESS: Per-class metrics were printed during training!")
    else:
        print("\n⚠️ Per-class metrics were not seen (may be in the log file)")

    if process.returncode == 0:
        print("✅ Training completed successfully")
    else:
        print(f"❌ Training failed with exit code: {process.returncode}")

    return process.returncode, log_file


if __name__ == "__main__":
    exit_code, log_file = test_dinov2_hub()
    sys.exit(exit_code)
