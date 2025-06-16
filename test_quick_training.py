#!/usr/bin/env python
"""Quick test script to verify per-class metrics with minimal training."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_quick_test():
    # Create logs directory
    logs_dir = Path("training_logs")
    logs_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"quick_test_{timestamp}.txt"

    print(f"Starting quick test. Output will be logged to: {log_file}")
    print(f"{'='*80}")

    # Training command with very small validation interval
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--batch_size",
        "1",
        "--epochs",
        "1",
        "--steps_per_validation",
        "5",  # Even smaller interval
        "--print_per_class_metrics",
        "--output_dir",
        "test_quick",
        "--num_workers",
        "0",  # Disable multiprocessing for faster startup
        "--lr",
        "1e-4",
        "--gradient_accumulation_steps",
        "1",
    ]

    print(f"Running command: {' '.join(cmd)}")

    # Run training and capture output
    with open(log_file, "w") as log:
        log.write(f"Quick test started at: {datetime.now()}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        # Run subprocess with both stdout and stderr redirected
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        # Wait for process to complete
        process.wait()

        log.write(f"\n{'='*80}\n")
        log.write(f"Quick test completed at: {datetime.now()}\n")
        log.write(f"Exit code: {process.returncode}\n")

    print(f"\n{'='*80}")
    print(f"Quick test complete. Full log saved to: {log_file}")

    # Check if per-class metrics were printed
    with open(log_file, "r") as f:
        content = f.read()
        if "Per-Class" in content:
            print("\n✅ SUCCESS: Per-class metrics were printed!")
        else:
            print("\n❌ WARNING: Per-class metrics were NOT found in the output!")

    return process.returncode, log_file


if __name__ == "__main__":
    exit_code, log_file = run_quick_test()
    sys.exit(exit_code)
