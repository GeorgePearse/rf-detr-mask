#!/usr/bin/env python
"""Test evaluation with per-class metrics only (no training)."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def test_eval_only():
    # Create logs directory
    logs_dir = Path("training_logs")
    logs_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"eval_only_{timestamp}.txt"

    print(f"Starting evaluation test. Output will be logged to: {log_file}")
    print(f"{'='*80}")

    # Run evaluation only
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--eval",  # Evaluation only mode
        "--batch_size",
        "1",
        "--print_per_class_metrics",
        "--output_dir",
        "test_eval",
        "--num_workers",
        "0",
    ]

    print(f"Running command: {' '.join(cmd)}")

    # Run and capture output
    with open(log_file, "w") as log:
        log.write(f"Evaluation test started at: {datetime.now()}\n")
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
        log.write(f"Evaluation test completed at: {datetime.now()}\n")
        log.write(f"Exit code: {process.returncode}\n")

    print(f"\n{'='*80}")
    print(f"Evaluation test complete. Full log saved to: {log_file}")

    # Check results
    with open(log_file, "r") as f:
        content = f.read()
        if "Per-Class" in content:
            print("\n✅ SUCCESS: Per-class metrics were printed!")
            # Show a snippet of the metrics
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "Per-Class" in line:
                    print("\nFound per-class metrics:")
                    print("\n".join(lines[i : min(i + 20, len(lines))]))
                    break
        else:
            print("\n❌ WARNING: Per-class metrics were NOT found in the output!")

    return process.returncode, log_file


if __name__ == "__main__":
    exit_code, log_file = test_eval_only()
    sys.exit(exit_code)
