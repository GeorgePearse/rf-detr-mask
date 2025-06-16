#!/usr/bin/env python
"""Run training with full output logging to file."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_training_with_logging():
    # Create logs directory
    logs_dir = Path("training_logs")
    logs_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"training_log_{timestamp}.txt"

    print(f"Starting training run. All output will be logged to: {log_file}")
    print(f"{'='*80}")

    # Training command with small validation interval
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--batch_size",
        "2",
        "--epochs",
        "10",
        "--steps_per_validation",
        "10",
        "--print_per_class_metrics",
        "--output_dir",
        "test_output",
        "--num_workers",
        "2",
        "--lr",
        "1e-4",
        "--lr_encoder",
        "1e-5",
        "--lr_projector",
        "1e-5",
        "--gradient_accumulation_steps",
        "2",
    ]

    # Run training and capture output
    with open(log_file, "w") as log:
        log.write(f"Training started at: {datetime.now()}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        # Run subprocess with both stdout and stderr redirected to log file
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="")  # Print to console
            log.write(line)  # Write to log file
            log.flush()

        # Wait for process to complete
        process.wait()

        log.write(f"\n{'='*80}\n")
        log.write(f"Training completed at: {datetime.now()}\n")
        log.write(f"Exit code: {process.returncode}\n")

    print(f"\n{'='*80}")
    print(f"Training complete. Full log saved to: {log_file}")

    return process.returncode, log_file


if __name__ == "__main__":
    exit_code, log_file = run_training_with_logging()
    sys.exit(exit_code)
