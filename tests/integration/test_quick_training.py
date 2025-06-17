#!/usr/bin/env python
"""Quick test script to verify per-class metrics with minimal training."""

import pytest
import subprocess
import sys
from pathlib import Path


@pytest.mark.e2e
@pytest.mark.slow
def test_quick_training_with_metrics(tmp_path):
    """Test minimal training run with per-class metrics enabled."""

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Create log file in tmp directory
    log_file = tmp_path / "quick_test.txt"

    # Training command with very small validation interval
    script_path = project_root / "scripts" / "train.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--batch_size",
        "1",
        "--epochs",
        "1",
        "--steps_per_validation",
        "5",  # Small interval for quick test
        "--print_per_class_metrics",
        "--output_dir",
        str(tmp_path / "test_output"),
        "--num_workers",
        "0",  # Disable multiprocessing for faster startup
        "--lr",
        "1e-4",
        "--gradient_accumulation_steps",
        "1",
        "--test_limit",
        "10",  # Limit dataset size for speed
    ]

    # Run training and capture output
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            log.write(line)
            log.flush()

        # Wait for process to complete
        process.wait()

    # Check that training completed successfully
    assert (
        process.returncode == 0
    ), f"Training failed with exit code {process.returncode}"

    # Check if per-class metrics were printed
    output = "".join(output_lines)
    assert "Per-Class" in output, "Per-class metrics were not printed in the output"

    # Verify that output directory was created
    output_dir = tmp_path / "test_output"
    assert output_dir.exists(), "Output directory was not created"
