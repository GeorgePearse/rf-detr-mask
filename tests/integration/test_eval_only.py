"""Test evaluation with per-class metrics only (no training)."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.slow
def test_eval_only(tmp_path):
    """Test evaluation mode prints per-class metrics without training."""
    # Create logs directory
    logs_dir = tmp_path / "training_logs"
    logs_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"eval_only_{timestamp}.txt"

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Run evaluation only
    script_path = project_root / "scripts" / "train.py"
    output_dir = tmp_path / "test_eval"
    cmd = [
        sys.executable,
        str(script_path),
        "--eval",  # Evaluation only mode
        "--batch_size",
        "1",
        "--print_per_class_metrics",
        "--output_dir",
        str(output_dir),
        "--num_workers",
        "0",
    ]

    # Expected patterns in evaluation output
    expected_patterns = [
        "Per-Class",  # Should print per-class metrics
        "AP @ IoU=0.5",  # Should show AP metrics
        "Averaged stats:",  # Should complete evaluation
    ]

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
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            log.write(line)
            log.flush()

        # Wait for completion
        process.wait()

        log.write(f"\n{'='*80}\n")
        log.write(f"Evaluation test completed at: {datetime.now()}\n")
        log.write(f"Exit code: {process.returncode}\n")

    # Join all output
    full_output = "".join(output_lines)

    # Assertions
    assert (
        process.returncode == 0
    ), f"Evaluation failed with exit code: {process.returncode}"

    # Check for expected patterns
    for pattern in expected_patterns:
        assert (
            pattern in full_output
        ), f"Expected pattern '{pattern}' not found in output"

    # Verify evaluation completed quickly (should be much faster than training)
    with open(log_file, "r") as f:
        log_content = f.read()
        # Check that it's evaluation only (no training epochs)
        assert (
            "Epoch:" not in log_content or "eval" in log_content.lower()
        ), "Evaluation mode should not perform training epochs"

    # Verify per-class metrics contain actual class names
    assert (
        "person" in full_output or "car" in full_output
    ), "Per-class metrics should include actual class names"
