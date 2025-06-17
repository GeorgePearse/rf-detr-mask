"""Test training with pretrained weights to get meaningful metrics faster."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.slow
def test_with_pretrained(tmp_path):
    """Test training with pretrained weights downloads weights and produces better metrics."""
    # Create logs directory
    logs_dir = tmp_path / "training_logs"
    logs_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pretrained_test_{timestamp}.txt"

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Training command with pretrained weights
    script_path = project_root / "scripts" / "train.py"
    output_dir = tmp_path / "test_pretrained"
    cmd = [
        sys.executable,
        str(script_path),
        "--batch_size",
        "4",  # Increased batch size for faster evaluation
        "--epochs",
        "1",
        "--steps_per_validation",
        "20",  # Less frequent validation
        "--print_per_class_metrics",
        "--output_dir",
        str(output_dir),
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

    # Expected patterns
    expected_patterns = [
        "Per-Class",  # Should print per-class metrics
        "Averaged stats:",  # Should complete validation
    ]

    # Patterns that indicate pretrained weights were used
    pretrained_indicators = [
        "Loading pretrained weights",
        "Downloading pretrained weights",
        "rf-detr-base.pth",
    ]

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
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            log.write(line)
            log.flush()

        # Wait for completion
        process.wait()

        log.write(f"\n{'='*80}\n")
        log.write(f"Pretrained test completed at: {datetime.now()}\n")
        log.write(f"Exit code: {process.returncode}\n")

    # Join all output
    full_output = "".join(output_lines)

    # Assertions
    assert (
        process.returncode == 0
    ), f"Training failed with exit code: {process.returncode}"

    # Check for expected patterns
    for pattern in expected_patterns:
        assert (
            pattern in full_output
        ), f"Expected pattern '{pattern}' not found in output"

    # Check that pretrained weights were used
    pretrained_found = any(
        indicator in full_output for indicator in pretrained_indicators
    )
    assert pretrained_found, "No indication that pretrained weights were loaded"

    # Verify output directory was created
    assert output_dir.exists(), "Output directory was not created"

    # Check that checkpoint was saved
    checkpoints = list(output_dir.glob("checkpoint*.pth"))
    assert len(checkpoints) > 0, "No checkpoints were saved"

    # Verify AP@0.5 metrics are present
    assert "AP @ IoU=0.5" in full_output, "AP@0.5 metrics not found in output"

    # Check that metrics show actual detection (not all zeros)
    # With pretrained weights, we should see non-zero AP values
    lines = full_output.split("\n")
    ap_lines = [line for line in lines if "AP" in line and ":" in line]

    # At least one AP value should be non-zero
    has_nonzero_ap = False
    for line in ap_lines:
        if any(
            float(part) > 0.0
            for part in line.split()
            if part.replace(".", "").isdigit()
        ):
            has_nonzero_ap = True
            break

    assert (
        has_nonzero_ap
    ), "All AP values are zero - pretrained weights may not be working correctly"
