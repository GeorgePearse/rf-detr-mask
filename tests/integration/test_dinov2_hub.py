"""Test training by loading DINOv2 weights from hub."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.slow
def test_dinov2_hub(tmp_path):
    """Test that DINOv2 weights can be loaded from Hugging Face hub."""
    # Create logs directory
    logs_dir = tmp_path / "training_logs"
    logs_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"dinov2_hub_test_{timestamp}.txt"

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Training command without pretrain_weights (will load DINOv2 from hub)
    script_path = project_root / "scripts" / "train.py"
    output_dir = tmp_path / "test_dinov2"
    cmd = [
        sys.executable,
        str(script_path),
        "--batch_size",
        "2",
        "--epochs",
        "1",
        "--steps_per_validation",
        "10",
        "--print_per_class_metrics",
        "--output_dir",
        str(output_dir),
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

    # Expected to find in output
    expected_patterns = [
        "Loading DINOv2",  # Should see DINOv2 loading message
        "Per-Class",  # Should print per-class metrics
        "Averaged stats:",  # Should complete validation
    ]

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
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            log.write(line)
            log.flush()

        # Wait for completion
        process.wait()

        log.write(f"\n{'='*80}\n")
        log.write(f"DINOv2 hub test completed at: {datetime.now()}\n")
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

    # Verify output directory was created
    assert output_dir.exists(), "Output directory was not created"

    # Check that checkpoint was saved
    checkpoints = list(output_dir.glob("checkpoint*.pth"))
    assert len(checkpoints) > 0, "No checkpoints were saved"
