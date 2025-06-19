#!/usr/bin/env python
"""Validate that refactored components work correctly.

This script runs a series of tests to ensure the refactored architecture
is working properly before full integration.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and check its exit status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {description} passed!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ {description} failed!")
        print(f"Error output:\n{result.stderr}")
        return False
    
    return True


def main():
    """Run validation tests for refactored components."""
    print("RF-DETR Refactoring Validation")
    print("==============================")
    
    all_passed = True
    
    # 1. Test component integration
    if not run_command(
        ["python", "-m", "pytest", "tests/integration/test_refactored_integration.py", "-v"],
        "Component integration tests"
    ):
        all_passed = False
    
    # 2. Test backward compatibility
    if not run_command(
        ["python", "-m", "pytest", "tests/integration/test_refactored_compatibility.py", "-v"],
        "Backward compatibility tests"
    ):
        all_passed = False
    
    # 3. Test existing unit tests still pass
    if not run_command(
        ["python", "-m", "pytest", "tests/test_model_loading.py", "-v"],
        "Existing model loading tests"
    ):
        all_passed = False
    
    # 4. Quick training test with legacy approach
    print("\n" + "="*60)
    print("Testing legacy training pipeline...")
    print("="*60)
    
    legacy_cmd = [
        "python", "scripts/train.py",
        "--batch_size", "2",
        "--epochs", "1",
        "--steps_per_validation", "10",
        "--test_limit", "20",
        "--num_workers", "0",
        "--output_dir", "/tmp/rfdetr_legacy_test"
    ]
    
    if not run_command(legacy_cmd, "Legacy training pipeline"):
        all_passed = False
    
    # 5. Type checking with mypy
    if not run_command(
        ["mypy", "rfdetr/core/", "--ignore-missing-imports"],
        "Type checking for refactored components"
    ):
        print("Warning: Type checking failed (non-critical)")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if all_passed:
        print("✓ All validation tests passed!")
        print("\nNext steps:")
        print("1. Review the refactored component implementations in rfdetr/core/")
        print("2. Run the example training script: python examples/train_with_refactored_components.py")
        print("3. Gradually migrate existing code to use refactored components")
        print("4. Add feature flag to train.py for A/B testing")
        return 0
    else:
        print("✗ Some validation tests failed!")
        print("\nPlease fix the failing tests before proceeding with migration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())