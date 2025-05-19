#!/usr/bin/env python3
"""
Script to fix common mypy errors in the codebase
"""

import os
import re
from typing import List, Tuple

# Map of files to fix and their issues
FILES_TO_FIX = {
    # Key deployment files
    "rfdetr/deploy/_onnx/symbolic.py": {
        "fixes": [
            ("_OPTIMIZER", "_OPTIMIZER: list[str] = "),
            (r"def register_custom_op_symbolic\(\):", "def register_custom_op_symbolic() -> None:"),
            (r"def custom_op_symbolic\(", "def custom_op_symbolic("),
            (r"def optimizer\(\):", "def optimizer() -> None:"),
        ]
    },
    "rfdetr/deploy/_onnx/optimizer.py": {
        "fixes": [
            (r"def set_severity\(level\):", "def set_severity(level: int) -> None:"),
            (r"def cleanup\(\):", "def cleanup() -> None:"),
            (r"def onnx_simplify\(onnx_dir, ", "def onnx_simplify(onnx_dir: str, "),
            (r"def export_onnx\(", "def export_onnx("),
            (r"def clean_onnx_model\(", "def clean_onnx_model("),
            (r"def optimize_onnx_model\(", "def optimize_onnx_model("),
            (r"def replace_node_with_identity\(", "def replace_node_with_identity("),
            (r"def remove_initializer_if_unused\(", "def remove_initializer_if_unused("),
            (r"def process_slice_nodes\(", "def process_slice_nodes("),
            (r"def fix_gather_op\(", "def fix_gather_op("),
        ]
    },
    # Utility files
    "rfdetr/util/early_stopping.py": {
        "fixes": [
            (r"def __init__\(self", "def __init__(self"),
            (r"def update\(self", "def update(self"),
        ]
    },
    "rfdetr/util/logging_config.py": {
        "fixes": [
            (
                r"def configure_handlers\(console_level=logging.INFO, file_level=logging.DEBUG, log_file=None\):",
                "def configure_handlers(console_level=logging.INFO, file_level=logging.DEBUG, log_file=None) -> dict[str, logging.Handler]:",
            ),
        ]
    },
    # Add more files as needed
}


def fix_file(file_path: str, fixes: List[Tuple[str, str]]) -> bool:
    """Apply regex replacements to fix mypy errors in a file.

    Args:
        file_path: Path to the file to fix
        fixes: List of (pattern, replacement) tuples

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read file content
        with open(file_path) as f:
            content = f.read()

        # Apply fixes
        modified = False
        for pattern, replacement in fixes:
            # Special case for adding type annotations to variables
            if pattern == "_OPTIMIZER" and "_OPTIMIZER = " in content:
                content = content.replace("_OPTIMIZER = ", replacement)
                modified = True
            else:
                # Regular expression replacement
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    modified = True

        # Write back to file if changes were made
        if modified:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"✓ Fixed {file_path}")
            return True
        else:
            print(f"No changes needed for {file_path}")
            return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main() -> None:
    """Apply fixes to all specified files"""
    print("Starting mypy error fixes...")

    fixed_count = 0
    for file_path, fix_info in FILES_TO_FIX.items():
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(full_path):
            if fix_file(full_path, fix_info["fixes"]):
                fixed_count += 1
        else:
            print(f"File not found: {full_path}")

    print(f"Completed fixes for {fixed_count} files")
    print(
        "Note: This script only fixes basic type annotation issues. You may need to manually fix more complex issues."
    )


if __name__ == "__main__":
    main()
