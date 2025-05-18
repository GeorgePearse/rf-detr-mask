"""
Initialize the logs directory structure.

This module is imported and run early in the application lifecycle to ensure
the logs directory is created and accessible.
"""

import os
from pathlib import Path


def init_logs_directory(base_dir: str = None) -> str:
    """
    Create the logs directory structure if it doesn't exist.

    Args:
        base_dir: Optional base directory, if None, creates in current directory

    Returns:
        Absolute path to the created logs directory
    """
    # Determine the base directory for logs
    if base_dir is None:
        # Use the directory where the project is installed/running from
        base_dir = os.getcwd()

    # Create the logs directory
    logs_dir = os.path.join(base_dir, "logs")
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    # Return the absolute path to the logs directory
    return os.path.abspath(logs_dir)


if __name__ == "__main__":
    # This allows the script to be run directly to initialize the logs directory
    logs_path = init_logs_directory()
    print(f"Logs directory initialized at: {logs_path}")
