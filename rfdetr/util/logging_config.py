"""
Logging configuration for RF-DETR.

This module configures the Python logging system for the project, setting up
handlers for different log levels and implementing log rotation.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Optional


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    info_log_backup_count: int = 5,
    debug_backup_count: int = 3,
    error_backup_count: int = 10,
    console_level: int = logging.INFO,
) -> Dict[str, logging.Handler]:
    """
    Set up logging configuration for the project.

    Args:
        log_dir: Directory to store log files
        log_level: Default logging level for file handlers
        max_file_size: Maximum size in bytes before rotating a log file
        info_log_backup_count: Number of info log backups to keep
        debug_backup_count: Number of debug log backups to keep
        error_backup_count: Number of error log backups to keep
        console_level: Logging level for console output

    Returns:
        Dictionary containing the created handlers
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs but filter at handler level

    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Setup file handlers for different log levels
    debug_file = os.path.join(log_dir, "debug.log")
    debug_handler = logging.handlers.RotatingFileHandler(
        debug_file, maxBytes=max_file_size, backupCount=debug_backup_count
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    root_logger.addHandler(debug_handler)

    info_file = os.path.join(log_dir, "info.log")
    info_handler = logging.handlers.RotatingFileHandler(
        info_file, maxBytes=max_file_size, backupCount=info_log_backup_count
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    root_logger.addHandler(info_handler)

    warning_file = os.path.join(log_dir, "warning.log")
    warning_handler = logging.handlers.RotatingFileHandler(
        warning_file, maxBytes=max_file_size, backupCount=error_backup_count
    )
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(formatter)
    root_logger.addHandler(warning_handler)

    error_file = os.path.join(log_dir, "error.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_file, maxBytes=max_file_size, backupCount=error_backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Create separate file for exceptions
    exception_file = os.path.join(log_dir, "exception.log")
    exception_handler = logging.FileHandler(exception_file, mode="a")
    exception_handler.setLevel(logging.ERROR)

    # Custom formatter for exceptions that includes traceback
    exception_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s"
    )
    exception_handler.setFormatter(exception_formatter)

    # Create a filter to only log records with exc_info
    class ExceptionFilter(logging.Filter):
        def filter(self, record):
            return record.exc_info is not None

    exception_handler.addFilter(ExceptionFilter())
    root_logger.addHandler(exception_handler)

    # Keep track of the handlers
    handlers = {
        "console": console_handler,
        "debug": debug_handler,
        "info": info_handler,
        "warning": warning_handler,
        "error": error_handler,
        "exception": exception_handler,
    }

    # Log setup completed
    logging.info(f"Logging is configured. Log files are in {os.path.abspath(log_dir)}")

    return handlers


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the given name. If logging hasn't been set up yet,
    configure it with default settings.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Configured logger instance
    """
    if not logging.getLogger().handlers:
        # Logging hasn't been set up yet, set it up with defaults
        setup_logging()

    return logging.getLogger(name)
