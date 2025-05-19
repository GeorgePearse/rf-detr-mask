"""
Error handling utilities for RF-DETR.

This module provides standardized error handling functions and custom exception
classes for the RF-DETR codebase. Use these utilities to ensure consistent
error handling and logging throughout the project.
"""

import functools
import logging
import sys
import traceback
from typing import Any, Callable, Optional, Type, TypeVar, cast

from rfdetr.util.logging_config import get_logger

# Type variable for function return type
T = TypeVar("T")


class RFDETRError(Exception):
    """Base exception class for all RF-DETR specific exceptions."""

    pass


class ConfigurationError(RFDETRError):
    """Exception raised for errors in the configuration settings."""

    pass


class DataError(RFDETRError):
    """Exception raised for errors related to data loading or processing."""

    pass


class ModelError(RFDETRError):
    """Exception raised for errors in model construction or execution."""

    pass


class TrainingError(RFDETRError):
    """Exception raised for errors during model training."""

    pass


class ExportError(RFDETRError):
    """Exception raised for errors during model export."""

    pass


def log_exception(
    logger: logging.Logger,
    exception: Exception,
    message: str = "An error occurred",
    level: int = logging.ERROR,
) -> None:
    """
    Log an exception with a custom message at the specified logging level.

    Args:
        logger: The logger to use
        exception: The exception that was caught
        message: Custom message to provide context for the error
        level: Logging level (defaults to ERROR)
    """
    # Format exception with traceback for debugging
    exc_info = sys.exc_info()
    exc_tb = "".join(traceback.format_exception(*exc_info))

    # Log the error with the custom message and full traceback
    logger.log(
        level,
        f"{message}: {type(exception).__name__}: {exception!s}\n{exc_tb}",
    )


def handle_exception(
    exception_type: Type[Exception] = Exception,
    message: str = "An error occurred",
    reraise: bool = True,
    fallback_return: Any = None,
    log_level: int = logging.ERROR,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator for standardized exception handling.

    Args:
        exception_type: The type of exception to catch
        message: Custom message to log when an exception occurs
        reraise: Whether to re-raise the exception after logging
        fallback_return: Value to return if exception is caught and reraise is False
        log_level: Logging level to use for the error message

    Returns:
        The decorated function

    Example:
        @handle_exception(DataError, message="Failed to load dataset", reraise=False)
        def load_dataset(path):
            # function implementation
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                # Construct a more detailed message that includes function context
                detailed_message = f"{message} in {func.__name__}"

                # Log the exception with appropriate context
                log_exception(logger, e, detailed_message, log_level)

                # Either reraise or return the fallback value
                if reraise:
                    raise
                return fallback_return

        return cast(Callable[..., T], wrapper)

    return decorator


def try_except(
    operation: str,
    logger: Optional[logging.Logger] = None,
    exception_types: tuple[Type[Exception], ...] = (Exception,),
    reraise: bool = True,
    fallback: Any = None,
    log_level: int = logging.ERROR,
) -> Any:
    """
    Context manager for standardized exception handling.

    Args:
        operation: Description of the operation being performed (for logging)
        logger: Logger to use, if None a logger will be created based on the caller's module
        exception_types: Tuple of exception types to catch
        reraise: Whether to re-raise the exception after logging
        fallback: Value to return if exception is caught and reraise is False
        log_level: Logging level to use for the error message

    Example:
        with try_except("loading configuration file", logger=my_logger) as result:
            result.value = json.loads(config_text)
    """

    # Use a class to hold the result so it can be modified within the context
    class Result:
        def __init__(self) -> None:
            self.value: Any = None
            self.exception: Optional[Exception] = None
            self.success: bool = False

    result = Result()

    class TryExcept:
        def __enter__(self) -> Result:
            return result

        def __exit__(self, exc_type: Optional[Type[BaseException]], 
                    exc_val: Optional[BaseException], 
                    exc_tb: Optional[traceback.TracebackType]) -> bool:
            if exc_type is not None and issubclass(exc_type, exception_types):
                # Get the logger if not provided
                nonlocal logger
                if logger is None:
                    # Get the calling module's name
                    frame = sys._getframe(1)
                    module = frame.f_globals.get("__name__", "__main__")
                    logger = get_logger(module)

                # Log the exception
                assert exc_val is not None  # for type checking
                log_exception(logger, exc_val, f"Error while {operation}", log_level)

                # Store exception info
                result.exception = exc_val
                result.success = False

                # Set fallback value
                result.value = fallback

                # Return True to suppress the exception, False to propagate it
                return not reraise

            # If no exception or not of the specified types, don't handle
            if exc_type is None:
                result.success = True

            return False

    return TryExcept()
