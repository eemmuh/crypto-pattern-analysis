"""
Error handling utilities for crypto trading analysis.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Callable
from functools import wraps
import time
import sys

logger = logging.getLogger(__name__)


class CryptoAnalysisError(Exception):
    """Base exception for crypto analysis errors."""
    pass


class DataCollectionError(CryptoAnalysisError):
    """Raised when data collection fails."""
    pass


class ValidationError(CryptoAnalysisError):
    """Raised when data validation fails."""
    pass


class ModelError(CryptoAnalysisError):
    """Raised when model operations fail."""
    pass


class ConfigurationError(CryptoAnalysisError):
    """Raised when configuration is invalid."""
    pass


class NetworkError(CryptoAnalysisError):
    """Raised when network operations fail."""
    pass


class RetryableError(CryptoAnalysisError):
    """Raised for errors that can be retried."""
    pass


class NonRetryableError(CryptoAnalysisError):
    """Raised for errors that should not be retried."""
    pass


def handle_exceptions(
    error_types: tuple = (Exception,),
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False
) -> Callable:
    """
    Decorator to handle exceptions gracefully.
    
    Args:
        error_types: Tuple of exception types to catch
        default_return: Value to return if exception occurs
        log_error: Whether to log the error
        reraise: Whether to reraise the exception after handling
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (RetryableError, ConnectionError, TimeoutError),
    on_failure: Optional[Callable] = None
) -> Callable:
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to retry on
        on_failure: Function to call if all retries fail
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {wait_time:.2f} seconds..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                        )
                        if on_failure:
                            return on_failure(*args, **kwargs)
                        raise last_exception
            
            return None  # Should never reach here
        return wrapper
    return decorator


def validate_data(
    data: Any,
    required_fields: Optional[list] = None,
    data_type: Optional[type] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_empty: bool = False
) -> bool:
    """
    Validate data structure and content.
    
    Args:
        data: Data to validate
        required_fields: List of required fields/columns
        data_type: Expected data type
        min_length: Minimum length/size
        max_length: Maximum length/size
        allow_empty: Whether empty data is allowed
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check if data is empty
        if not allow_empty and (data is None or (hasattr(data, '__len__') and len(data) == 0)):
            raise ValidationError("Data cannot be empty")
        
        # Check data type
        if data_type and not isinstance(data, data_type):
            raise ValidationError(f"Expected {data_type.__name__}, got {type(data).__name__}")
        
        # Check length constraints
        if hasattr(data, '__len__'):
            if min_length is not None and len(data) < min_length:
                raise ValidationError(f"Data length {len(data)} is less than minimum {min_length}")
            
            if max_length is not None and len(data) > max_length:
                raise ValidationError(f"Data length {len(data)} exceeds maximum {max_length}")
        
        # Check required fields (for dict-like objects)
        if required_fields and hasattr(data, 'keys'):
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValidationError(f"Missing required fields: {missing_fields}")
        
        return True
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Validation failed: {str(e)}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def format_error_message(error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Format error message with context.
    
    Args:
        error: Exception that occurred
        context: Additional context information
        
    Returns:
        Formatted error message
    """
    message = f"{type(error).__name__}: {str(error)}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        message += f" | Context: {context_str}"
    
    return message


def log_error_with_context(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "ERROR"
) -> None:
    """
    Log error with context information.
    
    Args:
        error: Exception that occurred
        context: Additional context information
        level: Logging level
    """
    message = format_error_message(error, context)
    
    if level.upper() == "DEBUG":
        logger.debug(message)
    elif level.upper() == "INFO":
        logger.info(message)
    elif level.upper() == "WARNING":
        logger.warning(message)
    elif level.upper() == "ERROR":
        logger.error(message)
    elif level.upper() == "CRITICAL":
        logger.critical(message)
    
    # Log traceback at debug level
    logger.debug(f"Traceback: {traceback.format_exc()}")


def create_error_report(
    error: Exception,
    function_name: str,
    args: tuple,
    kwargs: dict,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive error report.
    
    Args:
        error: Exception that occurred
        function_name: Name of the function where error occurred
        args: Function arguments
        kwargs: Function keyword arguments
        context: Additional context
        
    Returns:
        Error report dictionary
    """
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "function_name": function_name,
        "timestamp": time.time(),
        "args": str(args),
        "kwargs": str(kwargs),
        "context": context or {},
        "traceback": traceback.format_exc(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        }
    }


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is retryable
    """
    retryable_types = (
        ConnectionError,
        TimeoutError,
        RetryableError,
        OSError,
        NetworkError
    )
    
    return isinstance(error, retryable_types)


def handle_critical_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Handle critical errors that require immediate attention.
    
    Args:
        error: Critical error
        context: Additional context
    """
    logger.critical("CRITICAL ERROR DETECTED")
    log_error_with_context(error, context, "CRITICAL")
    
    # Could add additional handling here:
    # - Send notifications
    # - Create incident tickets
    # - Trigger alerts
    # - Save error to persistent storage 