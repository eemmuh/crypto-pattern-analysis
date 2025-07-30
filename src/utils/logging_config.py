"""
Logging configuration for crypto trading analysis.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json

from config.settings import LOGGING_SETTINGS


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


def setup_logging(
    log_level: str = None,
    log_file: str = None,
    log_format: str = None,
    max_bytes: int = None,
    backup_count: int = None,
    console_output: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Log message format
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output to console
        json_format: Whether to use JSON format for file logging
        
    Returns:
        Configured logger
    """
    # Use settings if not provided
    log_level = log_level or LOGGING_SETTINGS['level']
    log_file = log_file or LOGGING_SETTINGS['file']
    log_format = log_format or LOGGING_SETTINGS['format']
    max_bytes = max_bytes or LOGGING_SETTINGS['max_bytes']
    backup_count = backup_count or LOGGING_SETTINGS['backup_count']
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if json_format:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColoredFormatter(log_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        if json_format:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_formatter = JSONFormatter()
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_formatter = logging.Formatter(log_format)
        
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None, level: str = "DEBUG"):
    """
    Log function call details.
    
    Args:
        func_name: Name of the function
        args: Function arguments
        kwargs: Function keyword arguments
        level: Logging level
    """
    logger = get_logger(__name__)
    
    message = f"Function call: {func_name}"
    if args:
        message += f" | Args: {args}"
    if kwargs:
        message += f" | Kwargs: {kwargs}"
    
    getattr(logger, level.lower())(message)


def log_performance(func_name: str, execution_time: float, level: str = "INFO"):
    """
    Log function performance.
    
    Args:
        func_name: Name of the function
        execution_time: Execution time in seconds
        level: Logging level
    """
    logger = get_logger(__name__)
    
    message = f"Performance: {func_name} took {execution_time:.4f} seconds"
    getattr(logger, level.lower())(message)


def log_data_info(data: Any, data_name: str = "data", level: str = "DEBUG"):
    """
    Log information about data.
    
    Args:
        data: Data to log info about
        data_name: Name of the data
        level: Logging level
    """
    logger = get_logger(__name__)
    
    if hasattr(data, 'shape'):
        message = f"{data_name} shape: {data.shape}"
    elif hasattr(data, '__len__'):
        message = f"{data_name} length: {len(data)}"
    else:
        message = f"{data_name} type: {type(data).__name__}"
    
    getattr(logger, level.lower())(message)


def log_model_info(model: Any, model_name: str = "model", level: str = "INFO"):
    """
    Log information about a model.
    
    Args:
        model: Model to log info about
        model_name: Name of the model
        level: Logging level
    """
    logger = get_logger(__name__)
    
    message = f"{model_name} type: {type(model).__name__}"
    
    # Add model-specific information
    if hasattr(model, 'n_clusters'):
        message += f" | Clusters: {model.n_clusters}"
    if hasattr(model, 'n_regimes'):
        message += f" | Regimes: {model.n_regimes}"
    if hasattr(model, 'n_components'):
        message += f" | Components: {model.n_components}"
    
    getattr(logger, level.lower())(message)


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any] = None,
    level: str = "ERROR"
):
    """
    Log error with context information.
    
    Args:
        error: Exception that occurred
        context: Additional context information
        level: Logging level
    """
    logger = get_logger(__name__)
    
    message = f"Error: {type(error).__name__}: {str(error)}"
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        message += f" | Context: {context_str}"
    
    getattr(logger, level.lower())(message)


def log_startup_info():
    """Log startup information."""
    logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("Crypto Trading Analysis - Starting Up")
    logger.info("=" * 50)
    logger.info(f"Startup time: {datetime.now().isoformat()}")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Platform: {os.sys.platform}")
    logger.info("=" * 50)


def log_shutdown_info():
    """Log shutdown information."""
    logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("Crypto Trading Analysis - Shutting Down")
    logger.info("=" * 50)
    logger.info(f"Shutdown time: {datetime.now().isoformat()}")
    logger.info("=" * 50)


# Performance monitoring decorator
def log_execution_time(level: str = "INFO"):
    """
    Decorator to log function execution time.
    
    Args:
        level: Logging level for performance log
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            log_performance(func.__name__, execution_time, level)
            
            return result
        return wrapper
    return decorator


# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging() 