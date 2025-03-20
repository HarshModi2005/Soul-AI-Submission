"""
Logging Configuration Module for NER API Service

This module provides a comprehensive logging setup for the Named Entity Recognition (NER) API:
- Configurable log levels via environment variables
- Multiple output formats (JSON structured logging or standard text)
- Rotating file handlers with size limits
- Request ID tracking across asynchronous operations
- Performance measurement utilities
- Exception handling and reporting

Usage:
    from api.logging_config import logger, log_request, log_performance, log_prediction

    @log_request
    async def handler(request):
        with log_performance("database_query"):
            # ... code ...
        
        log_prediction(len(text), entities, "v1.0.0")
        return response
"""

import logging
import logging.handlers
import os
import json
import uuid
import time
import traceback
import sys
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Any, Dict, List, Callable, TypeVar, Union, cast

# -----------------------------------------------------------------------------
# Configuration Settings
# -----------------------------------------------------------------------------

# Configuration from environment variables with sensible defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "standard")  # Options: "standard" or "json"
LOG_RETENTION = int(os.getenv("LOG_RETENTION_DAYS", "30"))
MAX_LOG_SIZE_MB = int(os.getenv("MAX_LOG_SIZE_MB", "10"))
LOGS_DIR = os.getenv("LOGS_DIR", "logs")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Map string log levels to Python logging level constants
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# -----------------------------------------------------------------------------
# Custom Formatters
# -----------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON objects.
    
    Provides structured logging with standardized fields and
    automatic inclusion of context attributes (request_id, user, etc.)
    """
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.
        
        Args:
            record: The log record to format
            
        Returns:
            A JSON string representation of the log record
        """
        # Build the base log data with standard fields
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "path": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add thread and process info in debug mode for more detailed troubleshooting
        if record.levelno <= logging.DEBUG:
            log_data.update({
                "thread": record.threadName,
                "process": record.processName
            })
            
        # Add context attributes if they exist to enrich the log data
        for attr in ["request_id", "user", "execution_time", "model_version", "entity_count"]:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)
                
        # Add exception info if present for better error tracking
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "trace": traceback.format_exception(*record.exc_info)
            }
                
        return json.dumps(log_data)

# -----------------------------------------------------------------------------
# Logger Setup
# -----------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    """
    Configure and set up the application logger with appropriate handlers and formatters.
    
    Returns:
        A configured Logger instance
    """
    # Determine the actual log level from configuration
    log_level = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)
    
    # Initialize the logger with app name
    logger = logging.getLogger("ner_api")
    logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicates on reinitialization
    # This is important when the function might be called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Define log file paths
    general_log_file = f"{LOGS_DIR}/api.log"
    error_log_file = f"{LOGS_DIR}/api_errors.log"
    
    # Rotating file handler for general logs
    # This prevents log files from growing unbounded
    general_handler = logging.handlers.RotatingFileHandler(
        general_log_file,
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,  # Convert MB to bytes
        backupCount=10  # Keep up to 10 rotated files
    )
    general_handler.setLevel(log_level)
    
    # Dedicated rotating file handler for errors and above
    # This makes it easier to track and respond to errors
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    
    # Console handler for immediate feedback during development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create appropriate formatter based on configuration
    if LOG_FORMAT.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(request_id)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Apply formatters to all handlers
    general_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add all handlers to the logger
    logger.addHandler(general_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    # Add filter to ensure request_id is always available
    # This prevents KeyError when using request_id in format strings
    class RequestContextFilter(logging.Filter):
        """Filter that ensures request_id is available in log records."""
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, 'request_id'):
                record.request_id = 'no-request-id'
            return True
    
    logger.addFilter(RequestContextFilter())
    
    return logger

# Initialize the application logger
logger = setup_logger()

# -----------------------------------------------------------------------------
# Request Context Management
# -----------------------------------------------------------------------------

class RequestContext:
    """
    Static class to manage request context throughout the lifecycle of a request.
    
    Maintains request_id and user information across asynchronous boundaries,
    allowing consistent logging even in complex async workflows.
    """
    _request_id: Optional[str] = None
    _user: Optional[str] = None
    
    @classmethod
    def get_request_id(cls) -> str:
        """
        Get the current request ID or generate a new one if none exists.
        
        Returns:
            The current request ID string
        """
        if cls._request_id is None:
            cls._request_id = str(uuid.uuid4())
        return cls._request_id
    
    @classmethod
    def set_request_id(cls, request_id: str) -> None:
        """
        Set the request ID for the current context.
        
        Args:
            request_id: The request ID to set
        """
        cls._request_id = request_id
    
    @classmethod
    def clear_request_id(cls) -> None:
        """Clear the request ID when a request is complete."""
        cls._request_id = None
    
    @classmethod
    def set_user(cls, user: str) -> None:
        """
        Set the user for the current context.
        
        Args:
            user: User identifier (e.g., username or user ID)
        """
        cls._user = user
        
    @classmethod
    def get_user(cls) -> Optional[str]:
        """
        Get the current user context.
        
        Returns:
            The current user identifier or None if not set
        """
        return cls._user

# -----------------------------------------------------------------------------
# Logging Decorators and Context Managers
# -----------------------------------------------------------------------------

# Type variable for preserving the return type of decorated functions
F = TypeVar('F', bound=Callable[..., Any])

def log_request(func: F) -> F:
    """
    Decorator to log information about API requests.
    
    Automatically generates and tracks request IDs, measures execution time,
    and logs both successful and failed requests with appropriate context.
    
    Args:
        func: The async function to decorate
        
    Returns:
        Wrapped function that logs request details
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Generate and set unique request ID
        request_id = str(uuid.uuid4())
        RequestContext.set_request_id(request_id)
        
        # Start timing the request
        start_time = time.time()
        
        # Log the start of the request
        logger.info(f"Request started: {func.__name__}", 
                   extra={'request_id': request_id})
        
        try:
            # Execute the actual handler function
            response = await func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log successful completion with timing info
            logger.info(
                f"Request completed in {execution_time:.3f}s",
                extra={
                    'request_id': request_id,
                    'execution_time': execution_time,
                    'status_code': getattr(response, 'status_code', 0)
                }
            )
            
            return response
            
        except Exception as e:
            # Calculate execution time for error case
            execution_time = time.time() - start_time
            
            # Log the exception with detailed context
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    'request_id': request_id,
                    'execution_time': execution_time
                },
                exc_info=True
            )
            
            # Re-raise the exception for proper error handling
            raise
        finally:
            # Always clear the request context to prevent leaks between requests
            RequestContext.clear_request_id()
    
    # Cast to preserve the return type information
    return cast(F, wrapper)

@contextmanager
def log_performance(operation_name: str) -> None:
    """
    Context manager for measuring and logging performance of specific operations.
    
    Usage:
        with log_performance("database_query"):
            results = db.execute(query)
    
    Args:
        operation_name: Name of the operation being measured
    """
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger.debug(
            f"Performance: {operation_name} completed in {execution_time:.3f}s",
            extra={
                'request_id': RequestContext.get_request_id(),
                'operation': operation_name,
                'execution_time': execution_time
            }
        )

# -----------------------------------------------------------------------------
# Specialized Logging Functions
# -----------------------------------------------------------------------------

def log_prediction(text_length: int, entities: List[Dict[str, Any]], model_version: str) -> None:
    """
    Log details about NER model predictions.
    
    Args:
        text_length: Length of the input text in characters
        entities: List of detected entities with their details
        model_version: Version of the NER model used
    """
    logger.info(
        f"Prediction processed: {len(entities)} entities found in {text_length} chars",
        extra={
            'request_id': RequestContext.get_request_id(),
            'text_length': text_length,
            'entity_count': len(entities),
            'model_version': model_version,
            'entity_types': {e['label']: 1 for e in entities}
        }
    )

# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------

def log_unhandled_exception(exc_type: type, exc_value: Exception, exc_traceback: traceback) -> None:
    """
    Custom exception hook for logging unhandled exceptions.
    
    Args:
        exc_type: Type of the exception
        exc_value: Exception instance
        exc_traceback: Traceback object
    """
    logger.critical(
        f"Unhandled exception: {exc_type.__name__}: {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
# Register the exception hook to catch unhandled exceptions
sys.excepthook = log_unhandled_exception