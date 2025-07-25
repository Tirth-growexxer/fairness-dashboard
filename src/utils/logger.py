"""Centralized logging configuration for the fairness dashboard."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        logging.DEBUG: '\033[0;36m',    # Cyan
        logging.INFO: '\033[0;32m',     # Green
        logging.WARNING: '\033[0;33m',  # Yellow
        logging.ERROR: '\033[0;31m',    # Red
        logging.CRITICAL: '\033[0;35m'  # Purple
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelno)
        record.levelname = f'{color}{record.levelname}{self.RESET}'
        record.msg = f'{color}{record.msg}{self.RESET}'
        return super().format(record)

def setup_logger(
    name: str = "fairness_dashboard",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file. If None, timestamp-based file is created
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    console_formatter = CustomFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(f'logs/fairness_dashboard_{timestamp}.log')
    
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logger()

def log_step(step_name: str, total_steps: Optional[int] = None) -> None:
    """Log a step in a multi-step process."""
    if total_steps:
        logger.info(f"Step: {step_name} ({total_steps} steps total)")
    else:
        logger.info(f"Step: {step_name}")

def log_progress(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50
) -> None:
    """
    Log a progress bar.
    
    Args:
        current: Current progress value
        total: Total value for 100% progress
        prefix: Prefix string
        suffix: Suffix string
        decimals: Number of decimal places for percentage
        length: Character length of the progress bar
    """
    percent = f"{100 * (current / float(total)):.{decimals}f}"
    filled = int(length * current // total)
    bar = "█" * filled + "-" * (length - filled)
    logger.info(f"\r{prefix} |{bar}| {percent}% {suffix}")

def log_error_trace(error: Exception) -> None:
    """Log an error with its full traceback."""
    import traceback
    logger.error(f"Error: {str(error)}")
    logger.debug(f"Traceback:\n{''.join(traceback.format_tb(error.__traceback__))}")

def log_model_metrics(model_name: str, metrics: dict) -> None:
    """Log model evaluation metrics."""
    logger.info(f"\nMetrics for {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

def log_fairness_metrics(model_name: str, attribute: str, metrics: dict) -> None:
    """Log fairness metrics for a specific attribute."""
    logger.info(f"\nFairness Metrics for {model_name} - {attribute}:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}") 