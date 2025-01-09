import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(
    name: str = "main",
    log_level: int = logging.INFO,
    log_file: str = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure and return a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        console_output: Whether to output to console (default: True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if console_output is True
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Example usage
if __name__ == "__main__":
    # Create a logger with both console and file output
    logger = setup_logger(
        name="example",
        log_level=logging.DEBUG,
        log_file=f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    # Test logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")