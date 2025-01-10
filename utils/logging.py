import logging
import sys
from pathlib import Path

def setup_logger(
    name: str = "main",
    log_level: int = logging.INFO,
    log_file: str = None,
    console_output: bool = True,
    log_format: str = '%(message)s'  # Changed default format to just show message
) -> logging.Logger:
    """Configure and return a logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
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