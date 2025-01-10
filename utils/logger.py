import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

class Logger:
    """Centralized logger for managing application logging state"""
    
    _instances: Dict[str, logging.Logger] = {}
    _log_dir: Path = Path('logs')  # Default base log directory
    _default_format: str = '%(message)s'
    _initialized: bool = False
    _loggers_created: set = set()  # Add set to track created loggers
    
    @classmethod
    def setup(cls, 
              log_dir: Optional[str] = None,
              log_format: Optional[str] = None,
              default_level: int = logging.INFO) -> None:
        """Initialize global logging configuration"""
        if log_dir:
            cls._log_dir = Path(log_dir)
        cls._log_dir = cls._log_dir.resolve()  # Convert to absolute path
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_format:
            cls._default_format = log_format
            
        # Set basic configuration for root logger
        logging.basicConfig(
            level=default_level,
            format=cls._default_format
        )
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, 
                   name: str,
                   level: int = logging.INFO,
                   filename: Optional[str] = None,
                   console_output: bool = True) -> logging.Logger:
        """Get or create a logger instance"""
        if not cls._initialized:
            cls.setup()
            
        # Return existing logger if already created
        if name in cls._instances:
            return cls._instances[name]
            
        # Create new logger only if it hasn't been created before
        logger_key = f"{name}_{filename}" if filename else name
        if logger_key in cls._loggers_created:
            return cls._instances[name]
            
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(cls._default_format)
        
        # Add file handler if filename is specified
        if filename:
            # Handle both absolute and relative paths
            file_path = Path(filename)
            if not file_path.is_absolute():
                file_path = cls._log_dir / file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(file_path))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Store logger instance
        cls._instances[name] = logger
        cls._loggers_created.add(logger_key)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        return logger
    
    @classmethod
    def get_timestamp_logger(cls,
                            name: str,
                            log_dir: Optional[Path] = None) -> logging.Logger:
        """Get a logger with timestamped file output"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Normalize the log directory path
        if log_dir:
            log_dir = Path(log_dir)
            # Remove 'logs' from path if it exists to prevent duplication
            parts = [p for p in log_dir.parts if p != 'logs']
            clean_path = Path(*parts)
            final_path = cls._log_dir / clean_path
        else:
            final_path = cls._log_dir
            
        filename = f'{name}_{timestamp}.log'
        return cls.get_logger(name, filename=str(final_path / filename))

    @classmethod
    def update_format(cls, new_format: str) -> None:
        """Update format for all existing loggers"""
        cls._default_format = new_format
        formatter = logging.Formatter(new_format)
        
        for logger in cls._instances.values():
            for handler in logger.handlers:
                handler.setFormatter(formatter)
