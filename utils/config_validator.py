from typing import Dict, Any, Optional
import os
import yaml

class ConfigValidationError(Exception):
    """Custom exception for config validation errors"""
    pass

class ConfigValidator:
    """Centralized configuration validation"""
    
    REQUIRED_MONITORING = {
        'enabled': bool,
        'metrics': {
            'save_interval': int,
            'plot_interval': int
        },
        'memory': {
            'track': bool
        },
        'performance': {
            'track_time': bool,
            'track_throughput': bool
        }
    }
    
    @staticmethod
    def validate_paths(config: Dict[str, Any]) -> None:
        """Validate all file paths in config"""
        required_paths = {
            'data.train_path': 'Training data file',
            'data.val_path': 'Validation data file',
            'model.architecture_yaml': 'Model architecture YAML',
            'model.save_path': 'Model checkpoint path'
        }
        
        for path_key, description in required_paths.items():
            sections = path_key.split('.')
            current = config
            for section in sections:
                if section not in current:
                    raise ConfigValidationError(f"Missing {description} path in config: {path_key}")
                current = current[section]
            
            if not isinstance(current, str):
                raise ConfigValidationError(f"{description} path must be a string: {path_key}")
            
            if path_key != 'model.save_path' and not os.path.exists(current):
                raise ConfigValidationError(f"{description} not found at: {current}")

    @staticmethod
    def validate_training_params(config: Dict[str, Any]) -> None:
        """Validate training parameters"""
        required_params = {
            'training.batch_size': (int, 'Batch size'),
            'training.epochs': (int, 'Number of epochs'),
            'training.optimizer.name': (str, 'Optimizer name'),
            'training.metric': (str, 'Training metric'),
            'model.input_size': (int, 'Model input size'),
            'model.num_classes': (int, 'Number of classes')
        }
        
        for param_key, (expected_type, description) in required_params.items():
            sections = param_key.split('.')
            current = config
            for section in sections:
                if section not in current:
                    raise ConfigValidationError(f"Missing {description} in config: {param_key}")
                current = current[section]
            
            if not isinstance(current, expected_type):
                raise ConfigValidationError(
                    f"{description} must be of type {expected_type.__name__}: {param_key}")
            
            if expected_type == int and current <= 0:
                raise ConfigValidationError(f"{description} must be positive: {param_key}")

    @staticmethod
    def validate_early_stopping(config: Dict[str, Any]) -> None:
        """Validate early stopping configuration"""
        if 'early_stopping' not in config:
            raise ConfigValidationError("Missing early stopping configuration")
        
        required_params = {
            'patience': (int, 'must be positive'),
            'min_delta': (float, 'must be non-negative')
        }
        
        for param, (param_type, condition) in required_params.items():
            if param not in config['early_stopping']:
                raise ConfigValidationError(f"Missing early stopping parameter: {param}")
            
            value = config['early_stopping'][param]
            if not isinstance(value, param_type):
                raise ConfigValidationError(
                    f"Early stopping {param} must be of type {param_type.__name__}")
            
            if (param_type == int and value <= 0) or (param_type == float and value < 0):
                raise ConfigValidationError(f"Early stopping {param} {condition}")

    @classmethod
    def validate_monitoring(cls, config: Dict[str, Any]) -> None:
        """Validate monitoring configuration"""
        if 'monitoring' not in config:
            raise ConfigValidationError("Missing required 'monitoring' section in config")
            
        monitoring = config['monitoring']
        
        # If monitoring is disabled, don't validate further
        if not monitoring.get('enabled', False):
            required_structure = {
                'enabled': False,
                'metrics': {'save_interval': int, 'plot_interval': int},
                'memory': {'track': bool},
                'performance': {'track_time': bool, 'track_throughput': bool}
            }
        else:
            # Full validation for enabled monitoring
            required_structure = {
                'enabled': True,
                'metrics': {'save_interval': int, 'plot_interval': int},
                'memory': {'track': bool},
                'performance': {'track_time': bool, 'track_throughput': bool}
            }
            
        # Validate structure
        cls._validate_structure(monitoring, required_structure, path='monitoring')

    @classmethod
    def _validate_structure(cls, config: Dict, required: Dict, path: str = '') -> None:
        """Recursively validate configuration structure"""
        for key, value in required.items():
            if key not in config:
                raise ConfigValidationError(f"Missing required key: {path}.{key}")
            
            if isinstance(value, dict):
                if not isinstance(config[key], dict):
                    raise ConfigValidationError(f"{path}.{key} must be a dictionary")
                cls._validate_structure(config[key], value, f"{path}.{key}")
            elif isinstance(value, type):
                if not isinstance(config[key], value):
                    raise ConfigValidationError(
                        f"{path}.{key} must be of type {value.__name__}"
                    )

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """Validate entire configuration"""
        validators = [
            cls.validate_paths,
            cls.validate_training_params,
            cls.validate_early_stopping,
            cls.validate_monitoring
        ]
        
        for validator in validators:
            validator(config)
