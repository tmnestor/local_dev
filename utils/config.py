import logging
import sys
import torch
from torch.utils.data import Dataset
from typing import Dict, Any
import yaml
import os

def setup_logger(name='MLPTrainer', log_dir='logs', filename_template='{name}.log', 
                log_format='%(message)s'):
    """Setup logger with proper propagation control"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler with configured path
        log_path = os.path.join(log_dir, filename_template.format(name=name))
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter with configured format
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

def _get_nested(d, path, default=None):
    """Safely get a nested dictionary value."""
    try:
        for key in path.split('.'):
            d = d[key]
        return d
    except (KeyError, TypeError, AttributeError):
        return default

def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten a nested dictionary with dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _set_nested(d: Dict, path: str, value: Any) -> None:
    """Set a nested dictionary value using dot notation."""
    keys = path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def validate_config(config: Dict) -> None:
    """Validate configuration structure and required fields with defaults."""
    required_fields = {
        'model': {
            'type': 'resnet_mlp',
            'input_size': None,  # Must be specified
            'num_classes': None,  # Must be specified
            'architecture_yaml': 'models/resnet_mlp.yaml',
            'save_path': 'models/best_model.pt'
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'optimizer': {
                'name': 'Adam',
                'params': {'lr': 0.001, 'weight_decay': 0.0}
            },
            'loss': {
                'name': 'CrossEntropyLoss',
                'label_smoothing': {'enabled': False, 'factor': 0.1}
            },
            'metric': 'accuracy',
            'device': 'cpu',
            'seed': 42
        },
        'data': {
            'train_path': None,  # Must be specified
            'val_path': None,    # Must be specified
            'target_column': None # Must be specified
        },
        'dataloader': {
            'num_workers': 'auto'
        },
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001
        },
        'tuning': {
            'n_trials': 50,
            'pruning': {
                'warm_up_epochs': 5,
                'min_trials_complete': 10
            }
        },
        'logging': {
            'directory': 'logs',
            'filename': '{name}.log',
            'format': '%(message)s'
        }
    }
    
    # First validate required sections
    for section in required_fields:
        if section not in config:
            config[section] = {}
    
    # Then validate and set defaults for all fields
    for section, fields in required_fields.items():
        for field, default in _flatten_dict(fields).items():
            current = _get_nested(config, f"{section}.{field}")
            if current is None:
                if default is None:
                    raise ValueError(f"Required field has no value: {section}.{field}")
                # Set default value
                _set_nested(config, f"{section}.{field}", default)

def load_config(config_path: str) -> Dict:
    """Load and validate configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config

def resolve_optimizer_args(config: Dict) -> Dict:
    """Helper to get optimizer arguments from new config structure"""
    return {
        'name': config['training']['optimizer']['name'],
        'params': config['training']['optimizer']['params']
    }

def resolve_loss_args(config: Dict) -> Dict:
    """Helper to get loss function arguments from new config structure"""
    return {
        'name': config['training']['loss']['name'],
        'label_smoothing': config['training']['loss']['label_smoothing']
    }

class CustomDataset(Dataset):
    """Shared dataset class for loading model data"""
    def __init__(self, df, target_column):
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")
            
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        self.features = torch.FloatTensor(df.drop(target_column, axis=1).values)
        self.labels = torch.LongTensor(df[target_column].values)
        
        if self.features.nelement() == 0 or self.labels.nelement() == 0:
            raise ValueError("Failed to create data tensors")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
