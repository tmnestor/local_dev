import os
import logging
import torch
import torch.nn as nn
import yaml
from typing import Dict, Any, List, Optional

from utils.config import setup_logger
from .architectures.registry import ArchitectureRegistry

# Import ResidualBlock and ModuleFactory classes
class ResidualBlock(nn.Module):
    """A residual block with optional projection"""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.needs_projection = input_size != output_size
        if self.needs_projection:
            self.projection = nn.Linear(input_size, output_size)
        
    def forward(self, x: torch.Tensor, residual_input: torch.Tensor) -> torch.Tensor:
        if self.needs_projection:
            return x + self.projection(residual_input)
        return x + residual_input

class ModuleFactory:
    """Factory class for creating PyTorch modules from YAML specifications"""
    
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,  # Add ELU activation
        'tanh': nn.Tanh,
        # 'sigmoid': nn.Sigmoid,
        'none': nn.Identity,  # Add Identity for no activation
        None: nn.Identity    # Handle None case
    }
    
    LAYERS = {
        'linear': nn.Linear,
        'dropout': nn.Dropout,
        'batch_norm': nn.BatchNorm1d,
        'residual': ResidualBlock
    }
    
    @classmethod
    def create_activation(cls, name: str, **kwargs) -> nn.Module:
        """Create activation layer, returning Identity for None/none"""
        name = name.lower() if isinstance(name, str) else name
        if name not in cls.ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {name}")
        return cls.ACTIVATIONS[name](**kwargs)
    
    @classmethod
    def create_layer(cls, spec: Dict[str, Any]) -> nn.Module:
        layer_spec = spec.copy()
        layer_type = layer_spec.pop('type').lower()
        
        # Remove non-layer parameters
        _ = layer_spec.pop('activation', None)  # Remove but don't use here
        _ = layer_spec.pop('residual', False)   # Remove but don't use here
        
        if layer_type not in cls.LAYERS:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        return cls.LAYERS[layer_type](**layer_spec)

def load_model_from_yaml(yaml_path: str) -> nn.Module:
    """Create model from YAML using architecture registry"""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    # Handle both direct architecture specification and layered config
    if 'architecture' not in config:
        if 'layers' in config:
            # Infer architecture type from layers
            has_residual = any(
                isinstance(layer, dict) and layer.get('residual', False)
                for layer in config['layers']
            )
            config['architecture'] = 'resnet_mlp' if has_residual else 'complex_mlp'
        else:
            raise ValueError("YAML must specify 'architecture' type or contain 'layers' specification")
    
    # Get architecture implementation
    architecture = ArchitectureRegistry.get_architecture(config['architecture'])
    
    # Validate architecture-specific config
    architecture.validate_config(config)
    
    # Create layer specifications
    layers = architecture.create_layers(config)
    
    # Create model using existing DynamicModel class
    return DynamicModel({'layers': layers})

class DynamicModel(nn.Module):
    def __init__(self, model_spec: Dict[str, Any]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        
        current_size = None
        residual_input_size = None
        
        for layer_spec in model_spec['layers']:
            if isinstance(layer_spec, dict):
                # Track input/output sizes for residual connections
                if layer_spec['type'] == 'linear':
                    current_size = layer_spec['out_features']
                    if residual_input_size is None:
                        residual_input_size = layer_spec['in_features']
                
                # Create the main layer
                layer = ModuleFactory.create_layer(layer_spec)
                self.layers.append(layer)
                
                # Add activation if specified
                if 'activation' in layer_spec:
                    activation = ModuleFactory.create_activation(
                        layer_spec['activation']
                    )
                    self.layers.append(activation)
                
                # Add residual connection if specified
                if layer_spec.get('residual', False) and current_size is not None:
                    residual = ResidualBlock(residual_input_size, current_size)
                    self.residual_blocks.append(residual)
                    residual_input_size = current_size
                else:
                    self.residual_blocks.append(None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply residual connection if exists for this layer
            if i < len(self.residual_blocks) and self.residual_blocks[i] is not None:
                x = self.residual_blocks[i](x, residual_input)
                residual_input = x
        return x

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(pred, dim=1), dim=1))

def restore_best_model(config):
    """Restore best model from config with architecture type inference."""
    if not os.path.exists(config['model']['save_path']):
        raise FileNotFoundError(f"Model checkpoint not found at {config['model']['save_path']}")
    
    checkpoint = torch.load(config['model']['save_path'], map_location='cpu')
    
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Invalid checkpoint: missing model_state_dict")
    
    # Load base architecture from YAML
    with open(config['model']['architecture_yaml'], 'r') as f:
        base_architecture = yaml.safe_load(f)
    
    model = load_model_from_yaml(config['model']['architecture_yaml'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Infer architecture type from the architecture configuration
    if 'architecture' in base_architecture:
        arch_type = base_architecture['architecture']
    else:
        arch_type = "ResNet MLP" if any(
            isinstance(layer, dict) and layer.get('residual', False)
            for layer in base_architecture.get('layers', [])
        ) else "Complex MLP"
    
    # Create architecture description
    arch_desc = f"{arch_type} with {len([l for l in base_architecture.get('layers', []) if l['type'] == 'linear'])} layers"
    
    optimizer = getattr(torch.optim, config['training']['optimizer']['name'])(
        model.parameters(),
        **config['training']['optimizer']['params']
    )
    
    return {
        'model': model,
        'optimizer': optimizer,
        'metric_name': config['training']['metric'],
        'metric_value': checkpoint.get('metric_value', 0.0),
        'hyperparameters': checkpoint.get('hyperparameters', {
            'architecture': base_architecture,
            'lr': config['training']['optimizer']['params']['lr'],
            'weight_decay': config['training']['optimizer']['params'].get('weight_decay', 0.0)
        }),
        'architecture_type': arch_type,
        'architecture_description': arch_desc
    }

# Example YAML model specification:
EXAMPLE_YAML = """
# Model architecture specification
layers:
  - type: linear
    in_features: 7
    out_features: 128
    activation: gelu
    residual: false  # First layer typically doesn't have residual
    
  - type: batch_norm
    num_features: 128
    
  - type: dropout
    p: 0.2
    
  - type: linear
    in_features: 128
    out_features: 128  # Same size for proper residual connection
    activation: gelu
    residual: true  # Enable residual connection
    
  - type: batch_norm
    num_features: 128
    
  - type: dropout
    p: 0.2
    
  - type: linear
    in_features: 128
    out_features: 5
    residual: false  # Output layer typically doesn't have residual
"""

if __name__ == "__main__":
    model = load_model_from_yaml("resnet_mlp.yaml")
    x = torch.randn(32, 7)
    output = model(x)
    print(f"Output shape: {output.shape}")