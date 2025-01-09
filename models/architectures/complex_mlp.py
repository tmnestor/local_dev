from typing import Dict, List
from .base import BaseArchitecture

class ComplexMLPArchitecture(BaseArchitecture):
    """Complex MLP with tapering width and dropout"""
    
    def __init__(self):
        super().__init__()
        self.required_fields.update({
            'dropout_rate': float,
            'activation': str,
        })
    
    def create_layers(self, config: Dict) -> List[Dict]:
        """Create Complex MLP layer specifications"""
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        output_size = config['num_classes']
        n_layers = config.get('n_layers', 3)
        
        # Calculate decreasing layer sizes
        layer_sizes = []
        for i in range(n_layers):
            if i == 0:
                # First hidden layer
                layer_sizes.append(hidden_size)
            elif i == n_layers - 1:
                # Last layer (output)
                layer_sizes.append(output_size)
            else:
                # Hidden layers with decreasing size
                current_size = int(hidden_size * (1 - (i / n_layers)))
                # Ensure size is at least double the output size
                current_size = max(current_size, output_size * 2)
                layer_sizes.append(current_size)
        
        layers = []
        prev_size = input_size  # Start with input size
        
        # Create layers with proper connections
        for i, size in enumerate(layer_sizes):
            # Add linear layer
            layers.append({
                'type': 'linear',
                'in_features': prev_size,
                'out_features': size,
                'activation': config['activation'] if i < len(layer_sizes) - 1 else None
            })
            
            # Add dropout after all layers except the last
            if i < len(layer_sizes) - 1:
                layers.append({
                    'type': 'dropout',
                    'p': config['dropout_rate']
                })
            
            prev_size = size  # Update for next layer
        
        return layers

    def validate_config(self, config: Dict) -> None:
        """Additional validation for Complex MLP"""
        super().validate_config(config)
        
        # Ensure hidden size is appropriate
        if config['hidden_size'] < config['num_classes'] * 2:
            config['hidden_size'] = config['num_classes'] * 4
            
        # Ensure reasonable layer count
        if config.get('n_layers', 0) < 2:
            config['n_layers'] = 3
