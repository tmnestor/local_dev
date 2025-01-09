from typing import Dict, List
from .base import BaseArchitecture

class ResNetMLPArchitecture(BaseArchitecture):
    """ResNet-style MLP with residual connections and batch normalization"""
    
    def __init__(self):
        super().__init__()
        self.required_fields.update({
            'batch_norm': bool,
            'activation': str,
        })
    
    def create_layers(self, config: Dict) -> List[Dict]:
        """Create ResNet MLP layer specifications"""
        layers = []
        prev_size = config['input_size']
        hidden_size = config['hidden_size']
        
        # Input layer
        layers.extend([
            {
                'type': 'linear',
                'in_features': prev_size,
                'out_features': hidden_size,
                'activation': config['activation'],
                'residual': False
            }
        ])
        
        if config['batch_norm']:
            layers.append({
                'type': 'batch_norm',
                'num_features': hidden_size
            })
        
        # Hidden layers with residual connections
        for _ in range(config.get('n_layers', 2)):
            layers.extend([
                {
                    'type': 'linear',
                    'in_features': hidden_size,
                    'out_features': hidden_size,
                    'activation': config['activation'],
                    'residual': True
                }
            ])
            
            if config['batch_norm']:
                layers.append({
                    'type': 'batch_norm',
                    'num_features': hidden_size
                })
        
        # Output layer
        layers.append({
            'type': 'linear',
            'in_features': hidden_size,
            'out_features': config['num_classes'],
            'residual': False
        })
        
        return layers
