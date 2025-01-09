from typing import Dict, List, Any
import yaml

class BaseArchitecture:
    """Base class for model architectures"""
    
    def __init__(self):
        self.required_fields = {
            'input_size': int,
            'hidden_size': int,
            'num_classes': int,
        }
    
    def validate_config(self, config: Dict) -> None:
        """Validate configuration has required fields with correct types"""
        for field, field_type in self.required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Field {field} must be of type {field_type}, got {type(config[field])}")
            
            # Additional validation for numeric fields
            if field_type in (int, float):
                if config[field] <= 0:
                    raise ValueError(f"Field {field} must be positive, got {config[field]}")

    def create_layers(self, config: Dict) -> List[Dict]:
        """Create layer specifications from config"""
        raise NotImplementedError("Subclasses must implement create_layers")
    
    @staticmethod
    def save_architecture(layers: List[Dict], filepath: str) -> None:
        """Save architecture specification to YAML"""
        with open(filepath, 'w') as f:
            yaml.dump({'layers': layers}, f)
    
    @staticmethod
    def load_architecture(filepath: str) -> Dict:
        """Load architecture specification from YAML"""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    def infer_architecture_type(self, layers: List[Dict]) -> str:
        """Infer architecture type from layer specifications"""
        if any(isinstance(layer, dict) and layer.get('residual', False) 
               for layer in layers):
            return 'resnet_mlp'
        if any(isinstance(layer, dict) and layer.get('type') == 'dropout'
               for layer in layers):
            return 'complex_mlp'
        return 'basic_mlp'