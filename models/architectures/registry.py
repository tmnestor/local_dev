from typing import Dict, Type
from .base import BaseArchitecture
from .resnet_mlp import ResNetMLPArchitecture
from .complex_mlp import ComplexMLPArchitecture

class ArchitectureRegistry:
    """Registry for available model architectures"""
    
    _architectures: Dict[str, Type[BaseArchitecture]] = {
        'resnet_mlp': ResNetMLPArchitecture,
        'complex_mlp': ComplexMLPArchitecture
    }
    
    @classmethod
    def register(cls, name: str, architecture: Type[BaseArchitecture]) -> None:
        """Register a new architecture"""
        if not issubclass(architecture, BaseArchitecture):
            raise TypeError("Architecture must inherit from BaseArchitecture")
        cls._architectures[name] = architecture
    
    @classmethod
    def get_architecture(cls, name: str) -> BaseArchitecture:
        """Get architecture instance by name"""
        if name not in cls._architectures:
            available = list(cls._architectures.keys())
            raise ValueError(f"Unknown architecture: {name}. Available architectures: {available}")
        return cls._architectures[name]()
    
    @classmethod
    def list_architectures(cls) -> list:
        """List all registered architectures"""
        return list(cls._architectures.keys())
