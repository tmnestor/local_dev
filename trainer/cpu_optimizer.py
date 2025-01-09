import logging
import psutil
import cpuinfo
import torch
import numpy as np
import random
import os

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class CPUOptimizer:
    """Handles CPU-specific optimizations for PyTorch training."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('CPUOptimizer')
        self.cpu_info = cpuinfo.get_cpu_info()
        
    def detect_cpu_features(self):
        """Detect CPU features and capabilities."""
        features = {
            'processor': self.cpu_info.get('brand_raw', 'Unknown'),
            'architecture': self.cpu_info.get('arch', 'Unknown'),
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'avx512': 'avx512' in self.cpu_info.get('flags', []),
            'avx2': 'avx2' in self.cpu_info.get('flags', []),
            'mkl': hasattr(torch, 'backends') and hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available(),
            'ipex': hasattr(torch, 'xpu') or hasattr(torch, 'ipex')
        }
        return features
        
    def configure_optimizations(self):
        """Configure CPU-specific optimizations based on detected features."""
        features = self.detect_cpu_features()
        optimizations = {}
        
        # Configure number of threads
        if self.config['training']['cpu_optimization']['num_threads'] == 'auto':
            optimizations['num_threads'] = features['threads']
        else:
            optimizations['num_threads'] = self.config['training']['cpu_optimization']['num_threads']
        
        # Configure MKL-DNN
        optimizations['enable_mkldnn'] = (
            features['avx512'] or features['avx2']
        ) and self.config['training']['cpu_optimization']['enable_mkldnn']
        
        # Configure data types
        optimizations['use_bfloat16'] = (
            features['avx512'] and 
            self.config['training']['cpu_optimization']['use_bfloat16']
        )
        
        # Set thread configurations
        torch.set_num_threads(optimizations['num_threads'])
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(min(4, optimizations['num_threads']))
        
        # Enable MKL-DNN if available
        if optimizations['enable_mkldnn']:
            torch.backends.mkldnn.enabled = True
        
        self.log_optimization_config(features, optimizations)
        return optimizations
        
    def log_optimization_config(self, features, optimizations):
        """Log CPU features and applied optimizations."""
        self.logger.info("CPU Configuration:")
        self.logger.info(f"Processor: {features['processor']}")
        self.logger.info(f"Architecture: {features['architecture']}")
        self.logger.info(f"Physical cores: {features['cores']}")
        self.logger.info(f"Logical threads: {features['threads']}")
        self.logger.info("\nCPU Features:")
        self.logger.info(f"AVX-512 support: {features['avx512']}")
        self.logger.info(f"AVX2 support: {features['avx2']}")
        self.logger.info(f"MKL support: {features['mkl']}")
        self.logger.info(f"IPEX support: {features['ipex']}")
        self.logger.info("\nApplied Optimizations:")
        self.logger.info(f"Number of threads: {optimizations['num_threads']}")
        self.logger.info(f"MKL-DNN enabled: {optimizations['enable_mkldnn']}")
        self.logger.info(f"BFloat16 enabled: {optimizations['use_bfloat16']}")
