import os
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

class ExperimentTracker:
    """YAML-based experiment tracking system"""
    
    def __init__(self, experiments_dir: str = 'experiments'):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.current_experiment = None
        self.logger = logging.getLogger('ExperimentTracker')
    
    def start_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """Start a new experiment with given config"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"{name}_{timestamp}"
        
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save initial config
        self.current_experiment = {
            'id': experiment_id,
            'name': name,
            'timestamp': timestamp,
            'config': config,
            'metrics': {},
            'checkpoints': [],
            'status': 'running',
            'dir': experiment_dir
        }
        
        self._save_experiment()
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for current experiment"""
        if not self.current_experiment:
            raise ValueError("No active experiment")
            
        if step is None:
            step = len(self.current_experiment['metrics'])
            
        self.current_experiment['metrics'][step] = metrics
        self._save_experiment()
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]):
        """Log model checkpoint"""
        if not self.current_experiment:
            raise ValueError("No active experiment")
            
        self.current_experiment['checkpoints'].append({
            'path': checkpoint_path,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        })
        self._save_experiment()
    
    def end_experiment(self, status: str = 'completed'):
        """End current experiment"""
        if not self.current_experiment:
            raise ValueError("No active experiment")
            
        self.current_experiment['status'] = status
        self._save_experiment()
        self.current_experiment = None
    
    def _save_experiment(self):
        """Save experiment data to YAML"""
        if not self.current_experiment:
            return
            
        experiment_file = os.path.join(
            self.current_experiment['dir'], 
            'experiment.yaml'
        )
        
        with open(experiment_file, 'w') as f:
            yaml.dump(self.current_experiment, f)
    
    @staticmethod
    def load_experiment(experiment_dir: str) -> Dict[str, Any]:
        """Load experiment data from directory"""
        experiment_file = os.path.join(experiment_dir, 'experiment.yaml')
        if not os.path.exists(experiment_file):
            raise FileNotFoundError(f"No experiment found in {experiment_dir}")
            
        with open(experiment_file, 'r') as f:
            return yaml.safe_load(f)
