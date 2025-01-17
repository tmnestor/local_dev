import time
import logging
import yaml  # Change from json to yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt
import seaborn as sns  # Add this import
from datetime import datetime
from utils.logger import Logger  # Add this import

class PerformanceMonitor:
    """Monitors training performance and resource usage"""
    
    def __init__(self, config: Optional[Dict[str, Any]], model: torch.nn.Module):
        self.config = config if config else {}
        self.model = model
        self.metrics_history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_f1': [], 'val_f1': [],
            'learning_rate': [], 'memory_usage': [],
            'epoch_time': []
        }
        self.start_time = time.time()
        self.epoch_start_time = None
        self.epoch_count = 0  # Add epoch counter
        self.is_tuning = False  # Add flag for optuna tuning
        
        # Get monitoring config with defaults
        monitoring_config = self.config.get('monitoring', {})
        self.save_interval = monitoring_config.get('metrics', {}).get('save_interval', 1)
        self.plot_interval = monitoring_config.get('metrics', {}).get('plot_interval', 5)
        
        # Create monitoring directory with consistent timestamp format
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M')  # Remove seconds
        self.monitor_dir = Path(monitoring_config['log_dir']) / self.timestamp
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.monitor_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize storage for plots
        self._training_plots = {}

        # Make timestamp accessible to other components
        config['monitoring']['current_timestamp'] = self.timestamp
        
        # Setup logging with new Logger class
        self.logger = Logger.get_timestamp_logger(
            'Performance',
            log_dir=self.monitor_dir
        )

    def start_epoch(self) -> None:
        """Record start of epoch"""
        self.epoch_start_time = time.time()
        self.record_memory_usage()

    def end_epoch(self, metrics: Dict[str, float]) -> None:
        """Record end of epoch and save metrics"""
        if self.epoch_start_time is None:
            return
            
        self.epoch_count += 1  # Increment epoch counter
        epoch_time = time.time() - self.epoch_start_time
        self.metrics_history['epoch_time'].append(epoch_time)
        
        # Record metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Log epoch summary
        self.logger.info(
            f"Epoch {self.epoch_count} completed in {epoch_time:.2f}s - "
            f"Loss: {metrics.get('val_loss', 0):.4f}, "
            f"Accuracy: {metrics.get('val_accuracy', 0):.2f}%, "
            f"F1: {metrics.get('val_f1', 0):.4f}"
        )
        
        # Skip saving metrics during tuning
        if not self.is_tuning and self.epoch_count % self.save_interval == 0:
            self.save_metrics()
            
        # Skip generating plots during tuning
        if not self.is_tuning and self.epoch_count % self.plot_interval == 0:
            self.generate_plots()

    def set_tuning_mode(self, enabled: bool = True) -> None:
        """Enable or disable tuning mode to prevent saving during optuna trials"""
        self.is_tuning = enabled

    def record_memory_usage(self) -> None:
        """Record current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.metrics_history['memory_usage'].append(memory_info.rss / 1024 / 1024)  # MB

    def save_metrics(self) -> None:
        """Save metrics to YAML file"""
        metrics_file = self.monitor_dir / 'metrics.yaml'
        # Convert numpy arrays/values to Python native types
        cleaned_metrics = {}
        for key, values in self.metrics_history.items():
            cleaned_metrics[key] = [float(v) if hasattr(v, 'dtype') else v for v in values]
        
        with open(metrics_file, 'w') as f:
            yaml.dump(cleaned_metrics, f, default_flow_style=False, sort_keys=False)

    # Method to load metrics (new)
    def load_metrics(self, filepath: str) -> Dict[str, List[float]]:
        """Load metrics from YAML file"""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    def generate_plots(self) -> None:
        """Generate performance plots"""
        plots = [
            ('loss', ['train_loss', 'val_loss'], 'Loss'),
            ('accuracy', ['train_accuracy', 'val_accuracy'], 'Accuracy (%)'),
            ('f1', ['train_f1', 'val_f1'], 'F1 Score'),
        ]
        
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # Generate all plots and store them
        training_plots = {}
        
        for name, metrics, ylabel in plots:
            fig = plt.figure(figsize=(10, 6))
            for metric in metrics:
                if metric in self.metrics_history and len(self.metrics_history[metric]) > 0:
                    plt.plot(self.metrics_history[metric], 
                            label=metric.replace('_', ' ').title(),
                            marker='o',
                            markersize=4,
                            linestyle='-',
                            alpha=0.8)
            
            plt.xlabel('Epoch')
            plt.ylabel(ylabel)
            plt.title(f'Training {ylabel} Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            training_plots[f'{name}_plot.png'] = fig
            plt.close()  # Close figure but keep it in memory
            
        # Store the plots to be saved later
        self._training_plots = training_plots
        
    def save_all_plots(self, metrics_manager=None) -> None:
        """Save all plots from both training and CV in a single directory"""
        # Save training plots
        if self._training_plots:
            for name, fig in self._training_plots.items():
                fig.savefig(self.plots_dir / name, dpi=300, bbox_inches='tight')
                plt.close(fig)
            self._training_plots = {}
            
        # Save CV plots if metrics manager is provided and has plots
        if metrics_manager:
            metrics_manager.save_plots(self.plots_dir)
        
        self.logger.info(f"All plots saved to: {self.plots_dir}")

    def get_summary(self) -> Dict[str, float]:
        """Get training summary statistics."""
        total_time = time.time() - self.start_time
        
        # Format metrics without logging
        metrics = [
            ("Runtime", f"{total_time:.2f}s"),
            ("Avg Epoch Time", f"{np.mean(self.metrics_history['epoch_time']):.2f}s"),
            ("Peak Memory", f"{max(self.metrics_history['memory_usage']):.1f}MB"),
        ]
        
        if self.metrics_history['val_accuracy']:
            metrics.append(("Best Accuracy", f"{max(self.metrics_history['val_accuracy']):.2f}%"))
        if self.metrics_history['val_f1']:
            f1_value = max(self.metrics_history['val_f1']) * 100
            metrics.append(("Best F1 Score", f"{f1_value:.2f}%"))
        
        if self.metrics_history['train_loss']:
            metrics.append(("Final Train Loss", f"{self.metrics_history['train_loss'][-1]:.4f}"))
        if self.metrics_history['val_loss']:
            metrics.append(("Final Val Loss", f"{self.metrics_history['val_loss'][-1]:.4f}"))
        
        return metrics  # Return just the metrics list
