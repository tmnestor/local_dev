#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import copy
import logging
import datetime
from typing import Dict, Any

# Data handling
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

# ML/DL imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import optuna

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# System utilities
import psutil
import cpuinfo
import multiprocessing

# Local imports
from utils.logging import setup_logger  # Fix: Change from utils.config to utils.logging
from utils.data import CustomDataset
from models.model_loader import (
    ModuleFactory, 
    load_model_from_yaml, 
    ResidualBlock,
    LabelSmoothingLoss
)
from utils.performance_monitor import PerformanceMonitor
from utils.config_validator import ConfigValidator, ConfigValidationError
from utils.metrics_manager import MetricsManager
from pathlib import Path  # Also need this for Path objects

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
# Set up logging
def setup_logger(name='MLPTrainer'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(f'{name}.log')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class PyTorchTrainer:
    """A generic PyTorch trainer class.
    
    Attributes:trial
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (CPU/GPU)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    
    def __init__(self, model, criterion, optimizer, device='cpu', verbose=False, 
                 config=None, tuning_mode=False, metrics_tuning_mode=False):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        self.best_model_state = None
        self.best_metric = float('-inf')
        self.best_epoch = 0
        self.logger = logging.getLogger('PyTorchTrainer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        
        # Validate config for required fields
        if config is not None:
            required_fields = ['model']
            required_model_fields = ['num_classes', 'input_size']
            
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
                    
            for field in required_model_fields:
                if field not in config['model']:
                    raise ValueError(f"Missing required model config field: {field}")
        
        self.config = config  # Store config
        
        # Only create performance monitor if not in tuning mode and config exists
        self.performance_monitor = None
        if not tuning_mode and config and config.get('monitoring', {}).get('enabled', False):
            self.performance_monitor = PerformanceMonitor(config, model)
        
        # Modify metrics manager initialization to work with tuning mode
        if config:
            self.metrics_manager = MetricsManager(
                num_classes=config['model']['num_classes'],
                metrics_dir=Path('metrics/tuning') if tuning_mode else Path('metrics'),
                tuning_mode=metrics_tuning_mode,  # Pass tuning mode to metrics manager
                config=config  # Pass config to metrics manager
            )
        else:
            self.metrics_manager = None
        
        # Add tuning mode flag
        self.tuning_mode = tuning_mode
        self.current_epoch = 0  # Add this to track current epoch
        
        # Add validation tracking
        self.model_validated = False
        self.initial_validation_done = False
        
        # Add model validation tracking
        self.model_validated = False
        self.validation_results = {}
        
        # Initialize validation thresholds
        self.validation_thresholds = {
            'min_variance': 1e-5,  # Reduced from 1e-4
            'max_accuracy': 0.98,  # Increased from 0.95
            'min_loss': 1e-5   # Reduced from 1e-4
        }
        
    def reset_performance_monitor(self, config=None) -> None:
        """Reset or create new performance monitor"""
        if config is not None:
            ConfigValidator.validate_monitoring(config)
            self.performance_monitor = PerformanceMonitor(config, self.model)
        else:
            self.performance_monitor = None
        
    def train_epoch(self, train_loader, val_loader=None):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        return train_loss, train_accuracy
    
    def evaluate(self, val_loader):
        """Evaluates the model on validation data."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_y.size(0)
                total_samples += batch_y.size(0)
                
                all_outputs.append(outputs)
                all_targets.append(batch_y)
        
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets)
        metrics = self.metrics_manager.update(outputs, targets, phase='val')
        
        return (
            total_loss / total_samples,
            metrics['accuracy'] * 100,
            metrics['f1_macro']
        )

    def _validate_model_sanity(self, val_loader):
        """Validate model behavior on a small subset of data."""
        if not self.config or 'model' not in self.config:
            self.logger.warning("No config provided - skipping model validation")
            return True
            
        self.logger.info("Performing initial model validation...")
        
        # Add cross-input validation
        def check_model_consistency():
            """Check if model gives consistent but different outputs for different inputs"""
            with torch.no_grad():
                input_size = self.config['model']['input_size']
                x1 = torch.zeros(4, input_size).to(self.device)
                x2 = torch.ones(4, input_size).to(self.device)
                x3 = torch.randn(4, input_size).to(self.device)
                
                out1 = self.model(x1)
                out2 = self.model(x2)
                out3 = self.model(x3)
                
                # Check if outputs are different
                diff12 = not torch.allclose(out1, out2, atol=1e-3)
                diff23 = not torch.allclose(out2, out3, atol=1e-3)
                diff13 = not torch.allclose(out1, out3, atol=1e-3)
                
                return diff12 and diff23 and diff13
        
        # Get a small batch of data
        val_subset = torch.utils.data.Subset(
            val_loader.dataset,
            indices=range(min(32, len(val_loader.dataset)))
        )
        subset_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
        
        validation_failed = False
        validation_messages = []
        
        # Check model consistency
        if not check_model_consistency():
            msg = "Model gives similar outputs for different inputs"
            validation_failed = True
            validation_messages.append(msg)
        
        with torch.no_grad():
            outputs_list = []
            targets_list = []
            losses = []
            
            for batch_X, batch_y in subset_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                outputs_list.append(outputs)
                targets_list.append(batch_y)
                losses.append(loss.item())
            
            # Concatenate all outputs
            outputs = torch.cat(outputs_list)
            targets = torch.cat(targets_list)
            
            # 1. Check output dimensions
            if outputs.shape[1] != self.config['model']['num_classes']:
                msg = f"Model output dimension mismatch: {outputs.shape[1]} != {self.config['model']['num_classes']}"
                validation_failed = True
                validation_messages.append(msg)
            
            # 2. Check probability distribution
            probs = torch.softmax(outputs, dim=1)
            prob_stats = {
                'mean': probs.mean().item(),
                'std': probs.std().item(),
                'min': probs.min().item(),
                'max': probs.max().item()
            }
            
            # 3. Check for low variance (potential mode collapse)
            if prob_stats['std'] < self.validation_thresholds['min_variance']:
                msg = f"Model outputs show very low variance: {prob_stats['std']:.6f}"
                validation_failed = True
                validation_messages.append(msg)
            
            # 4. Check loss values
            mean_loss = sum(losses) / len(losses)
            if mean_loss < self.validation_thresholds['min_loss']:
                msg = f"Suspiciously low loss value: {mean_loss:.6f}"
                validation_failed = True
                validation_messages.append(msg)
            
            # Add data leakage checks
            all_inputs = []
            all_targets = []
            for batch_X, batch_y in subset_loader:
                all_inputs.append(batch_X.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
            
            inputs = np.concatenate(all_inputs)
            targets = np.concatenate(all_targets)
            
            # Check for duplicates
            input_hashes = [hash(arr.tobytes()) for arr in inputs]
            if len(set(input_hashes)) < len(input_hashes):
                msg = "Duplicate inputs detected in validation data"
                validation_failed = True
                validation_messages.append(msg)
            
            # Check target distribution
            target_counts = np.bincount(targets, minlength=self.config['model']['num_classes'])
            if len(set(target_counts)) == 1:
                msg = "Suspicious: Perfectly balanced class distribution"
                validation_failed = True
                validation_messages.append(msg)
            
            # Store validation results
            self.validation_results = {
                'output_stats': {
                    'shape': outputs.shape,
                    'mean': outputs.mean().item(),
                    'std': outputs.std().item()
                },
                'probability_stats': prob_stats,
                'loss_stats': {
                    'mean': mean_loss,
                    'min': min(losses),
                    'max': max(losses)
                }
            }
            
            # Log validation results
            self.logger.info("\nModel Validation Results:")
            self.logger.info(f"Output shape: {outputs.shape}")
            self.logger.info("\nProbability distribution:")
            for k, v in prob_stats.items():
                self.logger.info(f"  {k}: {v:.4f}")
            self.logger.info(f"\nLoss statistics:")
            self.logger.info(f"  Mean: {mean_loss:.4f}")
            self.logger.info(f"  Range: [{min(losses):.4f}, {max(losses):.4f}]")
            
            if validation_failed:
                self.logger.warning("\nValidation Issues Detected:")
                for msg in validation_messages:
                    self.logger.warning(f"  - {msg}")
            
            self.model_validated = True
            return not validation_failed

    def train(self, train_loader, val_loader, epochs, metric='accuracy',
              early_stop_patience=10, early_stop_delta=0.001, max_iter=100):
        """Trains the model with early stopping and best model saving."""
        # Get dataset sizes for logging
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        total_size = train_size + val_size
        
        self.logger.info("\nStarting final training phase...")
        self.logger.info(f"Dataset: {train_size:,} train + {val_size:,} val = {total_size:,} total samples")
        self.logger.info("Starting training...")
        
        # Initialize performance monitoring only if it doesn't already exist
        if self.performance_monitor is None and self.config and self.config.get('monitoring', {}).get('enabled', False):
            self.performance_monitor = PerformanceMonitor(self.config, self.model)
        
        if self.performance_monitor:
            self.performance_monitor.start_epoch()
            
        # Rest of train method...
        # ...existing code...
        
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        patience_counter = 0
        
        try:
            for epoch in tqdm(range(min(epochs, max_iter)), desc='Training'):
                train_loss, train_accuracy = self.train_epoch(train_loader, val_loader)
                val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
                
                # Record validation metrics during training
                if self.performance_monitor:
                    self.performance_monitor.end_epoch({
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                        'val_f1': val_f1,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
                # Select metric based on config
                train_metric = train_accuracy
                val_metric = val_f1 if metric == 'f1' else val_accuracy
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_metrics.append(train_metric)
                val_metrics.append(val_metric)
                
                # Save best model if improved
                if val_metric > self.best_metric + early_stop_delta:
                    self.best_metric = val_metric
                    self.best_epoch = epoch
                    self.best_model_state = {
                        'model_state': copy.deepcopy(self.model.state_dict()),
                        'epoch': epoch,
                        'metric_value': val_metric
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.verbose:
                    metric_name = 'F1' if metric == 'f1' else 'Accuracy'
                    metric_value = val_f1 if metric == 'f1' else val_accuracy
                    print(f'Epoch {epoch+1}/{epochs}: Val {metric_name}: {metric_value:.2f}% '
                          f'(Best: {self.best_metric:.2f}% @ epoch {self.best_epoch+1})')
                
                # Early stopping check
                if patience_counter >= early_stop_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            # Restore best model before returning
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state['model_state'])
                print(f"Restored best model from epoch {self.best_model_state['epoch']+1} "
                      f"with {metric}={self.best_model_state['metric_value']:.2f}%")
            
            self.plot_learning_curves(train_losses, val_losses, train_metrics, val_metrics,
                                    metric_name='F1-Score' if metric == 'f1' else 'Accuracy')
            
            # Get summary only if monitoring exists and avoid duplicate logging
            if self.performance_monitor:
                summary = self.performance_monitor.get_summary()
            else:
                # Only log if we don't have a performance monitor
                self.logger.info("\n" + "=" * 60)
                self.logger.info("Saving trained model...")
                self.logger.info(f"Model saved to {self.config['model']['save_path']}")
                self.logger.info(f"Final performance: {self.best_metric:.4f}")
            
            # Save model without logging
            torch.save(self.best_model_state, self.config['model']['save_path'])
            
            return train_losses, val_losses, train_metrics, val_metrics, self.best_metric
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    @staticmethod
    def plot_learning_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='Accuracy'):
        """Plots the learning curves for loss and chosen metric (accuracy or F1)."""
        # Create figures directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Create unique filename using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'figures/learning_curves_{timestamp}.png'
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Normalize values for better visualization
        max_loss = max(max(train_losses), max(val_losses))
        max_metric = max(max(train_metrics), max(val_metrics))
        
        epochs = range(1, len(train_losses) + 1)
        
        sns.lineplot(data={
            f"Training {metric_name}": [x/max_metric for x in train_metrics],
            f"Validation {metric_name}": [x/max_metric for x in val_metrics],
            "Training Loss": [x/max_loss for x in train_losses],
            "Validation Loss": [x/max_loss for x in val_losses]
        })
        
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Value")
        plt.title(f"Training and Validation Loss and {metric_name} Curves")
        plt.legend()
        plt.savefig(filename)
        plt.close()
        
        return filename  # Return filename for reference

    def cross_validate(self, dataset, n_splits=5, epochs=None):
        """Perform k-fold cross validation"""
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        import copy
        from models.model_loader import load_model_from_yaml
        
        # Add dataset inspection
        self.logger.debug(f"Dataset size: {len(dataset)}")
        self.logger.debug(f"Target distribution: {pd.Series(dataset[:][1].numpy()).value_counts()}")
        
        epochs = epochs or self.config['training']['epochs']
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Get all data
        X = [dataset[i][0] for i in range(len(dataset))]
        y = [dataset[i][1] for i in range(len(dataset))]
        X = torch.stack(X)
        y = torch.tensor(y)
        
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X, y), 
                                                       total=n_splits,
                                                       desc='Cross Validation')):
            self.logger.info(f"\nFold {fold+1}/{n_splits}")
            
            self.logger.debug(f"\nFold {fold+1} splits:")
            self.logger.debug(f"Train idx: {len(train_idx)}, Val idx: {len(val_idx)}")
            self.logger.debug(f"Train class dist: {np.unique(y[train_idx], return_counts=True)}")
            self.logger.debug(f"Val class dist: {np.unique(y[val_idx], return_counts=True)}")
            
            # Create a fresh model for each fold using the same architecture
            model = load_model_from_yaml(self.config['model']['architecture_yaml'])
            model = model.to(self.device)
            
            # Create fresh optimizer
            optimizer = getattr(torch.optim, self.config['training']['optimizer']['name'])(
                model.parameters(),
                **self.config['training']['optimizer']['params']
            )
            
            # Update trainer with new model and optimizer
            self.model = model
            self.optimizer = optimizer
            
            # Reset metrics manager
            if self.metrics_manager:
                self.metrics_manager.predictions = []
                self.metrics_manager.true_labels = []
                self.metrics_manager.train_metrics = []
                self.metrics_manager.val_metrics = []
            
            # Create data loaders
            train_loader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=torch.utils.data.SubsetRandomSampler(train_idx)
            )
            val_loader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                sampler=torch.utils.data.SubsetRandomSampler(val_idx)
            )
            
            # Training variables for this fold
            best_fold_metrics = None
            best_metric = float('-inf')
            patience_counter = 0
            
            # Train for this fold
            for epoch in range(epochs):
                self.logger.debug(f"Fold {fold+1}, Epoch {epoch+1}:")
                # Train one epoch
                self.train_epoch(train_loader)
                
                # Evaluate on validation set
                self.model.eval()
                with torch.no_grad():
                    all_outputs = []
                    all_targets = []
                    
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        all_outputs.append(outputs)
                        all_targets.append(batch_y)
                    
                    # Concatenate all batches
                    outputs = torch.cat(all_outputs)
                    targets = torch.cat(all_targets)
                    
                    # Calculate metrics
                    metrics = self.metrics_manager.update(outputs, targets, phase='val')
                    current_metric = metrics['f1_macro']
                
                # Log intermediate metrics
                self.logger.debug(f"Intermediate metrics: {metrics}")
                
                # Update best metrics if improved
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_fold_metrics = metrics.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= min(5, self.config['early_stopping']['patience']):
                    break
                
                # Print progress periodically
                if self.verbose and (epoch + 1) % 5 == 0:
                    print(f'Fold {fold+1}, Epoch {epoch+1}: F1={current_metric:.4f}')
            
            # Store results for this fold
            if best_fold_metrics is not None:
                best_fold_metrics['fold'] = fold
                cv_results.append(best_fold_metrics)
        
        # Restore original model state after CV
        model = load_model_from_yaml(self.config['model']['architecture_yaml'])
        model = model.to(self.device)
        optimizer = getattr(torch.optim, self.config['training']['optimizer']['name'])(
            model.parameters(),
            **self.config['training']['optimizer']['params']
        )
        self.model = model
        self.optimizer = optimizer
        
        return cv_results

def main():
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Set up logging
    setup_logger()
    
    # Initialize CPU optimization
    cpu_optimizer = CPUOptimizer(config)
    optimizations = cpu_optimizer.configure_optimizations()
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Create datasets and dataloaders
    train_df = pd.read_csv(config['data']['train_path'])
    val_df = pd.read_csv(config['data']['val_path'])
    train_dataset = CustomDataset(train_df, config['data']['target_column'])
    val_dataset = CustomDataset(val_df, config['data']['target_column'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # If best parameters don't exist in config, run hyperparameter tuning
    if 'best_model' not in config:
        tuner = HyperparameterTuner(config)
        best_trial, best_params = tuner.tune(train_loader, val_loader)
        save_best_params_to_config(config_path, best_trial, best_params)
        # Reload config with saved parameters
        config = load_config(config_path)
    
    print("\nBest model parameters from config:")
    for key, value in config['best_model'].items():
        print(f"    {key}: {value}")
    
    # Restore best model from checkpoint
    print("\nRestoring best model from checkpoint...")
    restored = restore_best_model(config)
    model = restored['model']
    optimizer = restored['optimizer']
    
    # Print architecture details
    print(f"\nArchitecture: {restored['architecture_description']}")
    print("Hyperparameters:")
    for key, value in restored['hyperparameters'].items():
        if key not in ['architecture']:  # Skip full architecture
            print(f"    {key}: {value}")
    
    # Create criterion for evaluation
    criterion = getattr(nn, config['training']['loss_function'])()
    
    # Create trainer for evaluation
    trainer = PyTorchTrainer(
        model, criterion, optimizer,
        device=config['device'],  # Changed from training.device to root level device
        verbose=True
    )
    
    # Evaluate restored model
    print("\nEvaluating restored model on validation set...")
    val_loss, val_accuracy, val_f1 = trainer.evaluate(val_loader)
    

    metric_name = config['training']['metric']
    metric_value = val_f1 if metric_name == 'f1' else val_accuracy
    
    print(f"\nRestored model performance:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation F1-Score: {val_f1:.4f}")
    print(f"\nBest {metric_name.upper()} from tuning: {restored['metric_value']:.4f}")
    print(f"Current {metric_name.upper()}: {metric_value:.4f}")

if __name__ == "__main__":
    main()
