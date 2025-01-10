import os
import time
import copy
import logging
import yaml
import numpy as np
import torch
import torch.nn as nn
import optuna
import warnings
import datetime  # Add this import
from optuna._experimental import ExperimentalWarning
from typing import Dict, Any, Tuple, List
from pathlib import Path
from utils.metrics_manager import MetricsManager

from .base_trainer import PyTorchTrainer
from models.model_loader import load_model_from_yaml
from utils.logger import Logger  # Only import Logger

def restore_best_model(config):
    """Restore best model with architecture description."""
    logger = Logger.get_logger('ModelRestoration')  # Use Logger directly
    
    try:
        if os.path.exists(config['model']['save_path']):
            logger.info(f"Loading checkpoint from {config['model']['save_path']}")
            
            checkpoint = torch.load(
                config['model']['save_path'],
                map_location='cpu',
                weights_only=True
            )
            
            # Load and validate architecture
            arch_config = checkpoint.get('hyperparameters', {}).get('architecture', None)
            if not arch_config:
                with open(config['model']['architecture_yaml'], 'r') as f:
                    arch_config = yaml.safe_load(f)
            
            # Create and validate model
            model = load_model_from_yaml(config['model']['architecture_yaml'])
            model.train(False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create optimizer
            optimizer = getattr(torch.optim, config['training']['optimizer']['name'])(
                model.parameters(),
                **config['training']['optimizer']['params']
            )
            
            # Get architecture type
            arch_type = "ResNet MLP" if any(
                isinstance(layer, dict) and layer.get('residual', False)
                for layer in arch_config.get('layers', [])
            ) else "Complex MLP"
            
            return {
                'model': model,
                'optimizer': optimizer,
                'metric_name': config['training']['metric'],
                'metric_value': checkpoint.get('metric_value', 0.0),
                'hyperparameters': checkpoint.get('hyperparameters', {}),
                'architecture_type': arch_type,
                'architecture_description': f"{arch_type} with {len([l for l in arch_config['layers'] if l.get('type') == 'linear'])} layers"
            }
            
    except Exception as e:
        logger.error(f"Failed to restore model: {str(e)}")
        return _create_default_model(config)

def _create_default_model(config):
    """Create a new model with default configuration."""
    logger = Logger.get_logger('ModelRestoration')  # Use Logger directly
    
    with open(config['model']['architecture_yaml'], 'r') as f:
        architecture = yaml.safe_load(f)
    
    model = load_model_from_yaml(config['model']['architecture_yaml'])
    model.train(False)
    
    optimizer = getattr(torch.optim, config['training']['optimizer']['name'])(
        model.parameters(),
        **config['training']['optimizer']['params']
    )
    
    logger.info("Created new model with default configuration")
    
    return {
        'model': model,
        'optimizer': optimizer,
        'metric_name': config['training']['metric'],
        'metric_value': 0.0,
        'hyperparameters': {
            'architecture': architecture,
            'lr': config['training']['optimizer']['params']['lr'],
            'weight_decay': config['training']['optimizer']['params'].get('weight_decay', 0.0)
        },
        'architecture_type': 'Default',
        'architecture_description': 'Default Architecture',
        'needs_validation': True
    }

def save_best_params_to_config(config_path, best_trial, best_params):
    """Save best parameters to config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create best_model section if it doesn't exist
    if 'best_model' not in config:
        config['best_model'] = {}
    
    # Extract parameters based on architecture type
    arch_type = best_trial.params['architecture']
    param_prefix = 'resnet/' if arch_type == 'resnet_mlp' else 'complex/'
    
    # Get parameters with correct prefix
    hidden_size = best_trial.params[f'{param_prefix}hidden_size']
    n_layers = best_trial.params[f'{param_prefix}n_layers']
    lr = best_trial.params[f'{param_prefix}lr']
    weight_decay = best_trial.params[f'{param_prefix}weight_decay']
    activation = best_trial.params[f'{param_prefix}activation']
    
    # For Complex MLP, get dropout rate
    dropout_rate = (
        best_trial.params[f'{param_prefix}dropout_rate'] 
        if arch_type == 'complex_mlp' 
        else 0.0
    )
    
    config['best_model'].update({
        'hidden_layers': [hidden_size] * n_layers,
        'dropout_rate': dropout_rate,
        'learning_rate': lr,
        'use_batch_norm': arch_type == 'resnet_mlp',
        'weight_decay': weight_decay,
        'best_metric_name': config['training']['metric'],
        'best_metric_value': best_trial.value,
        'n_layers': n_layers,
        'hidden_size': hidden_size,
        'activation': activation
    })
    
    # Save the best architecture from the YAML file
    with open(config['model']['architecture_yaml'], 'r') as f:
        base_architecture = yaml.safe_load(f)
    config['best_model']['architecture'] = base_architecture
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

class HyperparameterTuner:
    """Class for hyperparameter tuning using Optuna"""
    def __init__(self, config: Dict[str, Any], initial_model=None):
        self.config = config
        self.best_trial_value = float('-inf')
        self.best_params = None
        self.initial_model = initial_model
        
        # Setup single logger instance with correct path
        self.logger = Logger.get_timestamp_logger(
            'HyperTuner',
            log_dir='tuning'  # Simply pass 'tuning' as the subdirectory
        )
        
        # Load architectures
        models_dir = os.path.dirname(config['model']['architecture_yaml'])
        with open(config['model']['architecture_yaml'], 'r') as f:
            self.resnet_architecture = yaml.safe_load(f)
        
        complex_mlp_path = os.path.join(models_dir, 'complex_mlp.yaml')
        with open(complex_mlp_path, 'r') as f:
            self.complex_architecture = yaml.safe_load(f)
        
        # Remove duplicate logger setup
        
        # Initialize optimization statistics
        self.pruning_stats = {
            'pruned_trials': 0,
            'early_stopped_trials': 0,
            'completed_trials': 0
        }
        
        # Cache for storing trial results
        self.trial_history = {}
        
        # Create tuning-specific config with properly structured monitoring config
        self.tuning_config = copy.deepcopy(config)
        self.tuning_config['monitoring'] = {
            'enabled': False,
            'metrics': {
                'save_interval': 1,
                'plot_interval': 1
            },
            'memory': {
                'track': False
            },
            'performance': {
                'track_time': False,
                'track_throughput': False
            }
        }
        
        # Store model configuration
        self.input_size = config['model']['input_size']
        self.num_classes = config['model']['num_classes']
        self.device = config.get('device', 'cpu')
        
        # Setup proper model config
        self.model_config = {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'device': self.device
        }
        
        self.pruning_strategies = {
            'median': optuna.pruners.MedianPruner,
            'percentile': optuna.pruners.PercentilePruner,
            'hyperband': optuna.pruners.HyperbandPruner,
            'threshold': optuna.pruners.ThresholdPruner
        }
        
        # Add customizable parameter spaces
        self.param_spaces = {
            'learning_rate': {
                'type': 'float',
                'range': (1e-5, 1e-2),
                'log': True
            },
            'hidden_size': {
                'type': 'int',
                'range': (32, 512),
                'step': 32
            }
        }

    def create_model_and_optimizer(self, trial):
        """Create model and optimizer based on trial parameters."""
        # First suggest architecture type as it determines other parameters
        arch_type = trial.suggest_categorical('architecture', ['resnet_mlp', 'complex_mlp'])
        
        # Supported activations
        ACTIVATIONS = ['relu', 'gelu', 'leaky_relu', 'elu']
        
        # Group related parameters by architecture type
        if arch_type == 'resnet_mlp':
            params = {
                'hidden_size': trial.suggest_int('resnet/hidden_size', 64, 512, step=32),
                'n_layers': trial.suggest_int('resnet/n_layers', 2, 6),
                'lr': trial.suggest_float('resnet/lr', 5e-5, 5e-2, log=True),
                'weight_decay': trial.suggest_float('resnet/weight_decay', 1e-6, 1e-3, log=True),
                'activation': trial.suggest_categorical('resnet/activation', ACTIVATIONS),
                'batch_norm_momentum': trial.suggest_float('resnet/bn_momentum', 0.1, 0.9)
            }
            dropout_rate = 0.0
            use_batch_norm = True
        else:
            params = {
                'hidden_size': trial.suggest_int('complex/hidden_size', 128, 1024, step=32),
                'n_layers': trial.suggest_int('complex/n_layers', 3, 7),
                'lr': trial.suggest_float('complex/lr', 5e-5, 5e-2, log=True),
                'weight_decay': trial.suggest_float('complex/weight_decay', 1e-6, 1e-3, log=True),
                'dropout_rate': trial.suggest_float('complex/dropout_rate', 0.1, 0.7),
                'activation': trial.suggest_categorical('complex/activation', ACTIVATIONS),
                'layer_shrinkage': trial.suggest_float('complex/layer_shrinkage', 0.3, 0.7)
            }
            dropout_rate = params['dropout_rate']
            use_batch_norm = False

        # Create temporary architecture config with required fields
        architecture = {
            'architecture': arch_type,
            'input_size': self.input_size,  # Use stored input_size
            'hidden_size': params['hidden_size'],
            'num_classes': self.num_classes,  # Use stored num_classes
            'activation': params['activation'],
            'batch_norm': use_batch_norm,
            'dropout_rate': dropout_rate,
            'n_layers': params['n_layers']
        }
        
        # Create layers based on architecture type
        layers = (self._create_resnet_layers(params['n_layers'], params['hidden_size'], params['activation'])
                 if arch_type == 'resnet_mlp'
                 else self._create_complex_mlp_layers(params['n_layers'], params['hidden_size'],
                                                    dropout_rate, params['activation']))
        
        # Add layers to architecture
        architecture['layers'] = layers
        
        # Save temporary YAML and create model
        trial_yaml = f'trial_{trial.number}.yaml'
        try:
            with open(trial_yaml, 'w') as f:
                yaml.dump(architecture, f)
            
            # Create model with proper config
            model = load_model_from_yaml(trial_yaml)
            model = model.to(self.device)
            
        finally:
            if os.path.exists(trial_yaml):
                os.remove(trial_yaml)
        
        # Create optimizer
        optimizer = getattr(torch.optim, self.config['training']['optimizer']['name'])(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        
        return model, optimizer, {
            'architecture': architecture,
            'architecture_type': 'ResNet MLP' if arch_type == 'resnet_mlp' else 'Complex MLP',
            'lr': params['lr'],
            'weight_decay': params['weight_decay'],
            'n_layers': params['n_layers'],
            'hidden_size': params['hidden_size'],
            'dropout_rate': dropout_rate,
            'activation': params['activation'],
            'use_batch_norm': use_batch_norm,
            'input_size': self.input_size,  # Add input_size
            'num_classes': self.num_classes  # Add num_classes
        }

    def _create_resnet_layers(self, n_layers: int, hidden_size: int, activation: str) -> List[Dict]:
        """Create ResNet layers with chosen activation."""
        input_size = self.config['model']['input_size']
        output_size = self.config['model']['num_classes']
        
        layers = []
        
        # Input layer with batch norm
        layers.extend([
            {
                'type': 'linear',
                'in_features': input_size,
                'out_features': hidden_size,
                'activation': activation,
                'residual': False
            },
            {
                'type': 'batch_norm',
                'num_features': hidden_size
            }
        ])
        
        # Hidden layers with residual connections and batch norm
        for _ in range(n_layers - 1):
            layers.extend([
                {
                    'type': 'linear',
                    'in_features': hidden_size,  # Fixed: Use correct input size
                    'out_features': hidden_size,
                    'activation': activation,
                    'residual': True
                },
                {
                    'type': 'batch_norm',
                    'num_features': hidden_size
                }
            ])
        
        # Output layer
        layers.append({
            'type': 'linear',
            'in_features': hidden_size,  # Fixed: Use correct input size
            'out_features': output_size,
            'residual': False
        })
        
        return layers

    def _create_complex_mlp_layers(self, n_layers: int, hidden_size: int, dropout_rate: float, activation: str) -> List[Dict]:
        """Create Complex MLP layers with mandatory dropout and tapering width."""
        input_size = self.config['model']['input_size']
        output_size = self.config['model']['num_classes']
        
        # Calculate layer sizes with geometric reduction
        layer_sizes = [input_size, hidden_size]  # Start with input size
        current_size = hidden_size
        
        # Calculate reduction ratio for hidden layers
        if n_layers > 2:
            ratio = (output_size / hidden_size) ** (1.0 / (n_layers - 2))
            for _ in range(n_layers - 3):  # -3 because we already have input, first hidden, and will add output
                current_size = int(current_size * ratio)
                layer_sizes.append(current_size)
        
        layer_sizes.append(output_size)  # Add output size
        
        # Create layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            layers.append({
                'type': 'linear',
                'in_features': layer_sizes[i],
                'out_features': layer_sizes[i + 1],
                'activation': activation if i < len(layer_sizes) - 2 else None
            })
            
            # Add dropout after all layers except the last
            if i < len(layer_sizes) - 2:
                layers.append({
                    'type': 'dropout',
                    'p': dropout_rate
                })
        
        return layers

    def save_best_model(self, model, optimizer, metric_value, trial_params):
        """Save best model with architecture info."""
        # Get architecture type from trial params
        arch_type = trial_params.get('architecture_type', 
            "ResNet MLP" if trial_params['architecture']['architecture'] == 'resnet_mlp'
            else "Complex MLP")
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_name': self.config['training']['optimizer']['name'],
            'metric_name': self.config['training']['metric'],
            'metric_value': metric_value,
            'hyperparameters': trial_params,
            'architecture_type': arch_type,
            'architecture': trial_params['architecture']
        }
        # Save without logging - logging will be handled by trainer
        torch.save(checkpoint, self.config['model']['save_path'])

    def objective(self, trial, train_loader, val_loader):
        """Objective function for Optuna optimization."""
        start_time = time.time()
        model, optimizer, trial_params = self.create_model_and_optimizer(trial)
        arch_type = trial_params['architecture_type']
        trial.set_user_attr('architecture_type', arch_type)
        
        # Create criterion
        criterion = getattr(nn, self.config['training']['loss']['name'])()
        
        # Create proper configuration for trainer
        trainer_config = {
            'model': {
                'num_classes': self.config['model']['num_classes'],
                'input_size': self.config['model']['input_size'],
                'architecture': trial_params['architecture']
            },
            'training': self.config['training'],
            'device': self.config['device']
        }
        
        # Create trainer in tuning mode with proper configuration
        trainer = PyTorchTrainer(
            model, criterion, optimizer,
            device=self.config['device'],
            verbose=False,
            config=trainer_config,
            tuning_mode=True,
            metrics_tuning_mode=True
        )
        
        # Early performance check after first epoch
        trainer.train_epoch(train_loader)
        _, accuracy, f1 = trainer.evaluate(val_loader)
        metric = f1 if self.config['training']['metric'] == 'f1' else accuracy
        
        # Early pruning if first epoch performance is poor
        if len(self.trial_history) > 3:
            avg_first_metric = sum(
                t['first_epoch_metric'] 
                for t in self.trial_history.values()
            ) / len(self.trial_history)
            
            if metric < avg_first_metric * 0.8:  # 20% worse than average
                self.pruning_stats['pruned_trials'] += 1
                raise optuna.TrialPruned(
                    f"First epoch metric {metric:.4f} below threshold {avg_first_metric * 0.8:.4f}"
                )
        
        # Continue with rest of training
        patience = self.config['early_stopping']['patience']
        min_delta = self.config['early_stopping']['min_delta']
        best_metric = metric
        patience_counter = 0
        running_metrics = [metric]
        
        # Store first epoch metric
        trial.set_user_attr('first_epoch_metric', metric)
        
        for epoch in range(1, self.config['training']['epochs']):
            trainer.train_epoch(train_loader)
            _, accuracy, f1 = trainer.evaluate(val_loader)
            metric = f1 if self.config['training']['metric'] == 'f1' else accuracy
            
            trial.report(metric, epoch)
            
            if trial.should_prune():
                self.pruning_stats['pruned_trials'] += 1
                raise optuna.TrialPruned()
            
            running_metrics.append(metric)
            if len(running_metrics) > 3:
                running_metrics.pop(0)
            
            if metric > best_metric + min_delta:
                best_metric = metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            # More aggressive early stopping during tuning
            if patience_counter >= min(patience, 5):  # Use shorter patience during tuning
                self.pruning_stats['early_stopped_trials'] += 1
                # Set the completed epochs before breaking
                trial.set_user_attr('completed_epochs', epoch + 1)
                break
            
            # Dynamic pruning based on running average
            if epoch >= 3:
                avg_metric = sum(running_metrics) / len(running_metrics)
                if (best_metric - avg_metric) / best_metric > 0.2:  # 20% deterioration
                    self.pruning_stats['pruned_trials'] += 1
                    raise optuna.TrialPruned()
        
        # Set completed epochs if we completed all epochs
        if epoch + 1 >= self.config['training']['epochs']:
            trial.set_user_attr('completed_epochs', self.config['training']['epochs'])
        
        # Store trial results for parameter optimization
        self.trial_history[trial.number] = {
            'metric': best_metric,
            'params': trial_params,
            'first_epoch_metric': trial.user_attrs['first_epoch_metric'],
            'duration': time.time() - start_time,
            'completed_epochs': trial.user_attrs['completed_epochs'],
            'cv_stats': {
                'std': trial.user_attrs.get('cv_std', 0.0),
                'variance': trial.user_attrs.get('cv_variance', 0.0),
                'confidence_interval': trial.user_attrs.get('cv_confidence_interval', [0.0, 0.0]),
                'stability_score': trial.user_attrs.get('cv_stability', 0.0)
            }
        }
        
        self.pruning_stats['completed_trials'] += 1
        
        # Log efficiency statistics periodically
        if len(self.trial_history) % 5 == 0:
            self.logger.info("\nTuning Efficiency Stats:")
            self.logger.info(f"Completed: {self.pruning_stats['completed_trials']}")
            self.logger.info(f"Pruned: {self.pruning_stats['pruned_trials']}")
            self.logger.info(f"Early Stopped: {self.pruning_stats['early_stopped_trials']}")
            avg_duration = sum(t['duration'] for t in self.trial_history.values()) / len(self.trial_history)
            self.logger.info(f"Average trial duration: {avg_duration:.2f}s")
            self.logger.info("\nCross Validation Stats:")
            self.logger.info(f"Avg CV Std: {np.mean([t['cv_stats']['std'] for t in self.trial_history.values()]):.4f}")
            self.logger.info(f"Most Stable Trial: {min(self.trial_history.items(), key=lambda x: x[1]['cv_stats']['std'])[0]}")
        
        return best_metric

    def tune(self, train_loader, val_loader):
        """Run hyperparameter tuning with Optuna."""
        # Suppress experimental warning
        warnings.filterwarnings('ignore', category=ExperimentalWarning)
        
        # Use multivariate TPE sampler with proper parameter grouping
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=5,
            consider_prior=True,
            consider_magic_clip=True,
            consider_endpoints=True,
            n_ei_candidates=24,
            group=True  # Enable parameter grouping
        )
        
        # Create a study with parameter grouping enabled
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1,
                n_min_trials=4
            ),
            sampler=sampler
        )
        
        def log_trial(study, trial):
            if hasattr(trial, 'user_attrs'):
                arch_type = trial.user_attrs.get('architecture_type', 'Unknown')
                epochs = trial.user_attrs.get('completed_epochs', '?')
                first_epoch_metric = trial.user_attrs.get('first_epoch_metric', 0.0)
                
                if trial.state == optuna.trial.TrialState.PRUNED:
                    return  # Don't log pruned trials here
                
                msg = f"Trial {trial.number:02d} [{arch_type}] completed after {epochs} epochs"
                msg += f" (First epoch: {first_epoch_metric:.4f}, Final: {trial.value:.4f})"
                
                if study.best_trial.number == trial.number:
                    msg += " [NEW BEST]"
                
                self.logger.info(msg)
                print("=" * 80)  # Print separator directly without timestamp
        
        study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader),
            n_trials=self.config['tuning']['n_trials'],
            callbacks=[log_trial]
        )
        
        best_trial = study.best_trial
        self.logger.info(f"\nBest Trial: #{best_trial.number}")
        self.logger.info(f"Value: {best_trial.value:.4f}")
        
        return best_trial, best_trial.params
