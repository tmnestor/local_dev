import os
import sys
import argparse
import logging  # Add this import
from pathlib import Path
from typing import Dict, Any

# Data handling
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader

# Local imports
from models.model_loader import load_model_from_yaml, LabelSmoothingLoss
from utils.data import CustomDataset
from utils.config import (
    load_config,
    resolve_optimizer_args,
    resolve_loss_args
)
from utils.logger import Logger  # Only import Logger, remove setup_logger
from trainer.base_trainer import PyTorchTrainer
from trainer.hyperparameter_tuner import (
    HyperparameterTuner,
    restore_best_model,
    save_best_params_to_config
)
from trainer.cpu_optimizer import CPUOptimizer, set_seed
from utils.config_validator import ConfigValidator, ConfigValidationError
import click
from functools import wraps

def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ConfigValidationError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.debug("Stack trace:", exc_info=True)
            sys.exit(1)
    return wrapper

@click.group()
def cli():
    """ML Training and Inference CLI"""
    pass

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--force-retrain', is_flag=True, help='Force retraining')
@handle_errors
def train(config, force_retrain):
    """Train the model"""
    train_mode(config, force_retrain)

def _configure_dataloader_params(config):
    """Helper function to configure dataloader parameters."""
    params = config['dataloader'].copy()  # Use dataloader from root level
    if params['num_workers'] == 'auto':
        params['num_workers'] = min(os.cpu_count(), 8)
    return params

def train_mode(config_path: str, force_retrain: bool = False):
    try:
        config = load_config(config_path)
        ConfigValidator.validate_config(config)
        # Get logger with specific name to avoid duplication
        logger = Logger.get_logger('TrainMode', console_output=True)
        
        # Ensure architecture yaml exists and has proper content
        if not os.path.exists(config['model']['architecture_yaml']):
            raise FileNotFoundError(f"Architecture YAML not found: {config['model']['architecture_yaml']}")
            
        try:
            with open(config['model']['architecture_yaml']) as f:
                arch_config = yaml.safe_load(f)
                if not arch_config:
                    raise ValueError("Empty architecture configuration")
                if 'architecture' not in arch_config and 'layers' not in arch_config:
                    raise ValueError("Architecture YAML must specify 'architecture' type or 'layers'")
        except Exception as e:
            logger.error(f"Failed to load architecture YAML: {e}")
            raise
        
        dataloader_params = _configure_dataloader_params(config)
        
        train_df = pd.read_csv(config['data']['train_path'])
        val_df = pd.read_csv(config['data']['val_path'])
        
        train_dataset = CustomDataset(train_df, config['data']['target_column'])
        val_dataset = CustomDataset(val_df, config['data']['target_column'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            **dataloader_params
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            **dataloader_params
        )
        
        logger.info("\n=== Stage 1: Model Setup ===")
        # Always try to load previous best model first, or create from yaml spec
        logger.info("Loading model...")
        try:
            if os.path.exists(config['model']['save_path']):
                logger.info("Found previous best model, loading it...")
                restored = restore_best_model(config)
            else:
                logger.info("No previous model found, creating from yaml specification...")
                model = load_model_from_yaml(config['model']['architecture_yaml'])
                
                # Load architecture YAML to determine type
                with open(config['model']['architecture_yaml']) as f:
                    architecture = yaml.safe_load(f)
                
                # Determine architecture type from layers
                arch_type = "ResNet MLP" if any(
                    isinstance(layer, dict) and layer.get('residual', False)
                    for layer in architecture['layers']
                ) else "Complex MLP"
                
                arch_desc = f"{arch_type} with default configuration"
                
                # Get hidden size safely from architecture
                hidden_size = None
                for layer in architecture['layers']:
                    if isinstance(layer, dict) and layer['type'] == 'linear':
                        if layer['in_features'] == config['model']['input_size']:
                            hidden_size = layer['out_features']
                            break
                
                if hidden_size is None:
                    raise ValueError("Could not determine hidden size from architecture. Check layer specifications.")
                
                optimizer = getattr(torch.optim, config['training']['optimizer']['name'])(
                    model.parameters(),
                    **config['training']['optimizer']['params']
                )
                
                restored = {
                    'model': model,
                    'optimizer': optimizer,
                    'metric_name': config['training']['metric'],
                    'metric_value': 0.0,
                    'hyperparameters': {
                        'architecture': architecture,
                        'lr': config['training']['optimizer']['params']['lr'],
                        'weight_decay': config['training']['optimizer']['params'].get('weight_decay', 0.0),
                        'n_layers': len([l for l in architecture['layers'] if l.get('type') == 'linear']),
                        'hidden_size': hidden_size
                    },
                    'architecture_type': arch_type,
                    'architecture_description': arch_desc
                }
            
            model = restored['model']
            optimizer = restored['optimizer']
            logger.info(f"Model ready with metric value: {restored['metric_value']:.4f}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
        
        logger.info("\n=== Stage 2: Hyperparameter Tuning ===")
        # Always run hyperparameter tuning to find best configuration
        logger.info("Starting hyperparameter tuning...")
        tuner = HyperparameterTuner(config, initial_model=model if restored else None)
        best_trial, best_params = tuner.tune(train_loader, val_loader)
        
        # Add cross-validation stability analysis
        cv_stats = tuner.trial_history[best_trial.number]['cv_stats']
        logger.info("\nBest Model Cross Validation Analysis:")
        logger.info(f"Standard Deviation: {cv_stats['std']:.4f}")
        logger.info(f"95% Confidence Interval: [{cv_stats['confidence_interval'][0]:.4f}, {cv_stats['confidence_interval'][1]:.4f}]")
        logger.info(f"Stability Score: {cv_stats['stability_score']:.4f}")
        
        # Only save if the model is stable enough
        if cv_stats['std'] < config.get('cv_stability_threshold', 0.1):
            save_best_params_to_config(config_path, best_trial, best_params)
            logger.info("Model saved - passed stability checks")
        else:
            logger.warning("Model not saved - failed stability checks")
        
        logger.info("\n=== Stage 3: Final Training with Cross Validation ===")
        
        # Get optimizer and loss args first
        optimizer_args = resolve_optimizer_args(config)
        loss_args = resolve_loss_args(config)
        
        # Create trainer for cross validation with newly tuned model
        criterion = getattr(torch.nn, loss_args['name'])()
        if loss_args['label_smoothing']['enabled']:
            criterion = LabelSmoothingLoss(
                num_classes=config['model']['num_classes'],
                smoothing=loss_args['label_smoothing']['factor']
            )
        
        # Create trainer once and reuse it
        trainer = PyTorchTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=config['device'],
            verbose=True,
            config=config  # Pass full config for metrics and monitoring
        )
        
        # Create complete dataset for cross validation
        complete_dataset = CustomDataset(
            pd.concat([train_df, val_df], ignore_index=True),
            config['data']['target_column']
        )
        
        # Now perform cross validation with the trainer
        cv_results = trainer.cross_validate(
            complete_dataset,
            n_splits=config['training'].get('cross_validation', {}).get('n_splits', 5),  # Get from config with default
            epochs=min(config['training']['epochs'], 
                      config['training'].get('cross_validation', {}).get('max_epochs', 20))  # Get from config with default
        )
        
        # Get CV summary
        cv_summary = trainer.metrics_manager.summarize_cross_validation(cv_results)
        
        # Log CV results
        logger.info("\nCross Validation Results:")
        logger.info("=" * 60)
        for metric, stats in cv_summary.items():
            if isinstance(stats, dict) and 'mean' in stats:
                logger.info(f"\n{metric}:")
                logger.info(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                logger.info(f"  95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
                logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Plot CV results
        trainer.metrics_manager.plot_cv_results(
            cv_results,
            Path(config['monitoring']['log_dir'])
        )
        
        # Continue with final training on full dataset
        logger.info("\nProceeding with final training on complete dataset...")
        
        logger.info("\nBest Model Configuration:")
        logger.info("=" * 60)
        logger.info("Architecture:")
        
        # Safely handle architecture information
        arch_config = restored['hyperparameters']['architecture']
        arch_type = restored.get('architecture_type')  # Get from restored state
        if not arch_type:
            # Fallback to inference from layers
            arch_type = "ResNet MLP" if any(
                isinstance(layer, dict) and layer.get('residual', False)
                for layer in arch_config.get('layers', [])
            ) else "Complex MLP"
        
        logger.info(f"  Type:        {arch_type}")
        logger.info(f"  Input Size:  {arch_config.get('input_size', config['model']['input_size'])}")
        logger.info(f"  Hidden Size: {arch_config.get('hidden_size', 128)}")
        logger.info(f"  Num Layers:  {len([l for l in arch_config['layers'] if l.get('type') == 'linear'])}")
        logger.info(f"  Output Size: {config['model']['num_classes']}")
        
        logger.info("\nFeatures:")
        # Get activation from architecture configuration
        activation = None
        for layer in arch_config['layers']:
            if isinstance(layer, dict) and layer.get('activation'):
                activation = layer['activation']
                break
        if activation:
            logger.info(f"  Activation:    {activation}")
        
        # Determine regularization type and details
        if arch_type == "Complex MLP":
            dropout_rate = next(
                (layer['p'] for layer in arch_config['layers'] 
                 if isinstance(layer, dict) and layer.get('type') == 'dropout'),
                None
            )
            if dropout_rate is not None:
                logger.info(f"  Regularization: Dropout ({dropout_rate:.4f})")
        else:
            logger.info("  Regularization: BatchNorm + Residual")
        
        logger.info("\nFeatures:")
        # Get activation from first linear layer or config
        activation = None
        for layer in arch_config.get('layers', []):
            if isinstance(layer, dict) and layer.get('activation'):
                activation = layer['activation']
                break
        if activation:
            logger.info(f"  Activation:    {activation}")
        
        if 'activation' in restored['hyperparameters']:
            logger.info(f"  Activation:    {restored['hyperparameters']['activation']}")
        if restored['architecture_type'] == 'Complex MLP':
            logger.info(f"  Regularization: Dropout ({restored['hyperparameters']['dropout_rate']:.4f})")
            logger.info("  Layer Structure:")
            for layer in model.children():
                logger.info(f"    {layer}")
        else:
            logger.info("  Regularization: BatchNorm + Residual")
        
        logger.info("\nOptimization:")
        logger.info(f"  Learning Rate: {restored['hyperparameters']['lr']:.6f}")
        logger.info(f"  Weight Decay:  {restored['hyperparameters']['weight_decay']:.8f}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("\nModel Size:")
        logger.info(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        logger.info("\nPrevious Performance:")
        logger.info(f"  {restored['metric_name']}: {restored['metric_value']:.4f}")
        logger.info("=" * 60)
        
        logger.info("\nStarting final training phase...")
        
        # Create criterion
        optimizer_args = resolve_optimizer_args(config)
        loss_args = resolve_loss_args(config)
        
        criterion = getattr(torch.nn, loss_args['name'])()
        if loss_args['label_smoothing']['enabled']:
            criterion = LabelSmoothingLoss(
                num_classes=config['model']['num_classes'],
                smoothing=loss_args['label_smoothing']['factor']
            )
        
        # Train model without resetting performance monitor
        # Continue with final training using the same trainer instance
        logger.info("\nStarting final training phase...")
        _, _, _, _, best_metric = trainer.train(
            train_loader,
            val_loader,
            config['training']['epochs'],
            metric=config['training']['metric']
        )
        
        # Save trained model
        logger.info("Saving trained model...")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_name': config['training']['optimizer']['name'],  # Updated path
            'metric_name': config['training']['metric'],  # Changed from 'optimization_metric' to 'metric'
            'metric_value': best_metric,
            'hyperparameters': restored['hyperparameters']
        }
        torch.save(checkpoint, config['model']['save_path'])
        logger.info(f"Model saved to {config['model']['save_path']}")
        logger.info(f"Final performance: {best_metric:.4f}")
    except ConfigValidationError as e:
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

def inference_mode(config_path: str):
    """Run inference mode."""
    config = load_config(config_path)
    logger = Logger.get_logger('InferenceMode', level=logging.DEBUG)  # Replace setup_logger
    
    # Add data inspection
    test_df = pd.read_csv(config['data']['test_path'])
    logger.debug(f"Test data shape: {test_df.shape}")
    logger.debug(f"Test class distribution:\n{test_df[config['data']['target_column']].value_counts()}")
    
    if not os.path.exists(config['model']['save_path']):
        raise FileNotFoundError(f"Model checkpoint not found at {config['model']['save_path']}")
    
    # Restore best model
    logger.info("Restoring best model...")
    restored = restore_best_model(config)
    model = restored['model']
    
    # Log model information
    logger.info("\nModel Information:")
    arch_type = restored.get('architecture_type', 
        "ResNet MLP" if any(
            isinstance(layer, dict) and layer.get('residual', False)
            for layer in restored['hyperparameters']['architecture'].get('layers', [])
        ) else "Complex MLP"
    )
    logger.info(f"Architecture: {arch_type}")
    logger.info(f"Best {restored['metric_name']}: {restored['metric_value']:.4f}")
    
    logger.info("\nArchitecture Details:")
    arch_config = restored['hyperparameters']['architecture']
    logger.info(f"  Input Size:  {arch_config.get('input_size', config['model']['input_size'])}")
    
    # Show proper layer sizes for Complex MLP
    if arch_type == "Complex MLP":
        # Get layer sizes from the architecture's layers
        layer_sizes = [
            layer['out_features'] 
            for layer in arch_config.get('layers', [])
            if layer['type'] == 'linear'
        ]
        input_size = config['model']['input_size']
        logger.info(f"  Layer Sizes: {input_size} → " + 
                   " → ".join(str(size) for size in layer_sizes))
    else:
        # For ResNet MLP, show single hidden size
        logger.info(f"  Hidden Size: {arch_config.get('hidden_size', config['model'].get('hidden_size', 256))}")
    
    logger.info(f"  Num Layers:  {arch_config.get('n_layers', len([l for l in arch_config.get('layers', []) if l.get('type') == 'linear']))}")
    logger.info(f"  Output Size: {arch_config.get('num_classes', config['model']['num_classes'])}")
    
    # Print model structure
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel Parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    
    # Load test data instead of validation data
    test_df = pd.read_csv(config['data']['test_path'])
    test_dataset = CustomDataset(test_df, config['data']['target_column'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()
    if config['training'].get('label_smoothing', {}).get('enabled', False):
        from test_load_model_from_yaml import LabelSmoothingLoss
        criterion = LabelSmoothingLoss(
            num_classes=config['model']['num_classes'],
            smoothing=config['training']['label_smoothing']['factor']
        )
    
    # Create trainer with proper metrics manager initialization
    trainer = PyTorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=restored['optimizer'],
        device=config['device'],
        verbose=True,
        config=config,  # Add config to enable metrics manager
        tuning_mode=False,  # Ensure tuning mode is off
        metrics_tuning_mode=False  # Ensure metrics tuning mode is off
    )
    
    # Ensure monitoring directory exists
    monitoring_dir = Path(config['monitoring']['log_dir'])
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Evaluating model on test set...")  # Updated message
    test_loss, test_accuracy, test_f1 = trainer.evaluate(test_loader)  # Changed var names
    
    # Generate and print comprehensive report
    if trainer.metrics_manager:
        report = trainer.metrics_manager.generate_comprehensive_report()
        logger.info("\nComprehensive Evaluation Report:")
        for line in report.split('\n'):
            logger.info(line)
        
        # Save report to file
        report_path = monitoring_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"\nDetailed evaluation report saved to: {report_path}")
    
    # Log basic results
    logger.info(f"\nSummary Results:")
    logger.info(f"Loss: {test_loss:.4f}")
    logger.info(f"Accuracy: {test_accuracy:.2f}%")
    logger.info(f"F1-Score: {test_f1:.4f}")

def online_learning_mode(config_path: str):
    """Mode for online learning with new data"""
    config = load_config(config_path)
    logger = Logger.get_logger('OnlineLearning')  # Replace setup_logger
    
    # Initialize experiment tracker
    experiment_tracker = ExperimentTracker(
        experiments_dir=config.get('experiments_dir', 'experiments')
    )
    
    # Start new experiment
    experiment_id = experiment_tracker.start_experiment(
        name="online_learning",
        config=config
    )
    
    # Restore best model
    restored = restore_best_model(config)
    model = restored['model']
    optimizer = restored['optimizer']
    
    # Create trainer
    trainer = OnlineTrainer(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=config['device'],
        experiment_tracker=experiment_tracker,
        verbose=True
    )
    
    # Load new data
    new_data_df = pd.read_csv(config['data']['new_data_path'])
    val_df = pd.read_csv(config['data']['val_path'])
    
    new_data_dataset = CustomDataset(new_data_df, config['data']['target_column'])
    val_dataset = CustomDataset(val_df, config['data']['target_column'])
    
    new_data_loader = DataLoader(
        new_data_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Update model with new data
    metrics = trainer.update_model(
        new_data_loader=new_data_loader,
        val_loader=val_loader,
        epochs=config['online_learning']['epochs'],
        checkpoint_dir=config['online_learning']['checkpoint_dir']
    )
    
    # End experiment
    experiment_tracker.end_experiment()
    
    logger.info("\nOnline Learning Results:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

# Update main() function
def main():
    parser = argparse.ArgumentParser(description='Train or run inference with PyTorch model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to config YAML file')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['train', 'infer', 'online'],
                      help='Mode to run: train, infer, or online')
    parser.add_argument('--force-retrain', action='store_true',
                      help='Force retraining even if best model exists')
    
    args = parser.parse_args()
    
    # Initialize logging only once at the start
    logging.basicConfig(level=logging.INFO)  # Add basic logging configuration
    Logger.setup()
    logger = Logger.get_logger('Main', console_output=True)
    
    try:
        config = load_config(args.config)
        seed = config.get('seed', config['training'].get('seed', 42))
        set_seed(seed)
        
        if args.mode == 'train':
            train_mode(args.config, args.force_retrain)
        elif args.mode == 'infer':
            inference_mode(args.config)
        else:
            online_learning_mode(args.config)
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
