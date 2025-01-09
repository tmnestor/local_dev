import os
import sys
import argparse
import logging
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
from utils.logging import setup_logger
from trainer.base_trainer import PyTorchTrainer
from trainer.hyperparameter_tuner import (
    HyperparameterTuner,
    restore_best_model,
    save_best_params_to_config
)
from trainer.cpu_optimizer import CPUOptimizer, set_seed

def _configure_dataloader_params(config):
    """Helper function to configure dataloader parameters."""
    params = config['dataloader'].copy()  # Use dataloader from root level
    if params['num_workers'] == 'auto':
        params['num_workers'] = min(os.cpu_count(), 8)
    return params

def train_mode(config_path: str, force_retrain: bool = False):
    config = load_config(config_path)
    logger = setup_logger('TrainMode')
    
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
    save_best_params_to_config(config_path, best_trial, best_params)
    
    logger.info("\n=== Stage 3: Final Training ===")
    logger.info("\nBest Model Configuration:")
    logger.info("=" * 60)
    logger.info("Architecture:")
    arch_type = restored.get('architecture_type', 
        "ResNet MLP" if any(
            isinstance(layer, dict) and layer.get('residual', False)
            for layer in restored['hyperparameters']['architecture'].get('layers', [])
        ) else "Complex MLP"
    )
    logger.info(f"  Type:        {arch_type}")
    logger.info(f"  Input Size:  {config['model']['input_size']}")
    
    # Modified section to show correct layer sizes for Complex MLP
    if arch_type == "Complex MLP":
        # Get layer sizes from the architecture
        layer_sizes = [
            layer['out_features'] 
            for layer in restored['hyperparameters']['architecture']['layers']
            if layer['type'] == 'linear'
        ]
        logger.info(f"  Layer Sizes:  {config['model']['input_size']} → " + 
                   " → ".join(str(size) for size in layer_sizes))
    else:
        # For ResNet MLP, show single hidden size as before
        logger.info(f"  Hidden Size: {restored['hyperparameters'].get('hidden_size', config['model'].get('hidden_size', 256))}")
    
    logger.info(f"  Num Layers:  {restored['hyperparameters'].get('n_layers', len([l for l in restored['hyperparameters']['architecture'].get('layers', []) if l['type'] == 'linear']))}")
    logger.info(f"  Output Size: {config['model']['num_classes']}")
    
    logger.info("\nFeatures:")
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
    
    # Train model
    trainer = PyTorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=config['device'],  # Changed from training.device to root level
        verbose=True
    )
    
    logger.info("Starting training...")
    _, _, _, _, best_metric = trainer.train(
        train_loader,
        val_loader,
        config['training']['epochs'],
        metric=config['training']['metric']  # Changed from 'optimization_metric' to 'metric'
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

def inference_mode(config_path: str):
    """Run inference mode."""
    config = load_config(config_path)
    logger = setup_logger('InferenceMode')
    
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
    
    # Load validation data
    val_df = pd.read_csv(config['data']['val_path'])
    val_dataset = CustomDataset(val_df, config['data']['target_column'])
    val_loader = DataLoader(
        val_dataset,
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
    
    # Create trainer and evaluate
    trainer = PyTorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=restored['optimizer'],
        device=config['device'],  # Changed from training.device to root level
        verbose=True
    )
    
    logger.info("Evaluating model on validation set...")
    val_loss, val_accuracy, val_f1 = trainer.evaluate(val_loader)
    
    # Log results
    logger.info(f"\nValidation Results:")
    logger.info(f"Loss: {val_loss:.4f}")
    logger.info(f"Accuracy: {val_accuracy:.2f}%")
    logger.info(f"F1-Score: {val_f1:.4f}")

def online_learning_mode(config_path: str):
    """Mode for online learning with new data"""
    config = load_config(config_path)
    logger = setup_logger('OnlineLearning')
    
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
    config = load_config(args.config)
    seed = config.get('seed', config['training'].get('seed', 42))
    set_seed(seed)
    
    if args.mode == 'train':
        train_mode(args.config, args.force_retrain)
    elif args.mode == 'infer':
        inference_mode(args.config)
    else:
        online_learning_mode(args.config)

if __name__ == "__main__":
    main()
