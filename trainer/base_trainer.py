#!/usr/bin/env python
# coding: utf-8

import os
import sys  # Add this import
import random
import copy
import logging
import datetime  # Add this import
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
from utils.config import setup_logger
from utils.data import CustomDataset
from models.model_loader import (
    ModuleFactory, 
    load_model_from_yaml, 
    ResidualBlock,
    LabelSmoothingLoss
)


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
    
    def __init__(self, model, criterion, optimizer, device='cpu', verbose=False):
        self.model = model.to(device)
        self.criterion = criterion  # Fix typo: was 'criteriontrial'
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        self.best_model_state = None
        self.best_metric = float('-inf')
        self.best_epoch = 0
        
    def train_epoch(self, train_loader):
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
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            # Save the best model and optimizer from hyperparameter tuning. Reload this best model after hyperparameter tuning. apply the reloaded model to the validation dataset as the final step, to compare its performance with the results of the train_final_model step.
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        return total_loss / len(train_loader), accuracy
    
    def evaluate(self, val_loader):
        """Evaluates the model on validation data."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return total_loss / len(val_loader), accuracy, f1

    def train(self, train_loader, val_loader, epochs, metric='accuracy',
              early_stop_patience=10, early_stop_delta=0.001, max_iter=100):
        """Trains the model with early stopping and best model saving."""
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        patience_counter = 0
        
        for epoch in tqdm(range(min(epochs, max_iter)), desc='Training'):
            train_loss, train_accuracy = self.train_epoch(train_loader)
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
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
        
        return train_losses, val_losses, train_metrics, val_metrics, self.best_metric
    
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






    print(f"Validation F1-Score: {val_f1:.4f}")
    print(f"\nBest {metric_name.upper()} from tuning: {restored['metric_value']:.4f}")

    print(f"Current {metric_name.upper()}: {metric_value:.4f}")

if __name__ == "__main__":
    main()





    print(f"Current {metric_name.upper()}: {metric_value:.4f}")

if __name__ == "__main__":
    main()




