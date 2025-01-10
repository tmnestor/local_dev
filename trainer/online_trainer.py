import torch
from typing import Dict, Any
import logging

import torch.nn as nn
import torch.optim as optim

class OnlineTrainer:
    def __init__(self, model: nn.Module, optimizer_config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize the online trainer.
        
        Args:
            model: PyTorch model to train
            optimizer_config: Dictionary containing optimizer parameters
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = self._setup_optimizer(optimizer_config)
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _setup_optimizer(self, config: Dict[str, Any]) -> optim.Optimizer:
        """Setup the optimizer based on config."""
        optimizer_type = config.get('type', 'Adam')
        lr = config.get('learning_rate', 0.001)
        
        if optimizer_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'SGD':
            return optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def train_step(self, batch_data, batch_labels):
        """Perform a single training step."""
        self.model.train()
        
        # Move data to device
        inputs = batch_data.to(self.device)
        labels = batch_labels.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, val_loader):
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy

    def update_model(self, new_data_loader, val_loader, epochs):
        """Remove checkpoint_dir parameter and related code"""
        # ...existing code without checkpoint saving...

if __name__ == "__main__":
    # Example usage
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    optimizer_config = {
        'type': 'Adam',
        'learning_rate': 0.001
    }
    
    trainer = OnlineTrainer(model, optimizer_config)