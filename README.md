# PyTorch Training Framework

A robust deep learning framework optimized for automated model architecture selection and hyperparameter tuning.

## Key Features

### Advanced Architecture Support
- **ResNet MLP**
  - Residual connections for deep networks
  - Configurable batch normalization
  - Fixed-width or expanding layers
  - Gradient-friendly skip connections

- **Complex MLP**
  - Automatic width tapering
  - Progressive layer size reduction
  - Dynamic dropout rates
  - Adaptive regularization

### Training Optimizations
- Learning rate scheduling with warmup
- Gradient clipping (max norm: 1.0)
- Early stopping with patience
- Dynamic pruning of poor trials
- Multivariate hyperparameter sampling
- Trial history tracking
- Performance-based early termination

### Hyperparameter Tuning
- Optuna-based optimization
- TPE sampler with:
  - Multivariate parameter relationships
  - Automated pruning
  - Cross-trial learning
  - Performance tracking
  - Resource optimization

### Monitoring & Validation
- Real-time metrics tracking
- Unique learning curve plots
- Trial efficiency statistics
- Detailed architecture logging
- Cross-validation support

## Installation

```bash
git clone https://github.com/yourusername/pytorch-training-framework.git
cd pytorch-training-framework
pip install -r requirements.txt
```

## Quick Start

1. Create a configuration file (`config.yaml`):
```yaml
model:
  architecture_yaml: "models/architectures/resnet_mlp.yaml"
  save_path: "checkpoints/model.pt"
  input_size: 784
  num_classes: 10

training:
  epochs: 100
  batch_size: 32
  optimizer:
    name: "Adam"
    params:
      lr: 0.001
      weight_decay: 0.0001
  metric: "accuracy"
  loss:
    name: "CrossEntropyLoss"
  early_stopping:
    patience: 10
    min_delta: 0.001

tuning:
  n_trials: 50
  pruning:
    warm_up_epochs: 5
    min_trials_complete: 10

data:
  train_path: "data/train.csv"
  val_path: "data/val.csv"
  target_column: "label"
  
dataloader:
  num_workers: "auto"
  pin_memory: true
```

2. Run training:
```bash
python main.py --mode train --config config.yaml
```

## Usage Modes

### Training
```bash
python main.py --mode train --config config.yaml [--force-retrain]
```

### Inference
```bash
python main.py --mode infer --config config.yaml
```

### Online Learning
```bash
python main.py --mode online --config config.yaml
```

## Architecture Configuration

### ResNet MLP Example
```yaml
architecture: resnet_mlp
input_size: 784
hidden_size: 256
num_classes: 10
batch_norm: true
activation: "relu"
n_layers: 3
```

### Complex MLP Example
```yaml
architecture: complex_mlp
input_size: 784
hidden_size: 512
num_classes: 10
dropout_rate: 0.3
activation: "gelu"
n_layers: 4
layer_shrinkage: 0.5
```

## Performance Features

### Auto-Optimization
- Learning rate scheduling
- Batch size optimization
- Layer width adaptation
- Dropout rate tuning
- Early stopping thresholds
- Pruning criteria

### Hardware Utilization
- CPU thread optimization
- Memory usage monitoring
- Batch size auto-scaling
- Worker count adaptation

### Monitoring
- Real-time metrics
- Resource utilization
- Training progression
- Model comparisons

## Project Structure
```
pytorch-training-framework/
├── main.py                # Entry point
├── trainer/
│   ├── base_trainer.py    # Core training logic
│   ├── hyperparameter_tuner.py
│   ├── online_trainer.py  # Online learning
│   └── cpu_optimizer.py
├── models/
│   ├── architectures/
│   │   ├── base.py
│   │   ├── resnet_mlp.py
│   │   ├── complex_mlp.py
│   │   └── registry.py
│   └── model_loader.py
└── utils/
    ├── config.py
    ├── data.py
    └── logging.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{pytorch_training_framework,
  author = {Tod M. Nestor},
  title = {PyTorch Training Framework},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/tmnestor/pytorch-training-framework}
}