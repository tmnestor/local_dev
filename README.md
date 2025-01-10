# PyTorch Training Framework

A robust deep learning framework with advanced model architecture selection, hyperparameter tuning, and comprehensive performance analysis.

## Key Features

### Model Architectures
- **ResNet MLP**
  - Residual connections for deep networks
  - Configurable batch normalization
  - Adaptive layer sizing
  - Gradient-friendly skip connections

- **Complex MLP**
  - Dynamic width adaptation
  - Progressive layer size reduction
  - Optimized dropout rates
  - Intelligent regularization

### Advanced Training Features
- Early stopping with configurable patience
- Learning rate scheduling with warmup
- Gradient clipping and normalization
- Cross-validation with stability analysis
- Performance-based pruning
- Data leakage detection

### Comprehensive Metrics Analysis
- **Performance Metrics**
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC and PR-AUC curves
  - Cohen's Kappa coefficient
  - Per-class performance analysis

- **Statistical Analysis**
  - Chi-square independence tests
  - McNemar's test for model comparison
  - Cross-validation stability metrics
  - Confidence intervals for all metrics

- **Visualization**
  - Normalized confusion matrices
  - Learning curve plots
  - Per-class performance plots
  - Cross-validation analysis plots

### Hyperparameter Optimization
- Optuna integration with TPE sampler
- Multi-objective optimization
- Cross-trial learning
- Early pruning of poor trials
- Resource-aware scheduling

## Quick Start

1. Installation:
```bash
git clone https://github.com/tmnestor/pytorch-training-framework.git
cd pytorch-training-framework
pip install -r requirements.txt
```

2. Create configuration (config.yaml):
```yaml
model:
  architecture_yaml: "models/architectures/resnet_mlp.yaml"
  save_path: "checkpoints/model.pt"
  input_size: 7
  num_classes: 5

training:
  epochs: 100
  batch_size: 32
  optimizer:
    name: "Adam"
    params:
      lr: 0.001
  metric: "f1"  # or "accuracy"
  
monitoring:
  log_dir: "logs"
  enabled: true
  metrics:
    save_interval: 1
    plot_interval: 5
```

3. Run training:
```bash
python main.py --mode train --config config.yaml
```

## Usage Modes

### Training
```bash
python main.py --mode train --config config.yaml [--force-retrain]
```

### Inference with Analysis
```bash
python main.py --mode infer --config config.yaml
```
- Generates comprehensive performance report
- Saves detailed metrics and visualizations
- Performs statistical significance testing

### Online Learning
```bash
python main.py --mode online --config config.yaml
```

## Performance Reports

The framework generates detailed performance reports including:

1. Overall Metrics
   - Accuracy, Precision, Recall, F1-Score
   - Confidence intervals and statistical tests

2. Per-Class Analysis
   - Individual class performance metrics
   - Class distribution analysis
   - Error analysis

3. Statistical Tests
   - Chi-square test results
   - McNemar's test
   - Cross-validation stability

4. Visualizations
   - Confusion matrices
   - ROC and PR curves
   - Learning curves
   - Cross-validation plots

## Configuration Guide

### Model Architecture
```yaml
architecture: resnet_mlp  # or complex_mlp
input_size: 7
hidden_size: 256
num_classes: 5
batch_norm: true
activation: "relu"
n_layers: 3
```

### Training Settings
```yaml
training:
  epochs: 100
  batch_size: 32
  early_stopping:
    patience: 10
    min_delta: 0.001
  cross_validation:
    n_splits: 5
    max_epochs: 20
```

### Monitoring Configuration
```yaml
monitoring:
  enabled: true
  log_dir: "logs"
  metrics:
    save_interval: 1
    plot_interval: 5
  performance:
    track_time: true
    track_memory: true
```

## Project Structure
```
pytorch-training-framework/
├── main.py
├── trainer/
│   ├── base_trainer.py
│   ├── hyperparameter_tuner.py
│   └── online_trainer.py
├── models/
│   ├── architectures/
│   │   ├── base.py
│   │   ├── resnet_mlp.py
│   │   └── complex_mlp.py
│   └── model_loader.py
├── utils/
│   ├── metrics_manager.py
│   ├── performance_monitor.py
│   ├── config.py
│   └── logger.py
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@software{pytorch_training_framework,
  author = {Tod M. Nestor},
  title = {PyTorch Training Framework},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/tmnestor/pytorch-training-framework}
}
```