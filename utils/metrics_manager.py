from typing import Dict, List, Any
import numpy as np
import pandas as pd  # Add pandas import
import torch
from pathlib import Path
import datetime
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
    average_precision_score,
    cohen_kappa_score,
    f1_score  # Add f1_score import
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns  # Add this import
import logging
from utils.logger import Logger  # Add this import

class MetricsManager:
    """Manages comprehensive metrics collection and validation"""
    
    def __init__(self, num_classes: int, metrics_dir: Path = Path('metrics'), tuning_mode: bool = False, config: Dict = None):
        self.num_classes = num_classes
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.config = config  # Store config for use in plot_cv_results
        
        # Store predictions for statistical analysis
        self.predictions = []
        self.true_labels = []
        self.train_metrics = []
        self.val_metrics = []
        self.current_epoch = 0
        
        # Simplified logging setup with new Logger class
        self.logger = Logger.get_logger(
            'MetricsManager',
            filename='metrics.log' if not tuning_mode else 'tuning_metrics.log'
        )
        
        self.tuning_mode = tuning_mode  # Add tuning mode flag
        self.accuracy_threshold = 0.99  # Adjust threshold to be more reasonable
        self.store_predictions = True  # Always store predictions

        self.data_checks = {
            'last_seen_inputs': None,
            'input_hashes': set(),
            'target_distribution': {}
        }
    
    def validate_inputs(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Validate inputs for data leakage"""
        # Convert inputs to numpy for analysis
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Check 1: Input duplication
        current_batch_hashes = set(hash(inputs_np[i].tobytes()) for i in range(len(inputs_np)))
        duplicate_inputs = len(current_batch_hashes) < len(inputs_np)
        
        # Check 2: Distribution analysis
        target_counts = np.bincount(targets_np, minlength=self.num_classes)
        perfectly_balanced = len(set(target_counts)) == 1
        
        # Check 3: Cross-batch duplication
        cross_batch_duplicates = False
        if self.data_checks['last_seen_inputs'] is not None:
            previous_inputs = self.data_checks['last_seen_inputs']
            for current_input in inputs_np:
                if any(np.allclose(current_input, prev_input) for prev_input in previous_inputs):
                    cross_batch_duplicates = True
                    break
        
        # Store current batch for next comparison
        self.data_checks['last_seen_inputs'] = inputs_np
        
        # Update target distribution
        for t in targets_np:
            self.data_checks['target_distribution'][t] = self.data_checks['target_distribution'].get(t, 0) + 1
        
        return {
            'duplicate_inputs': duplicate_inputs,
            'perfectly_balanced': perfectly_balanced,
            'cross_batch_duplicates': cross_batch_duplicates,
            'target_distribution': dict(self.data_checks['target_distribution'])
        }
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor, phase: str='val') -> Dict[str, float]:
        """Calculate comprehensive metrics for a batch"""
        if outputs is None or targets is None:
            return self._get_default_metrics()

        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Calculate accuracy
        accuracy = (predictions == targets).mean()
        
        # Store predictions for validation phase
        if phase == 'val':
            self.predictions.extend(predictions)
            self.true_labels.extend(targets)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions,
            average='macro',
            zero_division=0,
            labels=range(self.num_classes)
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'kappa': float(cohen_kappa_score(targets, predictions))
        }

        # Calculate per-class metrics
        for i in range(self.num_classes):
            class_targets = (targets == i).astype(int)
            class_preds = (predictions == i).astype(int)
            class_precision, class_recall, _, _ = precision_recall_fscore_support(
                class_targets, class_preds, 
                average='binary',
                zero_division=0
            )
            metrics[f'precision_class_{i}'] = float(class_precision)
            metrics[f'recall_class_{i}'] = float(class_recall)
        
        # Store metrics
        if phase == 'val':
            self.val_metrics.append(metrics)
        else:
            self.train_metrics.append(metrics)
            
        return metrics
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when validation fails"""
        # Add basic metrics that are always needed
        base_metrics = {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'kappa': 0.0
        }
        
        # Add per-class metrics
        for i in range(self.num_classes):
            base_metrics[f'precision_class_{i}'] = 0.0
            base_metrics[f'recall_class_{i}'] = 0.0
        
        return base_metrics

    def compute_statistical_significance(self) -> Dict[str, Any]:
        """Compute statistical significance of model performance"""
        if len(self.predictions) == 0:
            return {}
        
        # Convert to numpy arrays
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        # Chi-square test of independence
        contingency = pd.crosstab(y_true, y_pred)
        chi2, chi2_p = stats.chi2_contingency(contingency)[:2]
        
        # Build contingency table for manual McNemar calculation
        # This replaces the deprecated stats.mcnemar
        correct_pred = y_true == y_pred
        if len(y_true) > 1:
            b = np.sum(~correct_pred[:-1] & correct_pred[1:])  # only first wrong
            c = np.sum(correct_pred[:-1] & ~correct_pred[1:])  # only second wrong
            
            # Manual McNemar's test calculation
            if b + c > 0:
                mcnemar_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
                mcnemar_p = stats.chi2.sf(mcnemar_stat, df=1)
            else:
                mcnemar_stat, mcnemar_p = 0, 1.0
        else:
            mcnemar_stat, mcnemar_p = 0, 1.0
            
        return {
            'chi2_stat': chi2,
            'chi2_pvalue': chi2_p,
            'mcnemar_stat': mcnemar_stat,
            'mcnemar_pvalue': mcnemar_p,
            'significant_at_05': chi2_p < 0.05 or mcnemar_p < 0.05
        }
    
    def plot_confusion_matrix(self, epoch: int) -> None:
        """Plot confusion matrix with advanced visualization"""
        if len(self.predictions) == 0:
            return
            
        cm = confusion_matrix(self.true_labels, self.predictions)
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=range(self.num_classes),
            yticklabels=range(self.num_classes)
        )
        
        plt.title(f'Normalized Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy score
        acc = np.trace(cm) / np.sum(cm)
        plt.text(
            0.02, 0.98, f'Accuracy: {acc:.2%}',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.savefig(self.metrics_dir / f'confusion_matrix_epoch_{epoch}.png')
        plt.close()
    
    def generate_epoch_report(self, epoch: int) -> Dict[str, Any]:
        """Generate comprehensive report for epoch"""
        if not self.predictions or not self.true_labels:
            self.logger.warning("No predictions collected - cannot generate report")
            return {
                'epoch': epoch,
                'current_metrics': self._get_default_metrics(),
                'statistical_tests': {},
                'detailed_report': {},
                'metric_trends': {}
            }
            
        # Get latest metrics - ensure we have them
        if not self.val_metrics:
            self.logger.warning("No validation metrics found - calculating from stored predictions")
            metrics = self.calculate_metrics_from_stored()
        else:
            metrics = self.val_metrics[-1]
        
        # Get statistical significance
        significance = self.compute_statistical_significance()
        
        # Get detailed classification report
        report = classification_report(
            self.true_labels,
            self.predictions,
            output_dict=True
        )
        
        # Plot confusion matrix
        self.plot_confusion_matrix(epoch)
        
        # Calculate metric trends
        metric_trends = {
            key: [m[key] for m in self.val_metrics]
            for key in metrics.keys()
        }
        
        return {
            'epoch': epoch,
            'current_metrics': metrics,
            'statistical_tests': significance,
            'detailed_report': report,
            'metric_trends': metric_trends
        }

    def calculate_metrics_from_stored(self) -> Dict[str, float]:
        """Calculate metrics from stored predictions if needed"""
        if not self.predictions or not self.true_labels:
            return self._get_default_metrics()
            
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average='macro',
            zero_division=0,
            labels=range(self.num_classes)
        )
        
        return {
            'accuracy': float((y_pred == y_true).mean()),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'kappa': float(cohen_kappa_score(y_true, y_pred))
        }

    def summarize_cross_validation(self, cv_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Summarize cross validation results with statistical analysis"""
        if not cv_results:
            self.logger.warning("Empty CV results")
            return {}
            
        if not all(isinstance(r, dict) for r in cv_results):
            self.logger.error("Invalid CV results format")
            return {}

        import numpy as np
        from scipy import stats

        # Collect metrics across folds
        metrics_by_fold = {
            metric: [fold[metric] for fold in cv_results]
            for metric in cv_results[0].keys()
            if not metric == 'fold'  # Exclude fold number from statistics
        }
        
        summary = {}
        for metric, values in metrics_by_fold.items():
            values = np.array(values)
            try:
                ci = stats.t.interval(
                    confidence=0.95,  # Explicitly specify confidence
                    df=len(values)-1,
                    loc=np.mean(values),
                    scale=stats.sem(values)
                )
            except (ValueError, ZeroDivisionError):
                ci = (np.nan, np.nan)  # Fallback if CI calculation fails
                
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_95': ci,
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Only add normality tests if we have enough samples and no constant values
        if len(cv_results) >= 5:
            summary['statistical_tests'] = {}
            for metric, values in metrics_by_fold.items():
                if len(np.unique(values)) > 1:  # Only test if we have variation
                    try:
                        summary['statistical_tests'][metric] = {
                            'shapiro': stats.shapiro(values)
                        }
                    except ValueError:
                        # Skip if test fails
                        continue
        
        return summary

    def plot_cv_results(self, cv_results: List[Dict[str, float]], output_dir: Path) -> None:
        """Create visualizations for cross validation results"""
        # Generate the plots but don't save them yet
        cv_plots = {}
        
        try:
            plt.style.use('default')
            sns.set_theme(style="whitegrid")
            colors = sns.color_palette("husl", n_colors=2)
            
            # Generate all CV plots and store them in memory
            for plot_name, plot_func in [
                ('cv_precision.png', self._plot_class_precision),
                ('cv_recall.png', self._plot_class_recall),
                ('cv_overall.png', self._plot_overall_metrics)
            ]:
                # Create figure
                fig = plt.figure(figsize=(12, 6))
                plot_func(cv_results, colors)
                cv_plots[plot_name] = fig
                plt.close()  # Close figure but keep it in memory
            
            # Store the plots to be saved later
            self._cv_plots = cv_plots
            
        except Exception as e:
            self.logger.error(f"Error creating CV plots: {str(e)}")
            self.logger.debug("Error details:", exc_info=True)
    
    def save_plots(self, plots_dir: Path) -> None:
        """Save all collected plots to the specified directory"""
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save any CV plots if they exist
        if hasattr(self, '_cv_plots'):
            for name, fig in self._cv_plots.items():
                fig.savefig(plots_dir / name, dpi=300, bbox_inches='tight')
                plt.close(fig)  # Clean up
            
            # Clear stored plots
            self._cv_plots = {}
            
        self.logger.info(f"All plots saved to: {plots_dir}")

    def _plot_class_precision(self, cv_results: List[Dict[str, float]], colors):
        """Helper method for plotting class precision"""
        precision_data = []
        for i in range(self.num_classes):
            key = f'precision_class_{i}'
            if key in cv_results[0]:
                values = [fold[key] for fold in cv_results]
                precision_data.extend([{'Class': f'Class {i}', 'Value': v} for v in values])
        
        if precision_data:
            df_precision = pd.DataFrame(precision_data)
            sns.boxplot(data=df_precision, x='Class', y='Value', color=colors[0])
            plt.title('Per-Class Precision Across Folds', pad=20)
            plt.ylabel('Precision')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

    def _plot_class_recall(self, cv_results: List[Dict[str, float]], colors):
        """Helper method for plotting class recall"""
        recall_data = []
        for i in range(self.num_classes):
            key = f'recall_class_{i}'
            if key in cv_results[0]:
                values = [fold[key] for fold in cv_results]
                recall_data.extend([{'Class': f'Class {i}', 'Value': v} for v in values])
        
        if recall_data:
            df_recall = pd.DataFrame(recall_data)
            sns.boxplot(data=df_recall, x='Class', y='Value', color=colors[1])
            plt.title('Per-Class Recall Across Folds', pad=20)
            plt.ylabel('Recall')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

    def _plot_overall_metrics(self, cv_results: List[Dict[str, float]], colors):
        """Helper method for plotting overall metrics"""
        overall_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'kappa']
        overall_data = []
        
        for metric in overall_metrics:
            if metric in cv_results[0]:
                values = [fold[metric] for fold in cv_results]
                overall_data.extend([{'Metric': metric.replace('_', ' ').title(), 'Value': v} for v in values])
        
        if overall_data:
            df_overall = pd.DataFrame(overall_data)
            sns.boxplot(data=df_overall, x='Metric', y='Value', hue='Metric', legend=False)
            plt.title('Overall Metrics Across Folds', pad=20)
            plt.xticks(rotation=45)
            plt.ylabel('Value')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
