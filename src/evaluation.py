"""
Model Evaluation Module

Comprehensive evaluation with precision, recall, F1, AUC-ROC, confusion matrices, and cost analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score
)

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for threat severity prediction.
    
    Provides regression and classification metrics, visualizations,
    and cost-based analysis.
    """
    
    def __init__(self, threshold: float = 5.0):
        """
        Initialize evaluator.
        
        Args:
            threshold: Severity threshold for binary classification (attack vs normal)
        """
        self.threshold = threshold
        self.metrics = {}
        
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            
        Returns:
            Dictionary of regression metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
        }
        
        logger.info("Regression Metrics:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE:  {metrics['mae']:.4f}")
        logger.info(f"  R²:   {metrics['r2']:.4f}")
        
        return metrics
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate as binary classification (attack vs normal).
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            threshold: Classification threshold (default: self.threshold)
            
        Returns:
            Dictionary of classification metrics
        """
        if threshold is None:
            threshold = self.threshold
        
        # Convert to binary (0 = normal, 1 = attack)
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'fpr': fpr,  # False positive rate
            'fnr': fnr,  # False negative rate
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
        }
        
        logger.info(f"\nClassification Metrics (threshold={threshold:.2f}):")
        logger.info(f"  Precision:   {precision:.4f}")
        logger.info(f"  Recall:      {recall:.4f}")
        logger.info(f"  F1 Score:    {f1:.4f}")
        logger.info(f"  Specificity: {specificity:.4f}")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"  FPR:         {fpr:.4f}")
        logger.info(f"  FNR:         {fnr:.4f}")
        
        return metrics
    
    def calculate_cost_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fp_cost: float = 50.0,
        fn_cost: float = 5000000.0,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate cost-based metrics.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            fp_cost: Cost per false positive ($50 per investigation)
            fn_cost: Cost per false negative ($5M per missed breach)
            threshold: Classification threshold
            
        Returns:
            Dictionary of cost metrics
        """
        if threshold is None:
            threshold = self.threshold
        
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        total_fp_cost = fp * fp_cost
        total_fn_cost = fn * fn_cost
        total_cost = total_fp_cost + total_fn_cost
        
        metrics = {
            'fp_cost': total_fp_cost,
            'fn_cost': total_fn_cost,
            'total_cost': total_cost,
            'cost_per_prediction': total_cost / len(y_true),
        }
        
        logger.info(f"\nCost Analysis (FP=${fp_cost}, FN=${fn_cost:,.0f}):")
        logger.info(f"  False Positives: {fp} × ${fp_cost} = ${total_fp_cost:,.2f}")
        logger.info(f"  False Negatives: {fn} × ${fn_cost:,.0f} = ${total_fn_cost:,.2f}")
        logger.info(f"  Total Cost: ${total_cost:,.2f}")
        logger.info(f"  Cost per Prediction: ${metrics['cost_per_prediction']:.2f}")
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: Optional[float] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            threshold: Classification threshold
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if threshold is None:
            threshold = self.threshold
        
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, float]:
        """
        Plot ROC curve and calculate AUC.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            save_path: Path to save plot (optional)
            
        Returns:
            Tuple of (figure, auc_score)
        """
        y_true_binary = (y_true > 0).astype(int)
        
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        logger.info(f"AUC-ROC: {roc_auc:.4f}")
        
        return fig, roc_auc
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        y_true_binary = (y_true > 0).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        return fig
    
    def comprehensive_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Perform comprehensive evaluation with all metrics and plots.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            output_dir: Directory to save plots (optional)
            
        Returns:
            Dictionary containing all metrics and figures
        """
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE MODEL EVALUATION")
        logger.info("="*60)
        
        results = {}
        
        # Regression metrics
        results['regression_metrics'] = self.evaluate_regression(y_true, y_pred)
        
        # Classification metrics
        results['classification_metrics'] = self.evaluate_classification(y_true, y_pred)
        
        # Cost metrics
        results['cost_metrics'] = self.calculate_cost_metrics(y_true, y_pred)
        
        # Plots
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results['confusion_matrix_fig'] = self.plot_confusion_matrix(
                y_true, y_pred, save_path=output_dir / 'confusion_matrix.png'
            )
            
            results['roc_fig'], results['auc_roc'] = self.plot_roc_curve(
                y_true, y_pred, save_path=output_dir / 'roc_curve.png'
            )
            
            results['pr_fig'] = self.plot_precision_recall_curve(
                y_true, y_pred, save_path=output_dir / 'precision_recall_curve.png'
            )
        else:
            results['confusion_matrix_fig'] = self.plot_confusion_matrix(y_true, y_pred)
            results['roc_fig'], results['auc_roc'] = self.plot_roc_curve(y_true, y_pred)
            results['pr_fig'] = self.plot_precision_recall_curve(y_true, y_pred)
        
        # Store all metrics
        self.metrics = results
        
        return results


if __name__ == "__main__":
    # Test evaluation
    logging.basicConfig(level=logging.INFO)
    
    # Create sample predictions
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate realistic predictions
    y_true = np.concatenate([
        np.zeros(700),  # Normal traffic
        np.random.uniform(7, 10, 300)  # Attacks
    ])
    
    # Model predictions with some errors
    y_pred = y_true + np.random.normal(0, 1.0, n_samples)
    y_pred = np.clip(y_pred, 0, 10)
    
    evaluator = ModelEvaluator(threshold=5.0)
    results = evaluator.comprehensive_evaluation(y_true, y_pred)
    
    print("\n=== Evaluation Test Complete ===")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
