"""
Threshold Optimization Module

Optimizes decision thresholds for different SLOs (SOC teams, IR teams, executives).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

from src.config import THRESHOLD_SLOS

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes classification thresholds for different operational requirements.
    
    Supports optimization for:
    - High recall (SOC Tier 1)
    - Balanced F1 (SOC Tier 2)
    - High precision (Incident Response)
    - Cost-optimal (Executive/Business)
    """
    
    def __init__(self):
        """Initialize threshold optimizer."""
        self.optimal_thresholds = {}
        
    def find_threshold_for_recall(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_recall: float = 0.95
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find threshold that achieves target recall.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            target_recall: Target recall value (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (threshold, metrics)
        """
        y_true_binary = (y_true > 0).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred)
        
        # Find threshold closest to target recall
        idx = np.argmin(np.abs(recall - target_recall))
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        
        # Calculate metrics at this threshold
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        metrics = {
            'threshold': float(optimal_threshold),
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1_score': f1_score(y_true_binary, y_pred_binary),
        }
        
        logger.info(f"Threshold for {target_recall:.1%} recall:")
        logger.info(f"  Threshold: {optimal_threshold:.4f}")
        logger.info(f"  Achieved Recall: {metrics['recall']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  F1: {metrics['f1_score']:.4f}")
        
        return optimal_threshold, metrics
    
    def find_threshold_for_precision(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_precision: float = 0.98
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find threshold that achieves target precision.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            target_precision: Target precision value (e.g., 0.98 for 98%)
            
        Returns:
            Tuple of (threshold, metrics)
        """
        y_true_binary = (y_true > 0).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred)
        
        # Find threshold closest to target precision
        idx = np.argmin(np.abs(precision - target_precision))
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        
        # Calculate metrics at this threshold
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        metrics = {
            'threshold': float(optimal_threshold),
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1_score': f1_score(y_true_binary, y_pred_binary),
        }
        
        logger.info(f"Threshold for {target_precision:.1%} precision:")
        logger.info(f"  Threshold: {optimal_threshold:.4f}")
        logger.info(f"  Achieved Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1_score']:.4f}")
        
        return optimal_threshold, metrics
    
    def find_threshold_for_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find threshold that maximizes F1 score.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            
        Returns:
            Tuple of (threshold, metrics)
        """
        y_true_binary = (y_true > 0).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred)
        
        # Calculate F1 for all thresholds
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find threshold with max F1
        idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        
        # Calculate metrics at this threshold
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        metrics = {
            'threshold': float(optimal_threshold),
            'precision': precision_score(y_true_binary, y_pred_binary),
            'recall': recall_score(y_true_binary, y_pred_binary),
            'f1_score': f1_score(y_true_binary, y_pred_binary),
        }
        
        logger.info(f"Threshold for maximum F1:")
        logger.info(f"  Threshold: {optimal_threshold:.4f}")
        logger.info(f"  F1: {metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        
        return optimal_threshold, metrics
    
    def find_cost_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fp_cost: float = 50.0,
        fn_cost: float = 5000000.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find threshold that minimizes total cost.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            fp_cost: Cost per false positive
            fn_cost: Cost per false negative
            
        Returns:
            Tuple of (threshold, metrics)
        """
        y_true_binary = (y_true > 0).astype(int)
        
        # Test range of thresholds
        thresholds_to_test = np.linspace(y_pred.min(), y_pred.max(), 100)
        
        best_threshold = None
        best_cost = float('inf')
        best_metrics = None
        
        for threshold in thresholds_to_test:
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate confusion matrix
            tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
            fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
            tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
            fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
            
            # Calculate total cost
            total_cost = fp * fp_cost + fn * fn_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
                best_metrics = {
                    'threshold': float(threshold),
                    'fp_cost': fp * fp_cost,
                    'fn_cost': fn * fn_cost,
                    'total_cost': total_cost,
                    'fp': int(fp),
                    'fn': int(fn),
                    'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                    'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                }
        
        logger.info(f"Cost-optimal threshold (FP=${fp_cost}, FN=${fn_cost:,.0f}):")
        logger.info(f"  Threshold: {best_threshold:.4f}")
        logger.info(f"  Total Cost: ${best_cost:,.2f}")
        logger.info(f"  FP Cost: ${best_metrics['fp_cost']:,.2f} ({best_metrics['fp']} FPs)")
        logger.info(f"  FN Cost: ${best_metrics['fn_cost']:,.2f} ({best_metrics['fn']} FNs)")
        logger.info(f"  Precision: {best_metrics['precision']:.4f}")
        logger.info(f"  Recall: {best_metrics['recall']:.4f}")
        
        return best_threshold, best_metrics
    
    def optimize_all_slos(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize thresholds for all SLO types.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            
        Returns:
            Dictionary mapping SLO name to threshold and metrics
        """
        logger.info("\n" + "="*60)
        logger.info("THRESHOLD OPTIMIZATION FOR ALL SLOS")
        logger.info("="*60)
        
        results = {}
        
        # SOC Tier 1: High Recall (95%)
        logger.info("\n--- SOC Tier 1 (High Recall) ---")
        threshold, metrics = self.find_threshold_for_recall(
            y_true, y_pred,
            target_recall=THRESHOLD_SLOS['tier1_recall']['target_value']
        )
        results['tier1_recall'] = metrics
        self.optimal_thresholds['tier1_recall'] = threshold
        
        # SOC Tier 2: Balanced F1
        logger.info("\n--- SOC Tier 2 (Balanced F1) ---")
        threshold, metrics = self.find_threshold_for_f1(y_true, y_pred)
        results['tier2_balanced'] = metrics
        self.optimal_thresholds['tier2_balanced'] = threshold
        
        # Incident Response: High Precision (98%)
        logger.info("\n--- Incident Response (High Precision) ---")
        threshold, metrics = self.find_threshold_for_precision(
            y_true, y_pred,
            target_precision=THRESHOLD_SLOS['ir_precision']['target_value']
        )
        results['ir_precision'] = metrics
        self.optimal_thresholds['ir_precision'] = threshold
        
        # Executive: Cost-Optimal
        logger.info("\n--- Executive (Cost-Optimal) ---")
        threshold, metrics = self.find_cost_optimal_threshold(
            y_true, y_pred,
            fp_cost=THRESHOLD_SLOS['cost_optimal']['fp_cost'],
            fn_cost=THRESHOLD_SLOS['cost_optimal']['fn_cost']
        )
        results['cost_optimal'] = metrics
        self.optimal_thresholds['cost_optimal'] = threshold
        
        # Print summary table
        self._print_summary_table(results)
        
        return results
    
    def _print_summary_table(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print summary table of all SLO thresholds."""
        logger.info("\n" + "="*60)
        logger.info("THRESHOLD OPTIMIZATION SUMMARY")
        logger.info("="*60)
        
        summary_data = []
        for slo_name, metrics in results.items():
            summary_data.append({
                'SLO': slo_name,
                'Threshold': metrics['threshold'],
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1_score', 0),
            })
        
        df_summary = pd.DataFrame(summary_data)
        logger.info("\n" + df_summary.to_string(index=False))
    
    def plot_threshold_curves(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot precision, recall, and F1 vs threshold.
        
        Args:
            y_true: True severity scores
            y_pred: Predicted severity scores
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        y_true_binary = (y_true > 0).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred)
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Align lengths (precision/recall have one more element than thresholds)
        precision = precision[:-1]
        recall = recall[:-1]
        f1_scores = f1_scores[:-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, precision, label='Precision', linewidth=2)
        ax.plot(thresholds, recall, label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
        
        # Mark optimal thresholds
        if hasattr(self, 'optimal_thresholds'):
            for slo_name, threshold in self.optimal_thresholds.items():
                ax.axvline(threshold, alpha=0.3, linestyle=':', label=f'{slo_name}')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, and F1 vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold curves saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Test threshold optimization
    logging.basicConfig(level=logging.INFO)
    
    # Create sample predictions
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.concatenate([
        np.zeros(700),
        np.random.uniform(7, 10, 300)
    ])
    
    y_pred = y_true + np.random.normal(0, 1.0, n_samples)
    y_pred = np.clip(y_pred, 0, 10)
    
    optimizer = ThresholdOptimizer()
    results = optimizer.optimize_all_slos(y_true, y_pred)
    
    print("\n=== Threshold Optimization Test Complete ===")
    print(f"Optimal thresholds: {optimizer.optimal_thresholds}")
