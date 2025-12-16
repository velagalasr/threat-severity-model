"""
Monitoring and Model Drift Detection Module

Tracks model performance, data drift, and operational metrics in production.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
from scipy import stats

from src.config import MONITORING_CONFIG

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Monitors model performance and data drift in production.
    
    Tracks:
    - Model precision/recall degradation
    - Data drift using Kolmogorov-Smirnov test
    - Inference latency (p50, p95, p99)
    - Cost tracking (false positives)
    """
    
    def __init__(self):
        """Initialize model monitor."""
        self.alerts = []
        self.metrics_history = []
        
    def check_performance_degradation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 5.0
    ) -> Dict[str, any]:
        """
        Check for model performance degradation.
        
        Args:
            y_true: True labels from production
            y_pred: Model predictions
            threshold: Classification threshold
            
        Returns:
            Dictionary with performance metrics and alerts
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_samples': len(y_true),
        }
        
        # Check for alerts
        if precision < MONITORING_CONFIG['precision_threshold']:
            alert = {
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'HIGH',
                'message': f"Precision dropped to {precision:.4f} (threshold: {MONITORING_CONFIG['precision_threshold']})",
                'timestamp': datetime.now().isoformat(),
            }
            self.alerts.append(alert)
            logger.warning(f"[ALERT] {alert['message']}")
        
        self.metrics_history.append(metrics)
        
        logger.info(f"Performance Metrics:")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        return metrics
    
    def detect_data_drift(
        self,
        X_reference: pd.DataFrame,
        X_production: pd.DataFrame,
        p_value_threshold: float = None
    ) -> Dict[str, any]:
        """
        Detect data drift using Kolmogorov-Smirnov test.
        
        Args:
            X_reference: Reference dataset (training data)
            X_production: Production dataset
            p_value_threshold: Significance level (default from config)
            
        Returns:
            Dictionary with drift detection results
        """
        if p_value_threshold is None:
            p_value_threshold = MONITORING_CONFIG['drift_p_value']
        
        drift_results = {}
        drifted_features = []
        
        for col in X_reference.columns:
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(
                X_reference[col].dropna(),
                X_production[col].dropna()
            )
            
            drift_detected = p_value < p_value_threshold
            
            drift_results[col] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': drift_detected,
            }
            
            if drift_detected:
                drifted_features.append(col)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(X_reference.columns),
            'drifted_features': len(drifted_features),
            'drift_percentage': 100 * len(drifted_features) / len(X_reference.columns),
            'drifted_feature_names': drifted_features,
            'details': drift_results,
        }
        
        if len(drifted_features) > 0:
            alert = {
                'type': 'DATA_DRIFT',
                'severity': 'MEDIUM' if len(drifted_features) < 5 else 'HIGH',
                'message': f"Data drift detected in {len(drifted_features)} features: {', '.join(drifted_features[:5])}...",
                'timestamp': datetime.now().isoformat(),
            }
            self.alerts.append(alert)
            logger.warning(f"[ALERT] {alert['message']}")
        
        logger.info(f"Data Drift Detection:")
        logger.info(f"  Total Features: {summary['total_features']}")
        logger.info(f"  Drifted Features: {summary['drifted_features']} ({summary['drift_percentage']:.1f}%)")
        
        if drifted_features:
            logger.info(f"  Top Drifted: {', '.join(drifted_features[:5])}")
        
        return summary
    
    def track_latency(
        self,
        latencies_ms: np.ndarray
    ) -> Dict[str, float]:
        """
        Track inference latency metrics.
        
        Args:
            latencies_ms: Array of inference times in milliseconds
            
        Returns:
            Dictionary with latency metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'p50_ms': np.percentile(latencies_ms, 50),
            'p95_ms': np.percentile(latencies_ms, 95),
            'p99_ms': np.percentile(latencies_ms, 99),
            'mean_ms': np.mean(latencies_ms),
            'max_ms': np.max(latencies_ms),
            'n_samples': len(latencies_ms),
        }
        
        # Check for SLO violations
        if metrics['p95_ms'] > MONITORING_CONFIG['latency_p95_ms']:
            alert = {
                'type': 'LATENCY_SLO_VIOLATION',
                'severity': 'MEDIUM',
                'message': f"p95 latency {metrics['p95_ms']:.2f}ms exceeds SLO {MONITORING_CONFIG['latency_p95_ms']}ms",
                'timestamp': datetime.now().isoformat(),
            }
            self.alerts.append(alert)
            logger.warning(f"[ALERT] {alert['message']}")
        
        if metrics['p99_ms'] > MONITORING_CONFIG['latency_p99_ms']:
            alert = {
                'type': 'LATENCY_SLO_VIOLATION',
                'severity': 'MEDIUM',
                'message': f"p99 latency {metrics['p99_ms']:.2f}ms exceeds SLO {MONITORING_CONFIG['latency_p99_ms']}ms",
                'timestamp': datetime.now().isoformat(),
            }
            self.alerts.append(alert)
            logger.warning(f"[ALERT] {alert['message']}")
        
        logger.info(f"Latency Metrics:")
        logger.info(f"  p50: {metrics['p50_ms']:.2f}ms")
        logger.info(f"  p95: {metrics['p95_ms']:.2f}ms")
        logger.info(f"  p99: {metrics['p99_ms']:.2f}ms")
        logger.info(f"  mean: {metrics['mean_ms']:.2f}ms")
        
        return metrics
    
    def calculate_cost_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 5.0,
        fp_cost: float = 50.0,
        fn_cost: float = 5000000.0
    ) -> Dict[str, float]:
        """
        Calculate operational cost metrics.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            threshold: Classification threshold
            fp_cost: Cost per false positive
            fn_cost: Cost per false negative
            
        Returns:
            Dictionary with cost metrics
        """
        from sklearn.metrics import confusion_matrix
        
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        total_fp_cost = fp * fp_cost
        total_fn_cost = fn * fn_cost
        total_cost = total_fp_cost + total_fn_cost
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'fp_cost': total_fp_cost,
            'fn_cost': total_fn_cost,
            'total_cost': total_cost,
            'daily_fp_cost': total_fp_cost,  # Assuming daily batch
            'projected_annual_cost': total_cost * 365,
        }
        
        logger.info(f"Cost Metrics:")
        logger.info(f"  False Positives: {fp} × ${fp_cost} = ${total_fp_cost:,.2f}/day")
        logger.info(f"  False Negatives: {fn} × ${fn_cost:,.0f} = ${total_fn_cost:,.2f}/day")
        logger.info(f"  Total Daily Cost: ${total_cost:,.2f}")
        logger.info(f"  Projected Annual Cost: ${metrics['projected_annual_cost']:,.2f}")
        
        return metrics
    
    def generate_daily_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X_reference: Optional[pd.DataFrame] = None,
        X_production: Optional[pd.DataFrame] = None,
        latencies_ms: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive daily monitoring report.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            X_reference: Reference features (for drift detection)
            X_production: Production features (for drift detection)
            latencies_ms: Inference latencies
            output_path: Path to save report (optional)
            
        Returns:
            Dictionary with all monitoring metrics
        """
        logger.info("\n" + "="*60)
        logger.info("DAILY MONITORING REPORT")
        logger.info("="*60)
        logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_predictions': len(y_true),
        }
        
        # Performance metrics
        logger.info("\n--- Performance Metrics ---")
        report['performance'] = self.check_performance_degradation(y_true, y_pred)
        
        # Data drift
        if X_reference is not None and X_production is not None:
            logger.info("\n--- Data Drift Detection ---")
            report['drift'] = self.detect_data_drift(X_reference, X_production)
        
        # Latency metrics
        if latencies_ms is not None:
            logger.info("\n--- Latency Metrics ---")
            report['latency'] = self.track_latency(latencies_ms)
        
        # Cost metrics
        logger.info("\n--- Cost Metrics ---")
        report['cost'] = self.calculate_cost_metrics(y_true, y_pred)
        
        # Alerts summary
        logger.info("\n--- Alerts Summary ---")
        report['alerts'] = self.alerts
        logger.info(f"  Total Alerts: {len(self.alerts)}")
        
        for alert in self.alerts:
            logger.info(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")
        
        # Save report
        if output_path:
            import json
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"\n[SUCCESS] Report saved to {output_path}")
        
        return report
    
    def get_alerts(self) -> list:
        """Get all alerts."""
        return self.alerts
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []


if __name__ == "__main__":
    # Test monitoring
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.concatenate([
        np.zeros(700),
        np.random.uniform(7, 10, 300)
    ])
    
    y_pred = y_true + np.random.normal(0, 1.0, n_samples)
    y_pred = np.clip(y_pred, 0, 10)
    
    # Sample features
    X_reference = pd.DataFrame(np.random.randn(n_samples, 10))
    X_production = pd.DataFrame(np.random.randn(n_samples, 10) + 0.1)  # Slight drift
    
    # Sample latencies
    latencies_ms = np.random.exponential(15, n_samples)
    
    # Run monitoring
    monitor = ModelMonitor()
    report = monitor.generate_daily_report(
        y_true, y_pred,
        X_reference, X_production,
        latencies_ms
    )
    
    print("\n=== Monitoring Test Complete ===")
    print(f"Alerts generated: {len(report['alerts'])}")
