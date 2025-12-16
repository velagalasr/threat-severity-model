"""
Production Monitoring Script

Runs monitoring checks on production predictions.
"""

import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from monitoring import ModelMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run production monitoring')
    parser.add_argument('--predictions-path', type=str, required=True,
                       help='Path to CSV with production predictions (columns: y_true, y_pred)')
    parser.add_argument('--reference-path', type=str, default=None,
                       help='Path to reference feature data (for drift detection)')
    parser.add_argument('--production-path', type=str, default=None,
                       help='Path to production feature data (for drift detection)')
    parser.add_argument('--output-path', type=str, default='monitoring_report.json',
                       help='Path to save monitoring report')
    
    args = parser.parse_args()
    
    # Load predictions
    logger.info(f"Loading predictions from {args.predictions_path}...")
    df_pred = pd.read_csv(args.predictions_path)
    
    y_true = df_pred['y_true'].values
    y_pred = df_pred['y_pred'].values
    
    # Load features if provided
    X_reference = None
    X_production = None
    if args.reference_path and args.production_path:
        logger.info("Loading feature data for drift detection...")
        X_reference = pd.read_csv(args.reference_path)
        X_production = pd.read_csv(args.production_path)
    
    # Extract latencies if available
    latencies_ms = None
    if 'latency_ms' in df_pred.columns:
        latencies_ms = df_pred['latency_ms'].values
    
    # Run monitoring
    monitor = ModelMonitor()
    report = monitor.generate_daily_report(
        y_true, y_pred,
        X_reference, X_production,
        latencies_ms,
        output_path=Path(args.output_path)
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("MONITORING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Alerts: {len(report['alerts'])}")
    
    for alert in report['alerts']:
        logger.info(f"  [{alert['severity']}] {alert['type']}")
    
    logger.info(f"\n[SUCCESS] Monitoring complete! Report saved to {args.output_path}")


if __name__ == "__main__":
    main()
