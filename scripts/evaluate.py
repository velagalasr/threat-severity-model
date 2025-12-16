"""
Model Evaluation Script

Evaluates a trained model on the test set.
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import NSLKDDDataLoader
from feature_engineering import SecurityFeatureEngineer
from model_training import ThreatSeverityModel
from evaluation import ModelEvaluator
from config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model-path', type=str, default=str(MODELS_DIR / 'xgboost_model.pkl'),
                       help='Path to trained model')
    parser.add_argument('--scaler-path', type=str, default=str(MODELS_DIR / 'scaler.pkl'),
                       help='Path to feature scaler')
    parser.add_argument('--output-dir', type=str, default=str(MODELS_DIR / 'evaluation'),
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    logger.info("Loading data...")
    data_loader = NSLKDDDataLoader()
    _, _, _, _, X_test, y_test = data_loader.load_and_preprocess()
    
    logger.info("Loading feature engineer...")
    feature_engineer = SecurityFeatureEngineer.load(args.scaler_path)
    X_test_eng = feature_engineer.transform(X_test)
    
    logger.info(f"Loading model from {args.model_path}...")
    model = ThreatSeverityModel.load(args.model_path)
    
    logger.info("Making predictions...")
    y_pred = model.predict(X_test_eng)
    
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(threshold=5.0)
    results = evaluator.comprehensive_evaluation(
        y_test.values, y_pred,
        output_dir=Path(args.output_dir)
    )
    
    logger.info(f"\n[SUCCESS] Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
