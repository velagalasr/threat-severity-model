"""
End-to-End Training Script

Downloads data, engineers features, trains models, and saves artifacts.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import NSLKDDDataLoader
from feature_engineering import SecurityFeatureEngineer
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from explainability import SHAPExplainer
from threshold_optimization import ThresholdOptimizer
from config import MODELS_DIR, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run end-to-end training pipeline."""
    logger.info("="*60)
    logger.info("THREAT SEVERITY MODEL - TRAINING PIPELINE")
    logger.info("="*60)
    
    # Step 1: Load and preprocess data
    logger.info("\n[1/7] Loading and preprocessing data...")
    data_loader = NSLKDDDataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_and_preprocess()
    
    # Step 2: Engineer features
    logger.info("\n[2/7] Engineering security features...")
    feature_engineer = SecurityFeatureEngineer()
    X_train_eng = feature_engineer.fit_transform(X_train)
    X_val_eng = feature_engineer.transform(X_val)
    X_test_eng = feature_engineer.transform(X_test)
    
    # Save feature engineer
    feature_engineer.save(MODELS_DIR / 'scaler.pkl')
    
    # Step 3: Train models
    logger.info("\n[3/7] Training models...")
    trainer = ModelTrainer()
    metrics = trainer.train_all_models(
        X_train_eng, y_train,
        X_val_eng, y_val,
        model_types=['xgboost', 'lightgbm', 'random_forest', 'linear']
    )
    
    # Save all models
    trainer.save_all_models(MODELS_DIR)
    
    # Step 4: Evaluate on test set
    logger.info("\n[4/7] Evaluating on test set...")
    best_model_type, best_model = trainer.get_best_model(metric='val_rmse')
    
    y_test_pred = best_model.predict(X_test_eng)
    
    evaluator = ModelEvaluator(threshold=5.0)
    eval_results = evaluator.comprehensive_evaluation(
        y_test.values, y_test_pred,
        output_dir=MODELS_DIR / 'evaluation_plots'
    )
    
    logger.info(f"\nTest Set Results ({best_model_type}):")
    logger.info(f"  RMSE: {eval_results['regression_metrics']['rmse']:.4f}")
    logger.info(f"  Precision: {eval_results['classification_metrics']['precision']:.4f}")
    logger.info(f"  Recall: {eval_results['classification_metrics']['recall']:.4f}")
    logger.info(f"  F1 Score: {eval_results['classification_metrics']['f1_score']:.4f}")
    logger.info(f"  AUC-ROC: {eval_results['auc_roc']:.4f}")
    
    # Step 5: SHAP explainability
    logger.info("\n[5/7] Creating SHAP explainer...")
    shap_explainer = SHAPExplainer(
        best_model.model,
        X_train_eng.columns.tolist()
    )
    shap_explainer.fit(X_train_eng.sample(min(1000, len(X_train_eng)), random_state=42))
    
    # Calculate SHAP values for test set sample
    test_sample = X_test_eng.sample(min(500, len(X_test_eng)), random_state=42)
    shap_values = shap_explainer.calculate_shap_values(test_sample)
    
    # Save plots
    plots_dir = MODELS_DIR / 'shap_plots'
    plots_dir.mkdir(exist_ok=True)
    shap_explainer.plot_bar(test_sample, shap_values, save_path=plots_dir / 'feature_importance.png')
    shap_explainer.plot_summary(test_sample, shap_values, save_path=plots_dir / 'summary.png')
    
    # Save explainer
    shap_explainer.save(MODELS_DIR / 'shap_explainer.pkl')
    
    # Step 6: Threshold optimization
    logger.info("\n[6/7] Optimizing thresholds for different SLOs...")
    optimizer = ThresholdOptimizer()
    threshold_results = optimizer.optimize_all_slos(y_test.values, y_test_pred)
    
    # Save threshold curves
    optimizer.plot_threshold_curves(
        y_test.values, y_test_pred,
        save_path=MODELS_DIR / 'threshold_curves.png'
    )
    
    # Step 7: Save summary report
    logger.info("\n[7/7] Saving training summary...")
    
    import json
    summary = {
        'best_model': best_model_type,
        'test_metrics': {
            'rmse': eval_results['regression_metrics']['rmse'],
            'mae': eval_results['regression_metrics']['mae'],
            'r2': eval_results['regression_metrics']['r2'],
            'precision': eval_results['classification_metrics']['precision'],
            'recall': eval_results['classification_metrics']['recall'],
            'f1_score': eval_results['classification_metrics']['f1_score'],
            'auc_roc': eval_results['auc_roc'],
        },
        'optimal_thresholds': {
            name: result['threshold']
            for name, result in threshold_results.items()
        },
        'feature_count': X_train_eng.shape[1],
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
    }
    
    with open(MODELS_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"\nModels saved to: {MODELS_DIR}")
    logger.info(f"Best model: {best_model_type}")
    logger.info(f"Test F1 Score: {eval_results['classification_metrics']['f1_score']:.4f}")
    logger.info(f"Test AUC-ROC: {eval_results['auc_roc']:.4f}")
    logger.info("\n[SUCCESS] Training complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Review evaluation plots in models/evaluation_plots/")
    logger.info("  2. Review SHAP plots in models/shap_plots/")
    logger.info("  3. Start API server: python scripts/serve.py")
    logger.info("  4. Run notebooks for detailed analysis")


if __name__ == "__main__":
    main()
