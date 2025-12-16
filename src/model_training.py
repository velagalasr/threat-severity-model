"""
Model Training Module

Trains multiple models for threat severity prediction with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
from pathlib import Path
import time

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import MODEL_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)


class ThreatSeverityModel:
    """
    Threat severity prediction model with multiple algorithm support.
    
    Supported models:
    - XGBoost (primary)
    - LightGBM
    - Random Forest
    - Linear Regression (baseline)
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize model.
        
        Args:
            model_type: One of ['xgboost', 'lightgbm', 'random_forest', 'linear']
        """
        self.model_type = model_type
        self.model = None
        self.training_history = {}
        self.feature_importance_ = None
        
    def _get_base_model(self) -> Any:
        """Get base model instance based on model type."""
        if self.model_type == 'xgboost':
            config = MODEL_CONFIG['xgboost'].copy()
            early_stopping = config.pop('early_stopping_rounds', 10)
            return xgb.XGBRegressor(**config)
            
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(**MODEL_CONFIG['lightgbm'])
            
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(**MODEL_CONFIG['random_forest'])
            
        elif self.model_type == 'linear':
            return LinearRegression()
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        tune_hyperparameters: bool = False
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional)
            tune_hyperparameters: If True, perform grid search for hyperparameters
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        start_time = time.time()
        
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            self.model = self._get_base_model()
            
            # Train with early stopping if validation data provided
            if X_val is not None and self.model_type in ['xgboost', 'lightgbm']:
                eval_set = [(X_val, y_val)]
                
                if self.model_type == 'xgboost':
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=False
                    )
                elif self.model_type == 'lightgbm':
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                    )
            else:
                self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Calculate training metrics
        y_train_pred = self.predict(X_train)
        train_metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'training_time': training_time,
        }
        
        # Calculate validation metrics if provided
        if X_val is not None:
            y_val_pred = self.predict(X_val)
            val_metrics = {
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_r2': r2_score(y_val, y_val_pred),
            }
            train_metrics.update(val_metrics)
        
        self.training_history = train_metrics
        
        # Extract feature importance
        self._extract_feature_importance(X_train.columns)
        
        logger.info(f"Training complete in {training_time:.2f}s")
        logger.info(f"  Train RMSE: {train_metrics['train_rmse']:.4f}")
        logger.info(f"  Train MAE: {train_metrics['train_mae']:.4f}")
        logger.info(f"  Train R²: {train_metrics['train_r2']:.4f}")
        
        if 'val_rmse' in train_metrics:
            logger.info(f"  Val RMSE: {train_metrics['val_rmse']:.4f}")
            logger.info(f"  Val MAE: {train_metrics['val_mae']:.4f}")
            logger.info(f"  Val R²: {train_metrics['val_r2']:.4f}")
        
        return train_metrics
    
    def _tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best model from grid search
        """
        param_grids = {
            'xgboost': {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
            },
            'lightgbm': {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
            },
            'random_forest': {
                'max_depth': [6, 10, 15],
                'n_estimators': [50, 100, 200],
            },
        }
        
        if self.model_type not in param_grids:
            logger.warning(f"No param grid for {self.model_type}, using default params")
            return self._get_base_model()
        
        base_model = self._get_base_model()
        grid_search = GridSearchCV(
            base_model,
            param_grids[self.model_type],
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _extract_feature_importance(self, feature_names: list) -> None:
        """Extract and store feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Most Important Features:")
            for idx, row in self.feature_importance_.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions (threat severity scores 0-10)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        
        # Clip predictions to valid range [0, 10]
        predictions = np.clip(predictions, 0, 10)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance DataFrame."""
        return self.feature_importance_
    
    def save(self, filepath: Path) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'ThreatSeverityModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded model instance
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


class ModelTrainer:
    """
    Manages training of multiple model types.
    """
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.metrics = {}
        
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_types: list = ['xgboost', 'lightgbm', 'random_forest', 'linear']
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all specified model types.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_types: List of model types to train
            
        Returns:
            Dictionary mapping model type to metrics
        """
        logger.info(f"Training {len(model_types)} models...")
        
        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type.upper()} model")
            logger.info(f"{'='*60}")
            
            model = ThreatSeverityModel(model_type=model_type)
            metrics = model.train(X_train, y_train, X_val, y_val)
            
            self.models[model_type] = model
            self.metrics[model_type] = metrics
        
        logger.info("\n" + "="*60)
        logger.info("Training Summary")
        logger.info("="*60)
        self._print_comparison()
        
        return self.metrics
    
    def _print_comparison(self) -> None:
        """Print comparison table of all models."""
        if not self.metrics:
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_type, metrics in self.metrics.items():
            comparison_data.append({
                'Model': model_type,
                'Train RMSE': metrics.get('train_rmse', np.nan),
                'Val RMSE': metrics.get('val_rmse', np.nan),
                'Train MAE': metrics.get('train_mae', np.nan),
                'Val MAE': metrics.get('val_mae', np.nan),
                'Train R²': metrics.get('train_r2', np.nan),
                'Val R²': metrics.get('val_r2', np.nan),
                'Time (s)': metrics.get('training_time', np.nan),
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Val RMSE')
        
        logger.info("\n" + df_comparison.to_string(index=False))
        
        # Find best model
        best_model = df_comparison.iloc[0]['Model']
        logger.info(f"\n🏆 Best model (by Val RMSE): {best_model}")
    
    def save_all_models(self, output_dir: Path = MODELS_DIR) -> None:
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_type, model in self.models.items():
            filepath = output_dir / f"{model_type}_model.pkl"
            model.save(filepath)
        
        logger.info(f"Saved {len(self.models)} models to {output_dir}")
    
    def get_best_model(self, metric: str = 'val_rmse') -> Tuple[str, ThreatSeverityModel]:
        """
        Get the best model by specified metric.
        
        Args:
            metric: Metric to use for selection (lower is better for RMSE/MAE)
            
        Returns:
            Tuple of (model_type, model)
        """
        if not self.metrics:
            raise ValueError("No models trained yet")
        
        # Find best model
        best_type = min(
            self.metrics.keys(),
            key=lambda k: self.metrics[k].get(metric, float('inf'))
        )
        
        return best_type, self.models[best_type]


if __name__ == "__main__":
    # Test model training
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 49
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.uniform(0, 10, n_samples))
    
    X_val = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(np.random.uniform(0, 10, 200))
    
    # Train single model
    model = ThreatSeverityModel(model_type='xgboost')
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    print("\n=== Model Training Test Complete ===")
    print(f"Metrics: {metrics}")
