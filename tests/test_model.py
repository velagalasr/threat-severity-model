"""
Unit Tests for Model Training Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_training import ThreatSeverityModel, ModelTrainer


class TestThreatSeverityModel:
    """Test suite for ThreatSeverityModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        
        self.X_train = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.y_train = pd.Series(np.random.uniform(0, 10, self.n_samples))
        
        self.X_val = pd.DataFrame(
            np.random.randn(20, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.y_val = pd.Series(np.random.uniform(0, 10, 20))
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = ThreatSeverityModel(model_type='xgboost')
        
        assert model.model_type == 'xgboost'
        assert model.model is None
        assert model.training_history == {}
    
    def test_model_training(self):
        """Test model training."""
        model = ThreatSeverityModel(model_type='xgboost')
        metrics = model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        assert model.model is not None
        assert 'train_rmse' in metrics
        assert 'val_rmse' in metrics
        assert metrics['train_rmse'] >= 0
    
    def test_model_prediction(self):
        """Test model predictions."""
        model = ThreatSeverityModel(model_type='linear')
        model.train(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_val)
        
        assert len(predictions) == len(self.X_val)
        assert np.all((predictions >= 0) & (predictions <= 10))
    
    def test_model_save_load(self):
        """Test model save and load."""
        model = ThreatSeverityModel(model_type='linear')
        model.train(self.X_train, self.y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            model.save(save_path)
            
            loaded_model = ThreatSeverityModel.load(save_path)
            
            assert loaded_model.model_type == model.model_type
            
            # Check predictions match
            pred_original = model.predict(self.X_val)
            pred_loaded = loaded_model.predict(self.X_val)
            
            np.testing.assert_array_almost_equal(pred_original, pred_loaded)


class TestModelTrainer:
    """Test suite for ModelTrainer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        
        self.X_train = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.y_train = pd.Series(np.random.uniform(0, 10, self.n_samples))
        
        self.X_val = pd.DataFrame(
            np.random.randn(20, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.y_val = pd.Series(np.random.uniform(0, 10, 20))
    
    def test_train_multiple_models(self):
        """Test training multiple models."""
        trainer = ModelTrainer()
        metrics = trainer.train_all_models(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            model_types=['linear', 'random_forest']
        )
        
        assert len(metrics) == 2
        assert 'linear' in metrics
        assert 'random_forest' in metrics
    
    def test_get_best_model(self):
        """Test getting best model."""
        trainer = ModelTrainer()
        trainer.train_all_models(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            model_types=['linear', 'random_forest']
        )
        
        best_type, best_model = trainer.get_best_model()
        
        assert best_type in ['linear', 'random_forest']
        assert best_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
