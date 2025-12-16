"""
SHAP Explainability Module

Implements SHAP TreeExplainer for model interpretability.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import joblib

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainability for threat severity predictions.
    
    Provides:
    - Feature importance via SHAP values
    - Individual prediction explanations
    - Waterfall plots
    - Summary plots
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (XGBoost, LightGBM, or Random Forest)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def fit(self, X_background: pd.DataFrame) -> None:
        """
        Fit SHAP explainer on background data.
        
        Args:
            X_background: Background dataset for SHAP (typically training data sample)
        """
        logger.info("Fitting SHAP TreeExplainer...")
        
        # Use TreeExplainer for tree-based models
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer fitted successfully")
        except Exception as e:
            logger.warning(f"TreeExplainer failed, using KernelExplainer: {e}")
            # Fallback to KernelExplainer for non-tree models
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                X_background.sample(min(100, len(X_background)), random_state=42)
            )
    
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for dataset.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Must call fit() before calculating SHAP values")
        
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)
        
        return self.shap_values
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 20,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot SHAP summary showing feature importance.
        
        Args:
            X: Features
            shap_values: Precomputed SHAP values (optional)
            max_display: Maximum features to display
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if shap_values is None:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            shap_values = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        fig = plt.gcf()
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        return fig
    
    def plot_bar(
        self,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        max_display: int = 20,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot SHAP bar chart of mean absolute feature importance.
        
        Args:
            X: Features
            shap_values: Precomputed SHAP values (optional)
            max_display: Maximum features to display
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if shap_values is None:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            shap_values = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        
        fig = plt.gcf()
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP bar plot saved to {save_path}")
        
        return fig
    
    def plot_waterfall(
        self,
        X: pd.DataFrame,
        index: int = 0,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot SHAP waterfall for single prediction.
        
        Args:
            X: Features
            index: Sample index to explain
            shap_values: Precomputed SHAP values (optional)
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if shap_values is None:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            shap_values = self.shap_values
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[index],
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=X.iloc[index].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        
        fig = plt.gcf()
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP waterfall plot saved to {save_path}")
        
        return fig
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        index: int = 0,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Get top contributing features for a single prediction.
        
        Args:
            X: Features
            index: Sample index to explain
            top_k: Number of top features to return
            
        Returns:
            DataFrame with top contributors
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Get SHAP values for this sample
        sample_shap = self.shap_values[index]
        
        # Create DataFrame of features and their contributions
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X.iloc[index].values,
            'shap_value': sample_shap,
            'abs_shap_value': np.abs(sample_shap)
        })
        
        # Sort by absolute SHAP value
        contributions = contributions.sort_values('abs_shap_value', ascending=False)
        
        return contributions.head(top_k)
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values.
        
        Args:
            X: Features
            shap_values: Precomputed SHAP values (optional)
            
        Returns:
            DataFrame with feature importance rankings
        """
        if shap_values is None:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            shap_values = self.shap_values
        
        # Calculate mean absolute SHAP values
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        })
        
        importance = importance.sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Features by SHAP Importance:")
        for idx, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance
    
    def save(self, filepath: Path) -> None:
        """Save explainer to disk."""
        joblib.dump(self, filepath)
        logger.info(f"SHAP explainer saved to {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'SHAPExplainer':
        """Load explainer from disk."""
        explainer = joblib.load(filepath)
        logger.info(f"SHAP explainer loaded from {filepath}")
        return explainer


if __name__ == "__main__":
    # Test SHAP explainer
    logging.basicConfig(level=logging.INFO)
    
    # Create sample model and data
    from sklearn.ensemble import RandomForestRegressor
    
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = np.random.uniform(0, 10, n_samples)
    
    # Train simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = SHAPExplainer(model, X.columns.tolist())
    explainer.fit(X)
    
    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(X)
    
    # Get feature importance
    importance = explainer.get_feature_importance(X)
    
    print("\n=== SHAP Explainer Test Complete ===")
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Top feature: {importance.iloc[0]['feature']}")
