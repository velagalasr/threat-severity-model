"""
Data Loading and Preprocessing Module

Handles loading NSL-KDD dataset, preprocessing, and train/val/test splits.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    COLUMN_NAMES, TRAIN_FILE, TEST_FILE, ATTACK_SEVERITY_MAP,
    ATTACK_CATEGORY_MAP, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    CATEGORICAL_FEATURES
)

logger = logging.getLogger(__name__)


class NSLKDDDataLoader:
    """NSL-KDD dataset loader with preprocessing capabilities."""
    
    def __init__(self):
        """Initialize data loader with label encoders."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names = None
        
    def load_raw_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load raw NSL-KDD data file.
        
        Args:
            filepath: Path to NSL-KDD data file (KDDTrain+.txt or KDDTest+.txt)
            
        Returns:
            DataFrame with raw data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if not filepath.exists():
            logger.error(f"Data file not found: {filepath}")
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                f"Run: python data/download_dataset.py"
            )
        
        logger.info(f"Loading data from {filepath}")
        
        try:
            # Load data without header
            df = pd.read_csv(filepath, names=COLUMN_NAMES, header=None)
            
            # Remove trailing periods from attack_type if present
            df['attack_type'] = df['attack_type'].str.replace('.', '', regex=False)
            
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create threat severity score (0-10) from attack types.
        
        Args:
            df: DataFrame with attack_type column
            
        Returns:
            DataFrame with added threat_severity column
        """
        # Map attack types to severity scores
        df['threat_severity'] = df['attack_type'].map(ATTACK_SEVERITY_MAP)
        
        # Handle unknown attack types (assign medium severity)
        unknown_attacks = df['threat_severity'].isna()
        if unknown_attacks.any():
            logger.warning(f"Found {unknown_attacks.sum()} unknown attack types, assigning severity 5")
            df.loc[unknown_attacks, 'threat_severity'] = 5.0
        
        # Add attack category
        df['attack_category'] = df['attack_type'].map(ATTACK_CATEGORY_MAP)
        df['attack_category'] = df['attack_category'].fillna('unknown')
        
        # Create binary classification target (0=normal, 1=attack)
        df['is_attack'] = (df['attack_type'] != 'normal').astype(int)
        
        logger.info(f"Threat severity distribution:\n{df['threat_severity'].value_counts().sort_index()}")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: DataFrame with categorical features
            fit: If True, fit new encoders; if False, use existing encoders
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
                
            if fit:
                # Fit new encoder
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Use existing encoder
                if col not in self.label_encoders:
                    raise ValueError(f"No encoder found for {col}. Must fit first.")
                
                # Handle unseen categories
                known_classes = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known_classes else 'unknown'
                )
                
                # Add 'unknown' class if needed
                if 'unknown' not in known_classes:
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 'unknown'
                    )
                
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # NSL-KDD typically has no missing values, but check anyway
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
            
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        return df
    
    def prepare_features_and_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Columns to exclude from features
        exclude_cols = ['attack_type', 'difficulty_level', 'threat_severity', 
                       'attack_category', 'is_attack']
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['threat_severity'].copy()
        
        self.feature_names = feature_cols
        
        logger.info(f"Prepared features: {X.shape[1]} columns, {len(X)} rows")
        
        return X, y
    
    def load_and_preprocess(
        self, 
        train_path: Path = TRAIN_FILE,
        test_path: Path = TEST_FILE,
        val_split: float = VAL_RATIO
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load and preprocess NSL-KDD dataset with train/val/test splits.
        
        Args:
            train_path: Path to training data file
            test_path: Path to test data file
            val_split: Proportion of training data to use for validation
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Starting data loading and preprocessing pipeline...")
        
        # Load training data
        train_df = self.load_raw_data(train_path)
        train_df = self.create_target_variable(train_df)
        train_df = self.handle_missing_values(train_df)
        train_df = self.encode_categorical_features(train_df, fit=True)
        
        # Load test data
        test_df = self.load_raw_data(test_path)
        test_df = self.create_target_variable(test_df)
        test_df = self.handle_missing_values(test_df)
        test_df = self.encode_categorical_features(test_df, fit=False)
        
        # Prepare features and targets
        X_train_full, y_train_full = self.prepare_features_and_target(train_df)
        X_test, y_test = self.prepare_features_and_target(test_df)
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_split,
            random_state=42,
            stratify=(y_train_full > 0).astype(int)  # Stratify by attack/normal
        )
        
        logger.info(f"Data splits:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Val:   {len(X_val)} samples")
        logger.info(f"  Test:  {len(X_test)} samples")
        
        # Log class distribution
        logger.info(f"Training set severity distribution:")
        for severity in sorted(y_train.unique()):
            count = (y_train == severity).sum()
            pct = 100 * count / len(y_train)
            logger.info(f"  Severity {severity:.0f}: {count} ({pct:.1f}%)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_feature_names(self) -> list:
        """Get list of feature names after preprocessing."""
        if self.feature_names is None:
            raise ValueError("Must call load_and_preprocess first")
        return self.feature_names


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Convenience function to load and preprocess data.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    loader = NSLKDDDataLoader()
    return loader.load_and_preprocess()


if __name__ == "__main__":
    # Test data loading
    logging.basicConfig(level=logging.INFO)
    
    loader = NSLKDDDataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_and_preprocess()
    
    print("\n=== Data Loading Complete ===")
    print(f"Feature names: {loader.get_feature_names()[:5]}... ({len(loader.get_feature_names())} total)")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
