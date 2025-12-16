"""
Feature Engineering Module

Creates domain-specific security features for threat detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import PRIVILEGED_PORTS

logger = logging.getLogger(__name__)


class SecurityFeatureEngineer:
    """
    Creates advanced security features from NSL-KDD base features.
    
    Features include:
    - Attack chain length indicators
    - Privilege escalation detection
    - Failed authentication spike detection
    - Geographic/temporal anomalies
    - Protocol anomalies
    - Port-based risk scoring
    - Service type anomalies
    - Connection duration anomalies
    - Byte transfer volume anomalies
    - Port scanning indicators
    """
    
    def __init__(self):
        """Initialize feature engineer with scaler."""
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_attack_chain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create attack chain length features.
        
        Rationale: Attackers often perform sequences of related actions.
        Long chains indicate systematic reconnaissance or exploitation.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with added attack chain features
        """
        df = df.copy()
        
        # Attack chain length approximation using count features
        df['attack_chain_length'] = (
            df['count'] + 
            df['srv_count'] + 
            df['dst_host_count']
        ) / 3.0
        
        # Connection persistence (long chains with same service)
        df['connection_persistence'] = df['same_srv_rate'] * df['srv_count']
        
        return df
    
    def create_privilege_escalation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create privilege escalation indicators.
        
        Rationale: Privilege escalation attempts show patterns like
        root access, su commands, and compromised accounts.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with privilege escalation features
        """
        df = df.copy()
        
        # Privilege escalation indicator
        df['privilege_escalation_indicator'] = (
            df['root_shell'] * 3 +
            df['su_attempted'] * 2 +
            df['num_root'] +
            df['num_compromised']
        )
        
        # Administrative action score
        df['admin_action_score'] = (
            df['num_file_creations'] +
            df['num_shells'] +
            df['num_access_files']
        )
        
        return df
    
    def create_failed_auth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create failed authentication spike detection.
        
        Rationale: Brute force attacks show high failed login rates.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with authentication anomaly features
        """
        df = df.copy()
        
        # Failed login spike (normalized by connection count)
        df['failed_login_rate'] = df['num_failed_logins'] / (df['count'] + 1)
        
        # Guest login anomaly
        df['guest_login_anomaly'] = df['is_guest_login'] * (1 - df['logged_in'])
        
        return df
    
    def create_temporal_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal anomaly features.
        
        Rationale: Unusual timing patterns indicate automated attacks
        or off-hours malicious activity.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with temporal anomaly features
        """
        df = df.copy()
        
        # Connection duration anomaly (z-score normalized)
        duration_mean = df['duration'].mean()
        duration_std = df['duration'].std() + 1e-6
        df['duration_anomaly'] = (df['duration'] - duration_mean) / duration_std
        
        # Rapid connection rate
        df['rapid_connection_rate'] = df['count'] / (df['duration'] + 1)
        
        return df
    
    def create_protocol_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create protocol anomaly features.
        
        Rationale: Unusual protocol usage patterns indicate attacks
        (e.g., ICMP floods, UDP storms).
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with protocol anomaly features
        """
        df = df.copy()
        
        # Error rate anomalies
        df['serror_anomaly'] = (df['serror_rate'] + df['srv_serror_rate']) / 2
        df['rerror_anomaly'] = (df['rerror_rate'] + df['srv_rerror_rate']) / 2
        
        # Overall error score
        df['total_error_rate'] = (
            df['serror_rate'] + 
            df['srv_serror_rate'] + 
            df['rerror_rate'] + 
            df['srv_rerror_rate']
        ) / 4
        
        return df
    
    def create_port_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create port-based risk scoring.
        
        Rationale: Privileged ports (0-1023) are higher risk targets.
        Unusual port patterns indicate scanning or exploitation.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with port risk features
        """
        df = df.copy()
        
        # Source port risk (assuming lower ports are riskier for attacks)
        # Note: NSL-KDD doesn't have explicit port numbers, using srv_count as proxy
        df['high_risk_service_indicator'] = (
            df['srv_count'] * df['dst_host_srv_count'] / 
            (df['count'] + 1)
        )
        
        # Same source port rate (port scanning indicator)
        df['same_src_port_rate'] = df['dst_host_same_src_port_rate']
        
        return df
    
    def create_service_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create service type anomaly features.
        
        Rationale: Unusual service access patterns indicate
        lateral movement or service exploitation.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with service anomaly features
        """
        df = df.copy()
        
        # Service diversity (potential reconnaissance)
        df['service_diversity'] = df['diff_srv_rate'] * df['srv_diff_host_rate']
        
        # Service concentration (focused attack)
        df['service_concentration'] = df['same_srv_rate'] * (1 - df['diff_srv_rate'])
        
        return df
    
    def create_byte_transfer_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create byte transfer volume anomaly features.
        
        Rationale: Data exfiltration shows unusual byte patterns.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with byte transfer anomaly features
        """
        df = df.copy()
        
        # Byte transfer asymmetry
        df['byte_transfer_ratio'] = (
            (df['src_bytes'] + 1) / (df['dst_bytes'] + 1)
        )
        
        # Total byte volume (log-scaled)
        df['total_bytes_log'] = np.log1p(df['src_bytes'] + df['dst_bytes'])
        
        # Unusual byte patterns (z-score)
        src_bytes_mean = df['src_bytes'].mean()
        src_bytes_std = df['src_bytes'].std() + 1e-6
        df['src_bytes_anomaly'] = (df['src_bytes'] - src_bytes_mean) / src_bytes_std
        
        return df
    
    def create_port_scanning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create port scanning detection features.
        
        Rationale: Port scans show connections to many unique destinations
        with low service repetition.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with port scanning features
        """
        df = df.copy()
        
        # Unique destination indicator (many hosts, few connections each)
        df['unique_destination_rate'] = (
            df['dst_host_count'] / (df['count'] + 1)
        )
        
        # Port scan score (high dst_host_count with low same_srv_rate)
        df['port_scan_score'] = (
            df['dst_host_count'] * 
            (1 - df['dst_host_same_srv_rate']) /
            (df['duration'] + 1)
        )
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all security features.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with all engineered features added
        """
        logger.info("Creating security features...")
        
        df = self.create_attack_chain_features(df)
        df = self.create_privilege_escalation_features(df)
        df = self.create_failed_auth_features(df)
        df = self.create_temporal_anomaly_features(df)
        df = self.create_protocol_anomaly_features(df)
        df = self.create_port_risk_features(df)
        df = self.create_service_anomaly_features(df)
        df = self.create_byte_transfer_anomaly_features(df)
        df = self.create_port_scanning_features(df)
        
        # Count new features
        new_feature_count = len([col for col in df.columns if col not in self.get_original_columns(df)])
        logger.info(f"Created {new_feature_count} new security features")
        
        return df
    
    @staticmethod
    def get_original_columns(df: pd.DataFrame) -> list:
        """Get list of original (non-engineered) column names."""
        # These are the base NSL-KDD features
        original_cols = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        return [col for col in original_cols if col in df.columns]
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform features.
        
        Args:
            X: DataFrame with engineered features
            
        Returns:
            Scaled DataFrame
        """
        # Create all features
        X_engineered = self.create_all_features(X)
        
        # Fit and transform scaler
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_engineered),
            columns=X_engineered.columns,
            index=X_engineered.index
        )
        
        self.is_fitted = True
        logger.info("Fitted scaler on training data")
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            X: DataFrame with features to transform
            
        Returns:
            Scaled DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_transform before transform")
        
        # Create all features
        X_engineered = self.create_all_features(X)
        
        # Transform using fitted scaler
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_engineered),
            columns=X_engineered.columns,
            index=X_engineered.index
        )
        
        return X_scaled
    
    def save(self, filepath: str) -> None:
        """Save feature engineer to disk."""
        joblib.dump(self, filepath)
        logger.info(f"Saved feature engineer to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'SecurityFeatureEngineer':
        """Load feature engineer from disk."""
        engineer = joblib.load(filepath)
        logger.info(f"Loaded feature engineer from {filepath}")
        return engineer


def engineer_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SecurityFeatureEngineer]:
    """
    Convenience function to engineer features for all splits.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, engineer)
    """
    engineer = SecurityFeatureEngineer()
    
    X_train_scaled = engineer.fit_transform(X_train)
    X_val_scaled = engineer.transform(X_val)
    X_test_scaled = engineer.transform(X_test)
    
    logger.info(f"Final feature count: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, engineer


if __name__ == "__main__":
    # Test feature engineering
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'duration': [0, 5, 10, 100],
        'src_bytes': [100, 200, 500, 1000],
        'dst_bytes': [50, 100, 250, 500],
        'count': [1, 5, 10, 50],
        'srv_count': [1, 3, 8, 40],
        'dst_host_count': [1, 2, 5, 20],
        'num_failed_logins': [0, 0, 5, 10],
        'logged_in': [1, 1, 0, 0],
        'root_shell': [0, 0, 1, 1],
        'su_attempted': [0, 0, 1, 1],
        'num_root': [0, 0, 2, 5],
        'num_compromised': [0, 0, 1, 3],
        'same_srv_rate': [1.0, 0.8, 0.5, 0.2],
        'diff_srv_rate': [0.0, 0.2, 0.5, 0.8],
        'serror_rate': [0.0, 0.1, 0.3, 0.7],
        'srv_serror_rate': [0.0, 0.1, 0.3, 0.7],
        'rerror_rate': [0.0, 0.05, 0.2, 0.5],
        'srv_rerror_rate': [0.0, 0.05, 0.2, 0.5],
        'srv_diff_host_rate': [0.0, 0.1, 0.3, 0.6],
        'dst_host_same_srv_rate': [1.0, 0.9, 0.6, 0.3],
        'dst_host_srv_diff_host_rate': [0.0, 0.1, 0.4, 0.7],
        'dst_host_same_src_port_rate': [1.0, 0.8, 0.5, 0.2],
        'is_guest_login': [0, 0, 1, 0],
        'num_file_creations': [0, 1, 2, 5],
        'num_shells': [0, 0, 1, 2],
        'num_access_files': [0, 0, 1, 3],
        'hot': [0, 1, 2, 5],
        'wrong_fragment': [0, 0, 1, 2],
        'urgent': [0, 0, 0, 1],
        'land': [0, 0, 0, 1],
        'num_outbound_cmds': [0, 0, 0, 0],
        'is_host_login': [0, 0, 0, 0],
    })
    
    engineer = SecurityFeatureEngineer()
    transformed = engineer.fit_transform(sample_data)
    
    print("\n=== Feature Engineering Complete ===")
    print(f"Original features: {sample_data.shape[1]}")
    print(f"Engineered features: {transformed.shape[1]}")
    print(f"New features added: {transformed.shape[1] - sample_data.shape[1]}")
    print(f"\nNew feature names: {[col for col in transformed.columns if col not in sample_data.columns]}")
