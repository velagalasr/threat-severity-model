"""
Unit Tests for Data Loader Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import NSLKDDDataLoader


class TestNSLKDDDataLoader:
    """Test suite for NSLKDDDataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = NSLKDDDataLoader()
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        assert self.loader.label_encoders == {}
        assert self.loader.feature_names is None
    
    def test_create_target_variable(self):
        """Test threat severity creation from attack types."""
        df = pd.DataFrame({
            'attack_type': ['normal', 'neptune', 'buffer_overflow', 'unknown_attack']
        })
        
        result = self.loader.create_target_variable(df)
        
        assert 'threat_severity' in result.columns
        assert result.loc[0, 'threat_severity'] == 0  # normal
        assert result.loc[1, 'threat_severity'] == 7  # neptune (DoS)
        assert result.loc[2, 'threat_severity'] == 10  # buffer_overflow (U2R)
        assert result.loc[3, 'threat_severity'] == 5  # unknown (default)
        
        assert 'is_attack' in result.columns
        assert result.loc[0, 'is_attack'] == 0
        assert result.loc[1, 'is_attack'] == 1
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        df = pd.DataFrame({
            'protocol_type': ['tcp', 'udp', 'tcp'],
            'service': ['http', 'ftp', 'http'],
            'flag': ['SF', 'REJ', 'SF']
        })
        
        # Fit encoders
        result = self.loader.encode_categorical_features(df, fit=True)
        
        assert result['protocol_type'].dtype in [np.int32, np.int64]
        assert result['service'].dtype in [np.int32, np.int64]
        assert len(self.loader.label_encoders) == 3
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, np.nan, 4.0],
            'col2': ['a', 'b', None, 'd']
        })
        
        result = self.loader.handle_missing_values(df)
        
        assert not result.isnull().any().any()
        assert result.loc[2, 'col1'] == df['col1'].median()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
