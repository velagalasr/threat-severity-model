"""
Unit Tests for Feature Engineering Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from feature_engineering import SecurityFeatureEngineer


class TestSecurityFeatureEngineer:
    """Test suite for SecurityFeatureEngineer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = SecurityFeatureEngineer()
        self.sample_data = pd.DataFrame({
            'duration': [0, 5, 10],
            'src_bytes': [100, 200, 300],
            'dst_bytes': [50, 100, 150],
            'count': [1, 5, 10],
            'srv_count': [1, 3, 8],
            'dst_host_count': [1, 2, 5],
            'num_failed_logins': [0, 0, 5],
            'logged_in': [1, 1, 0],
            'root_shell': [0, 0, 1],
            'su_attempted': [0, 0, 1],
            'num_root': [0, 0, 2],
            'num_compromised': [0, 0, 1],
            'same_srv_rate': [1.0, 0.8, 0.5],
            'diff_srv_rate': [0.0, 0.2, 0.5],
            'serror_rate': [0.0, 0.1, 0.3],
            'srv_serror_rate': [0.0, 0.1, 0.3],
            'rerror_rate': [0.0, 0.05, 0.2],
            'srv_rerror_rate': [0.0, 0.05, 0.2],
            'srv_diff_host_rate': [0.0, 0.1, 0.3],
            'dst_host_same_srv_rate': [1.0, 0.9, 0.6],
            'dst_host_srv_diff_host_rate': [0.0, 0.1, 0.4],
            'dst_host_same_src_port_rate': [1.0, 0.8, 0.5],
            'is_guest_login': [0, 0, 1],
            'num_file_creations': [0, 1, 2],
            'num_shells': [0, 0, 1],
            'num_access_files': [0, 0, 1],
            'hot': [0, 1, 2],
            'wrong_fragment': [0, 0, 1],
            'urgent': [0, 0, 0],
            'land': [0, 0, 0],
            'num_outbound_cmds': [0, 0, 0],
            'is_host_login': [0, 0, 0],
        })
    
    def test_create_attack_chain_features(self):
        """Test attack chain feature creation."""
        result = self.engineer.create_attack_chain_features(self.sample_data)
        
        assert 'attack_chain_length' in result.columns
        assert 'connection_persistence' in result.columns
        assert len(result) == len(self.sample_data)
    
    def test_create_privilege_escalation_features(self):
        """Test privilege escalation feature creation."""
        result = self.engineer.create_privilege_escalation_features(self.sample_data)
        
        assert 'privilege_escalation_indicator' in result.columns
        assert 'admin_action_score' in result.columns
        
        # Check that high privilege actions score higher
        assert result.loc[2, 'privilege_escalation_indicator'] > result.loc[0, 'privilege_escalation_indicator']
    
    def test_create_failed_auth_features(self):
        """Test failed authentication features."""
        result = self.engineer.create_failed_auth_features(self.sample_data)
        
        assert 'failed_login_rate' in result.columns
        assert 'guest_login_anomaly' in result.columns
    
    def test_fit_transform(self):
        """Test fit_transform scales features."""
        result = self.engineer.fit_transform(self.sample_data)
        
        assert self.engineer.is_fitted
        assert result.shape[1] > self.sample_data.shape[1]  # More features added
        
        # Check scaling (mean should be ~0, std ~1 for each feature)
        assert abs(result.mean().mean()) < 0.5
    
    def test_transform_requires_fit(self):
        """Test that transform requires prior fit."""
        with pytest.raises(ValueError, match="Must call fit_transform before transform"):
            self.engineer.transform(self.sample_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
