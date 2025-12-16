"""
Configuration Management

Centralized configuration for threat severity model project.
"""

from pathlib import Path
from typing import Dict, List
import os

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Dataset Configuration
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# NSL-KDD Column Names
COLUMN_NAMES = [
    # Basic features
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
    'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    # Content features
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 
    'is_host_login', 'is_guest_login',
    # Traffic features (time-based)
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate',
    # Traffic features (host-based)
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate',
    # Label
    'attack_type', 'difficulty_level'
]

# Attack Type Mapping to Severity Scores (0-10)
ATTACK_SEVERITY_MAP = {
    'normal': 0,
    # DoS attacks - High severity (7-8)
    'back': 7, 'land': 7, 'neptune': 7, 'pod': 7, 'smurf': 7, 
    'teardrop': 7, 'processtable': 7, 'mailbomb': 7, 'apache2': 7,
    'udpstorm': 7, 'worm': 8,
    # Probe attacks - Medium severity (4-6)
    'satan': 5, 'ipsweep': 5, 'nmap': 5, 'portsweep': 5, 
    'mscan': 5, 'saint': 6,
    # R2L attacks - Critical severity (8-9)
    'guess_passwd': 8, 'ftp_write': 8, 'imap': 8, 'phf': 8, 
    'multihop': 8, 'warezmaster': 7, 'warezclient': 7, 
    'spy': 9, 'xlock': 8, 'xsnoop': 8, 'snmpguess': 8, 
    'snmpgetattack': 8, 'httptunnel': 8, 'sendmail': 8, 'named': 8,
    # U2R attacks - Critical severity (9-10)
    'buffer_overflow': 10, 'loadmodule': 10, 'rootkit': 10, 
    'perl': 9, 'sqlattack': 9, 'xterm': 9, 'ps': 9,
}

# Attack Category Mapping
ATTACK_CATEGORY_MAP = {
    'normal': 'normal',
    # DoS
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 
    'smurf': 'dos', 'teardrop': 'dos', 'processtable': 'dos', 
    'mailbomb': 'dos', 'apache2': 'dos', 'udpstorm': 'dos', 'worm': 'dos',
    # Probe
    'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 
    'portsweep': 'probe', 'mscan': 'probe', 'saint': 'probe',
    # R2L
    'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 
    'phf': 'r2l', 'multihop': 'r2l', 'warezmaster': 'r2l', 
    'warezclient': 'r2l', 'spy': 'r2l', 'xlock': 'r2l', 
    'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l', 
    'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    # U2R
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 
    'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r',
}

# Model Training Configuration
MODEL_CONFIG = {
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 10,
    },
    'lightgbm': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'regression',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1,
    },
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000,
        'n_jobs': -1,
    }
}

# Train/Val/Test Split Ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Threshold Optimization SLOs
THRESHOLD_SLOS = {
    'tier1_recall': {
        'name': 'SOC Tier 1',
        'target_metric': 'recall',
        'target_value': 0.95,
        'description': 'High recall for initial triage'
    },
    'tier2_balanced': {
        'name': 'SOC Tier 2',
        'target_metric': 'f1',
        'target_value': None,  # Maximize F1
        'description': 'Balanced precision and recall'
    },
    'ir_precision': {
        'name': 'Incident Response',
        'target_metric': 'precision',
        'target_value': 0.98,
        'description': 'High precision for deep investigations'
    },
    'cost_optimal': {
        'name': 'Cost-Optimal',
        'target_metric': 'cost',
        'fp_cost': 50,  # $50 per false positive investigation
        'fn_cost': 5000000,  # $5M per missed breach
        'description': 'Minimize total cost'
    }
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'precision_threshold': 0.85,  # Alert if below 85%
    'drift_p_value': 0.05,  # KS test significance level
    'latency_p95_ms': 50,  # p95 latency SLO
    'latency_p99_ms': 100,  # p99 latency SLO
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'max_request_timeout': 5.0,  # seconds
    'model_path': MODELS_DIR / 'xgboost_model.pkl',
    'scaler_path': MODELS_DIR / 'scaler.pkl',
    'explainer_path': MODELS_DIR / 'shap_explainer.pkl',
}

# Risk Level Thresholds
RISK_LEVEL_THRESHOLDS = {
    'low': (0, 3),
    'medium': (3, 6),
    'high': (6, 8),
    'critical': (8, 10),
}

# Recommended Actions by Risk Level
RECOMMENDED_ACTIONS = {
    'low': 'Log and monitor',
    'medium': 'Alert SOC Tier 1 for review',
    'high': 'Escalate to SOC Tier 2 for investigation',
    'critical': 'Immediate isolation and forensic investigation',
}

# Feature Groups for Engineering
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERICAL_FEATURES = [col for col in COLUMN_NAMES if col not in CATEGORICAL_FEATURES + ['attack_type', 'difficulty_level']]

# Privileged Ports (security feature)
PRIVILEGED_PORTS = list(range(0, 1024))

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'app.log'),
            'formatter': 'standard',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file'],
    },
}
