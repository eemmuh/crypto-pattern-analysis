"""
Configuration settings for the crypto trading analysis project.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

# Environment variable support
def get_env_var(key: str, default: Any = None, type_cast=str):
    """Get environment variable with type casting."""
    value = os.getenv(key, default)
    if value is not None and type_cast != str:
        try:
            return type_cast(value)
        except (ValueError, TypeError):
            return default
    return value

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Data Collection Settings
DATA_SETTINGS = {
    'cache_dir': str(CACHE_DIR),
    'default_period': get_env_var('DEFAULT_PERIOD', '1y'),
    'default_interval': get_env_var('DEFAULT_INTERVAL', '1d'),
    'supported_symbols': [
        'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 
        'MATIC', 'LINK', 'UNI', 'XRP', 'LTC', 'BCH', 'XLM'
    ],
    'max_data_points': get_env_var('MAX_DATA_POINTS', 10000, int),
    'data_quality_threshold': get_env_var('DATA_QUALITY_THRESHOLD', 0.9, float)
}

# Technical Indicators Settings
INDICATOR_SETTINGS = {
    'rsi_periods': [14, 21],
    'moving_averages': [5, 10, 20, 50, 100, 200],
    'exponential_mas': [12, 26, 50, 100],
    'bollinger_period': 20,
    'bollinger_std': 2,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'stochastic_k': 14,
    'stochastic_d': 3,
    'atr_period': 14
}

# Pattern Detection Settings
PATTERN_SETTINGS = {
    'doji_threshold': 0.1,  # Percentage of average body size
    'hammer_shadow_ratio': 2.0,  # Lower shadow to body ratio
    'engulfing_threshold': 0.8,  # Minimum engulfing ratio
    'divergence_lookback': 10,  # Periods to look back for divergence
    'pattern_confidence': 0.7  # Minimum confidence for pattern detection
}

# Clustering Settings
CLUSTERING_SETTINGS = {
    'default_n_clusters': 5,
    'max_clusters': 10,
    'dbscan_eps': 0.5,
    'dbscan_min_samples': 5,
    'time_series_window': 20,
    'feature_groups': ['momentum', 'volatility', 'trend', 'volume']
}

# Regime Detection Settings
REGIME_SETTINGS = {
    'default_n_regimes': 4,
    'regime_types': ['bull_stable', 'bull_volatile', 'bear_stable', 'bear_volatile', 'sideways_stable', 'sideways_volatile'],
    'return_threshold': 0.01,  # 1% daily return threshold
    'volatility_threshold': 0.02,  # 2% daily volatility threshold
    'transition_smoothing': 0.1  # Smoothing factor for transition matrix
}

# Visualization Settings
VISUALIZATION_SETTINGS = {
    'default_height': 600,
    'default_width': 1000,
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'template': 'plotly_white',
    'font_family': 'Arial, sans-serif',
    'font_size': 12
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    'risk_free_rate': 0.02,  # 2% annual risk-free rate
    'trading_days': 252,  # Number of trading days per year
    'max_drawdown_threshold': 0.20,  # 20% maximum drawdown threshold
    'sharpe_threshold': 1.0,  # Minimum Sharpe ratio threshold
    'volatility_threshold': 0.50  # 50% annual volatility threshold
}

# Machine Learning Settings
ML_SETTINGS = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.1,
    'cross_validation_folds': 5,
    'early_stopping_patience': 10,
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 100
}

# Logging Settings
LOGGING_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/crypto_analysis.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5
}

# API Settings (for future extensions)
API_SETTINGS = {
    'yfinance_timeout': 30,
    'max_retries': 3,
    'retry_delay': 1,
    'rate_limit': 100,  # requests per minute
    'user_agent': 'CryptoAnalysis/1.0'
}

# Feature Engineering Settings
FEATURE_SETTINGS = {
    'return_periods': [1, 5, 10, 20],
    'volatility_periods': [5, 10, 20, 50],
    'momentum_periods': [5, 10, 20],
    'volume_periods': [5, 10, 20],
    'correlation_period': 20,
    'z_score_threshold': 3.0,
    'outlier_method': 'iqr'  # 'iqr' or 'zscore'
}

# Anomaly Detection Settings
ANOMALY_SETTINGS = {
    'isolation_forest_contamination': 0.1,
    'one_class_svm_nu': 0.1,
    'dbscan_eps_anomaly': 0.3,
    'dbscan_min_samples_anomaly': 3,
    'lof_n_neighbors': 20,
    'lof_contamination': 0.1
}

# Spectral Analysis Settings
SPECTRAL_SETTINGS = {
    'fft_window': 'hann',
    'fft_nperseg': 256,
    'fft_noverlap': 128,
    'periodogram_method': 'welch',
    'spectral_density_method': 'periodogram',
    'frequency_range': (0.001, 0.5),  # Daily to yearly cycles
    'dominant_frequency_threshold': 0.1
}

# Changepoint Detection Settings
CHANGEPOINT_SETTINGS = {
    'penalty': 'aic',  # 'aic', 'bic', 'manual'
    'penalty_value': 10,
    'min_segment_length': 20,
    'max_segments': 10,
    'method': 'pelt',  # 'pelt', 'binseg', 'segneigh'
    'cost_function': 'l2'  # 'l2', 'l1', 'rbf'
}

# Export all settings
ALL_SETTINGS = {
    'data': DATA_SETTINGS,
    'indicators': INDICATOR_SETTINGS,
    'patterns': PATTERN_SETTINGS,
    'clustering': CLUSTERING_SETTINGS,
    'regimes': REGIME_SETTINGS,
    'visualization': VISUALIZATION_SETTINGS,
    'performance': PERFORMANCE_SETTINGS,
    'ml': ML_SETTINGS,
    'logging': LOGGING_SETTINGS,
    'api': API_SETTINGS,
    'features': FEATURE_SETTINGS,
    'anomaly': ANOMALY_SETTINGS,
    'spectral': SPECTRAL_SETTINGS,
    'changepoint': CHANGEPOINT_SETTINGS
} 