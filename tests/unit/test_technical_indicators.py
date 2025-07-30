"""
Unit tests for technical indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.features.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test cases for TechnicalIndicators class."""
    
    def test_initialization(self, sample_crypto_data):
        """Test indicator initialization."""
        ti = TechnicalIndicators(sample_crypto_data)
        
        assert ti.data is not None
        assert len(ti.data) == len(sample_crypto_data)
        assert all(col in ti.data.columns for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
    
    def test_add_all_indicators(self, sample_crypto_data):
        """Test adding all technical indicators."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_indicators = ti.add_all_indicators()
        
        # Check that indicators were added
        assert len(data_with_indicators.columns) > len(sample_crypto_data.columns)
        
        # Check for specific indicator groups
        rsi_cols = [col for col in data_with_indicators.columns if 'RSI' in col]
        macd_cols = [col for col in data_with_indicators.columns if 'MACD' in col]
        bb_cols = [col for col in data_with_indicators.columns if 'BB_' in col]
        
        assert len(rsi_cols) > 0
        assert len(macd_cols) > 0
        assert len(bb_cols) > 0
    
    def test_rsi_calculation(self, sample_crypto_data):
        """Test RSI calculation."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_rsi = ti.add_rsi(period=14)
        
        assert 'RSI_14' in data_with_rsi.columns
        
        # RSI should be between 0 and 100
        rsi_values = data_with_rsi['RSI_14'].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)
        
        # RSI should not have extreme outliers
        assert rsi_values.std() < 50  # Reasonable standard deviation
    
    def test_moving_averages(self, sample_crypto_data):
        """Test moving average calculations."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_ma = ti.add_moving_averages(periods=[5, 10, 20])
        
        # Check SMA columns
        for period in [5, 10, 20]:
            col_name = f'SMA_{period}'
            assert col_name in data_with_ma.columns
            
            # Moving averages should be reasonable
            ma_values = data_with_ma[col_name].dropna()
            assert len(ma_values) > 0
            assert ma_values.min() > 0
    
    def test_bollinger_bands(self, sample_crypto_data):
        """Test Bollinger Bands calculation."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_bb = ti.add_bollinger_bands(period=20, std_dev=2)
        
        bb_cols = ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_WIDTH', 'BB_POSITION']
        for col in bb_cols:
            assert col in data_with_bb.columns
        
        # Bollinger Bands relationships
        bb_data = data_with_bb.dropna()
        assert all(bb_data['BB_UPPER'] >= bb_data['BB_MIDDLE'])
        assert all(bb_data['BB_MIDDLE'] >= bb_data['BB_LOWER'])
        assert all(bb_data['BB_WIDTH'] >= 0)
    
    def test_macd_calculation(self, sample_crypto_data):
        """Test MACD calculation."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_macd = ti.add_macd(fast=12, slow=26, signal=9)
        
        macd_cols = ['MACD', 'MACD_SIGNAL', 'MACD_HISTOGRAM']
        for col in macd_cols:
            assert col in data_with_macd.columns
        
        # MACD histogram should equal MACD - MACD_SIGNAL
        macd_data = data_with_macd.dropna()
        np.testing.assert_array_almost_equal(
            macd_data['MACD_HISTOGRAM'],
            macd_data['MACD'] - macd_data['MACD_SIGNAL'],
            decimal=10
        )
    
    def test_stochastic_oscillator(self, sample_crypto_data):
        """Test Stochastic Oscillator calculation."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_stoch = ti.add_stochastic_oscillator(k_period=14, d_period=3)
        
        stoch_cols = ['STOCH_K', 'STOCH_D']
        for col in stoch_cols:
            assert col in data_with_stoch.columns
        
        # Stochastic values should be between 0 and 100
        stoch_data = data_with_stoch.dropna()
        for col in stoch_cols:
            values = stoch_data[col]
            assert all(0 <= val <= 100 for val in values)
    
    def test_atr_calculation(self, sample_crypto_data):
        """Test Average True Range calculation."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_atr = ti.add_atr(period=14)
        
        assert 'ATR' in data_with_atr.columns
        
        # ATR should always be positive
        atr_values = data_with_atr['ATR'].dropna()
        assert all(val > 0 for val in atr_values)
    
    def test_volume_indicators(self, sample_crypto_data):
        """Test volume-based indicators."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_volume = ti.add_volume_indicators()
        
        volume_cols = ['OBV', 'VWAP', 'VOLUME_SMA', 'VOLUME_RATIO']
        for col in volume_cols:
            assert col in data_with_volume.columns
        
        # Volume indicators should be reasonable
        volume_data = data_with_volume.dropna()
        assert all(volume_data['OBV'] >= 0)  # OBV should be non-negative
        assert all(volume_data['VOLUME_RATIO'] >= 0)  # Volume ratio should be non-negative
    
    def test_momentum_indicators(self, sample_crypto_data):
        """Test momentum indicators."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_momentum = ti.add_momentum_indicators()
        
        momentum_cols = ['MOMENTUM', 'ROC', 'WILLIAMS_R']
        for col in momentum_cols:
            assert col in data_with_momentum.columns
        
        # Williams %R should be between -100 and 0
        williams_data = data_with_momentum['WILLIAMS_R'].dropna()
        assert all(-100 <= val <= 0 for val in williams_data)
    
    def test_trend_indicators(self, sample_crypto_data):
        """Test trend indicators."""
        ti = TechnicalIndicators(sample_crypto_data)
        data_with_trend = ti.add_trend_indicators()
        
        trend_cols = ['ADX', 'CCI', 'AROON_UP', 'AROON_DOWN']
        for col in trend_cols:
            assert col in data_with_trend.columns
        
        # Aroon indicators should be between 0 and 100
        aroon_data = data_with_trend.dropna()
        assert all(0 <= val <= 100 for val in aroon_data['AROON_UP'])
        assert all(0 <= val <= 100 for val in aroon_data['AROON_DOWN'])
    
    def test_get_feature_matrix(self, sample_crypto_data):
        """Test feature matrix generation."""
        ti = TechnicalIndicators(sample_crypto_data)
        ti.add_all_indicators()
        
        feature_matrix = ti.get_feature_matrix()
        
        assert isinstance(feature_matrix, pd.DataFrame)
        assert len(feature_matrix) > 0
        assert len(feature_matrix.columns) > 0
        
        # Should not contain NaN values
        assert not feature_matrix.isnull().any().any()
    
    def test_get_indicator_list(self, sample_crypto_data):
        """Test indicator list retrieval."""
        ti = TechnicalIndicators(sample_crypto_data)
        ti.add_all_indicators()
        
        indicator_list = ti.get_indicator_list()
        
        assert isinstance(indicator_list, list)
        assert len(indicator_list) > 0
        
        # Should contain expected indicator types
        indicator_names = [ind.lower() for ind in indicator_list]
        assert any('rsi' in name for name in indicator_names)
        assert any('macd' in name for name in indicator_names)
        assert any('bb_' in name for name in indicator_names)
    
    def test_data_validation(self):
        """Test data validation."""
        # Test with missing required columns
        invalid_data = pd.DataFrame({'OPEN': [100], 'CLOSE': [101]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            TechnicalIndicators(invalid_data)
    
    def test_nan_handling(self, sample_crypto_data):
        """Test handling of NaN values."""
        # Add some NaN values
        data_with_nans = sample_crypto_data.copy()
        data_with_nans.loc[10:15, 'CLOSE'] = np.nan
        
        ti = TechnicalIndicators(data_with_nans)
        data_with_indicators = ti.add_all_indicators()
        
        # Should handle NaN values gracefully
        assert isinstance(data_with_indicators, pd.DataFrame)
    
    def test_performance_with_large_data(self):
        """Test performance with larger datasets."""
        # Generate larger dataset
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        large_data = pd.DataFrame({
            'OPEN': np.random.uniform(40000, 60000, 1000),
            'HIGH': np.random.uniform(40000, 60000, 1000),
            'LOW': np.random.uniform(40000, 60000, 1000),
            'CLOSE': np.random.uniform(40000, 60000, 1000),
            'VOLUME': np.random.lognormal(15, 0.5, 1000)
        }, index=dates)
        
        # Ensure OHLC relationships
        large_data['HIGH'] = large_data[['OPEN', 'HIGH', 'CLOSE']].max(axis=1)
        large_data['LOW'] = large_data[['OPEN', 'LOW', 'CLOSE']].min(axis=1)
        
        ti = TechnicalIndicators(large_data)
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        data_with_indicators = ti.add_all_indicators()
        end_time = time.time()
        
        assert end_time - start_time < 10  # Should complete within 10 seconds
        assert len(data_with_indicators) == len(large_data)
    
    def test_indicator_consistency(self, sample_crypto_data):
        """Test consistency of indicator calculations."""
        ti = TechnicalIndicators(sample_crypto_data)
        
        # Calculate indicators multiple times
        data1 = ti.add_all_indicators()
        data2 = ti.add_all_indicators()
        
        # Results should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_custom_periods(self, sample_crypto_data):
        """Test custom period parameters."""
        ti = TechnicalIndicators(sample_crypto_data)
        
        # Test custom RSI period
        data_custom_rsi = ti.add_rsi(period=21)
        assert 'RSI_21' in data_custom_rsi.columns
        
        # Test custom moving average periods
        data_custom_ma = ti.add_moving_averages(periods=[7, 14, 30])
        for period in [7, 14, 30]:
            assert f'SMA_{period}' in data_custom_ma.columns 