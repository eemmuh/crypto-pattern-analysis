"""
Unit tests for data collection module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil

from src.data.data_collector import CryptoDataCollector


class TestCryptoDataCollector:
    """Test cases for CryptoDataCollector class."""
    
    def test_initialization(self, temp_cache_dir):
        """Test collector initialization."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        assert collector.cache_dir == temp_cache_dir
        assert os.path.exists(temp_cache_dir)
        assert isinstance(collector.crypto_symbols, dict)
        assert 'BTC' in collector.crypto_symbols
        assert collector.crypto_symbols['BTC'] == 'BTC-USD'
    
    def test_get_ohlcv_data_success(self, mock_yfinance, temp_cache_dir):
        """Test successful data retrieval."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        data = collector.get_ohlcv_data('BTC', period='1mo')
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
        assert len(data) > 0
    
    def test_get_ohlcv_data_invalid_symbol(self, temp_cache_dir):
        """Test handling of invalid symbol."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        with pytest.raises(ValueError, match="Invalid symbol"):
            collector.get_ohlcv_data('INVALID', period='1mo')
    
    def test_get_ohlcv_data_cache(self, mock_yfinance, temp_cache_dir):
        """Test data caching functionality."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        # First call should fetch and cache
        data1 = collector.get_ohlcv_data('BTC', period='1mo')
        
        # Second call should load from cache
        data2 = collector.get_ohlcv_data('BTC', period='1mo')
        
        assert data1.equals(data2)
        
        # Check cache file exists
        cache_files = os.listdir(temp_cache_dir)
        assert len(cache_files) > 0
    
    def test_get_multiple_cryptos(self, mock_yfinance, temp_cache_dir):
        """Test multiple cryptocurrency data retrieval."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        symbols = ['BTC', 'ETH']
        market_data = collector.get_multiple_cryptos(symbols, period='1mo')
        
        assert isinstance(market_data, dict)
        assert len(market_data) == len(symbols)
        assert all(symbol in market_data for symbol in symbols)
        assert all(isinstance(data, pd.DataFrame) for data in market_data.values())
    
    def test_get_market_data(self, mock_yfinance, temp_cache_dir):
        """Test market data aggregation."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        market_data = collector.get_market_data(['BTC', 'ETH'], period='1mo')
        
        assert isinstance(market_data, pd.DataFrame)
        assert not market_data.empty
        assert 'SYMBOL' in market_data.columns
    
    def test_add_basic_features(self, sample_crypto_data):
        """Test basic feature calculation."""
        collector = CryptoDataCollector()
        
        data_with_features = collector._add_basic_features(sample_crypto_data.copy())
        
        expected_features = ['RETURN', 'LOG_RETURN', 'VOLATILITY', 'HIGH_LOW_RATIO']
        for feature in expected_features:
            assert feature in data_with_features.columns
    
    def test_get_latest_data(self, mock_yfinance, temp_cache_dir):
        """Test latest data retrieval."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        latest_data = collector.get_latest_data('BTC', days=30)
        
        assert isinstance(latest_data, pd.DataFrame)
        assert not latest_data.empty
        assert len(latest_data) <= 30
    
    def test_cache_filename_generation(self, temp_cache_dir):
        """Test cache filename generation."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        filename = collector._get_cache_filename('BTC-USD', '1mo', '1d', None, None)
        
        assert 'BTC-USD' in filename
        assert '1mo' in filename
        assert '1d' in filename
        assert filename.endswith('.csv')
    
    def test_error_handling_network_failure(self, temp_cache_dir):
        """Test handling of network failures."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        # Mock yfinance to raise an exception
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("Network error")
            
            # Should fall back to sample data
            data = collector.get_ohlcv_data('BTC', period='1mo')
            
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
    
    def test_data_validation(self, temp_cache_dir):
        """Test data validation."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        # Test with invalid period
        with pytest.raises(ValueError):
            collector.get_ohlcv_data('BTC', period='invalid_period')
        
        # Test with invalid interval
        with pytest.raises(ValueError):
            collector.get_ohlcv_data('BTC', period='1mo', interval='invalid_interval')
    
    def test_retry_mechanism(self, temp_cache_dir):
        """Test retry mechanism for failed requests."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        call_count = 0
        
        def mock_history(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first two attempts
                raise Exception("Temporary error")
            return pd.DataFrame({
                'Open': [50000],
                'High': [51000],
                'Low': [49000],
                'Close': [50500],
                'Volume': [1000]
            })
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history = mock_history
            
            data = collector.get_ohlcv_data('BTC', period='1mo', max_retries=3)
            
            assert call_count == 3  # Should have retried
            assert isinstance(data, pd.DataFrame)
    
    def test_sample_data_generation(self, temp_cache_dir):
        """Test sample data generation when API fails."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        # Force sample data generation
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("API unavailable")
            
            data = collector.get_ohlcv_data('BTC', period='1mo')
            
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            assert all(col in data.columns for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
    
    def test_data_consistency(self, mock_yfinance, temp_cache_dir):
        """Test data consistency across multiple calls."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        data1 = collector.get_ohlcv_data('BTC', period='1mo')
        data2 = collector.get_ohlcv_data('BTC', period='1mo')
        
        # Data should be identical (from cache)
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_memory_efficiency(self, mock_yfinance, temp_cache_dir):
        """Test memory efficiency with large datasets."""
        collector = CryptoDataCollector(cache_dir=temp_cache_dir)
        
        # Test with longer period
        data = collector.get_ohlcv_data('BTC', period='1y')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 100  # Should have substantial data
        assert data.memory_usage(deep=True).sum() < 10 * 1024 * 1024  # Less than 10MB 