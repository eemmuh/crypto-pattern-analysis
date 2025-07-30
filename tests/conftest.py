"""
Pytest configuration and fixtures for crypto trading analysis tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import shutil

from tests import TEST_CONFIG


@pytest.fixture(scope="session")
def sample_crypto_data():
    """Generate sample cryptocurrency data for testing."""
    np.random.seed(42)
    
    # Generate 100 days of sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data with some volatility
    base_price = 50000  # Starting price
    returns = np.random.normal(0.001, 0.02, 100)  # Daily returns with 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'OPEN': prices * (1 + np.random.normal(0, 0.005, 100)),
        'HIGH': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'LOW': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'CLOSE': prices,
        'VOLUME': np.random.lognormal(15, 0.5, 100)  # Realistic volume data
    }, index=dates)
    
    # Ensure OHLC relationships are valid
    data['HIGH'] = data[['OPEN', 'HIGH', 'CLOSE']].max(axis=1)
    data['LOW'] = data[['OPEN', 'LOW', 'CLOSE']].min(axis=1)
    
    return data


@pytest.fixture(scope="session")
def sample_market_data():
    """Generate sample market data for multiple cryptocurrencies."""
    symbols = ['BTC', 'ETH', 'BNB']
    market_data = {}
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)  # Different seed for each symbol
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        base_price = 50000 if symbol == 'BTC' else (3000 if symbol == 'ETH' else 300)
        
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        data = pd.DataFrame({
            'OPEN': prices * (1 + np.random.normal(0, 0.005, 100)),
            'HIGH': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'LOW': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'CLOSE': prices,
            'VOLUME': np.random.lognormal(15, 0.5, 100)
        }, index=dates)
        
        data['HIGH'] = data[['OPEN', 'HIGH', 'CLOSE']].max(axis=1)
        data['LOW'] = data[['OPEN', 'LOW', 'CLOSE']].min(axis=1)
        
        market_data[symbol] = data
    
    return market_data


@pytest.fixture(scope="function")
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def mock_yfinance(monkeypatch):
    """Mock yfinance to avoid actual API calls during testing."""
    class MockTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        
        def history(self, period=None, start=None, end=None, interval='1d'):
            # Return sample data instead of making API calls
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            np.random.seed(hash(self.symbol) % 1000)
            
            base_price = 50000
            returns = np.random.normal(0.001, 0.02, 100)
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            prices = np.array(prices)
            
            data = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.005, 100)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                'Close': prices,
                'Volume': np.random.lognormal(15, 0.5, 100)
            }, index=dates)
            
            data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
            data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
            
            return data
    
    def mock_yf_ticker(symbol):
        return MockTicker(symbol)
    
    monkeypatch.setattr('yfinance.Ticker', mock_yf_ticker)


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'test_symbols': ['BTC', 'ETH', 'BNB'],
        'test_period': '1mo',
        'test_interval': '1d',
        'n_clusters': 3,
        'n_regimes': 3,
        'random_state': 42
    } 