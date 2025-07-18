"""
Data collection module for cryptocurrency market data.
Supports multiple data sources and timeframes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataCollector:
    """
    Collects cryptocurrency market data from various sources.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the data collector.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Common cryptocurrency symbols
        self.crypto_symbols = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD', 
            'BNB': 'BNB-USD',
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
            'DOT': 'DOT-USD',
            'AVAX': 'AVAX-USD',
            'MATIC': 'MATIC-USD',
            'LINK': 'LINK-USD',
            'UNI': 'UNI-USD'
        }
    
    def get_ohlcv_data(self, 
                       symbol: str, 
                       period: str = "2y",
                       interval: str = "1d",
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get the full symbol for yfinance
            full_symbol = self.crypto_symbols.get(symbol.upper(), f"{symbol.upper()}-USD")
            
            # Check cache first
            cache_file = self._get_cache_filename(full_symbol, period, interval, start_date, end_date)
            if os.path.exists(cache_file):
                logger.info(f"Loading cached data for {symbol}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            # Fetch data from yfinance
            ticker = yf.Ticker(full_symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Clean and standardize column names
            data.columns = [col.upper() for col in data.columns]
            
            # Add additional features
            data = self._add_basic_features(data)
            
            # Cache the data
            data.to_csv(cache_file)
            logger.info(f"Downloaded and cached data for {symbol}: {len(data)} records")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def get_multiple_cryptos(self, 
                           symbols: List[str], 
                           period: str = "1y",
                           interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to their OHLCV data
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data = self.get_ohlcv_data(symbol, period, interval)
                data_dict[symbol] = data
                logger.info(f"Successfully fetched data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return data_dict
    
    def get_market_data(self, 
                       symbols: List[str] = None,
                       period: str = "1y") -> pd.DataFrame:
        """
        Get market data for analysis with common timeframe.
        
        Args:
            symbols: List of symbols to fetch (default: all major cryptos)
            period: Data period
            
        Returns:
            DataFrame with market data for all symbols
        """
        if symbols is None:
            symbols = list(self.crypto_symbols.keys())
        
        data_dict = self.get_multiple_cryptos(symbols, period)
        
        # Combine data into a single DataFrame
        combined_data = pd.DataFrame()
        
        for symbol, data in data_dict.items():
            # Add symbol prefix to columns
            data_prefixed = data.copy()
            data_prefixed.columns = [f"{symbol}_{col}" for col in data_prefixed.columns]
            data_prefixed.index.name = 'Date'
            
            if combined_data.empty:
                combined_data = data_prefixed
            else:
                combined_data = combined_data.join(data_prefixed, how='outer')
        
        return combined_data
    
    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical features to the data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = data.copy()
        
        # Price changes
        df['RETURN'] = df['CLOSE'].pct_change()
        df['LOG_RETURN'] = np.log(df['CLOSE'] / df['CLOSE'].shift(1))
        
        # Volatility (rolling standard deviation of returns)
        df['VOLATILITY'] = df['RETURN'].rolling(window=20).std()
        
        # Price ranges
        df['HIGH_LOW_RATIO'] = df['HIGH'] / df['LOW']
        df['BODY_SIZE'] = abs(df['CLOSE'] - df['OPEN'])
        df['UPPER_SHADOW'] = df['HIGH'] - np.maximum(df['OPEN'], df['CLOSE'])
        df['LOWER_SHADOW'] = np.minimum(df['OPEN'], df['CLOSE']) - df['LOW']
        
        # Volume features
        df['VOLUME_MA'] = df['VOLUME'].rolling(window=20).mean()
        df['VOLUME_RATIO'] = df['VOLUME'] / df['VOLUME_MA']
        
        # Moving averages
        df['MA_20'] = df['CLOSE'].rolling(window=20).mean()
        df['MA_50'] = df['CLOSE'].rolling(window=50).mean()
        df['MA_200'] = df['CLOSE'].rolling(window=200).mean()
        
        return df
    
    def _get_cache_filename(self, 
                           symbol: str, 
                           period: str, 
                           interval: str,
                           start_date: Optional[str],
                           end_date: Optional[str]) -> str:
        """
        Generate cache filename for data.
        
        Args:
            symbol: Cryptocurrency symbol
            period: Data period
            interval: Data interval
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache filename
        """
        if start_date and end_date:
            filename = f"{symbol}_{start_date}_{end_date}_{interval}.csv"
        else:
            filename = f"{symbol}_{period}_{interval}.csv"
        
        return os.path.join(self.cache_dir, filename)
    
    def get_latest_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get the most recent data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to fetch
            
        Returns:
            DataFrame with recent data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_ohlcv_data(
            symbol, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )


def main():
    """Example usage of the data collector."""
    collector = CryptoDataCollector()
    
    # Fetch Bitcoin data
    btc_data = collector.get_ohlcv_data('BTC', period='1y')
    print(f"Bitcoin data shape: {btc_data.shape}")
    print(f"Columns: {btc_data.columns.tolist()}")
    print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
    
    # Fetch multiple cryptocurrencies
    symbols = ['BTC', 'ETH', 'ADA']
    market_data = collector.get_multiple_cryptos(symbols, period='6mo')
    
    for symbol, data in market_data.items():
        print(f"{symbol}: {len(data)} records")


if __name__ == "__main__":
    main() 