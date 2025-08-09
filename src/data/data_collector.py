"""
Data collection module for cryptocurrency market data.
Supports multiple data sources and timeframes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import os
import time
import requests
import glob

# Import project utilities
try:
    from ..utils.logging_config import get_logger, log_execution_time
    from ..utils.error_handling import (
        DataCollectionError, ValidationError, NetworkError, 
        RetryableError, NonRetryableError, retry_on_failure, 
        handle_exceptions, validate_data, log_error_with_context
    )
    from ...config.settings import DATA_SETTINGS, API_SETTINGS
except ImportError:
    # Fallback for direct execution
    import logging
    logger = logging.getLogger(__name__)
    
    # Define fallback settings
    DATA_SETTINGS = {
        'cache_dir': 'data/cache',
        'default_period': '1y',
        'default_interval': '1d',
        'supported_symbols': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
    }
    
    API_SETTINGS = {
        'yfinance_timeout': 30,
        'max_retries': 3,
        'retry_delay': 1,
        'rate_limit': 100,
        'user_agent': 'CryptoAnalysis/1.0'
    }
    
    # Define fallback error classes
    class DataCollectionError(Exception): pass
    class ValidationError(Exception): pass
    class NetworkError(Exception): pass
    class RetryableError(Exception): pass
    class NonRetryableError(Exception): pass
    
    def retry_on_failure(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def log_execution_time(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def log_error_with_context(error, context=None, level="ERROR"):
        logger.error(f"Error: {error} | Context: {context}")
    
    def get_logger(name):
        return logging.getLogger(name)

# Get logger
logger = get_logger(__name__)


class CryptoDataCollector:
    """
    Collects cryptocurrency market data from various sources.
    
    This class provides a robust interface for fetching cryptocurrency market data
    from Yahoo Finance API with caching, error handling, and data validation.
    
    Features:
    - Automatic caching of downloaded data
    - Comprehensive error handling with retry logic
    - Data quality validation and cleaning
    - Support for multiple cryptocurrencies
    - Configurable time periods and intervals
    - Cache management utilities
    
    Attributes:
        cache_dir (str): Directory for caching downloaded data
        crypto_symbols (Dict[str, str]): Mapping of symbol codes to full symbols
        max_retries (int): Maximum number of retry attempts for API calls
        retry_delay (float): Delay between retry attempts
        timeout (int): Timeout for API requests
    
    Example:
        >>> collector = CryptoDataCollector()
        >>> btc_data = collector.get_ohlcv_data('BTC', period='1y')
        >>> print(btc_data.shape)
        (365, 15)
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the data collector.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or DATA_SETTINGS['cache_dir']
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Common cryptocurrency symbols from settings
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
            'UNI': 'UNI-USD',
            'XRP': 'XRP-USD',
            'LTC': 'LTC-USD',
            'BCH': 'BCH-USD',
            'XLM': 'XLM-USD'
        }
        
        # API settings
        self.max_retries = API_SETTINGS['max_retries']
        self.retry_delay = API_SETTINGS['retry_delay']
        self.timeout = API_SETTINGS['yfinance_timeout']
    
    @log_execution_time()
    @retry_on_failure(
        max_retries=3,
        delay=1.0,
        backoff_factor=2.0,
        exceptions=(RetryableError, ConnectionError, TimeoutError)
    )
    def get_ohlcv_data(self, 
                       symbol: str, 
                       period: str = None,
                       interval: str = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       max_retries: int = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValidationError: If symbol, period, or interval is invalid
            DataCollectionError: If unable to collect data
            NetworkError: If network connection fails
        """
        # Use defaults from settings if not provided
        period = period or DATA_SETTINGS['default_period']
        interval = interval or DATA_SETTINGS['default_interval']
        max_retries = max_retries or self.max_retries
        
        # Validate inputs
        self._validate_symbol(symbol)
        self._validate_period(period)
        self._validate_interval(interval)
        self._validate_dates(start_date, end_date)
        self._validate_retries(max_retries)
        
        # Get the full symbol for yfinance
        full_symbol = self.crypto_symbols.get(symbol.upper(), f"{symbol.upper()}-USD")
        
        # Check cache first
        cache_file = self._get_cache_filename(full_symbol, period, interval, start_date, end_date)
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data for {symbol}")
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Successfully loaded cached data for {symbol}: {len(data)} records")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_file)
                except:
                    pass
        
        # Try to fetch data
        logger.info(f"Fetching data for {symbol} from API")
        
        try:
            # Fetch data from yfinance
            ticker = yf.Ticker(full_symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise DataCollectionError(f"No data found for {symbol}")
            
            # Clean and standardize column names
            data.columns = [col.upper() for col in data.columns]
            
            # Validate data quality
            self._validate_data_quality(data, symbol)
            
            # Clean the data
            data = self._clean_data(data)
            
            # Add additional features
            data = self._add_basic_features(data)
            
            # Cache the data
            try:
                data.to_csv(cache_file)
                logger.info(f"Downloaded and cached data for {symbol}: {len(data)} records")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
            
            return data
            
        except Exception as e:
            context = {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date
            }
            log_error_with_context(e, context)
            
            if isinstance(e, (ConnectionError, TimeoutError)):
                raise RetryableError(f"Network error for {symbol}: {str(e)}")
            else:
                raise DataCollectionError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def _validate_symbol(self, symbol: str) -> None:
        """
        Validate cryptocurrency symbol.
        
        Args:
            symbol: Symbol to validate
            
        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        if len(symbol) > 10:  # Reasonable max length for crypto symbols
            raise ValidationError("Symbol is too long")
    
    def _validate_period(self, period: str) -> None:
        """
        Validate data period.
        
        Args:
            period: Period to validate
            
        Raises:
            ValidationError: If period is invalid
        """
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if period not in valid_periods:
            raise ValidationError(f"Invalid period '{period}'. Must be one of: {valid_periods}")
    
    def _validate_interval(self, interval: str) -> None:
        """
        Validate data interval.
        
        Args:
            interval: Interval to validate
            
        Raises:
            ValidationError: If interval is invalid
        """
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval not in valid_intervals:
            raise ValidationError(f"Invalid interval '{interval}'. Must be one of: {valid_intervals}")
    
    def _validate_dates(self, start_date: Optional[str], end_date: Optional[str]) -> None:
        """
        Validate date parameters.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Raises:
            ValidationError: If dates are invalid
        """
        if start_date and end_date:
            try:
                from datetime import datetime
                datetime.strptime(start_date, '%Y-%m-%d')
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise ValidationError("Dates must be in 'YYYY-MM-DD' format")
            
            if start_date >= end_date:
                raise ValidationError("Start date must be before end date")
    
    def _validate_retries(self, max_retries: int) -> None:
        """
        Validate retry parameter.
        
        Args:
            max_retries: Number of retries
            
        Raises:
            ValidationError: If max_retries is invalid
        """
        if not isinstance(max_retries, int) or max_retries < 1 or max_retries > 10:
            raise ValidationError("max_retries must be an integer between 1 and 10")
    
    def _get_sample_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Generate sample data when API fails.
        
        Args:
            symbol: Cryptocurrency symbol
            period: Data period
            
        Returns:
            DataFrame with sample data
        """
        logger.info(f"Generating sample data for {symbol}")
        
        # Determine number of days based on period
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650
        }
        
        days = period_days.get(period, 365)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate sample data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Generate realistic price data
        base_price = 50000 if symbol.upper() == 'BTC' else 3000 if symbol.upper() == 'ETH' else 100
        np.random.seed(42)  # For reproducible results
        
        # Generate price series with some trend and volatility
        returns = np.random.normal(0.001, 0.03, n_days)  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.1))  # Prevent negative prices
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'OPEN': prices,
            'HIGH': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'LOW': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'CLOSE': prices,
            'VOLUME': np.random.lognormal(15, 0.5, n_days)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        data['HIGH'] = data[['OPEN', 'HIGH', 'CLOSE']].max(axis=1)
        data['LOW'] = data[['OPEN', 'LOW', 'CLOSE']].min(axis=1)
        
        # Add basic features
        data = self._add_basic_features(data)
        
        logger.info(f"Generated sample data for {symbol}: {len(data)} records")
        return data
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available cryptocurrency symbols.
        
        Returns:
            List of available symbols
        """
        return list(self.crypto_symbols.keys())
    
    def is_symbol_supported(self, symbol: str) -> bool:
        """
        Check if a symbol is supported.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if symbol is supported
        """
        return symbol.upper() in self.crypto_symbols
    
    @log_execution_time()
    def get_multiple_cryptos(self, 
                           symbols: List[str], 
                           period: str = None,
                           interval: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to their OHLCV data
        """
        # Use defaults from settings if not provided
        period = period or DATA_SETTINGS['default_period']
        interval = interval or DATA_SETTINGS['default_interval']
        
        data_dict = {}
        failed_symbols = []
        
        logger.info(f"Fetching data for {len(symbols)} symbols: {symbols}")
        
        for symbol in symbols:
            try:
                data = self.get_ohlcv_data(symbol, period, interval)
                data_dict[symbol] = data
                logger.info(f"Successfully fetched data for {symbol}: {len(data)} records")
            except Exception as e:
                failed_symbols.append(symbol)
                context = {'symbol': symbol, 'period': period, 'interval': interval}
                log_error_with_context(e, context, "WARNING")
                continue
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
        
        logger.info(f"Successfully fetched data for {len(data_dict)} out of {len(symbols)} symbols")
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

    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Validate the quality and integrity of fetched data.
        
        Args:
            data: DataFrame to validate
            symbol: Symbol for context
            
        Raises:
            ValidationError: If data quality issues are found
        """
        if data.empty:
            raise ValidationError(f"Empty dataset for {symbol}")
        
        # Check for required columns
        required_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # Check for negative prices
        price_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        for col in price_columns:
            if (data[col] < 0).any():
                raise ValidationError(f"Negative prices found in {col} column for {symbol}")
        
        # Check OHLC relationships
        invalid_high = (data['HIGH'] < data[['OPEN', 'CLOSE']].max(axis=1)).any()
        invalid_low = (data['LOW'] > data[['OPEN', 'CLOSE']].min(axis=1)).any()
        
        if invalid_high or invalid_low:
            raise ValidationError(f"Invalid OHLC relationships for {symbol}")
        
        # Check for excessive missing values
        missing_threshold = 0.1  # 10% missing data threshold
        for col in required_columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            if missing_ratio > missing_threshold:
                logger.warning(f"High missing data ratio in {col} for {symbol}: {missing_ratio:.2%}")
        
        # Check for suspicious volume (zero or extremely high)
        zero_volume_ratio = (data['VOLUME'] == 0).sum() / len(data)
        if zero_volume_ratio > 0.5:
            logger.warning(f"High ratio of zero volume for {symbol}: {zero_volume_ratio:.2%}")
        
        logger.info(f"Data quality validation passed for {symbol}: {len(data)} records")

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Forward fill small gaps in data
        df = df.fillna(method='ffill', limit=3)
        
        # Remove outliers in volume (extremely high values)
        volume_q99 = df['VOLUME'].quantile(0.99)
        df.loc[df['VOLUME'] > volume_q99 * 10, 'VOLUME'] = volume_q99
        
        # Ensure OHLC relationships are valid
        df['HIGH'] = df[['OPEN', 'HIGH', 'CLOSE']].max(axis=1)
        df['LOW'] = df[['OPEN', 'LOW', 'CLOSE']].min(axis=1)
        
        return df

    def clear_cache(self, symbol: str = None) -> None:
        """
        Clear cache files.
        
        Args:
            symbol: Specific symbol to clear (if None, clears all)
        """
        if symbol:
            # Clear cache for specific symbol
            pattern = f"{symbol.upper()}-USD_*.csv"
            cache_files = glob.glob(os.path.join(self.cache_dir, pattern))
            for file in cache_files:
                try:
                    os.remove(file)
                    logger.info(f"Removed cache file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {file}: {e}")
        else:
            # Clear all cache files
            cache_files = glob.glob(os.path.join(self.cache_dir, "*.csv"))
            for file in cache_files:
                try:
                    os.remove(file)
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {file}: {e}")
            logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache information
        """
        cache_files = glob.glob(os.path.join(self.cache_dir, "*.csv"))
        
        cache_info = {
            'total_files': len(cache_files),
            'total_size_mb': 0,
            'symbols': set(),
            'file_details': []
        }
        
        for file in cache_files:
            try:
                file_size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
                cache_info['total_size_mb'] += file_size
                
                filename = os.path.basename(file)
                symbol = filename.split('_')[0]
                cache_info['symbols'].add(symbol)
                
                cache_info['file_details'].append({
                    'filename': filename,
                    'size_mb': file_size,
                    'symbol': symbol
                })
            except Exception as e:
                logger.warning(f"Failed to get info for cache file {file}: {e}")
        
        cache_info['symbols'] = list(cache_info['symbols'])
        return cache_info
    
    def is_cache_valid(self, symbol: str, period: str, interval: str, 
                      start_date: Optional[str] = None, end_date: Optional[str] = None,
                      max_age_hours: int = 24) -> bool:
        """
        Check if cached data is still valid (not too old).
        
        Args:
            symbol: Cryptocurrency symbol
            period: Data period
            interval: Data interval
            start_date: Start date
            end_date: End date
            max_age_hours: Maximum age in hours for cache to be valid
            
        Returns:
            True if cache is valid
        """
        cache_file = self._get_cache_filename(symbol, period, interval, start_date, end_date)
        
        if not os.path.exists(cache_file):
            return False
        
        # Check file age
        file_age = time.time() - os.path.getmtime(cache_file)
        max_age_seconds = max_age_hours * 3600
        
        return file_age < max_age_seconds


def main():
    """Example usage of the data collector."""
    logger.info("Starting CryptoDataCollector demo")
    
    collector = CryptoDataCollector()
    
    # Show available symbols
    available_symbols = collector.get_available_symbols()
    logger.info(f"Available symbols: {available_symbols}")
    
    # Check cache info
    cache_info = collector.get_cache_info()
    logger.info(f"Cache info: {cache_info['total_files']} files, {cache_info['total_size_mb']:.2f} MB")
    
    # Fetch Bitcoin data
    try:
        btc_data = collector.get_ohlcv_data('BTC', period='1y')
        logger.info(f"Bitcoin data shape: {btc_data.shape}")
        logger.info(f"Columns: {btc_data.columns.tolist()}")
        logger.info(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
    except Exception as e:
        logger.error(f"Failed to fetch Bitcoin data: {e}")
    
    # Fetch multiple cryptocurrencies
    symbols = ['BTC', 'ETH', 'ADA']
    try:
        market_data = collector.get_multiple_cryptos(symbols, period='6mo')
        
        for symbol, data in market_data.items():
            logger.info(f"{symbol}: {len(data)} records")
    except Exception as e:
        logger.error(f"Failed to fetch multiple cryptos: {e}")
    
    # Test cache validation
    is_valid = collector.is_cache_valid('BTC', '1y', '1d')
    logger.info(f"BTC cache valid: {is_valid}")
    
    logger.info("CryptoDataCollector demo completed")


if __name__ == "__main__":
    main() 