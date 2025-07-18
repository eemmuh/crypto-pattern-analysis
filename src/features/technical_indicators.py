"""
Technical indicators for cryptocurrency analysis.
Includes momentum, trend, volatility, and volume indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Comprehensive technical analysis indicators for cryptocurrency data.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns (OPEN, HIGH, LOW, CLOSE, VOLUME)
        """
        self.data = data.copy()
        self.indicators = {}
        
    def add_all_indicators(self) -> pd.DataFrame:
        """
        Add all technical indicators to the data.
        
        Returns:
            DataFrame with all indicators added
        """
        logger.info("Adding all technical indicators...")
        
        # Trend indicators
        self._add_moving_averages()
        self._add_macd()
        
        # Momentum indicators
        self._add_rsi()
        self._add_stochastic()
        self._add_williams_r()
        
        # Volatility indicators
        self._add_bollinger_bands()
        self._add_atr()
        
        # Volume indicators
        self._add_volume_indicators()
        
        # Custom indicators
        self._add_custom_indicators()
        
        logger.info(f"Added {len(self.indicators)} technical indicators")
        return self.data
    
    def _add_moving_averages(self):
        """Add various moving averages."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            sma = SMAIndicator(close=self.data['CLOSE'], window=period)
            self.data[f'SMA_{period}'] = sma.sma_indicator()
            self.indicators[f'SMA_{period}'] = f'Simple Moving Average ({period})'
        
        # Exponential Moving Averages
        for period in [12, 26, 50, 100]:
            ema = EMAIndicator(close=self.data['CLOSE'], window=period)
            self.data[f'EMA_{period}'] = ema.ema_indicator()
            self.indicators[f'EMA_{period}'] = f'Exponential Moving Average ({period})'
        
        # Moving Average Crossovers
        self.data['SMA_CROSS_20_50'] = np.where(
            self.data['SMA_20'] > self.data['SMA_50'], 1, -1
        )
        self.data['EMA_CROSS_12_26'] = np.where(
            self.data['EMA_12'] > self.data['EMA_26'], 1, -1
        )
    
    def _add_macd(self):
        """Add MACD indicator."""
        macd = MACD(close=self.data['CLOSE'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_SIGNAL'] = macd.macd_signal()
        self.data['MACD_HISTOGRAM'] = macd.macd_diff()
        
        # MACD crossovers
        self.data['MACD_CROSS'] = np.where(
            self.data['MACD'] > self.data['MACD_SIGNAL'], 1, -1
        )
        
        self.indicators.update({
            'MACD': 'MACD Line',
            'MACD_SIGNAL': 'MACD Signal Line',
            'MACD_HISTOGRAM': 'MACD Histogram',
            'MACD_CROSS': 'MACD Crossover Signal'
        })
    
    def _add_rsi(self):
        """Add RSI indicator."""
        for period in [14, 21]:
            rsi = RSIIndicator(close=self.data['CLOSE'], window=period)
            self.data[f'RSI_{period}'] = rsi.rsi()
            self.indicators[f'RSI_{period}'] = f'Relative Strength Index ({period})'
        
        # RSI signals
        rsi_14 = self.data['RSI_14']
        self.data['RSI_OVERSOLD'] = np.where(rsi_14 < 30, 1, 0)
        self.data['RSI_OVERBOUGHT'] = np.where(rsi_14 > 70, 1, 0)
        self.data['RSI_DIVERGENCE'] = self._calculate_rsi_divergence()
    
    def _add_stochastic(self):
        """Add Stochastic Oscillator."""
        stoch = StochasticOscillator(
            high=self.data['HIGH'],
            low=self.data['LOW'],
            close=self.data['CLOSE']
        )
        self.data['STOCH_K'] = stoch.stoch()
        self.data['STOCH_D'] = stoch.stoch_signal()
        
        self.indicators.update({
            'STOCH_K': 'Stochastic %K',
            'STOCH_D': 'Stochastic %D'
        })
    
    def _add_williams_r(self):
        """Add Williams %R indicator."""
        williams_r = WilliamsRIndicator(
            high=self.data['HIGH'],
            low=self.data['LOW'],
            close=self.data['CLOSE']
        )
        self.data['WILLIAMS_R'] = williams_r.williams_r()
        self.indicators['WILLIAMS_R'] = 'Williams %R'
    
    def _add_bollinger_bands(self):
        """Add Bollinger Bands."""
        bb = BollingerBands(close=self.data['CLOSE'])
        self.data['BB_UPPER'] = bb.bollinger_hband()
        self.data['BB_MIDDLE'] = bb.bollinger_mavg()
        self.data['BB_LOWER'] = bb.bollinger_lband()
        self.data['BB_WIDTH'] = bb.bollinger_wband()
        self.data['BB_POSITION'] = bb.bollinger_pband()
        
        self.indicators.update({
            'BB_UPPER': 'Bollinger Bands Upper',
            'BB_MIDDLE': 'Bollinger Bands Middle',
            'BB_LOWER': 'Bollinger Bands Lower',
            'BB_WIDTH': 'Bollinger Bands Width',
            'BB_POSITION': 'Bollinger Bands Position'
        })
    
    def _add_atr(self):
        """Add Average True Range."""
        atr = AverageTrueRange(
            high=self.data['HIGH'],
            low=self.data['LOW'],
            close=self.data['CLOSE']
        )
        self.data['ATR'] = atr.average_true_range()
        self.indicators['ATR'] = 'Average True Range'
    
    def _add_volume_indicators(self):
        """Add volume-based indicators."""
        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(
            close=self.data['CLOSE'],
            volume=self.data['VOLUME']
        )
        self.data['OBV'] = obv.on_balance_volume()
        
        # Volume Weighted Average Price
        vwap = VolumeWeightedAveragePrice(
            high=self.data['HIGH'],
            low=self.data['LOW'],
            close=self.data['CLOSE'],
            volume=self.data['VOLUME']
        )
        self.data['VWAP'] = vwap.volume_weighted_average_price()
        
        # Volume moving averages
        self.data['VOLUME_SMA_20'] = self.data['VOLUME'].rolling(window=20).mean()
        self.data['VOLUME_RATIO'] = self.data['VOLUME'] / self.data['VOLUME_SMA_20']
        
        self.indicators.update({
            'OBV': 'On-Balance Volume',
            'VWAP': 'Volume Weighted Average Price',
            'VOLUME_RATIO': 'Volume Ratio'
        })
    
    def _add_custom_indicators(self):
        """Add custom technical indicators."""
        # Price action features
        self.data['PRICE_RANGE'] = self.data['HIGH'] - self.data['LOW']
        self.data['BODY_SIZE'] = abs(self.data['CLOSE'] - self.data['OPEN'])
        self.data['UPPER_SHADOW'] = self.data['HIGH'] - np.maximum(self.data['OPEN'], self.data['CLOSE'])
        self.data['LOWER_SHADOW'] = np.minimum(self.data['OPEN'], self.data['CLOSE']) - self.data['LOW']
        
        # Momentum features
        self.data['MOMENTUM_5'] = self.data['CLOSE'] / self.data['CLOSE'].shift(5) - 1
        self.data['MOMENTUM_10'] = self.data['CLOSE'] / self.data['CLOSE'].shift(10) - 1
        self.data['MOMENTUM_20'] = self.data['CLOSE'] / self.data['CLOSE'].shift(20) - 1
        
        # Volatility features
        self.data['VOLATILITY_5'] = self.data['RETURN'].rolling(window=5).std()
        self.data['VOLATILITY_10'] = self.data['RETURN'].rolling(window=10).std()
        self.data['VOLATILITY_20'] = self.data['RETURN'].rolling(window=20).std()
        
        # Support and Resistance levels (simplified)
        self.data['SUPPORT_20'] = self.data['LOW'].rolling(window=20).min()
        self.data['RESISTANCE_20'] = self.data['HIGH'].rolling(window=20).max()
        
        # Trend strength
        self.data['TREND_STRENGTH'] = abs(self.data['SMA_20'] - self.data['SMA_50']) / self.data['SMA_50']
        
        self.indicators.update({
            'MOMENTUM_5': '5-day Momentum',
            'MOMENTUM_10': '10-day Momentum',
            'MOMENTUM_20': '20-day Momentum',
            'VOLATILITY_5': '5-day Volatility',
            'VOLATILITY_10': '10-day Volatility',
            'VOLATILITY_20': '20-day Volatility',
            'TREND_STRENGTH': 'Trend Strength'
        })
    
    def _calculate_rsi_divergence(self) -> pd.Series:
        """
        Calculate RSI divergence (simplified version).
        
        Returns:
            Series with divergence signals
        """
        rsi = self.data['RSI_14']
        price = self.data['CLOSE']
        
        # Look for divergence over 20 periods
        divergence = pd.Series(0, index=self.data.index)
        
        for i in range(20, len(self.data)):
            # Check for bullish divergence (price lower lows, RSI higher lows)
            if (price.iloc[i] < price.iloc[i-10] and 
                rsi.iloc[i] > rsi.iloc[i-10] and 
                rsi.iloc[i] < 40):
                divergence.iloc[i] = 1
            
            # Check for bearish divergence (price higher highs, RSI lower highs)
            elif (price.iloc[i] > price.iloc[i-10] and 
                  rsi.iloc[i] < rsi.iloc[i-10] and 
                  rsi.iloc[i] > 60):
                divergence.iloc[i] = -1
        
        return divergence
    
    def get_indicator_list(self) -> Dict[str, str]:
        """
        Get list of all indicators with descriptions.
        
        Returns:
            Dictionary mapping indicator names to descriptions
        """
        return self.indicators
    
    def get_feature_matrix(self, 
                          exclude_ohlcv: bool = True,
                          exclude_nan: bool = True) -> pd.DataFrame:
        """
        Get feature matrix for machine learning.
        
        Args:
            exclude_ohlcv: Whether to exclude OHLCV columns
            exclude_nan: Whether to exclude columns with NaN values
            
        Returns:
            Feature matrix DataFrame
        """
        features = self.data.copy()
        
        if exclude_ohlcv:
            ohlcv_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
            features = features.drop(columns=ohlcv_cols, errors='ignore')
        
        if exclude_nan:
            # Remove columns with too many NaN values (>50%)
            nan_threshold = len(features) * 0.5
            features = features.dropna(axis=1, thresh=nan_threshold)
            
            # Remove rows with any remaining NaN values
            features = features.dropna()
        
        return features
    
    def get_signal_indicators(self) -> pd.DataFrame:
        """
        Get indicators that generate trading signals.
        
        Returns:
            DataFrame with signal indicators
        """
        signal_cols = [
            'MACD_CROSS', 'RSI_OVERSOLD', 'RSI_OVERBOUGHT', 'RSI_DIVERGENCE',
            'SMA_CROSS_20_50', 'EMA_CROSS_12_26', 'STOCH_K', 'STOCH_D',
            'WILLIAMS_R', 'BB_POSITION', 'VOLUME_RATIO'
        ]
        
        available_cols = [col for col in signal_cols if col in self.data.columns]
        return self.data[available_cols]


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to calculate all technical indicators.
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        DataFrame with all indicators added
    """
    ti = TechnicalIndicators(data)
    return ti.add_all_indicators()


def get_momentum_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract momentum-related features.
    
    Args:
        data: DataFrame with technical indicators
        
    Returns:
        DataFrame with momentum features
    """
    momentum_cols = [
        'RSI_14', 'RSI_21', 'STOCH_K', 'STOCH_D', 'WILLIAMS_R',
        'MOMENTUM_5', 'MOMENTUM_10', 'MOMENTUM_20', 'MACD', 'MACD_SIGNAL'
    ]
    
    available_cols = [col for col in momentum_cols if col in data.columns]
    return data[available_cols]


def get_volatility_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract volatility-related features.
    
    Args:
        data: DataFrame with technical indicators
        
    Returns:
        DataFrame with volatility features
    """
    volatility_cols = [
        'ATR', 'BB_WIDTH', 'VOLATILITY_5', 'VOLATILITY_10', 'VOLATILITY_20',
        'PRICE_RANGE', 'BODY_SIZE'
    ]
    
    available_cols = [col for col in volatility_cols if col in data.columns]
    return data[available_cols]


def get_trend_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trend-related features.
    
    Args:
        data: DataFrame with technical indicators
        
    Returns:
        DataFrame with trend features
    """
    trend_cols = [
        'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'EMA_50',
        'ADX', 'TREND_STRENGTH', 'SMA_CROSS_20_50', 'EMA_CROSS_12_26'
    ]
    
    available_cols = [col for col in trend_cols if col in data.columns]
    return data[available_cols]


if __name__ == "__main__":
    # Example usage
    from src.data.data_collector import CryptoDataCollector
    
    # Get sample data
    collector = CryptoDataCollector()
    btc_data = collector.get_ohlcv_data('BTC', period='6mo')
    
    # Calculate indicators
    ti = TechnicalIndicators(btc_data)
    data_with_indicators = ti.add_all_indicators()
    
    print(f"Original columns: {len(btc_data.columns)}")
    print(f"With indicators: {len(data_with_indicators.columns)}")
    print(f"Indicators added: {len(ti.get_indicator_list())}")
    
    # Get feature matrix
    features = ti.get_feature_matrix()
    print(f"Feature matrix shape: {features.shape}")
    
    # Get signal indicators
    signals = ti.get_signal_indicators()
    print(f"Signal indicators: {signals.columns.tolist()}") 