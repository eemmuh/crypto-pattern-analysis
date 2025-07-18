"""
Candlestick pattern recognition for cryptocurrency analysis.
Identifies common candlestick patterns and their reliability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CandlestickPatternDetector:
    """
    Detects various candlestick patterns in OHLCV data.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns (OPEN, HIGH, LOW, CLOSE, VOLUME)
        """
        self.data = data.copy()
        self.patterns = {}
        
    def detect_all_patterns(self) -> pd.DataFrame:
        """
        Detect all candlestick patterns.
        
        Returns:
            DataFrame with pattern signals added
        """
        logger.info("Detecting candlestick patterns...")
        
        # Single candlestick patterns
        self._detect_doji_patterns()
        self._detect_hammer_patterns()
        self._detect_shooting_star_patterns()
        self._detect_marubozu_patterns()
        
        # Two candlestick patterns
        self._detect_engulfing_patterns()
        self._detect_harami_patterns()
        self._detect_dark_cloud_cover()
        self._detect_piercing_pattern()
        
        # Three candlestick patterns
        self._detect_morning_star()
        self._detect_evening_star()
        self._detect_three_white_soldiers()
        self._detect_three_black_crows()
        
        # Multi-candlestick patterns
        self._detect_head_and_shoulders()
        self._detect_double_top_bottom()
        self._detect_triangle_patterns()
        
        logger.info(f"Detected {len(self.patterns)} pattern types")
        return self.data
    
    def _detect_doji_patterns(self):
        """Detect various doji patterns."""
        # Standard doji (open and close are very close)
        body_size = abs(self.data['CLOSE'] - self.data['OPEN'])
        avg_body = body_size.rolling(window=20).mean()
        doji_threshold = avg_body * 0.1
        
        self.data['DOJI'] = np.where(body_size <= doji_threshold, 1, 0)
        
        # Long-legged doji (long shadows, small body)
        shadow_length = self.data['HIGH'] - self.data['LOW']
        self.data['LONG_LEGGED_DOJI'] = np.where(
            (body_size <= doji_threshold) & (shadow_length > avg_body * 2), 1, 0
        )
        
        # Dragonfly doji (no upper shadow, long lower shadow)
        upper_shadow = self.data['HIGH'] - np.maximum(self.data['OPEN'], self.data['CLOSE'])
        lower_shadow = np.minimum(self.data['OPEN'], self.data['CLOSE']) - self.data['LOW']
        
        self.data['DRAGONFLY_DOJI'] = np.where(
            (body_size <= doji_threshold) & 
            (upper_shadow <= doji_threshold) & 
            (lower_shadow > avg_body), 1, 0
        )
        
        # Gravestone doji (no lower shadow, long upper shadow)
        self.data['GRAVESTONE_DOJI'] = np.where(
            (body_size <= doji_threshold) & 
            (lower_shadow <= doji_threshold) & 
            (upper_shadow > avg_body), 1, 0
        )
        
        self.patterns.update({
            'DOJI': 'Standard Doji',
            'LONG_LEGGED_DOJI': 'Long-legged Doji',
            'DRAGONFLY_DOJI': 'Dragonfly Doji',
            'GRAVESTONE_DOJI': 'Gravestone Doji'
        })
    
    def _detect_hammer_patterns(self):
        """Detect hammer and hanging man patterns."""
        body_size = abs(self.data['CLOSE'] - self.data['OPEN'])
        upper_shadow = self.data['HIGH'] - np.maximum(self.data['OPEN'], self.data['CLOSE'])
        lower_shadow = np.minimum(self.data['OPEN'], self.data['CLOSE']) - self.data['LOW']
        
        # Hammer (small body, long lower shadow, small upper shadow)
        self.data['HAMMER'] = np.where(
            (body_size < upper_shadow * 0.5) & 
            (lower_shadow > body_size * 2) & 
            (upper_shadow < body_size * 0.5), 1, 0
        )
        
        # Hanging man (hammer in downtrend)
        trend = self.data['CLOSE'].rolling(window=20).mean()
        self.data['HANGING_MAN'] = np.where(
            (self.data['HAMMER'] == 1) & 
            (self.data['CLOSE'] < trend), 1, 0
        )
        
        self.patterns.update({
            'HAMMER': 'Hammer Pattern',
            'HANGING_MAN': 'Hanging Man Pattern'
        })
    
    def _detect_shooting_star_patterns(self):
        """Detect shooting star and inverted hammer patterns."""
        body_size = abs(self.data['CLOSE'] - self.data['OPEN'])
        upper_shadow = self.data['HIGH'] - np.maximum(self.data['OPEN'], self.data['CLOSE'])
        lower_shadow = np.minimum(self.data['OPEN'], self.data['CLOSE']) - self.data['LOW']
        
        # Shooting star (small body, long upper shadow, small lower shadow)
        self.data['SHOOTING_STAR'] = np.where(
            (body_size < lower_shadow * 0.5) & 
            (upper_shadow > body_size * 2) & 
            (lower_shadow < body_size * 0.5), 1, 0
        )
        
        # Inverted hammer (shooting star in downtrend)
        trend = self.data['CLOSE'].rolling(window=20).mean()
        self.data['INVERTED_HAMMER'] = np.where(
            (self.data['SHOOTING_STAR'] == 1) & 
            (self.data['CLOSE'] < trend), 1, 0
        )
        
        self.patterns.update({
            'SHOOTING_STAR': 'Shooting Star Pattern',
            'INVERTED_HAMMER': 'Inverted Hammer Pattern'
        })
    
    def _detect_marubozu_patterns(self):
        """Detect marubozu patterns (no shadows)."""
        body_size = abs(self.data['CLOSE'] - self.data['OPEN'])
        upper_shadow = self.data['HIGH'] - np.maximum(self.data['OPEN'], self.data['CLOSE'])
        lower_shadow = np.minimum(self.data['OPEN'], self.data['CLOSE']) - self.data['LOW']
        
        # Marubozu (no shadows, strong trend)
        self.data['MARUBOZU'] = np.where(
            (upper_shadow < body_size * 0.1) & 
            (lower_shadow < body_size * 0.1), 1, 0
        )
        
        # Bullish marubozu (green marubozu)
        self.data['BULLISH_MARUBOZU'] = np.where(
            (self.data['MARUBOZU'] == 1) & 
            (self.data['CLOSE'] > self.data['OPEN']), 1, 0
        )
        
        # Bearish marubozu (red marubozu)
        self.data['BEARISH_MARUBOZU'] = np.where(
            (self.data['MARUBOZU'] == 1) & 
            (self.data['CLOSE'] < self.data['OPEN']), 1, 0
        )
        
        self.patterns.update({
            'MARUBOZU': 'Marubozu Pattern',
            'BULLISH_MARUBOZU': 'Bullish Marubozu',
            'BEARISH_MARUBOZU': 'Bearish Marubozu'
        })
    
    def _detect_engulfing_patterns(self):
        """Detect bullish and bearish engulfing patterns."""
        # Bullish engulfing (current green candle engulfs previous red candle)
        prev_open = self.data['OPEN'].shift(1)
        prev_close = self.data['CLOSE'].shift(1)
        curr_open = self.data['OPEN']
        curr_close = self.data['CLOSE']
        
        self.data['BULLISH_ENGULFING'] = np.where(
            (prev_close < prev_open) &  # Previous red candle
            (curr_close > curr_open) &  # Current green candle
            (curr_open < prev_close) &  # Current open below previous close
            (curr_close > prev_open),   # Current close above previous open
            1, 0
        )
        
        # Bearish engulfing (current red candle engulfs previous green candle)
        self.data['BEARISH_ENGULFING'] = np.where(
            (prev_close > prev_open) &  # Previous green candle
            (curr_close < curr_open) &  # Current red candle
            (curr_open > prev_close) &  # Current open above previous close
            (curr_close < prev_open),   # Current close below previous open
            1, 0
        )
        
        self.patterns.update({
            'BULLISH_ENGULFING': 'Bullish Engulfing Pattern',
            'BEARISH_ENGULFING': 'Bearish Engulfing Pattern'
        })
    
    def _detect_harami_patterns(self):
        """Detect bullish and bearish harami patterns."""
        prev_open = self.data['OPEN'].shift(1)
        prev_close = self.data['CLOSE'].shift(1)
        curr_open = self.data['OPEN']
        curr_close = self.data['CLOSE']
        
        # Bullish harami (small green candle inside previous red candle)
        self.data['BULLISH_HARAMI'] = np.where(
            (prev_close < prev_open) &  # Previous red candle
            (curr_close > curr_open) &  # Current green candle
            (curr_open > prev_close) &  # Current open above previous close
            (curr_close < prev_open),   # Current close below previous open
            1, 0
        )
        
        # Bearish harami (small red candle inside previous green candle)
        self.data['BEARISH_HARAMI'] = np.where(
            (prev_close > prev_open) &  # Previous green candle
            (curr_close < curr_open) &  # Current red candle
            (curr_open < prev_close) &  # Current open below previous close
            (curr_close > prev_open),   # Current close above previous open
            1, 0
        )
        
        self.patterns.update({
            'BULLISH_HARAMI': 'Bullish Harami Pattern',
            'BEARISH_HARAMI': 'Bearish Harami Pattern'
        })
    
    def _detect_dark_cloud_cover(self):
        """Detect dark cloud cover pattern."""
        prev_open = self.data['OPEN'].shift(1)
        prev_close = self.data['CLOSE'].shift(1)
        curr_open = self.data['OPEN']
        curr_close = self.data['CLOSE']
        
        # Dark cloud cover (red candle opens above previous high, closes below midpoint)
        prev_high = self.data['HIGH'].shift(1)
        midpoint = (prev_open + prev_close) / 2
        
        self.data['DARK_CLOUD_COVER'] = np.where(
            (prev_close > prev_open) &  # Previous green candle
            (curr_close < curr_open) &  # Current red candle
            (curr_open > prev_high) &   # Current open above previous high
            (curr_close < midpoint),    # Current close below midpoint
            1, 0
        )
        
        self.patterns['DARK_CLOUD_COVER'] = 'Dark Cloud Cover Pattern'
    
    def _detect_piercing_pattern(self):
        """Detect piercing pattern."""
        prev_open = self.data['OPEN'].shift(1)
        prev_close = self.data['CLOSE'].shift(1)
        curr_open = self.data['OPEN']
        curr_close = self.data['CLOSE']
        
        # Piercing pattern (green candle opens below previous low, closes above midpoint)
        prev_low = self.data['LOW'].shift(1)
        midpoint = (prev_open + prev_close) / 2
        
        self.data['PIERCING_PATTERN'] = np.where(
            (prev_close < prev_open) &  # Previous red candle
            (curr_close > curr_open) &  # Current green candle
            (curr_open < prev_low) &    # Current open below previous low
            (curr_close > midpoint),    # Current close above midpoint
            1, 0
        )
        
        self.patterns['PIERCING_PATTERN'] = 'Piercing Pattern'
    
    def _detect_morning_star(self):
        """Detect morning star pattern."""
        # Morning star: red candle, small candle (any color), green candle
        prev2_close = self.data['CLOSE'].shift(2)
        prev2_open = self.data['OPEN'].shift(2)
        prev_close = self.data['CLOSE'].shift(1)
        prev_open = self.data['OPEN'].shift(1)
        curr_close = self.data['CLOSE']
        curr_open = self.data['OPEN']
        
        # Small middle candle
        middle_body = abs(prev_close - prev_open)
        avg_body = abs(self.data['CLOSE'] - self.data['OPEN']).rolling(window=20).mean()
        
        self.data['MORNING_STAR'] = np.where(
            (prev2_close < prev2_open) &  # First red candle
            (middle_body < avg_body * 0.5) &  # Small middle candle
            (curr_close > curr_open) &  # Third green candle
            (curr_close > (prev2_open + prev2_close) / 2),  # Close above first candle midpoint
            1, 0
        )
        
        self.patterns['MORNING_STAR'] = 'Morning Star Pattern'
    
    def _detect_evening_star(self):
        """Detect evening star pattern."""
        prev2_close = self.data['CLOSE'].shift(2)
        prev2_open = self.data['OPEN'].shift(2)
        prev_close = self.data['CLOSE'].shift(1)
        prev_open = self.data['OPEN'].shift(1)
        curr_close = self.data['CLOSE']
        curr_open = self.data['OPEN']
        
        middle_body = abs(prev_close - prev_open)
        avg_body = abs(self.data['CLOSE'] - self.data['OPEN']).rolling(window=20).mean()
        
        self.data['EVENING_STAR'] = np.where(
            (prev2_close > prev2_open) &  # First green candle
            (middle_body < avg_body * 0.5) &  # Small middle candle
            (curr_close < curr_open) &  # Third red candle
            (curr_close < (prev2_open + prev2_close) / 2),  # Close below first candle midpoint
            1, 0
        )
        
        self.patterns['EVENING_STAR'] = 'Evening Star Pattern'
    
    def _detect_three_white_soldiers(self):
        """Detect three white soldiers pattern."""
        # Three consecutive green candles with higher opens
        prev2_close = self.data['CLOSE'].shift(2)
        prev2_open = self.data['OPEN'].shift(2)
        prev_close = self.data['CLOSE'].shift(1)
        prev_open = self.data['OPEN'].shift(1)
        curr_close = self.data['CLOSE']
        curr_open = self.data['OPEN']
        
        self.data['THREE_WHITE_SOLDIERS'] = np.where(
            (prev2_close > prev2_open) &  # First green candle
            (prev_close > prev_open) &    # Second green candle
            (curr_close > curr_open) &    # Third green candle
            (prev_open > prev2_open) &    # Higher opens
            (curr_open > prev_open),
            1, 0
        )
        
        self.patterns['THREE_WHITE_SOLDIERS'] = 'Three White Soldiers Pattern'
    
    def _detect_three_black_crows(self):
        """Detect three black crows pattern."""
        # Three consecutive red candles with lower opens
        prev2_close = self.data['CLOSE'].shift(2)
        prev2_open = self.data['OPEN'].shift(2)
        prev_close = self.data['CLOSE'].shift(1)
        prev_open = self.data['OPEN'].shift(1)
        curr_close = self.data['CLOSE']
        curr_open = self.data['OPEN']
        
        self.data['THREE_BLACK_CROWS'] = np.where(
            (prev2_close < prev2_open) &  # First red candle
            (prev_close < prev_open) &    # Second red candle
            (curr_close < curr_open) &    # Third red candle
            (prev_open < prev2_open) &    # Lower opens
            (curr_open < prev_open),
            1, 0
        )
        
        self.patterns['THREE_BLACK_CROWS'] = 'Three Black Crows Pattern'
    
    def _detect_head_and_shoulders(self):
        """Detect head and shoulders pattern (simplified)."""
        # This is a simplified version - full implementation would be more complex
        window = 20
        
        # Look for three peaks with middle peak higher
        highs = self.data['HIGH'].rolling(window=window, center=True).max()
        
        # Simplified detection - in practice would need more sophisticated peak detection
        self.data['HEAD_AND_SHOULDERS'] = 0  # Placeholder
        
        self.patterns['HEAD_AND_SHOULDERS'] = 'Head and Shoulders Pattern'
    
    def _detect_double_top_bottom(self):
        """Detect double top and double bottom patterns."""
        window = 20
        
        # Double top (two peaks at similar levels)
        highs = self.data['HIGH'].rolling(window=window, center=True).max()
        
        # Simplified detection
        self.data['DOUBLE_TOP'] = 0  # Placeholder
        self.data['DOUBLE_BOTTOM'] = 0  # Placeholder
        
        self.patterns.update({
            'DOUBLE_TOP': 'Double Top Pattern',
            'DOUBLE_BOTTOM': 'Double Bottom Pattern'
        })
    
    def _detect_triangle_patterns(self):
        """Detect triangle patterns (ascending, descending, symmetrical)."""
        # Simplified triangle detection
        window = 20
        
        # Calculate trend lines
        highs = self.data['HIGH'].rolling(window=window).max()
        lows = self.data['LOW'].rolling(window=window).min()
        
        # Placeholder for triangle detection
        self.data['ASCENDING_TRIANGLE'] = 0
        self.data['DESCENDING_TRIANGLE'] = 0
        self.data['SYMMETRICAL_TRIANGLE'] = 0
        
        self.patterns.update({
            'ASCENDING_TRIANGLE': 'Ascending Triangle Pattern',
            'DESCENDING_TRIANGLE': 'Descending Triangle Pattern',
            'SYMMETRICAL_TRIANGLE': 'Symmetrical Triangle Pattern'
        })
    
    def get_pattern_signals(self) -> pd.DataFrame:
        """
        Get all pattern signals as a DataFrame.
        
        Returns:
            DataFrame with pattern signals
        """
        pattern_cols = list(self.patterns.keys())
        available_cols = [col for col in pattern_cols if col in self.data.columns]
        return self.data[available_cols]
    
    def get_bullish_patterns(self) -> pd.DataFrame:
        """
        Get bullish pattern signals.
        
        Returns:
            DataFrame with bullish patterns
        """
        bullish_patterns = [
            'HAMMER', 'BULLISH_ENGULFING', 'BULLISH_HARAMI', 'PIERCING_PATTERN',
            'MORNING_STAR', 'THREE_WHITE_SOLDIERS', 'BULLISH_MARUBOZU'
        ]
        
        available_cols = [col for col in bullish_patterns if col in self.data.columns]
        return self.data[available_cols]
    
    def get_bearish_patterns(self) -> pd.DataFrame:
        """
        Get bearish pattern signals.
        
        Returns:
            DataFrame with bearish patterns
        """
        bearish_patterns = [
            'HANGING_MAN', 'SHOOTING_STAR', 'BEARISH_ENGULFING', 'BEARISH_HARAMI',
            'DARK_CLOUD_COVER', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'BEARISH_MARUBOZU'
        ]
        
        available_cols = [col for col in bearish_patterns if col in self.data.columns]
        return self.data[available_cols]
    
    def get_reversal_patterns(self) -> pd.DataFrame:
        """
        Get reversal pattern signals.
        
        Returns:
            DataFrame with reversal patterns
        """
        reversal_patterns = [
            'DOJI', 'LONG_LEGGED_DOJI', 'DRAGONFLY_DOJI', 'GRAVESTONE_DOJI',
            'HAMMER', 'HANGING_MAN', 'SHOOTING_STAR', 'INVERTED_HAMMER'
        ]
        
        available_cols = [col for col in reversal_patterns if col in self.data.columns]
        return self.data[available_cols]
    
    def get_pattern_summary(self) -> Dict[str, int]:
        """
        Get summary of detected patterns.
        
        Returns:
            Dictionary with pattern counts
        """
        pattern_cols = list(self.patterns.keys())
        available_cols = [col for col in pattern_cols if col in self.data.columns]
        
        summary = {}
        for col in available_cols:
            summary[self.patterns[col]] = self.data[col].sum()
        
        return summary


def detect_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to detect all candlestick patterns.
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        DataFrame with pattern signals added
    """
    detector = CandlestickPatternDetector(data)
    return detector.detect_all_patterns()


if __name__ == "__main__":
    # Example usage
    from src.data.data_collector import CryptoDataCollector
    
    # Get sample data
    collector = CryptoDataCollector()
    btc_data = collector.get_ohlcv_data('BTC', period='6mo')
    
    # Detect patterns
    detector = CandlestickPatternDetector(btc_data)
    data_with_patterns = detector.detect_all_patterns()
    
    print(f"Patterns detected: {len(detector.patterns)}")
    print(f"Pattern summary: {detector.get_pattern_summary()}")
    
    # Get pattern signals
    bullish_patterns = detector.get_bullish_patterns()
    bearish_patterns = detector.get_bearish_patterns()
    
    print(f"Bullish patterns found: {bullish_patterns.sum().sum()}")
    print(f"Bearish patterns found: {bearish_patterns.sum().sum()}") 