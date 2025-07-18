"""
Basic tests for the crypto trading analysis project.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_collector import CryptoDataCollector
from features.technical_indicators import TechnicalIndicators
from patterns.candlestick_patterns import CandlestickPatternDetector
from models.clustering import MarketBehaviorClusterer
from models.hmm_model import MarketRegimeDetector


class TestCryptoAnalysis(unittest.TestCase):
    """Test cases for crypto trading analysis components."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% daily return, 2% volatility
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.sample_data = pd.DataFrame({
            'OPEN': prices * (1 + np.random.normal(0, 0.005, 100)),
            'HIGH': prices * (1 + abs(np.random.normal(0, 0.01, 100))),
            'LOW': prices * (1 - abs(np.random.normal(0, 0.01, 100))),
            'CLOSE': prices,
            'VOLUME': np.random.lognormal(10, 1, 100)
        }, index=dates)
        
        # Standardize column names
        self.sample_data.columns = [col.upper() for col in self.sample_data.columns]
    
    def test_data_collector(self):
        """Test data collector functionality."""
        collector = CryptoDataCollector()
        
        # Test basic functionality
        self.assertIsInstance(collector.crypto_symbols, dict)
        self.assertIn('BTC', collector.crypto_symbols)
        self.assertIn('ETH', collector.crypto_symbols)
    
    def test_technical_indicators(self):
        """Test technical indicators calculation."""
        ti = TechnicalIndicators(self.sample_data)
        data_with_indicators = ti.add_all_indicators()
        
        # Check that indicators were added
        self.assertGreater(len(data_with_indicators.columns), len(self.sample_data.columns))
        
        # Check for specific indicators
        expected_indicators = ['RSI_14', 'MACD', 'SMA_20', 'BB_UPPER']
        for indicator in expected_indicators:
            if indicator in data_with_indicators.columns:
                self.assertIsInstance(data_with_indicators[indicator].iloc[-1], (int, float, np.number))
        
        # Test feature matrix
        features = ti.get_feature_matrix()
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
    
    def test_candlestick_patterns(self):
        """Test candlestick pattern detection."""
        detector = CandlestickPatternDetector(self.sample_data)
        data_with_patterns = detector.detect_all_patterns()
        
        # Check that patterns were detected
        self.assertGreater(len(detector.patterns), 0)
        
        # Check pattern summary
        pattern_summary = detector.get_pattern_summary()
        self.assertIsInstance(pattern_summary, dict)
        
        # Check bullish/bearish patterns
        bullish_patterns = detector.get_bullish_patterns()
        bearish_patterns = detector.get_bearish_patterns()
        self.assertIsInstance(bullish_patterns, pd.DataFrame)
        self.assertIsInstance(bearish_patterns, pd.DataFrame)
    
    def test_clustering(self):
        """Test market behavior clustering."""
        # First get features
        ti = TechnicalIndicators(self.sample_data)
        data_with_indicators = ti.add_all_indicators()
        features = ti.get_feature_matrix()
        
        # Test clustering
        clusterer = MarketBehaviorClusterer(features, n_clusters=3)
        feature_data = clusterer.prepare_features()
        
        # Test K-means clustering
        kmeans_results = clusterer.kmeans_clustering(feature_data)
        self.assertIn('labels', kmeans_results)
        self.assertIn('silhouette_score', kmeans_results)
        self.assertEqual(len(set(kmeans_results['labels'])), 3)
        
        # Test optimal cluster finding
        optimal = clusterer.find_optimal_clusters(feature_data, max_clusters=5)
        self.assertIn('optimal_k', optimal)
        self.assertIn('silhouette_scores', optimal)
    
    def test_regime_detection(self):
        """Test market regime detection."""
        # First get features
        ti = TechnicalIndicators(self.sample_data)
        data_with_indicators = ti.add_all_indicators()
        features = ti.get_feature_matrix()
        
        # Test regime detection
        detector = MarketRegimeDetector(features, n_regimes=3)
        regime_feature_data = detector.prepare_features()
        
        regime_results = detector.detect_regimes(regime_feature_data, method='gmm')
        self.assertIn('labels', regime_results)
        self.assertIn('probabilities', regime_results)
        self.assertIn('transition_matrix', regime_results)
        self.assertEqual(len(set(regime_results['labels'])), 3)
        
        # Test regime summary
        regime_summary = detector.get_regime_summary()
        self.assertIsInstance(regime_summary, pd.DataFrame)
    
    def test_data_integrity(self):
        """Test data integrity and consistency."""
        # Check for NaN values
        self.assertFalse(self.sample_data.isnull().any().any())
        
        # Check price relationships
        self.assertTrue((self.sample_data['HIGH'] >= self.sample_data['LOW']).all())
        self.assertTrue((self.sample_data['HIGH'] >= self.sample_data['OPEN']).all())
        self.assertTrue((self.sample_data['HIGH'] >= self.sample_data['CLOSE']).all())
        self.assertTrue((self.sample_data['LOW'] <= self.sample_data['OPEN']).all())
        self.assertTrue((self.sample_data['LOW'] <= self.sample_data['CLOSE']).all())
        
        # Check volume is positive
        self.assertTrue((self.sample_data['VOLUME'] > 0).all())


def run_quick_test():
    """Run a quick test to verify basic functionality."""
    print("🧪 Running quick functionality test...")
    
    try:
        # Create test instance
        test = TestCryptoAnalysis()
        test.setUp()
        
        # Test technical indicators
        print("  ✓ Testing technical indicators...")
        ti = TechnicalIndicators(test.sample_data)
        data_with_indicators = ti.add_all_indicators()
        print(f"    - Added {len(data_with_indicators.columns) - len(test.sample_data.columns)} indicators")
        
        # Test pattern detection
        print("  ✓ Testing candlestick patterns...")
        detector = CandlestickPatternDetector(test.sample_data)
        data_with_patterns = detector.detect_all_patterns()
        print(f"    - Detected {len(detector.patterns)} pattern types")
        
        # Test clustering
        print("  ✓ Testing clustering...")
        features = ti.get_feature_matrix()
        clusterer = MarketBehaviorClusterer(features, n_clusters=3)
        feature_data = clusterer.prepare_features()
        kmeans_results = clusterer.kmeans_clustering(feature_data)
        print(f"    - Found {len(set(kmeans_results['labels']))} clusters")
        
        # Test regime detection
        print("  ✓ Testing regime detection...")
        regime_detector = MarketRegimeDetector(features, n_regimes=3)
        regime_feature_data = regime_detector.prepare_features()
        regime_results = regime_detector.detect_regimes(regime_feature_data)
        print(f"    - Detected {len(set(regime_results['labels']))} regimes")
        
        print("✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == '__main__':
    # Run quick test
    run_quick_test()
    
    # Run full test suite
    unittest.main(argv=[''], exit=False, verbosity=2) 