"""
Main analysis script for crypto trading pattern recognition and market behavior clustering.
Demonstrates all capabilities of the project.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from data.data_collector import CryptoDataCollector
from features.technical_indicators import TechnicalIndicators
from patterns.candlestick_patterns import CandlestickPatternDetector
from models.clustering import MarketBehaviorClusterer
from models.hmm_model import MarketRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_comprehensive_analysis(symbols: list = None, period: str = "1y"):
    """
    Run comprehensive crypto trading analysis.
    
    Args:
        symbols: List of cryptocurrency symbols to analyze
        period: Data period to fetch
    """
    if symbols is None:
        symbols = ['BTC', 'ETH', 'ADA']
    
    logger.info("Starting comprehensive crypto trading analysis...")
    logger.info(f"Analyzing symbols: {symbols}")
    logger.info(f"Data period: {period}")
    
    # Initialize data collector
    collector = CryptoDataCollector()
    
    # Collect data for all symbols
    logger.info("Collecting market data...")
    market_data = collector.get_multiple_cryptos(symbols, period)
    
    results = {}
    
    for symbol in symbols:
        if symbol not in market_data:
            logger.warning(f"No data available for {symbol}")
            continue
            
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {symbol}")
        logger.info(f"{'='*50}")
        
        symbol_data = market_data[symbol]
        symbol_results = analyze_symbol(symbol, symbol_data)
        results[symbol] = symbol_results
    
    # Generate summary report
    generate_summary_report(results)
    
    return results


def analyze_symbol(symbol: str, data: pd.DataFrame) -> dict:
    """
    Perform comprehensive analysis for a single symbol.
    
    Args:
        symbol: Cryptocurrency symbol
        data: OHLCV data
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        'symbol': symbol,
        'data_points': len(data),
        'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    }
    
    # 1. Technical Indicators Analysis
    logger.info("1. Calculating technical indicators...")
    ti = TechnicalIndicators(data)
    data_with_indicators = ti.add_all_indicators()
    
    results['technical_indicators'] = {
        'total_indicators': len(ti.get_indicator_list()),
        'indicator_list': ti.get_indicator_list()
    }
    
    # 2. Candlestick Pattern Detection
    logger.info("2. Detecting candlestick patterns...")
    pattern_detector = CandlestickPatternDetector(data)
    data_with_patterns = pattern_detector.detect_all_patterns()
    
    pattern_summary = pattern_detector.get_pattern_summary()
    results['patterns'] = {
        'total_patterns': len(pattern_detector.patterns),
        'pattern_summary': pattern_summary,
        'bullish_patterns': pattern_detector.get_bullish_patterns().sum().sum(),
        'bearish_patterns': pattern_detector.get_bearish_patterns().sum().sum()
    }
    
    # 3. Market Behavior Clustering
    logger.info("3. Performing market behavior clustering...")
    features = ti.get_feature_matrix()
    
    clusterer = MarketBehaviorClusterer(features, n_clusters=5)
    feature_data = clusterer.prepare_features()
    
    # K-means clustering
    kmeans_results = clusterer.kmeans_clustering(feature_data)
    results['clustering'] = {
        'kmeans_silhouette': kmeans_results['silhouette_score'],
        'kmeans_clusters': len(set(kmeans_results['labels']))
    }
    
    # Find optimal clusters
    optimal = clusterer.find_optimal_clusters(feature_data)
    results['clustering']['optimal_clusters'] = optimal['optimal_k']
    
    # 4. Market Regime Detection
    logger.info("4. Detecting market regimes...")
    regime_detector = MarketRegimeDetector(features, n_regimes=4)
    regime_feature_data = regime_detector.prepare_features()
    
    regime_results = regime_detector.detect_regimes(regime_feature_data, method='gmm')
    results['regimes'] = {
        'n_regimes': len(set(regime_results['labels'])),
        'aic': regime_results['aic'],
        'bic': regime_results['bic']
    }
    
    # 5. Performance Metrics
    logger.info("5. Calculating performance metrics...")
    performance_metrics = calculate_performance_metrics(data)
    results['performance'] = performance_metrics
    
    logger.info(f"Analysis completed for {symbol}")
    
    return results


def calculate_performance_metrics(data: pd.DataFrame) -> dict:
    """
    Calculate key performance metrics.
    
    Args:
        data: OHLCV data
        
    Returns:
        Dictionary with performance metrics
    """
    returns = data['CLOSE'].pct_change().dropna()
    
    metrics = {
        'total_return': (data['CLOSE'].iloc[-1] / data['CLOSE'].iloc[0] - 1) * 100,
        'annualized_return': returns.mean() * 252 * 100,
        'volatility': returns.std() * np.sqrt(252) * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown(data['CLOSE']),
        'win_rate': (returns > 0).mean() * 100
    }
    
    return metrics


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Maximum drawdown percentage
    """
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100


def generate_summary_report(results: dict):
    """
    Generate a summary report of all analysis results.
    
    Args:
        results: Dictionary with analysis results
    """
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE ANALYSIS SUMMARY REPORT")
    logger.info("="*80)
    
    for symbol, result in results.items():
        logger.info(f"\n📊 {symbol.upper()} ANALYSIS SUMMARY")
        logger.info(f"   Data Points: {result['data_points']}")
        logger.info(f"   Date Range: {result['date_range']}")
        
        # Technical Indicators
        ti_info = result['technical_indicators']
        logger.info(f"   Technical Indicators: {ti_info['total_indicators']} calculated")
        
        # Patterns
        pattern_info = result['patterns']
        logger.info(f"   Candlestick Patterns: {pattern_info['total_patterns']} types detected")
        logger.info(f"   Bullish Patterns: {pattern_info['bullish_patterns']} instances")
        logger.info(f"   Bearish Patterns: {pattern_info['bearish_patterns']} instances")
        
        # Clustering
        cluster_info = result['clustering']
        logger.info(f"   Market Clusters: {cluster_info['kmeans_clusters']} (Silhouette: {cluster_info['kmeans_silhouette']:.3f})")
        logger.info(f"   Optimal Clusters: {cluster_info['optimal_clusters']}")
        
        # Regimes
        regime_info = result['regimes']
        logger.info(f"   Market Regimes: {regime_info['n_regimes']} detected")
        logger.info(f"   Model AIC: {regime_info['aic']:.2f}, BIC: {regime_info['bic']:.2f}")
        
        # Performance
        perf_info = result['performance']
        logger.info(f"   Total Return: {perf_info['total_return']:.2f}%")
        logger.info(f"   Annualized Return: {perf_info['annualized_return']:.2f}%")
        logger.info(f"   Volatility: {perf_info['volatility']:.2f}%")
        logger.info(f"   Sharpe Ratio: {perf_info['sharpe_ratio']:.3f}")
        logger.info(f"   Max Drawdown: {perf_info['max_drawdown']:.2f}%")
        logger.info(f"   Win Rate: {perf_info['win_rate']:.1f}%")
    
    logger.info("\n" + "="*80)
    logger.info("Analysis completed successfully!")
    logger.info("="*80)


def run_quick_analysis(symbol: str = 'BTC', period: str = "6mo"):
    """
    Run a quick analysis for demonstration purposes.
    
    Args:
        symbol: Cryptocurrency symbol
        period: Data period
    """
    logger.info(f"Running quick analysis for {symbol}...")
    
    # Get data
    collector = CryptoDataCollector()
    data = collector.get_ohlcv_data(symbol, period)
    
    # Calculate indicators
    ti = TechnicalIndicators(data)
    data_with_indicators = ti.add_all_indicators()
    
    # Detect patterns
    pattern_detector = CandlestickPatternDetector(data)
    data_with_patterns = pattern_detector.detect_all_patterns()
    
    # Get features for ML
    features = ti.get_feature_matrix()
    
    # Quick clustering
    clusterer = MarketBehaviorClusterer(features, n_clusters=3)
    feature_data = clusterer.prepare_features()
    kmeans_results = clusterer.kmeans_clustering(feature_data)
    
    # Quick regime detection
    regime_detector = MarketRegimeDetector(features, n_regimes=3)
    regime_feature_data = regime_detector.prepare_features()
    regime_results = regime_detector.detect_regimes(regime_feature_data)
    
    logger.info(f"Quick analysis completed for {symbol}")
    logger.info(f"  - {len(ti.get_indicator_list())} technical indicators calculated")
    logger.info(f"  - {len(pattern_detector.patterns)} pattern types detected")
    logger.info(f"  - {len(set(kmeans_results['labels']))} market clusters found")
    logger.info(f"  - {len(set(regime_results['labels']))} market regimes detected")
    
    return {
        'data': data,
        'indicators': data_with_indicators,
        'patterns': data_with_patterns,
        'clustering': kmeans_results,
        'regimes': regime_results
    }


def main():
    """Main function to run the analysis."""
    print("🚀 Crypto Trading Analysis: Pattern Recognition & Market Behavior Clustering")
    print("="*80)
    
    # Run quick analysis for demonstration
    try:
        results = run_quick_analysis('BTC', '6mo')
        print("\n✅ Quick analysis completed successfully!")
        print("📊 Check the logs above for detailed results.")
        print("\n💡 To run comprehensive analysis, call:")
        print("   run_comprehensive_analysis(['BTC', 'ETH', 'ADA'], '1y')")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\n❌ Analysis failed: {str(e)}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main() 