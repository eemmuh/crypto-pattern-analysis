#!/usr/bin/env python3
"""
Crypto Trading Analysis Demo
============================

This script demonstrates the comprehensive capabilities of our crypto trading analysis project.
It showcases pattern recognition, market behavior clustering, and regime detection.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data.data_collector import CryptoDataCollector
from features.technical_indicators import TechnicalIndicators
from patterns.candlestick_patterns import CandlestickPatternDetector
from models.clustering import MarketBehaviorClusterer
from models.hmm_model import MarketRegimeDetector


def main():
    """Main demonstration function."""
    print("🚀 Crypto Trading Analysis: Pattern Recognition & Market Behavior Clustering")
    print("="*80)
    print()
    
    # 1. Data Collection
    print("📊 1. DATA COLLECTION")
    print("-" * 40)
    collector = CryptoDataCollector()
    
    # Get Bitcoin data
    print("Fetching Bitcoin data...")
    btc_data = collector.get_ohlcv_data('BTC', period='6mo')
    print(f"✅ Collected {len(btc_data)} data points")
    print(f"   Date range: {btc_data.index[0].strftime('%Y-%m-%d')} to {btc_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Price range: ${btc_data['LOW'].min():.2f} - ${btc_data['HIGH'].max():.2f}")
    print()
    
    # 2. Technical Indicators
    print("📈 2. TECHNICAL INDICATORS")
    print("-" * 40)
    ti = TechnicalIndicators(btc_data)
    data_with_indicators = ti.add_all_indicators()
    
    print(f"✅ Calculated {len(ti.get_indicator_list())} technical indicators")
    
    # Show some key indicators
    key_indicators = ['RSI_14', 'MACD', 'BB_POSITION', 'ATR', 'VOLUME_RATIO']
    available_indicators = [col for col in key_indicators if col in data_with_indicators.columns]
    
    print("   Key indicators (latest values):")
    for indicator in available_indicators:
        value = data_with_indicators[indicator].iloc[-1]
        print(f"   - {indicator}: {value:.4f}")
    print()
    
    # 3. Candlestick Patterns
    print("🕯️ 3. CANDLESTICK PATTERNS")
    print("-" * 40)
    pattern_detector = CandlestickPatternDetector(btc_data)
    data_with_patterns = pattern_detector.detect_all_patterns()
    
    print(f"✅ Detected {len(pattern_detector.patterns)} pattern types")
    
    # Get pattern summary
    pattern_summary = pattern_detector.get_pattern_summary()
    print("   Pattern counts:")
    for pattern, count in pattern_summary.items():
        if count > 0:
            print(f"   - {pattern}: {count}")
    
    # Get bullish/bearish patterns
    bullish_patterns = pattern_detector.get_bullish_patterns()
    bearish_patterns = pattern_detector.get_bearish_patterns()
    print(f"   - Bullish patterns: {bullish_patterns.sum().sum()} instances")
    print(f"   - Bearish patterns: {bearish_patterns.sum().sum()} instances")
    print()
    
    # 4. Market Behavior Clustering
    print("🎯 4. MARKET BEHAVIOR CLUSTERING")
    print("-" * 40)
    features = ti.get_feature_matrix()
    
    clusterer = MarketBehaviorClusterer(features, n_clusters=5)
    feature_data = clusterer.prepare_features()
    
    # K-means clustering
    kmeans_results = clusterer.kmeans_clustering(feature_data)
    print(f"✅ K-means clustering completed")
    print(f"   - Clusters: {len(set(kmeans_results['labels']))}")
    print(f"   - Silhouette score: {kmeans_results['silhouette_score']:.3f}")
    print(f"   - Calinski score: {kmeans_results['calinski_score']:.3f}")
    
    # Find optimal clusters
    optimal = clusterer.find_optimal_clusters(feature_data, max_clusters=8)
    print(f"   - Optimal clusters: {optimal['optimal_k']}")
    print()
    
    # 5. Market Regime Detection
    print("🔄 5. MARKET REGIME DETECTION")
    print("-" * 40)
    regime_detector = MarketRegimeDetector(features, n_regimes=4)
    regime_feature_data = regime_detector.prepare_features()
    
    regime_results = regime_detector.detect_regimes(regime_feature_data, method='gmm')
    print(f"✅ Market regime detection completed")
    print(f"   - Regimes: {len(set(regime_results['labels']))}")
    print(f"   - AIC: {regime_results['aic']:.2f}")
    print(f"   - BIC: {regime_results['bic']:.2f}")
    
    # Get regime summary
    regime_summary = regime_detector.get_regime_summary()
    print("   Regime distribution:")
    for _, row in regime_summary.iterrows():
        print(f"   - {row['regime']}: {row['type']} ({row['percentage']:.1f}%)")
    print()
    
    # 6. Performance Analysis
    print("📊 6. PERFORMANCE ANALYSIS")
    print("-" * 40)
    returns = btc_data['CLOSE'].pct_change().dropna()
    
    performance_metrics = {
        'Total Return (%)': (btc_data['CLOSE'].iloc[-1] / btc_data['CLOSE'].iloc[0] - 1) * 100,
        'Annualized Return (%)': returns.mean() * 252 * 100,
        'Volatility (%)': returns.std() * np.sqrt(252) * 100,
        'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'Max Drawdown (%)': calculate_max_drawdown(btc_data['CLOSE']),
        'Win Rate (%)': (returns > 0).mean() * 100
    }
    
    print("   Performance metrics:")
    for metric, value in performance_metrics.items():
        print(f"   - {metric}: {value:.2f}")
    print()
    
    # 7. Summary
    print("🎉 7. ANALYSIS SUMMARY")
    print("-" * 40)
    print("✅ Successfully completed comprehensive crypto trading analysis!")
    print()
    print("📋 Key Achievements:")
    print(f"   • Collected {len(btc_data)} data points for Bitcoin")
    print(f"   • Calculated {len(ti.get_indicator_list())} technical indicators")
    print(f"   • Detected {len(pattern_detector.patterns)} candlestick pattern types")
    print(f"   • Identified {len(set(kmeans_results['labels']))} market behavior clusters")
    print(f"   • Discovered {len(set(regime_results['labels']))} market regimes")
    print()
    print("🔬 Analysis Techniques Used:")
    print("   • Technical Analysis (RSI, MACD, Bollinger Bands, etc.)")
    print("   • Pattern Recognition (Doji, Hammer, Engulfing, etc.)")
    print("   • Unsupervised Clustering (K-means, DBSCAN)")
    print("   • Hidden Markov Models (Market Regime Detection)")
    print("   • Statistical Analysis (Performance Metrics)")
    print()
    print("💡 Next Steps:")
    print("   • Run real-time analysis with live data feeds")
    print("   • Generate trading signals based on patterns")
    print("   • Implement portfolio optimization strategies")
    print("   • Add more cryptocurrencies for comparative analysis")
    print("   • Develop automated trading strategies")
    print()
    print("="*80)
    print("🎯 Project successfully demonstrates advanced crypto trading analysis!")
    print("="*80)


def calculate_max_drawdown(prices):
    """Calculate maximum drawdown."""
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100


if __name__ == "__main__":
    main() 