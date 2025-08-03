#!/usr/bin/env python3
"""
Backtesting Demo for Crypto Trading Analysis
============================================

This script demonstrates the comprehensive backtesting capabilities of our crypto trading analysis project.
It showcases pattern recognition, market regime detection, clustering, and combined strategies.
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
from models.backtester import Backtester, run_backtest


def main():
    """Main demonstration function."""
    print("🚀 Crypto Trading Analysis: Backtesting Framework Demo")
    print("="*80)
    print()
    
    # 1. Data Collection and Preparation
    print("📊 1. DATA COLLECTION & PREPARATION")
    print("-" * 40)
    collector = CryptoDataCollector()
    
    # Get Bitcoin data
    print("Fetching Bitcoin data...")
    btc_data = collector.get_ohlcv_data('BTC', period='1y')
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
    print()
    
    # 3. Backtesting Different Strategies
    print("🎯 3. BACKTESTING STRATEGIES")
    print("-" * 40)
    
    # Initialize backtester
    backtester = Backtester(data_with_indicators, initial_capital=10000)
    
    # Strategy 1: Pattern-based strategy
    print("\n🔍 Strategy 1: Pattern Recognition")
    print("-" * 30)
    pattern_results = backtester.run_pattern_strategy(
        pattern_threshold=0.5,
        stop_loss=0.05,
        take_profit=0.10
    )
    backtester.print_summary(pattern_results)
    
    # Strategy 2: Regime-based strategy
    print("\n📊 Strategy 2: Market Regime Detection")
    print("-" * 30)
    regime_results = backtester.run_regime_strategy(
        regime_bull='bull',
        regime_bear='bear',
        stop_loss=0.05,
        take_profit=0.10
    )
    backtester.print_summary(regime_results)
    
    # Strategy 3: Clustering-based strategy
    print("\n🎯 Strategy 3: Market Behavior Clustering")
    print("-" * 30)
    cluster_results = backtester.run_clustering_strategy(
        cluster_buy=0,
        cluster_sell=1,
        stop_loss=0.05,
        take_profit=0.10
    )
    backtester.print_summary(cluster_results)
    
    # Strategy 4: Combined strategy
    print("\n🔄 Strategy 4: Combined Approach")
    print("-" * 30)
    combined_results = backtester.run_combined_strategy(
        pattern_weight=0.4,
        regime_weight=0.4,
        cluster_weight=0.2,
        stop_loss=0.05,
        take_profit=0.10
    )
    backtester.print_summary(combined_results)
    
    # 4. Strategy Comparison
    print("\n📊 4. STRATEGY COMPARISON")
    print("-" * 40)
    
    strategies = {
        'Pattern Recognition': pattern_results,
        'Market Regime': regime_results,
        'Clustering': cluster_results,
        'Combined': combined_results
    }
    
    comparison_data = []
    for strategy_name, results in strategies.items():
        if results:
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': results.get('total_return', 0),
                'Sharpe Ratio': results.get('sharpe_ratio', 0),
                'Max Drawdown': results.get('max_drawdown', 0),
                'Win Rate': results.get('win_rate', 0),
                'Num Trades': results.get('num_trades', 0),
                'Excess Return': results.get('excess_return', 0)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # 5. Best Strategy Analysis
    print("\n🏆 5. BEST PERFORMING STRATEGY")
    print("-" * 40)
    
    if comparison_data:
        best_strategy = max(comparison_data, key=lambda x: x['Sharpe Ratio'])
        print(f"Best Strategy by Sharpe Ratio: {best_strategy['Strategy']}")
        print(f"Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")
        print(f"Total Return: {best_strategy['Total Return']:.2%}")
        print(f"Max Drawdown: {best_strategy['Max Drawdown']:.2%}")
        print(f"Win Rate: {best_strategy['Win Rate']:.2%}")
        
        # Plot the best strategy
        best_results = strategies[best_strategy['Strategy']]
        if best_results:
            print(f"\n📈 Generating visualization for {best_strategy['Strategy']}...")
            fig = backtester.plot_results(best_results)
            fig.show()
    
    # 6. Risk Analysis
    print("\n⚠️ 6. RISK ANALYSIS")
    print("-" * 40)
    
    if combined_results:
        print("Risk Metrics for Combined Strategy:")
        print(f"  - Volatility: {combined_results.get('volatility', 0):.2%}")
        print(f"  - Maximum Drawdown: {combined_results.get('max_drawdown', 0):.2%}")
        print(f"  - Profit Factor: {combined_results.get('profit_factor', 0):.2f}")
        print(f"  - Average Win: {combined_results.get('avg_win', 0):.2%}")
        print(f"  - Average Loss: {combined_results.get('avg_loss', 0):.2%}")
        
        # Calculate risk-adjusted metrics
        if combined_results.get('volatility', 0) > 0:
            calmar_ratio = combined_results.get('annualized_return', 0) / abs(combined_results.get('max_drawdown', 0))
            print(f"  - Calmar Ratio: {calmar_ratio:.3f}")
    
    # 7. Trade Analysis
    print("\n📋 7. TRADE ANALYSIS")
    print("-" * 40)
    
    if combined_results and combined_results.get('trades'):
        trades_df = pd.DataFrame(combined_results['trades'])
        print(f"Total Trades: {len(trades_df)}")
        
        if len(trades_df) > 0:
            print(f"Average Trade Duration: {(trades_df['exit_date'] - trades_df['entry_date']).mean().days:.1f} days")
            print(f"Best Trade: {trades_df['profit_pct'].max():.2%}")
            print(f"Worst Trade: {trades_df['profit_pct'].min():.2%}")
            
            # Monthly performance
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_df['month'] = trades_df['exit_date'].dt.to_period('M')
            monthly_performance = trades_df.groupby('month')['profit_pct'].sum()
            print(f"\nMonthly Performance:")
            for month, perf in monthly_performance.items():
                print(f"  {month}: {perf:.2%}")
    
    print("\n" + "="*80)
    print("✅ Backtesting Demo Complete!")
    print("="*80)


if __name__ == "__main__":
    main() 