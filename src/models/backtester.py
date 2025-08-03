"""
Backtesting framework for crypto trading strategies.
Tests pattern recognition, regime detection, and clustering signals against historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from ..patterns.candlestick_patterns import CandlestickPatternDetector
from ..models.hmm_model import MarketRegimeDetector
from ..models.clustering import MarketBehaviorClusterer
from ..features.technical_indicators import TechnicalIndicators


class Backtester:
    """
    Comprehensive backtesting framework for crypto trading strategies.
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000):
        """
        Initialize backtester.
        
        Args:
            data: OHLCV data with technical indicators and signals
            initial_capital: Starting capital for backtesting
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trades = []
        self.portfolio_values = []
        self.current_position = 0  # 0 = no position, 1 = long, -1 = short
        
    def run_pattern_strategy(self, 
                           pattern_threshold: float = 0.5,
                           stop_loss: float = 0.05,
                           take_profit: float = 0.10) -> Dict:
        """
        Run backtest using candlestick pattern signals.
        
        Args:
            pattern_threshold: Minimum confidence for pattern signals
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            
        Returns:
            Dictionary with backtest results
        """
        print("Running pattern-based strategy backtest...")
        
        # Detect patterns if not already present
        if not any(col.endswith('_PATTERN') for col in self.data.columns):
            pattern_detector = CandlestickPatternDetector(self.data)
            self.data = pattern_detector.detect_all_patterns()
        
        # Get pattern signals
        pattern_cols = [col for col in self.data.columns if col.endswith('_PATTERN')]
        bullish_patterns = [col for col in pattern_cols if 'BULLISH' in col or col in ['HAMMER', 'MORNING_STAR', 'PIERCING']]
        bearish_patterns = [col for col in pattern_cols if 'BEARISH' in col or col in ['SHOOTING_STAR', 'EVENING_STAR', 'DARK_CLOUD']]
        
        # Calculate signal strength
        self.data['BULLISH_SIGNAL'] = self.data[bullish_patterns].sum(axis=1)
        self.data['BEARISH_SIGNAL'] = self.data[bearish_patterns].sum(axis=1)
        self.data['SIGNAL_STRENGTH'] = self.data['BULLISH_SIGNAL'] - self.data['BEARISH_SIGNAL']
        
        # Generate trading signals
        self.data['SIGNAL'] = 0
        self.data.loc[self.data['SIGNAL_STRENGTH'] >= pattern_threshold, 'SIGNAL'] = 1  # Buy
        self.data.loc[self.data['SIGNAL_STRENGTH'] <= -pattern_threshold, 'SIGNAL'] = -1  # Sell
        
        return self._execute_trades(stop_loss, take_profit)
    
    def run_regime_strategy(self, 
                          regime_bull: str = 'bull',
                          regime_bear: str = 'bear',
                          stop_loss: float = 0.05,
                          take_profit: float = 0.10) -> Dict:
        """
        Run backtest using market regime signals.
        
        Args:
            regime_bull: Name of bullish regime
            regime_bear: Name of bearish regime
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            
        Returns:
            Dictionary with backtest results
        """
        print("Running regime-based strategy backtest...")
        
        # Detect regimes if not already present
        if 'REGIME' not in self.data.columns:
            regime_detector = MarketRegimeDetector(self.data)
            features = regime_detector.prepare_features()
            results = regime_detector.detect_regimes(features)
            self.data['REGIME'] = results['labels']
        
        # Generate trading signals based on regimes
        self.data['SIGNAL'] = 0
        self.data.loc[self.data['REGIME'] == regime_bull, 'SIGNAL'] = 1  # Buy in bull regime
        self.data.loc[self.data['REGIME'] == regime_bear, 'SIGNAL'] = -1  # Sell in bear regime
        
        return self._execute_trades(stop_loss, take_profit)
    
    def run_clustering_strategy(self, 
                              cluster_buy: int = 0,
                              cluster_sell: int = 1,
                              stop_loss: float = 0.05,
                              take_profit: float = 0.10) -> Dict:
        """
        Run backtest using clustering signals.
        
        Args:
            cluster_buy: Cluster number for buy signals
            cluster_sell: Cluster number for sell signals
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            
        Returns:
            Dictionary with backtest results
        """
        print("Running clustering-based strategy backtest...")
        
        # Perform clustering if not already present
        if 'CLUSTER' not in self.data.columns:
            ti = TechnicalIndicators(self.data)
            features = ti.get_feature_matrix()
            clusterer = MarketBehaviorClusterer(features, n_clusters=5)
            feature_data = clusterer.prepare_features()
            results = clusterer.kmeans_clustering(feature_data)
            self.data['CLUSTER'] = results['labels']
        
        # Generate trading signals based on clusters
        self.data['SIGNAL'] = 0
        self.data.loc[self.data['CLUSTER'] == cluster_buy, 'SIGNAL'] = 1  # Buy in favorable cluster
        self.data.loc[self.data['CLUSTER'] == cluster_sell, 'SIGNAL'] = -1  # Sell in unfavorable cluster
        
        return self._execute_trades(stop_loss, take_profit)
    
    def run_combined_strategy(self, 
                            pattern_weight: float = 0.4,
                            regime_weight: float = 0.4,
                            cluster_weight: float = 0.2,
                            stop_loss: float = 0.05,
                            take_profit: float = 0.10) -> Dict:
        """
        Run backtest using combined signals from all strategies.
        
        Args:
            pattern_weight: Weight for pattern signals
            regime_weight: Weight for regime signals
            cluster_weight: Weight for clustering signals
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            
        Returns:
            Dictionary with backtest results
        """
        print("Running combined strategy backtest...")
        
        # Ensure all signals are available
        if 'SIGNAL' not in self.data.columns:
            self.run_pattern_strategy()
        
        # Get regime signals
        if 'REGIME' not in self.data.columns:
            regime_detector = MarketRegimeDetector(self.data)
            features = regime_detector.prepare_features()
            results = regime_detector.detect_regimes(features)
            self.data['REGIME'] = results['labels']
        
        # Get cluster signals
        if 'CLUSTER' not in self.data.columns:
            ti = TechnicalIndicators(self.data)
            features = ti.get_feature_matrix()
            clusterer = MarketBehaviorClusterer(features, n_clusters=5)
            feature_data = clusterer.prepare_features()
            results = clusterer.kmeans_clustering(feature_data)
            self.data['CLUSTER'] = results['labels']
        
        # Calculate combined signal
        self.data['COMBINED_SIGNAL'] = (
            self.data['SIGNAL'] * pattern_weight +
            np.where(self.data['REGIME'] == 'bull', 1, 
                    np.where(self.data['REGIME'] == 'bear', -1, 0)) * regime_weight +
            np.where(self.data['CLUSTER'] == 0, 1, 
                    np.where(self.data['CLUSTER'] == 1, -1, 0)) * cluster_weight
        )
        
        # Generate final signals
        self.data['SIGNAL'] = np.where(self.data['COMBINED_SIGNAL'] > 0.3, 1,
                                      np.where(self.data['COMBINED_SIGNAL'] < -0.3, -1, 0))
        
        return self._execute_trades(stop_loss, take_profit)
    
    def _execute_trades(self, stop_loss: float, take_profit: float) -> Dict:
        """
        Execute trades based on signals and calculate performance.
        
        Args:
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            
        Returns:
            Dictionary with backtest results
        """
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.portfolio_values = []
        self.current_position = 0
        entry_price = 0
        entry_date = None
        
        for i, row in self.data.iterrows():
            current_price = row['CLOSE']
            
            # Check for stop loss or take profit
            if self.current_position != 0:
                if self.current_position == 1:  # Long position
                    if current_price <= entry_price * (1 - stop_loss) or current_price >= entry_price * (1 + take_profit):
                        # Close position
                        profit = (current_price - entry_price) / entry_price
                        self.current_capital *= (1 + profit)
                        self.trades.append({
                            'entry_date': entry_date,
                            'exit_date': i,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': 'LONG',
                            'profit_pct': profit,
                            'profit_abs': entry_price * profit
                        })
                        self.current_position = 0
                        entry_price = 0
                        entry_date = None
            
            # Check for new signals
            if self.current_position == 0 and row['SIGNAL'] != 0:
                if row['SIGNAL'] == 1:  # Buy signal
                    self.current_position = 1
                    entry_price = current_price
                    entry_date = i
                    self.positions.append({
                        'date': i,
                        'action': 'BUY',
                        'price': current_price,
                        'capital': self.current_capital
                    })
                elif row['SIGNAL'] == -1:  # Sell signal (short)
                    self.current_position = -1
                    entry_price = current_price
                    entry_date = i
                    self.positions.append({
                        'date': i,
                        'action': 'SELL',
                        'price': current_price,
                        'capital': self.current_capital
                    })
            
            # Record portfolio value
            portfolio_value = self.current_capital
            if self.current_position != 0:
                if self.current_position == 1:
                    portfolio_value = self.current_capital * (current_price / entry_price)
                else:  # Short position
                    portfolio_value = self.current_capital * (2 - current_price / entry_price)
            
            self.portfolio_values.append({
                'date': i,
                'value': portfolio_value,
                'price': current_price
            })
        
        # Close any remaining position
        if self.current_position != 0:
            final_price = self.data.iloc[-1]['CLOSE']
            if self.current_position == 1:
                profit = (final_price - entry_price) / entry_price
            else:
                profit = (entry_price - final_price) / entry_price
            
            self.current_capital *= (1 + profit)
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': self.data.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position': 'LONG' if self.current_position == 1 else 'SHORT',
                'profit_pct': profit,
                'profit_abs': entry_price * profit
            })
        
        return self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.portfolio_values:
            return {}
        
        # Convert to DataFrame for easier calculations
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        # Calculate metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        annualized_return = total_return * (252 / len(portfolio_df))
        volatility = portfolio_df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        portfolio_df['cumulative'] = (1 + portfolio_df['returns']).cumprod()
        portfolio_df['running_max'] = portfolio_df['cumulative'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['cumulative'] - portfolio_df['running_max']) / portfolio_df['running_max']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Calculate trade statistics
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            win_rate = (trades_df['profit_pct'] > 0).mean()
            avg_win = trades_df[trades_df['profit_pct'] > 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['profit_pct'] < 0]['profit_pct'].mean() if len(trades_df[trades_df['profit_pct'] < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Calculate buy-and-hold comparison
        buy_hold_return = (self.data['CLOSE'].iloc[-1] - self.data['CLOSE'].iloc[0]) / self.data['CLOSE'].iloc[0]
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': len(self.trades),
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'final_capital': self.current_capital,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'positions': self.positions
        }
    
    def plot_results(self, results: Dict) -> go.Figure:
        """
        Create interactive plot of backtest results.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Plotly figure with backtest visualization
        """
        if not results or not results.get('portfolio_values'):
            return go.Figure()
        
        portfolio_df = pd.DataFrame(results['portfolio_values'])
        portfolio_df.set_index('date', inplace=True)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Value', 'Price and Trades', 'Drawdown'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Price and trades
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['CLOSE'],
                mode='lines',
                name='Price',
                line=dict(color='gray', opacity=0.7)
            ),
            row=2, col=1
        )
        
        # Add trade markers
        if results.get('trades'):
            for trade in results['trades']:
                # Entry point
                fig.add_trace(
                    go.Scatter(
                        x=[trade['entry_date']],
                        y=[trade['entry_price']],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up' if trade['position'] == 'LONG' else 'triangle-down',
                            size=10,
                            color='green' if trade['position'] == 'LONG' else 'red'
                        ),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Exit point
                fig.add_trace(
                    go.Scatter(
                        x=[trade['exit_date']],
                        y=[trade['exit_price']],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color='black'
                        ),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Drawdown
        portfolio_df['cumulative'] = (1 + portfolio_df['value'].pct_change()).cumprod()
        portfolio_df['running_max'] = portfolio_df['cumulative'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['cumulative'] - portfolio_df['running_max']) / portfolio_df['running_max']
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['drawdown'] * 100,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red'),
                fill='tonexty'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Backtest Results - Total Return: {results.get("total_return", 0):.2%}',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def print_summary(self, results: Dict):
        """
        Print comprehensive backtest summary.
        
        Args:
            results: Backtest results dictionary
        """
        if not results:
            print("No results to display.")
            return
        
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${results.get('final_capital', 0):,.2f}")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
        print(f"Volatility: {results.get('volatility', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"Buy & Hold Return: {results.get('buy_hold_return', 0):.2%}")
        print(f"Excess Return: {results.get('excess_return', 0):.2%}")
        
        print(f"\nTRADING STATISTICS:")
        print(f"Number of Trades: {results.get('num_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Average Win: {results.get('avg_win', 0):.2%}")
        print(f"Average Loss: {results.get('avg_loss', 0):.2%}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        
        print("="*60)


def run_backtest(data: pd.DataFrame, 
                strategy: str = 'combined',
                initial_capital: float = 10000,
                **kwargs) -> Dict:
    """
    Convenience function to run backtests.
    
    Args:
        data: OHLCV data with technical indicators
        strategy: Strategy to use ('pattern', 'regime', 'clustering', 'combined')
        initial_capital: Starting capital
        **kwargs: Additional strategy parameters
        
    Returns:
        Backtest results dictionary
    """
    backtester = Backtester(data, initial_capital)
    
    if strategy == 'pattern':
        return backtester.run_pattern_strategy(**kwargs)
    elif strategy == 'regime':
        return backtester.run_regime_strategy(**kwargs)
    elif strategy == 'clustering':
        return backtester.run_clustering_strategy(**kwargs)
    elif strategy == 'combined':
        return backtester.run_combined_strategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}") 