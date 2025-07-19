"""
Hidden Markov Models for market regime detection.
Identifies different market states (bull, bear, sideways, volatile).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detects market regimes using Hidden Markov Models.
    """
    
    def __init__(self, data: pd.DataFrame, n_regimes: int = 4):
        """
        Initialize the regime detector.
        
        Args:
            data: Market data with features
            n_regimes: Number of market regimes to detect
        """
        self.data = data.copy()
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.hmm_model = None
        self.regime_labels = None
        self.regime_probabilities = None
        self.transition_matrix = None
        self.regime_characteristics = {}
        self.feature_indices = None  # Store indices of features used for detection
        
    def prepare_features(self, 
                        feature_groups: Optional[List[str]] = None,
                        normalize: bool = True) -> pd.DataFrame:
        """
        Prepare features for regime detection.
        
        Args:
            feature_groups: List of feature groups to include
            normalize: Whether to normalize features
            
        Returns:
            Prepared feature matrix
        """
        if feature_groups is None:
            feature_groups = ['returns', 'volatility', 'momentum', 'volume']
        
        # Select features based on groups
        selected_features = []
        
        if 'returns' in feature_groups:
            return_cols = [col for col in self.data.columns if 'RETURN' in col.upper()]
            selected_features.extend(return_cols)
        
        if 'volatility' in feature_groups:
            volatility_cols = [col for col in self.data.columns if any(x in col.upper() for x in ['VOLATILITY', 'ATR', 'BB_WIDTH'])]
            selected_features.extend(volatility_cols)
        
        if 'momentum' in feature_groups:
            momentum_cols = [col for col in self.data.columns if any(x in col.upper() for x in ['RSI', 'STOCH', 'MOMENTUM', 'MACD'])]
            selected_features.extend(momentum_cols)
        
        if 'volume' in feature_groups:
            volume_cols = [col for col in self.data.columns if any(x in col.upper() for x in ['VOLUME', 'OBV', 'VWAP'])]
            selected_features.extend(volume_cols)
        
        # Remove duplicates and ensure columns exist
        selected_features = list(set(selected_features))
        available_features = [col for col in selected_features if col in self.data.columns]
        
        if not available_features:
            logger.warning("No features found, using all numeric columns")
            available_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_data = self.data[available_features].copy()
        
        # Store the indices of the data used for feature preparation
        self.feature_indices = feature_data.index
        
        # Remove rows with NaN values
        feature_data = feature_data.dropna()
        
        # Update feature indices to match the cleaned data
        self.feature_indices = feature_data.index
        
        if normalize:
            self.scaled_data = self.scaler.fit_transform(feature_data)
            return pd.DataFrame(self.scaled_data, columns=available_features, index=feature_data.index)
        
        return feature_data
    
    def detect_regimes(self, 
                      feature_data: pd.DataFrame,
                      method: str = 'gmm',
                      n_regimes: Optional[int] = None) -> Dict:
        """
        Detect market regimes using HMM-like approach.
        
        Args:
            feature_data: Feature matrix
            method: Method to use ('gmm', 'pca_gmm')
            n_regimes: Number of regimes
            
        Returns:
            Dictionary with regime detection results
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        
        logger.info(f"Detecting market regimes using {method} with {n_regimes} regimes...")
        
        # Use scaled data if available
        if self.scaled_data is not None:
            data_for_modeling = self.scaled_data
        else:
            data_for_modeling = feature_data.values
        
        if method == 'gmm':
            return self._detect_regimes_gmm(data_for_modeling, n_regimes, feature_data)
        elif method == 'pca_gmm':
            return self._detect_regimes_pca_gmm(data_for_modeling, n_regimes, feature_data)
        else:
            logger.error(f"Unknown method: {method}")
            return {}
    
    def _detect_regimes_gmm(self, 
                           data: np.ndarray, 
                           n_regimes: int,
                           feature_data: pd.DataFrame) -> Dict:
        """
        Detect regimes using Gaussian Mixture Model.
        
        Args:
            data: Scaled feature data
            n_regimes: Number of regimes
            feature_data: Original feature data
            
        Returns:
            Dictionary with regime detection results
        """
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
        labels = gmm.fit_predict(data)
        probabilities = gmm.predict_proba(data)
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(labels, n_regimes)
        
        # Store results
        self.hmm_model = gmm
        self.regime_labels = labels
        self.regime_probabilities = probabilities
        self.transition_matrix = transition_matrix
        
        # Analyze regime characteristics
        self.regime_characteristics = self._analyze_regime_characteristics(
            labels, feature_data, n_regimes
        )
        
        results = {
            'model': gmm,
            'labels': labels,
            'probabilities': probabilities,
            'transition_matrix': transition_matrix,
            'regime_characteristics': self.regime_characteristics,
            'aic': gmm.aic(data),
            'bic': gmm.bic(data),
            'feature_names': feature_data.columns.tolist()
        }
        
        logger.info(f"GMM regime detection completed. AIC: {gmm.aic(data):.2f}, BIC: {gmm.bic(data):.2f}")
        
        return results
    
    def _detect_regimes_pca_gmm(self, 
                               data: np.ndarray, 
                               n_regimes: int,
                               feature_data: pd.DataFrame) -> Dict:
        """
        Detect regimes using PCA + Gaussian Mixture Model.
        
        Args:
            data: Scaled feature data
            n_regimes: Number of regimes
            feature_data: Original feature data
            
        Returns:
            Dictionary with regime detection results
        """
        # Apply PCA for dimensionality reduction
        n_components = min(5, data.shape[1])  # Use up to 5 components
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data)
        
        # Fit Gaussian Mixture Model on PCA data
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
        labels = gmm.fit_predict(pca_data)
        probabilities = gmm.predict_proba(pca_data)
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(labels, n_regimes)
        
        # Store results
        self.hmm_model = gmm
        self.pca_model = pca
        self.regime_labels = labels
        self.regime_probabilities = probabilities
        self.transition_matrix = transition_matrix
        
        # Analyze regime characteristics
        self.regime_characteristics = self._analyze_regime_characteristics(
            labels, feature_data, n_regimes
        )
        
        results = {
            'model': gmm,
            'pca_model': pca,
            'labels': labels,
            'probabilities': probabilities,
            'transition_matrix': transition_matrix,
            'regime_characteristics': self.regime_characteristics,
            'aic': gmm.aic(pca_data),
            'bic': gmm.bic(pca_data),
            'explained_variance': pca.explained_variance_ratio_,
            'feature_names': feature_data.columns.tolist()
        }
        
        logger.info(f"PCA+GMM regime detection completed. AIC: {gmm.aic(pca_data):.2f}, BIC: {gmm.bic(pca_data):.2f}")
        
        return results
    
    def _calculate_transition_matrix(self, 
                                   labels: np.ndarray, 
                                   n_regimes: int) -> np.ndarray:
        """
        Calculate transition matrix between regimes.
        
        Args:
            labels: Regime labels
            n_regimes: Number of regimes
            
        Returns:
            Transition matrix
        """
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(labels) - 1):
            current_regime = labels[i]
            next_regime = labels[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        # Handle rows with no transitions
        transition_matrix = np.nan_to_num(transition_matrix, nan=1.0/n_regimes)
        
        return transition_matrix
    
    def _analyze_regime_characteristics(self, 
                                      labels: np.ndarray, 
                                      feature_data: pd.DataFrame,
                                      n_regimes: int) -> Dict:
        """
        Analyze characteristics of each regime.
        
        Args:
            labels: Regime labels
            feature_data: Feature data
            n_regimes: Number of regimes
            
        Returns:
            Dictionary with regime characteristics
        """
        characteristics = {}
        
        for regime in range(n_regimes):
            regime_mask = labels == regime
            regime_data = feature_data.iloc[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate regime statistics
            regime_stats = {
                'size': len(regime_data),
                'percentage': len(regime_data) / len(feature_data) * 100,
                'mean_features': regime_data.mean().to_dict(),
                'std_features': regime_data.std().to_dict(),
                'min_features': regime_data.min().to_dict(),
                'max_features': regime_data.max().to_dict()
            }
            
            # Identify regime type based on characteristics
            regime_type = self._classify_regime(regime_data)
            regime_stats['type'] = regime_type
            
            characteristics[f'regime_{regime}'] = regime_stats
        
        return characteristics
    
    def _classify_regime(self, regime_data: pd.DataFrame) -> str:
        """
        Classify regime type based on characteristics.
        
        Args:
            regime_data: Data for a specific regime
            
        Returns:
            Regime type classification
        """
        # Get key metrics
        returns = regime_data.get('RETURN', regime_data.iloc[:, 0])  # Use first column if RETURN not found
        volatility = regime_data.get('VOLATILITY', regime_data.iloc[:, 0])
        
        if len(returns) == 0:
            return 'unknown'
        
        avg_return = returns.mean()
        avg_volatility = volatility.mean()
        
        # Classify based on return and volatility
        if avg_return > 0.01:  # High positive returns
            if avg_volatility > 0.02:  # High volatility
                return 'bull_volatile'
            else:
                return 'bull_stable'
        elif avg_return < -0.01:  # High negative returns
            if avg_volatility > 0.02:  # High volatility
                return 'bear_volatile'
            else:
                return 'bear_stable'
        else:  # Low returns
            if avg_volatility > 0.02:  # High volatility
                return 'sideways_volatile'
            else:
                return 'sideways_stable'
    
    def predict_regime(self, 
                      new_data: pd.DataFrame,
                      method: str = 'gmm') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime for new data.
        
        Args:
            new_data: New feature data
            method: Method used for training
            
        Returns:
            Tuple of (labels, probabilities)
        """
        if self.hmm_model is None:
            logger.error("Model not trained yet")
            return None, None
        
        # Scale new data
        if self.scaled_data is not None:
            scaled_new_data = self.scaler.transform(new_data)
        else:
            scaled_new_data = new_data.values
        
        # Apply PCA if used during training
        if method == 'pca_gmm' and hasattr(self, 'pca_model'):
            scaled_new_data = self.pca_model.transform(scaled_new_data)
        
        # Predict
        labels = self.hmm_model.predict(scaled_new_data)
        probabilities = self.hmm_model.predict_proba(scaled_new_data)
        
        return labels, probabilities
    
    def get_regime_summary(self) -> pd.DataFrame:
        """
        Get summary of detected regimes.
        
        Returns:
            DataFrame with regime summary
        """
        if not self.regime_characteristics:
            return pd.DataFrame()
        
        summary_data = []
        
        for regime_id, characteristics in self.regime_characteristics.items():
            summary = {
                'regime': regime_id,
                'type': characteristics.get('type', 'unknown'),
                'size': characteristics['size'],
                'percentage': characteristics['percentage']
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def get_aligned_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get price data aligned with regime detection results.
        
        Args:
            price_data: Original price data
            
        Returns:
            Aligned price data
        """
        if self.feature_indices is not None:
            return price_data.loc[self.feature_indices]
        else:
            return price_data.iloc[:len(self.regime_labels)] if self.regime_labels is not None else price_data
    
    def visualize_regimes(self, 
                         price_data: pd.DataFrame,
                         method: str = 'pca') -> go.Figure:
        """
        Visualize detected regimes.
        
        Args:
            price_data: Price data for visualization
            method: Visualization method
            
        Returns:
            Plotly figure with regime visualization
        """
        if self.regime_labels is None:
            logger.error("No regime labels available")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price with Regimes', 'Regime Probabilities'),
            vertical_spacing=0.1
        )
        
        # Use the helper method to get aligned data
        aligned_price_data = self.get_aligned_data(price_data)
        
        # Plot 1: Price with regime colors
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for regime in range(self.n_regimes):
            regime_mask = self.regime_labels == regime
            regime_dates = aligned_price_data.index[regime_mask]
            regime_prices = aligned_price_data['CLOSE'].iloc[regime_mask]
            
            fig.add_trace(
                go.Scatter(
                    x=regime_dates,
                    y=regime_prices,
                    mode='markers',
                    marker=dict(color=colors[regime % len(colors)], size=4),
                    name=f'Regime {regime}',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot 2: Regime probabilities
        if self.regime_probabilities is not None:
            for regime in range(self.n_regimes):
                fig.add_trace(
                    go.Scatter(
                        x=aligned_price_data.index,
                        y=self.regime_probabilities[:, regime],
                        mode='lines',
                        name=f'Regime {regime} Prob',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title=f"Market Regime Detection ({method.upper()})",
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    def plot_transition_matrix(self) -> go.Figure:
        """
        Plot transition matrix between regimes.
        
        Returns:
            Plotly figure with transition matrix
        """
        if self.transition_matrix is None:
            logger.error("No transition matrix available")
            return go.Figure()
        
        fig = go.Figure(data=go.Heatmap(
            z=self.transition_matrix,
            x=[f'Regime {i}' for i in range(self.n_regimes)],
            y=[f'Regime {i}' for i in range(self.n_regimes)],
            colorscale='Blues',
            text=np.round(self.transition_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Regime Transition Matrix",
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            template="plotly_white"
        )
        
        return fig


def detect_market_regimes(data: pd.DataFrame, 
                         n_regimes: int = 4,
                         method: str = 'gmm') -> MarketRegimeDetector:
    """
    Convenience function to detect market regimes.
    
    Args:
        data: Feature matrix
        n_regimes: Number of regimes
        method: Detection method
        
    Returns:
        MarketRegimeDetector instance with results
    """
    detector = MarketRegimeDetector(data, n_regimes)
    
    # Prepare features
    feature_data = detector.prepare_features()
    
    # Detect regimes
    detector.detect_regimes(feature_data, method)
    
    return detector


if __name__ == "__main__":
    # Example usage
    from src.data.data_collector import CryptoDataCollector
    from src.features.technical_indicators import TechnicalIndicators
    
    # Get sample data
    collector = CryptoDataCollector()
    btc_data = collector.get_ohlcv_data('BTC', period='1y')
    
    # Calculate technical indicators
    ti = TechnicalIndicators(btc_data)
    data_with_indicators = ti.add_all_indicators()
    
    # Get feature matrix
    features = ti.get_feature_matrix()
    
    # Detect market regimes
    detector = detect_market_regimes(features, n_regimes=4, method='gmm')
    
    # Analyze results
    print("Regime Summary:")
    print(detector.get_regime_summary())
    
    # Get regime characteristics
    print(f"\nRegime characteristics: {len(detector.regime_characteristics)} regimes found")
    
    # Visualize regimes
    fig = detector.visualize_regimes(btc_data)
    fig.show() 