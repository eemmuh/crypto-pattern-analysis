"""
Market behavior clustering using various algorithms.
Groups similar market conditions and behaviors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Some clustering methods will be disabled.")

try:
    from tslearn.clustering import TimeSeriesKMeans, silhouette_score as ts_silhouette_score
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    logger.warning("tslearn not available. Time series clustering will be disabled.")


class MarketBehaviorClusterer:
    """
    Clusters market behavior patterns using various algorithms.
    """
    
    def __init__(self, data: pd.DataFrame, n_clusters: int = 5):
        """
        Initialize the clusterer.
        
        Args:
            data: Feature matrix for clustering
            n_clusters: Number of clusters for K-means
        """
        self.data = data.copy()
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.clusters = {}
        self.cluster_labels = {}
        
    def prepare_features(self, 
                        feature_groups: Optional[List[str]] = None,
                        normalize: bool = True) -> pd.DataFrame:
        """
        Prepare features for clustering.
        
        Args:
            feature_groups: List of feature groups to include
            normalize: Whether to normalize features
            
        Returns:
            Prepared feature matrix
        """
        if feature_groups is None:
            feature_groups = ['momentum', 'volatility', 'trend', 'volume']
        
        # Select features based on groups
        selected_features = []
        
        if 'momentum' in feature_groups:
            momentum_cols = [col for col in self.data.columns if any(x in col.upper() for x in ['RSI', 'STOCH', 'MOMENTUM', 'MACD'])]
            selected_features.extend(momentum_cols)
        
        if 'volatility' in feature_groups:
            volatility_cols = [col for col in self.data.columns if any(x in col.upper() for x in ['VOLATILITY', 'ATR', 'BB_WIDTH'])]
            selected_features.extend(volatility_cols)
        
        if 'trend' in feature_groups:
            trend_cols = [col for col in self.data.columns if any(x in col.upper() for x in ['SMA', 'EMA', 'TREND'])]
            selected_features.extend(trend_cols)
        
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
        
        # Remove rows with NaN values
        feature_data = feature_data.dropna()
        
        if normalize:
            self.scaled_data = self.scaler.fit_transform(feature_data)
            return pd.DataFrame(self.scaled_data, columns=available_features, index=feature_data.index)
        
        return feature_data
    
    def kmeans_clustering(self, 
                         feature_data: pd.DataFrame,
                         n_clusters: Optional[int] = None) -> Dict:
        """
        Perform K-means clustering.
        
        Args:
            feature_data: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with clustering results
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Use scaled data if available
        if self.scaled_data is not None:
            data_for_clustering = self.scaled_data
        else:
            data_for_clustering = feature_data.values
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_for_clustering)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(data_for_clustering, labels)
        calinski_avg = calinski_harabasz_score(data_for_clustering, labels)
        
        # Store results
        self.clusters['kmeans'] = {
            'model': kmeans,
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_avg,
            'feature_names': feature_data.columns.tolist()
        }
        
        self.cluster_labels['kmeans'] = labels
        
        logger.info(f"K-means clustering completed. Silhouette score: {silhouette_avg:.3f}")
        
        return self.clusters['kmeans']
    
    def dbscan_clustering(self, 
                         feature_data: pd.DataFrame,
                         eps: float = 0.5,
                         min_samples: int = 5) -> Dict:
        """
        Perform DBSCAN clustering.
        
        Args:
            feature_data: Feature matrix
            eps: Epsilon parameter for DBSCAN
            min_samples: Minimum samples parameter for DBSCAN
            
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
        
        # Use scaled data if available
        if self.scaled_data is not None:
            data_for_clustering = self.scaled_data
        else:
            data_for_clustering = feature_data.values
        
        # Perform clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_for_clustering)
        
        # Calculate metrics (only if we have more than one cluster)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            silhouette_avg = silhouette_score(data_for_clustering, labels)
            calinski_avg = calinski_harabasz_score(data_for_clustering, labels)
        else:
            silhouette_avg = 0
            calinski_avg = 0
        
        # Store results
        self.clusters['dbscan'] = {
            'model': dbscan,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_avg,
            'feature_names': feature_data.columns.tolist()
        }
        
        self.cluster_labels['dbscan'] = labels
        
        logger.info(f"DBSCAN clustering completed. Clusters: {n_clusters}, Silhouette score: {silhouette_avg:.3f}")
        
        return self.clusters['dbscan']
    
    def time_series_clustering(self, 
                              feature_data: pd.DataFrame,
                              n_clusters: Optional[int] = None,
                              window_size: int = 20) -> Dict:
        """
        Perform time series clustering using sliding windows.
        
        Args:
            feature_data: Feature matrix
            n_clusters: Number of clusters
            window_size: Size of sliding window
            
        Returns:
            Dictionary with clustering results
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        logger.info(f"Performing time series clustering with {n_clusters} clusters, window={window_size}...")
        
        # Create time series segments
        segments = self._create_time_series_segments(feature_data, window_size)
        
        if len(segments) < n_clusters:
            logger.warning(f"Not enough segments ({len(segments)}) for {n_clusters} clusters")
            return {}
        
        # Perform time series clustering
        if not TSLEARN_AVAILABLE:
            logger.warning("tslearn not available, skipping time series clustering")
            return {}
            
        ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, random_state=42)
        labels = ts_kmeans.fit_predict(segments)
        
        # Calculate metrics
        silhouette_avg = ts_silhouette_score(segments, labels)
        
        # Store results
        self.clusters['timeseries'] = {
            'model': ts_kmeans,
            'labels': labels,
            'centroids': ts_kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg,
            'segments': segments,
            'window_size': window_size,
            'feature_names': feature_data.columns.tolist()
        }
        
        # Map segment labels back to original data
        segment_labels = self._map_segment_labels_to_data(labels, window_size, len(feature_data))
        self.cluster_labels['timeseries'] = segment_labels
        
        logger.info(f"Time series clustering completed. Silhouette score: {silhouette_avg:.3f}")
        
        return self.clusters['timeseries']
    
    def _create_time_series_segments(self, 
                                   data: pd.DataFrame, 
                                   window_size: int) -> np.ndarray:
        """
        Create time series segments for clustering.
        
        Args:
            data: Input data
            window_size: Size of sliding window
            
        Returns:
            Array of time series segments
        """
        segments = []
        
        for i in range(len(data) - window_size + 1):
            segment = data.iloc[i:i+window_size].values
            segments.append(segment)
        
        return np.array(segments)
    
    def _map_segment_labels_to_data(self, 
                                   segment_labels: np.ndarray, 
                                   window_size: int, 
                                   data_length: int) -> np.ndarray:
        """
        Map segment labels back to original data points.
        
        Args:
            segment_labels: Labels for segments
            window_size: Size of sliding window
            data_length: Length of original data
            
        Returns:
            Array of labels for each data point
        """
        labels = np.full(data_length, -1)  # -1 for unassigned
        
        for i, label in enumerate(segment_labels):
            start_idx = i
            end_idx = min(i + window_size, data_length)
            labels[start_idx:end_idx] = label
        
        return labels
    
    def find_optimal_clusters(self, 
                            feature_data: pd.DataFrame,
                            max_clusters: int = 10) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            feature_data: Feature matrix
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary with optimal clustering parameters
        """
        logger.info("Finding optimal number of clusters...")
        
        # Use scaled data if available
        if self.scaled_data is not None:
            data_for_clustering = self.scaled_data
        else:
            data_for_clustering = feature_data.values
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_for_clustering)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data_for_clustering, labels))
            calinski_scores.append(calinski_harabasz_score(data_for_clustering, labels))
        
        # Find optimal k based on silhouette score
        optimal_k = np.argmax(silhouette_scores) + 2
        
        results = {
            'optimal_k': optimal_k,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'k_range': list(range(2, max_clusters + 1))
        }
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        return results
    
    def analyze_clusters(self, 
                        cluster_type: str = 'kmeans',
                        feature_data: pd.DataFrame = None) -> Dict:
        """
        Analyze cluster characteristics.
        
        Args:
            cluster_type: Type of clustering ('kmeans', 'dbscan', 'timeseries')
            feature_data: Feature matrix
            
        Returns:
            Dictionary with cluster analysis
        """
        if cluster_type not in self.clusters:
            logger.error(f"Clustering {cluster_type} not found")
            return {}
        
        cluster_info = self.clusters[cluster_type]
        labels = cluster_info['labels']
        
        if feature_data is None:
            feature_data = self.data
        
        # Analyze each cluster
        cluster_analysis = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points for DBSCAN
                continue
                
            cluster_mask = labels == cluster_id
            cluster_data = feature_data.iloc[cluster_mask]
            
            analysis = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(feature_data) * 100,
                'mean_features': cluster_data.mean().to_dict(),
                'std_features': cluster_data.std().to_dict(),
                'min_features': cluster_data.min().to_dict(),
                'max_features': cluster_data.max().to_dict()
            }
            
            cluster_analysis[f'cluster_{cluster_id}'] = analysis
        
        return cluster_analysis
    
    def visualize_clusters(self, 
                          cluster_type: str = 'kmeans',
                          method: str = 'pca',
                          feature_data: pd.DataFrame = None) -> go.Figure:
        """
        Visualize clusters using dimensionality reduction.
        
        Args:
            cluster_type: Type of clustering to visualize
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            feature_data: Feature matrix
            
        Returns:
            Plotly figure with cluster visualization
        """
        if cluster_type not in self.clusters:
            logger.error(f"Clustering {cluster_type} not found")
            return go.Figure()
        
        if feature_data is None:
            feature_data = self.data
        
        cluster_info = self.clusters[cluster_type]
        labels = cluster_info['labels']
        
        # Prepare data for visualization
        if self.scaled_data is not None:
            data_for_viz = self.scaled_data
        else:
            data_for_viz = feature_data.values
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(data_for_viz)
            title = f"{cluster_type.upper()} Clusters - PCA"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            coords = reducer.fit_transform(data_for_viz)
            title = f"{cluster_type.upper()} Clusters - t-SNE"
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                logger.warning("UMAP not available, falling back to PCA")
                method = 'pca'
                reducer = PCA(n_components=2, random_state=42)
                coords = reducer.fit_transform(data_for_viz)
                title = f"{cluster_type.upper()} Clusters - PCA (UMAP fallback)"
            else:
                reducer = umap.UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(data_for_viz)
                title = f"{cluster_type.upper()} Clusters - UMAP"
        else:
            logger.error(f"Unknown visualization method: {method}")
            return go.Figure()
        
        # Create visualization
        fig = go.Figure()
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Noise points
                mask = labels == cluster_id
                fig.add_trace(go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode='markers',
                    marker=dict(color='black', size=3, opacity=0.5),
                    name=f'Noise',
                    showlegend=True
                ))
            else:
                mask = labels == cluster_id
                fig.add_trace(go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode='markers',
                    marker=dict(size=6),
                    name=f'Cluster {cluster_id}',
                    showlegend=True
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f"{method.upper()} 1",
            yaxis_title=f"{method.upper()} 2",
            template="plotly_white"
        )
        
        return fig
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary of all clustering results.
        
        Returns:
            DataFrame with clustering summary
        """
        summary_data = []
        
        for cluster_type, cluster_info in self.clusters.items():
            summary = {
                'method': cluster_type.upper(),
                'n_clusters': len(set(cluster_info['labels'])) - (1 if -1 in cluster_info['labels'] else 0),
                'silhouette_score': cluster_info.get('silhouette_score', 0),
                'calinski_score': cluster_info.get('calinski_score', 0),
                'n_features': len(cluster_info.get('feature_names', []))
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)


def cluster_market_behavior(data: pd.DataFrame, 
                           n_clusters: int = 5,
                           methods: List[str] = None) -> MarketBehaviorClusterer:
    """
    Convenience function to perform market behavior clustering.
    
    Args:
        data: Feature matrix
        n_clusters: Number of clusters
        methods: List of clustering methods to use
        
    Returns:
        MarketBehaviorClusterer instance with results
    """
    if methods is None:
        methods = ['kmeans', 'dbscan', 'timeseries']
    
    clusterer = MarketBehaviorClusterer(data, n_clusters)
    
    # Prepare features
    feature_data = clusterer.prepare_features()
    
    # Perform clustering
    if 'kmeans' in methods:
        clusterer.kmeans_clustering(feature_data)
    
    if 'dbscan' in methods:
        clusterer.dbscan_clustering(feature_data)
    
    if 'timeseries' in methods:
        clusterer.time_series_clustering(feature_data)
    
    return clusterer


if __name__ == "__main__":
    # Example usage
    from src.data.data_collector import CryptoDataCollector
    from src.features.technical_indicators import TechnicalIndicators
    
    # Get sample data
    collector = CryptoDataCollector()
    btc_data = collector.get_ohlcv_data('BTC', period='6mo')
    
    # Calculate technical indicators
    ti = TechnicalIndicators(btc_data)
    data_with_indicators = ti.add_all_indicators()
    
    # Get feature matrix
    features = ti.get_feature_matrix()
    
    # Perform clustering
    clusterer = cluster_market_behavior(features, n_clusters=5)
    
    # Analyze results
    print("Clustering Summary:")
    print(clusterer.get_cluster_summary())
    
    # Analyze K-means clusters
    kmeans_analysis = clusterer.analyze_clusters('kmeans', features)
    print(f"\nK-means cluster analysis: {len(kmeans_analysis)} clusters found")
    
    # Find optimal clusters
    optimal = clusterer.find_optimal_clusters(features)
    print(f"Optimal number of clusters: {optimal['optimal_k']}") 