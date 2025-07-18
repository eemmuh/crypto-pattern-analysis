# Crypto Trading Analysis: Pattern Recognition & Market Behavior Clustering

A comprehensive Python project for identifying candlestick patterns, trend reversals, and momentum shifts in cryptocurrency markets using advanced statistical and machine learning techniques.

## 🎯 Features

### Pattern Recognition
- **Candlestick Patterns**: Doji, Hammer, Shooting Star, Engulfing patterns
- **Trend Reversals**: Support/Resistance breaks, Double tops/bottoms
- **Momentum Shifts**: RSI divergences, MACD crossovers, Volume spikes

### Machine Learning Techniques
- **Hidden Markov Models (HMM)**: Market regime detection using `depmixS4`
- **Clustering**: K-means, DBSCAN, and time series clustering with `tsclust`
- **Anomaly Detection**: Statistical and ML-based anomaly detection
- **Spectral Analysis**: Frequency domain analysis for cyclical patterns
- **Dimensionality Reduction**: PCA, t-SNE, UMAP for visualization

### Market Behavior Analysis
- **Unsupervised Clustering**: Group similar market behaviors (bullish, bearish, sideways)
- **Feature Engineering**: Technical indicators, price action features, volume analysis
- **Real-time Analysis**: Live data processing and pattern detection

## 📁 Project Structure

```
crypto-project/
├── data/                   # Data storage and caching
├── src/
│   ├── data/              # Data collection and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models and algorithms
│   ├── patterns/          # Pattern recognition
│   ├── visualization/     # Plotting and charts
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── config/                # Configuration files
└── tests/                 # Unit tests
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Main Analysis**:
   ```bash
   python src/main.py
   ```

3. **Explore Jupyter Notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```

## 📊 Key Components

### 1. Data Preprocessing (`src/data/`)
- OHLCV data collection from multiple sources
- Data cleaning and normalization
- Time series resampling and alignment

### 2. Feature Engineering (`src/features/`)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price action features (returns, volatility, momentum)
- Volume analysis and market microstructure features

### 3. Pattern Recognition (`src/patterns/`)
- Candlestick pattern detection
- Support/resistance identification
- Trend reversal signals

### 4. Machine Learning Models (`src/models/`)
- Hidden Markov Models for regime detection
- Clustering algorithms for market behavior
- Anomaly detection models

### 5. Visualization (`src/visualization/`)
- Interactive charts with Plotly
- Pattern overlay on price charts
- Cluster visualization with dimensionality reduction

## 🔬 Analysis Examples

### Market Regime Detection
```python
from src.models.hmm_model import MarketRegimeDetector

detector = MarketRegimeDetector()
regimes = detector.detect_regimes(price_data)
```

### Market Behavior Clustering
```python
from src.models.clustering import MarketBehaviorClusterer

clusterer = MarketBehaviorClusterer()
clusters = clusterer.cluster_market_behavior(features)
```

### Pattern Detection
```python
from src.patterns.candlestick import CandlestickPatternDetector

detector = CandlestickPatternDetector()
patterns = detector.detect_patterns(ohlcv_data)
```

## 📈 Supported Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Cardano (ADA)
- Solana (SOL)
- And many more via yfinance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🔗 Dependencies

- **Data**: yfinance, pandas
- **ML**: scikit-learn, depmixS4, tslearn
- **Visualization**: plotly, matplotlib, seaborn
- **Analysis**: scipy, statsmodels, anomalize
- **Technical Analysis**: ta (Technical Analysis library) 