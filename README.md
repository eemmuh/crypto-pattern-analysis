# Crypto Trading Analysis: Pattern Recognition & Market Behavior Clustering

A comprehensive Python and R project for identifying candlestick patterns, trend reversals, and momentum shifts in cryptocurrency markets using advanced statistical and machine learning techniques.

## 🎯 Features

### Pattern Recognition
- **Candlestick Patterns**: Doji, Hammer, Shooting Star, Engulfing patterns
- **Trend Reversals**: Support/Resistance breaks, Double tops/bottoms
- **Momentum Shifts**: RSI divergences, MACD crossovers, Volume spikes

### Machine Learning Techniques
- **Hidden Markov Models (HMM)**: Market regime detection using Gaussian Mixture Models
- **Clustering**: K-means, DBSCAN, and time series clustering
- **Anomaly Detection**: Statistical and ML-based anomaly detection
- **Spectral Analysis**: Frequency domain analysis for cyclical patterns
- **Dimensionality Reduction**: PCA, t-SNE, UMAP for visualization

### Market Behavior Analysis
- **Unsupervised Clustering**: Group similar market behaviors (bullish, bearish, sideways)
- **Feature Engineering**: Technical indicators, price action features, volume analysis
- **Real-time Analysis**: Live data processing and pattern detection

## 🚀 Quick Start

### Option 1: Using Makefile (Recommended)
```bash
# Show all available commands
make help

# Full setup and quick start
make quick-start

# Or step by step:
make install      # Install Python dependencies
make setup-r      # Install R dependencies
make run-demo     # Run Python demo
make run-notebook # Start Jupyter notebook
```

### Option 2: Manual Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install R dependencies
Rscript install_dependencies.R

# Run Python demo
python demo.py

# Start Jupyter notebook
jupyter notebook notebooks/crypto_analysis_demo.ipynb

# Run R demo
Rscript demo.R
```

## 📁 Project Structure

```
crypto-project/
├── src/                    # Python source code
│   ├── data/              # Data collection and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models and algorithms
│   ├── patterns/          # Pattern recognition
│   ├── visualization/     # Plotting and charts
│   └── utils/             # Utility functions
├── R/                     # R source files
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Data storage and caching
├── tests/                 # Unit tests
├── config/                # Configuration files
├── demo.py                # Python demo script
├── demo.R                 # R demo script
├── requirements.txt       # Python dependencies
├── Makefile              # Development tasks
└── PROJECT_STRUCTURE.md  # Detailed project structure
```

## 📊 Key Components

### 1. Data Preprocessing (`src/data/`)
- OHLCV data collection from Yahoo Finance
- Data cleaning and normalization
- Time series resampling and alignment
- Robust error handling with fallback sample data

### 2. Feature Engineering (`src/features/`)
- 35+ technical indicators (RSI, MACD, Bollinger Bands)
- Price action features (returns, volatility, momentum)
- Volume analysis and market microstructure features

### 3. Pattern Recognition (`src/patterns/`)
- 27+ candlestick pattern detection
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

detector = MarketRegimeDetector(data, n_regimes=4)
regimes = detector.detect_regimes(features, method='gmm')
```

### Market Behavior Clustering
```python
from src.models.clustering import MarketBehaviorClusterer

clusterer = MarketBehaviorClusterer(data, n_clusters=5)
clusters = clusterer.kmeans_clustering(features)
```

### Pattern Detection
```python
from src.patterns.candlestick_patterns import CandlestickPatternDetector

detector = CandlestickPatternDetector(data)
patterns = detector.detect_all_patterns()
```

## 📈 Supported Cryptocurrencies

- Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB)
- Cardano (ADA), Solana (SOL), Polkadot (DOT)
- Avalanche (AVAX), Polygon (MATIC), Chainlink (LINK), Uniswap (UNI)
- And many more via yfinance

## 🛠️ Development

### Code Quality
```bash
make lint      # Run linting checks
make format    # Format code with black
make test      # Run tests with coverage
make clean     # Clean cache and temporary files
```

### Adding New Features
1. Add code to appropriate module in `src/`
2. Update `__init__.py` files for imports
3. Add tests in `tests/`
4. Update documentation

## 📝 Documentation

- [Project Structure](PROJECT_STRUCTURE.md) - Detailed project organization
- [R Documentation](README_R.md) - R-specific documentation
- [Jupyter Notebook](notebooks/crypto_analysis_demo.ipynb) - Interactive analysis

## 🔗 Dependencies

### Python
- **Data**: yfinance, pandas, numpy
- **ML**: scikit-learn, tslearn, umap-learn
- **Visualization**: plotly, matplotlib, seaborn
- **Analysis**: scipy, statsmodels
- **Technical Analysis**: ta (Technical Analysis library)

### R
- **Data**: quantmod, TTR
- **ML**: depmixS4, tsclust, cluster
- **Visualization**: ggplot2, plotly
- **Analysis**: PerformanceAnalytics, rugarch 

