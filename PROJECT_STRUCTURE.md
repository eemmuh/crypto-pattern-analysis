# Crypto Trading Analysis Project Structure

## 📁 Project Overview

This is a comprehensive cryptocurrency trading analysis project with both Python and R implementations.

## 🏗️ Directory Structure

```
crypto-project/
├── 📁 src/                          # Python source code
│   ├── 📁 data/                     # Data collection and preprocessing
│   │   ├── data_collector.py       # Crypto data fetching
│   │   └── __init__.py
│   ├── 📁 features/                 # Feature engineering
│   │   ├── technical_indicators.py  # Technical analysis indicators
│   │   └── __init__.py
│   ├── 📁 patterns/                 # Pattern recognition
│   │   ├── candlestick_patterns.py  # Candlestick pattern detection
│   │   └── __init__.py
│   ├── 📁 models/                   # Machine learning models
│   │   ├── clustering.py           # Market behavior clustering
│   │   ├── hmm_model.py            # Hidden Markov Models
│   │   └── __init__.py
│   ├── 📁 visualization/            # Plotting and charts
│   ├── 📁 utils/                    # Utility functions
│   ├── main.py                      # Main analysis script
│   └── __init__.py
├── 📁 R/                            # R source code
│   ├── data_collector.R            # R data collection
│   ├── technical_indicators.R      # R technical indicators
│   ├── candlestick_patterns.R      # R pattern detection
│   ├── clustering.R                # R clustering
│   ├── hmm_model.R                 # R HMM models
│   └── main.R                      # R main script
├── 📁 notebooks/                    # Jupyter notebooks
│   └── crypto_analysis_demo.ipynb  # Interactive analysis demo
├── 📁 data/                         # Data storage
│   └── 📁 cache/                   # Cached data files
├── 📁 tests/                        # Unit tests
├── 📁 config/                       # Configuration files
├── demo.py                          # Python demo script
├── demo.R                           # R demo script
├── requirements.txt                 # Python dependencies
├── DESCRIPTION                      # R package description
├── install_dependencies.R           # R dependency installer
└── README.md                        # Main documentation
```

## 🚀 Quick Start

### Python Version (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py

# Interactive analysis
jupyter notebook notebooks/crypto_analysis_demo.ipynb
```

### R Version
```bash
# Install dependencies
Rscript install_dependencies.R

# Run demo
Rscript demo.R
```

## 📊 Analysis Features

### Technical Analysis
- **35+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, etc.
- **Moving Averages**: SMA, EMA, WMA with crossovers
- **Volume Analysis**: OBV, VWAP, Volume ratios

### Pattern Recognition
- **27+ Candlestick Patterns**: Doji, Hammer, Engulfing, Morning/Evening stars
- **Support/Resistance**: Dynamic level detection
- **Trend Analysis**: Breakout and reversal detection

### Machine Learning
- **Market Clustering**: K-means, DBSCAN, Time series clustering
- **Regime Detection**: Hidden Markov Models for market states
- **Anomaly Detection**: Statistical and ML-based approaches

### Visualization
- **Interactive Charts**: Plotly-based visualizations
- **Pattern Overlays**: Candlestick patterns on price charts
- **Cluster Visualization**: Dimensionality reduction plots

## 🔧 Configuration

### Data Sources
- **Primary**: Yahoo Finance (yfinance)
- **Fallback**: Sample data generation
- **Caching**: Local cache for performance

### Supported Cryptocurrencies
- Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB)
- Cardano (ADA), Solana (SOL), Polkadot (DOT)
- Avalanche (AVAX), Polygon (MATIC), Chainlink (LINK), Uniswap (UNI)

## 📈 Performance Metrics

- **Total Return**: Absolute and percentage returns
- **Risk Metrics**: Volatility, Sharpe ratio, Maximum drawdown
- **Statistical Measures**: Skewness, kurtosis, VaR, CVaR
- **Win Rate**: Percentage of profitable periods

## 🛠️ Development

### Adding New Features
1. Add code to appropriate module in `src/`
2. Update `__init__.py` files for imports
3. Add tests in `tests/`
4. Update documentation

### Code Style
- **Python**: PEP 8 compliant
- **R**: Tidyverse style guide
- **Documentation**: Docstrings for all functions

## 📝 License

MIT License - see LICENSE file for details 