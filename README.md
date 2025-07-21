# Crypto Market Analysis: Clustering, Pattern Recognition, and Regime Detection

A cross-language (Python & R) toolkit for unsupervised clustering, pattern recognition, and market regime detection in cryptocurrency price data.

## Features

- **Dual Implementation**: All core analysis available in both Python and R
- **Pattern Recognition**: 27+ candlestick patterns, trend reversals, momentum shifts
- **Market Regime Detection**: Hidden Markov Models (HMM) and Gaussian Mixture Models (GMM) for identifying bull, bear, and sideways markets
- **Unsupervised Clustering**: K-means, DBSCAN, and time series clustering for market behavior
- **Robust Data Handling**: Automatic retry and fallback to sample data if API fails
- **Interactive Visualization**: Plotly, ggplot2, and more
- **Extensible & Modular**: Easy to add new features or models

## Key Technologies

- **Python**: `yfinance` (crypto data), `scikit-learn`, `tslearn`, `plotly`
- **R**: `depmixS4` (HMMs), `TSclust` (time series clustering), `quantmod`, `ggplot2`
- **Data Storage**: CSV cache (no SQL database required)

### Technology Highlights
- **depmixS4**: R package for Hidden Markov Models, used for market regime detection.
- **TSclust**: R package for time series clustering, grouping similar market behaviors.
- **yfinance**: Python package for downloading cryptocurrency and financial data from Yahoo Finance.

## Project Structure

```
crypto-project/
├── src/                    # Python source code
│   ├── data/               # Data collection and preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # ML models and algorithms
│   ├── patterns/           # Pattern recognition
│   ├── visualization/      # Plotting and charts
│   └── utils/              # Utility functions
├── R/                      # R source files
├── notebooks/              # Jupyter notebooks for analysis
├── data/                   # Data storage and caching
│   └── cache/
├── config/                 # Configuration files
├── demo.py                 # Python demo script
├── demo.R                  # R demo script
├── requirements.txt        # Python dependencies
├── install_dependencies.R  # R dependencies
├── README.md               # Main documentation
├── README_R.md             # R-specific documentation
├── DESCRIPTION             # R package description (optional)
├── Makefile                # Automation (optional)
└── .gitignore
```

## Documentation

- **README.md**: Main project documentation (this file)
- **README_R.md**: R-specific usage and setup
- **notebooks/crypto_analysis_demo.ipynb**: Interactive Python analysis
- **DESCRIPTION**: R package metadata (optional)

## Dependencies

### Python
- `yfinance`, `pandas`, `numpy`, `scikit-learn`, `tslearn`, `umap-learn`, `plotly`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `ta`

### R
- `quantmod`, `TTR`, `depmixS4`, `TSclust`, `cluster`, `ggplot2`, `plotly`, `PerformanceAnalytics`, `rugarch`

## Data Management

- All data is cached as CSV in `data/cache/`
- No raw or sensitive data is tracked in git

 

