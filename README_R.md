# Crypto Trading Analysis: R-Based Pattern Recognition & Market Behavior Clustering

A comprehensive R package for identifying candlestick patterns, trend reversals, and momentum shifts in cryptocurrency markets using advanced statistical and machine learning techniques.

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
├── R/                      # R source files
│   ├── data_collector.R    # Data collection and preprocessing
│   ├── technical_indicators.R # Feature engineering
│   ├── hmm_model.R         # HMM models and algorithms
│   ├── clustering.R        # Clustering analysis
│   ├── candlestick_patterns.R # Pattern recognition
│   └── main.R              # Main analysis script
├── data/                   # Data storage and caching
├── notebooks/              # R Markdown notebooks for analysis
├── config/                 # Configuration files
├── tests/                  # Unit tests
├── DESCRIPTION             # R package description
└── README_R.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

First, install the required R packages:

```r
# Install required packages
install.packages(c(
  "quantmod", "TTR", "depmixS4", "tsclust", "cluster", "factoextra",
  "ggplot2", "plotly", "dplyr", "tidyr", "lubridate", "zoo", "xts",
  "forecast", "changepoint", "strucchange", "PerformanceAnalytics",
  "rugarch", "fGarch", "moments", "psych", "corrplot", "viridis",
  "scales", "gridExtra", "knitr", "rmarkdown", "testthat"
))
```

### 2. Load the Package

```r
# Source all modules
source("R/data_collector.R")
source("R/technical_indicators.R")
source("R/hmm_model.R")
source("R/clustering.R")
source("R/candlestick_patterns.R")
source("R/main.R")
```

### 3. Run Analysis

```r
# Quick analysis on Bitcoin
results <- quick_analysis("BTC-USD", period = "6mo")

# Comprehensive analysis on multiple cryptocurrencies
results <- comprehensive_crypto_analysis(
  symbols = c("BTC-USD", "ETH-USD", "BNB-USD"),
  period = "1y",
  n_regimes = 4,
  n_clusters = 5
)
```

## 📊 Key Components

### 1. Data Collection (`R/data_collector.R`)
- OHLCV data collection from Yahoo Finance via `quantmod`
- Data cleaning and normalization
- Time series resampling and alignment
- Caching for improved performance

```r
# Initialize collector
collector <- CryptoDataCollector()

# Get Bitcoin data
btc_data <- collector$get_ohlcv_data("BTC-USD", period = "1y")

# Get multiple cryptocurrencies
symbols <- c("BTC-USD", "ETH-USD", "BNB-USD")
market_data <- collector$get_multiple_cryptos(symbols, period = "6mo")
```

### 2. Technical Indicators (`R/technical_indicators.R`)
- Comprehensive technical indicators using `TTR`
- Moving averages (SMA, EMA, WMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, VWAP, MFI)

```r
# Calculate all indicators
ti <- TechnicalIndicators(btc_data)
data_with_indicators <- ti$add_all_indicators()

# Get feature matrix for ML
features <- ti$get_feature_matrix()

# Get indicator summary
summary <- ti$get_indicator_summary()
```

### 3. Pattern Recognition (`R/candlestick_patterns.R`)
- Candlestick pattern detection
- Support/resistance identification
- Trend reversal signals
- Pattern classification (bullish/bearish/neutral)

```r
# Detect patterns
pattern_detector <- CandlestickPatternDetector(data_with_indicators)
data_with_patterns <- pattern_detector$detect_all_patterns()

# Get pattern summary
summary <- pattern_detector$get_pattern_summary()

# Get trading signals
signals <- pattern_detector$get_pattern_signals()

# Visualize patterns
fig <- pattern_detector$visualize_patterns("Hammer")
```

### 4. Market Regime Detection (`R/hmm_model.R`)
- Hidden Markov Models using `depmixS4`
- Gaussian Mixture Models as alternative
- Regime classification (bull/bear/sideways/volatile)
- Transition matrix analysis

```r
# Detect market regimes
detector <- detect_market_regimes(features, n_regimes = 4, method = "depmix")

# Get regime summary
summary <- detector$get_regime_summary()

# Visualize regimes
fig <- detector$visualize_regimes(btc_data)

# Plot transition matrix
transition_fig <- detector$plot_transition_matrix()
```

### 5. Market Behavior Clustering (`R/clustering.R`)
- K-means clustering for market behavior
- DBSCAN for density-based clustering
- Time series clustering with `tsclust`
- Optimal cluster number detection

```r
# Perform clustering
clusterer <- MarketBehaviorClusterer(data_with_patterns)
feature_data <- clusterer$prepare_features()

# K-means clustering
kmeans_results <- clusterer$kmeans_clustering(feature_data)

# Find optimal clusters
optimal <- clusterer$find_optimal_clusters(feature_data)

# Visualize clusters
fig <- clusterer$visualize_clusters("kmeans", feature_data, method = "pca")
```

## 🔬 Analysis Examples

### Market Regime Detection
```r
# Load data and calculate indicators
collector <- CryptoDataCollector()
btc_data <- collector$get_ohlcv_data("BTC-USD", period = "1y")

ti <- TechnicalIndicators(btc_data)
data_with_indicators <- ti$add_all_indicators()
features <- ti$get_feature_matrix()

# Detect market regimes
detector <- detect_market_regimes(features, n_regimes = 4, method = "depmix")

# Analyze results
print("Regime Summary:")
print(detector$get_regime_summary())

# Visualize regimes
fig <- detector$visualize_regimes(btc_data)
fig
```

### Market Behavior Clustering
```r
# Perform clustering analysis
clusterer <- MarketBehaviorClusterer(data_with_indicators)
feature_data <- clusterer$prepare_features()

# Multiple clustering methods
kmeans_results <- clusterer$kmeans_clustering(feature_data)
dbscan_results <- clusterer$dbscan_clustering(feature_data)
tsclust_results <- clusterer$time_series_clustering(feature_data)

# Find optimal clusters
optimal <- clusterer$find_optimal_clusters(feature_data)

# Get summary
print(clusterer$get_cluster_summary())

# Analyze clusters
kmeans_analysis <- clusterer$analyze_clusters("kmeans", features)
```

### Pattern Detection
```r
# Detect candlestick patterns
pattern_detector <- CandlestickPatternDetector(data_with_indicators)
data_with_patterns <- pattern_detector$detect_all_patterns()

# Get pattern summary
summary <- pattern_detector$get_pattern_summary()
print(summary)

# Get trading signals
signals <- pattern_detector$get_pattern_signals()

# Visualize specific patterns
fig <- pattern_detector$visualize_patterns("Hammer")
fig
```

## 📈 Supported Cryptocurrencies

- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)
- Binance Coin (BNB-USD)
- Cardano (ADA-USD)
- Solana (SOL-USD)
- And many more via Yahoo Finance

## 🎨 Visualization Features

### Interactive Charts
- **Plotly**: Interactive candlestick charts with pattern overlays
- **ggplot2**: Publication-quality static charts
- **Dashboard**: Multi-panel interactive dashboard

### Chart Types
- Price charts with technical indicators
- Candlestick patterns highlighted
- Market regime visualization
- Cluster analysis plots
- Performance metrics charts

## 📋 Performance Metrics

The package calculates comprehensive performance metrics:

- **Returns**: Total, annualized, and rolling returns
- **Risk Metrics**: Volatility, VaR, CVaR, maximum drawdown
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio
- **Distribution Metrics**: Skewness, kurtosis
- **Trading Metrics**: Win rate, profit factor

## 🔧 Configuration

### Data Sources
- **Primary**: Yahoo Finance (via `quantmod`)
- **Alternative**: Custom API integrations possible
- **Caching**: Local RDS files for improved performance

### Analysis Parameters
- **Time Periods**: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- **Intervals**: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
- **Regimes**: Configurable number (typically 3-6)
- **Clusters**: Configurable number (typically 3-10)

## 🧪 Testing

Run the test suite:

```r
# Run tests
library(testthat)
test_dir("tests/")
```

## 📚 Documentation

### Vignettes
- `crypto_analysis_vignette.Rmd`: Comprehensive usage guide
- `pattern_detection_vignette.Rmd`: Pattern recognition tutorial
- `regime_detection_vignette.Rmd`: HMM analysis guide

### Function Documentation
All functions include detailed Roxygen2 documentation with:
- Parameter descriptions
- Return value explanations
- Usage examples
- References to relevant literature

### Development Setup
```r
# Install development dependencies
install.packages(c("devtools", "roxygen2", "testthat", "covr"))

# Build and check package
devtools::document()
devtools::check()
devtools::test()
```

## 🔗 Dependencies

### Core Dependencies
- **Data**: `quantmod`, `TTR`, `dplyr`, `tidyr`
- **ML**: `depmixS4`, `tsclust`, `cluster`, `factoextra`
- **Visualization**: `ggplot2`, `plotly`, `viridis`
- **Analysis**: `PerformanceAnalytics`, `rugarch`, `moments`
- **Time Series**: `zoo`, `xts`, `forecast`

### Optional Dependencies
- **Advanced ML**: `mclust`, `Rtsne`, `umap`
- **Reporting**: `knitr`, `rmarkdown`
- **Testing**: `testthat`, `covr`

## 📊 Example Output

### Analysis Report
```
================================================================================
COMPREHENSIVE ANALYSIS SUMMARY REPORT
================================================================================

📊 BTC-USD ANALYSIS SUMMARY
   Data Points: 365
   Date Range: 2023-01-01 to 2024-01-01
   Total Return: 156.78%
   Technical Indicators: 45 calculated
   Candlestick Patterns: 10 types detected
   Bullish Patterns: 23 instances
   Bearish Patterns: 18 instances
   Market Clusters: 5 (Silhouette: 0.342)
   Optimal Clusters: 4
   Market Regimes: 4 detected
   Model AIC: 1245.67, BIC: 1289.34
   Total Return: 156.78%
   Annualized Return: 156.78%
   Volatility: 89.45%
   Sharpe Ratio: 1.753
   Max Drawdown: -45.23%
   Win Rate: 52.3%
================================================================================
Analysis completed successfully!
================================================================================
```

## 🚀 Getting Started with Examples

### Basic Usage
```r
# Load the package
source("R/main.R")

# Run quick analysis
results <- quick_analysis("BTC-USD", period = "6mo")

# Run comprehensive analysis
results <- comprehensive_crypto_analysis(
  symbols = c("BTC-USD", "ETH-USD"),
  period = "1y",
  n_regimes = 4,
  n_clusters = 5
)

# Create interactive dashboard
dashboard <- create_dashboard(results)
dashboard
```

This R-based crypto analysis package provides a comprehensive toolkit for cryptocurrency market analysis, combining traditional technical analysis with modern machine learning techniques for pattern recognition and market regime detection. 