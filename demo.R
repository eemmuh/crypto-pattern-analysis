#!/usr/bin/env Rscript

# Crypto Trading Analysis Demo (R Version)
# =======================================
# Simplified version that works around OpenGL issues on macOS

# Suppress warnings for cleaner output
options(warn = -1)

# Load required libraries
library(quantmod)
library(TTR)
library(dplyr)
library(ggplot2)
library(plotly)
library(depmixS4)
library(cluster)
library(factoextra)

cat("🚀 Crypto Trading Analysis: R Version\n")
cat("=====================================\n\n")

# Function to get crypto data with error handling
get_crypto_data <- function(symbol = "BTC-USD", period = "6mo") {
  tryCatch({
    cat(sprintf("📊 Fetching %s data...\n", symbol))
  
    # Get data from Yahoo Finance
    data <- getSymbols(symbol, src = "yahoo", period = period, auto.assign = FALSE)
  
    # Convert to data frame
    df <- data.frame(
      Date = index(data),
      Open = as.numeric(Op(data)),
      High = as.numeric(Hi(data)),
      Low = as.numeric(Lo(data)),
      Close = as.numeric(Cl(data)),
      Volume = as.numeric(Vo(data))
    )
    
    cat(sprintf("✅ Collected %d data points\n", nrow(df)))
    cat(sprintf("   Date range: %s to %s\n", 
                format(df$Date[1], "%Y-%m-%d"), 
                format(df$Date[nrow(df)], "%Y-%m-%d")))
    cat(sprintf("   Price range: $%.2f - $%.2f\n", 
                min(df$Low), max(df$High)))
    
    return(df)
    
  }, error = function(e) {
    cat(sprintf("❌ Error fetching %s: %s\n", symbol, e$message))
    cat("🔄 Using sample data instead...\n")
    
    # Generate sample data
    dates <- seq(as.Date("2024-01-01"), as.Date("2024-07-01"), by = "day")
    n <- length(dates)
    
    # Generate realistic price data
    set.seed(123)
    returns <- rnorm(n, mean = 0.001, sd = 0.03)
    prices <- cumprod(1 + returns) * 50000  # Start at $50k
    
    df <- data.frame(
      Date = dates,
      Open = prices * (1 + rnorm(n, 0, 0.01)),
      High = prices * (1 + abs(rnorm(n, 0.02, 0.01))),
      Low = prices * (1 - abs(rnorm(n, 0.02, 0.01))),
      Close = prices,
      Volume = rnorm(n, 1000000, 200000)
    )
    
    cat(sprintf("✅ Generated %d sample data points\n", nrow(df)))
    return(df)
  })
}

# Function to calculate technical indicators
calculate_indicators <- function(df) {
  cat("📈 Calculating technical indicators...\n")
  
  # Calculate returns
  df$Returns <- c(NA, diff(log(df$Close)))
  
  # RSI
  df$RSI <- RSI(df$Close, n = 14)
  
  # MACD
  macd_data <- MACD(df$Close)
  df$MACD <- macd_data[, 1]
  df$MACD_Signal <- macd_data[, 2]
  
  # Bollinger Bands
  bb_data <- BBands(df$Close)
  df$BB_Upper <- bb_data[, 3]
  df$BB_Middle <- bb_data[, 1]
  df$BB_Lower <- bb_data[, 2]
  df$BB_Position <- (df$Close - df$BB_Lower) / (df$BB_Upper - df$BB_Lower)
  
  # ATR
  df$ATR <- ATR(df[, c("High", "Low", "Close")])[, 2]
  
  # Volume indicators
  df$Volume_SMA <- SMA(df$Volume, n = 20)
  df$Volume_Ratio <- df$Volume / df$Volume_SMA
  
  cat(sprintf("✅ Calculated %d technical indicators\n", 8))
  return(df)
}

# Function to detect candlestick patterns
detect_patterns <- function(df) {
  cat("🕯️ Detecting candlestick patterns...\n")
  
  patterns <- list()
  
  # Doji pattern
  body_size <- abs(df$Close - df$Open)
  wick_size <- df$High - df$Low
  patterns$Doji <- body_size < (wick_size * 0.1)
  
  # Hammer pattern
  lower_wick <- pmin(df$Open, df$Close) - df$Low
  upper_wick <- df$High - pmax(df$Open, df$Close)
  patterns$Hammer <- (lower_wick > 2 * body_size) & (upper_wick < body_size)
  
  # Engulfing patterns
  patterns$Bullish_Engulfing <- (df$Open < lag(df$Close)) & 
                               (df$Close > lag(df$Open)) &
                               (body_size > lag(body_size))
  
  patterns$Bearish_Engulfing <- (df$Open > lag(df$Close)) & 
                               (df$Close < lag(df$Open)) &
                               (body_size > lag(body_size))
  
  # Count patterns
  pattern_counts <- sapply(patterns, sum, na.rm = TRUE)
  
  cat("✅ Detected candlestick patterns:\n")
  for (i in 1:length(pattern_counts)) {
    if (pattern_counts[i] > 0) {
      cat(sprintf("   - %s: %d instances\n", names(pattern_counts)[i], pattern_counts[i]))
    }
  }
  
  return(patterns)
}

# Function to perform clustering analysis
perform_clustering <- function(df) {
  cat("🎯 Performing market behavior clustering...\n")
  
  # Prepare features for clustering
  features <- df %>%
    dplyr::select(Returns, RSI, MACD, BB_Position, ATR, Volume_Ratio) %>%
    na.omit()
  
  # Standardize features
  features_scaled <- scale(features)
  
  # K-means clustering
  set.seed(123)
  kmeans_result <- kmeans(features_scaled, centers = 3, nstart = 25)
  
  # Calculate silhouette score
  sil_score <- silhouette(kmeans_result$cluster, dist(features_scaled))
  avg_silhouette <- mean(sil_score[, 3])
  
  cat(sprintf("✅ K-means clustering completed\n"))
  cat(sprintf("   - Clusters: %d\n", length(unique(kmeans_result$cluster))))
  cat(sprintf("   - Silhouette score: %.3f\n", avg_silhouette))
  
  return(list(
    clusters = kmeans_result$cluster,
    silhouette = avg_silhouette,
    centers = kmeans_result$centers
  ))
}

# Function to detect market regimes
detect_regimes <- function(df) {
  cat("🔄 Detecting market regimes...\n")
  
  # Prepare features
  features <- df %>%
    dplyr::select(Returns, RSI, MACD, BB_Position) %>%
    na.omit()
  
  # Use Gaussian Mixture Model (simpler than HMM)
  set.seed(123)
  
  # Try to fit GMM, fallback to k-means if it fails
  tryCatch({
    # For simplicity, use k-means as regime detector
    gmm_result <- kmeans(scale(features), centers = 4, nstart = 25)
    
    cat(sprintf("✅ Market regime detection completed\n"))
    cat(sprintf("   - Regimes: %d\n", length(unique(gmm_result$cluster))))
    
    # Analyze regimes
    regime_analysis <- data.frame(
      regime = gmm_result$cluster,
      returns = features$Returns
    ) %>%
      dplyr::group_by(regime) %>%
      dplyr::summarise(
        avg_return = mean(returns, na.rm = TRUE),
        volatility = sd(returns, na.rm = TRUE),
        count = n()
      )
    
    cat("   Regime characteristics:\n")
    for (i in 1:nrow(regime_analysis)) {
      cat(sprintf("   - Regime %d: Return=%.4f, Vol=%.4f, Count=%d\n",
                  regime_analysis$regime[i],
                  regime_analysis$avg_return[i],
                  regime_analysis$volatility[i],
                  regime_analysis$count[i]))
    }
    
    return(gmm_result)
    
  }, error = function(e) {
    cat(sprintf("❌ Regime detection failed: %s\n", e$message))
    return(NULL)
  })
  }
  
# Function to calculate performance metrics
calculate_performance <- function(df) {
  cat("📊 Calculating performance metrics...\n")
  
  returns <- df$Returns[!is.na(df$Returns)]
  
  metrics <- list(
    total_return = (df$Close[nrow(df)] / df$Close[1] - 1) * 100,
    annualized_return = mean(returns) * 252 * 100,
    volatility = sd(returns) * sqrt(252) * 100,
    sharpe_ratio = (mean(returns) * 252) / (sd(returns) * sqrt(252)),
    max_drawdown = min(cummin(cumprod(1 + returns)) - 1) * 100,
    win_rate = mean(returns > 0) * 100
  )
  
  cat("✅ Performance metrics:\n")
  cat(sprintf("   - Total Return: %.2f%%\n", metrics$total_return))
  cat(sprintf("   - Annualized Return: %.2f%%\n", metrics$annualized_return))
  cat(sprintf("   - Volatility: %.2f%%\n", metrics$volatility))
  cat(sprintf("   - Sharpe Ratio: %.3f\n", metrics$sharpe_ratio))
  cat(sprintf("   - Max Drawdown: %.2f%%\n", metrics$max_drawdown))
  cat(sprintf("   - Win Rate: %.1f%%\n", metrics$win_rate))
  
  return(metrics)
}

# Main analysis function
run_analysis <- function() {
  cat("🎯 Starting comprehensive crypto analysis...\n\n")
  
  # 1. Get data
  df <- get_crypto_data("BTC-USD", "6mo")
  cat("\n")
  
  # 2. Calculate indicators
  df <- calculate_indicators(df)
  cat("\n")
  
  # 3. Detect patterns
  patterns <- detect_patterns(df)
  cat("\n")
  
  # 4. Clustering
  clustering <- perform_clustering(df)
  cat("\n")
  
  # 5. Regime detection
  regimes <- detect_regimes(df)
  cat("\n")
  
  # 6. Performance metrics
  performance <- calculate_performance(df)
  cat("\n")
  
  # Summary
  cat("🎉 Analysis Summary\n")
  cat("==================\n")
  cat(sprintf("✅ Data points: %d\n", nrow(df)))
  cat(sprintf("✅ Technical indicators: 8\n"))
  cat(sprintf("✅ Candlestick patterns: %d types\n", length(patterns)))
  cat(sprintf("✅ Market clusters: %d\n", length(unique(clustering$clusters))))
  cat(sprintf("✅ Market regimes: %d\n", ifelse(is.null(regimes), 0, length(unique(regimes$cluster)))))
  cat("\n")
  cat("🔬 Analysis completed successfully!\n")
  cat("💡 The R version works around OpenGL issues on macOS\n")
}

# Run the analysis
if (!interactive()) {
  run_analysis()
} 