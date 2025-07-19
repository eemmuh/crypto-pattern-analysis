#' Main Crypto Analysis Script
#' 
#' Comprehensive cryptocurrency trading analysis using R
#' Combines data collection, technical indicators, pattern recognition,
#' clustering, and market regime detection.
#' 
#' @author Crypto Analyst
#' @keywords main, crypto analysis, comprehensive

# Load required libraries
library(dplyr)
library(ggplot2)
library(plotly)
library(zoo)
library(xts)
library(PerformanceAnalytics)

# Source all modules
source("R/data_collector.R")
source("R/technical_indicators.R")
source("R/hmm_model.R")
source("R/clustering.R")
source("R/candlestick_patterns.R")

#' Comprehensive Crypto Analysis Function
#' 
#' Performs complete analysis on cryptocurrency data
#' 
#' @param symbols Vector of cryptocurrency symbols
#' @param period Time period for analysis
#' @param n_regimes Number of market regimes to detect
#' @param n_clusters Number of clusters for market behavior
#' 
#' @return List with all analysis results
#' @export
comprehensive_crypto_analysis <- function(symbols = c("BTC-USD", "ETH-USD"), 
                                        period = "1y", 
                                        n_regimes = 4, 
                                        n_clusters = 5) {
  
  message("Starting comprehensive crypto trading analysis...")
  message(paste("Analyzing symbols:", paste(symbols, collapse = ", ")))
  message(paste("Data period:", period))
  
  # Initialize results storage
  results <- list()
  
  # Initialize data collector
  collector <- CryptoDataCollector()
  
  # Collect market data
  message("Collecting market data...")
  market_data <- collector$get_multiple_cryptos(symbols, period)
  
  # Analyze each symbol
  for (symbol in symbols) {
    if (is.null(market_data[[symbol]])) {
      message(paste("No data available for", symbol))
      next
    }
    
    message(paste("\n", "="*50))
    message(paste("Analyzing", symbol))
    message(paste("="*50))
    
    # Get data for current symbol
    data <- market_data[[symbol]]
    
    # Store basic info
    symbol_results <- list(
      symbol = symbol,
      data_points = nrow(data),
      date_range = paste(min(data$Date), "to", max(data$Date)),
      start_price = data$Close[1],
      end_price = data$Close[nrow(data)],
      total_return = (data$Close[nrow(data)] / data$Close[1] - 1) * 100
    )
    
    # 1. Calculate technical indicators
    message("1. Calculating technical indicators...")
    ti <- TechnicalIndicators(data)
    data_with_indicators <- ti$add_all_indicators()
    
    # Get feature matrix
    features <- ti$get_feature_matrix()
    indicator_summary <- ti$get_indicator_summary()
    
    symbol_results$technical_indicators <- list(
      total_indicators = ncol(features),
      indicator_summary = indicator_summary,
      feature_matrix = features
    )
    
    # 2. Detect candlestick patterns
    message("2. Detecting candlestick patterns...")
    pattern_detector <- CandlestickPatternDetector(data_with_indicators)
    data_with_patterns <- pattern_detector$detect_all_patterns()
    
    pattern_summary <- pattern_detector$get_pattern_summary()
    pattern_signals <- pattern_detector$get_pattern_signals()
    
    symbol_results$patterns <- list(
      total_patterns = nrow(pattern_summary),
      pattern_summary = pattern_summary,
      bullish_patterns = sum(pattern_detector$get_bullish_patterns(), na.rm = TRUE),
      bearish_patterns = sum(pattern_detector$get_bearish_patterns(), na.rm = TRUE),
      pattern_signals = pattern_signals
    )
    
    # 3. Perform market behavior clustering
    message("3. Performing market behavior clustering...")
    clusterer <- MarketBehaviorClusterer(data_with_patterns, n_clusters)
    feature_data <- clusterer$prepare_features()
    
    # K-means clustering
    kmeans_results <- clusterer$kmeans_clustering(feature_data)
    
    # Find optimal clusters
    optimal <- clusterer$find_optimal_clusters(feature_data)
    
    symbol_results$clustering <- list(
      kmeans_clusters = kmeans_results$n_clusters,
      kmeans_silhouette = kmeans_results$silhouette_score,
      optimal_clusters = optimal,
      cluster_summary = clusterer$get_cluster_summary()
    )
    
    # 4. Detect market regimes
    message("4. Detecting market regimes...")
    regime_detector <- MarketRegimeDetector(data_with_patterns, n_regimes)
    regime_feature_data <- regime_detector$prepare_features()
    regime_results <- regime_detector$detect_regimes(regime_feature_data, method = "depmix")
    
    symbol_results$regimes <- list(
      n_regimes = n_regimes,
      regime_summary = regime_detector$get_regime_summary(),
      aic = regime_results$aic,
      bic = regime_results$bic,
      transition_matrix = regime_results$transition_matrix
    )
    
    # 5. Calculate performance metrics
    message("5. Calculating performance metrics...")
    performance_metrics <- calculate_performance_metrics(data)
    symbol_results$performance <- performance_metrics
    
    # Store complete results
    results[[symbol]] <- symbol_results
    
    message(paste("Analysis completed for", symbol))
  }
  
  # Generate comprehensive report
  generate_analysis_report(results)
  
  return(results)
}

#' Calculate Performance Metrics
#' 
#' @param data OHLCV data
#' 
#' @return List with performance metrics
calculate_performance_metrics <- function(data) {
  # Calculate returns
  returns <- c(NA, diff(log(data$Close)))
  
  # Remove NA values
  returns_clean <- returns[!is.na(returns)]
  
  # Basic metrics
  total_return <- (tail(data$Close, 1) / head(data$Close, 1) - 1) * 100
  annualized_return <- total_return * (252 / length(returns_clean))
  volatility <- sd(returns_clean) * sqrt(252) * 100
  
  # Sharpe ratio (assuming risk-free rate of 0)
  sharpe_ratio <- mean(returns_clean) / sd(returns_clean) * sqrt(252)
  
  # Maximum drawdown
  cumulative_returns <- cumprod(1 + returns_clean)
  running_max <- cummax(cumulative_returns)
  drawdown <- (cumulative_returns - running_max) / running_max
  max_drawdown <- min(drawdown) * 100
  
  # Win rate
  positive_returns <- sum(returns_clean > 0)
  win_rate <- positive_returns / length(returns_clean) * 100
  
  # Additional metrics
  skewness <- moments::skewness(returns_clean)
  kurtosis <- moments::kurtosis(returns_clean)
  
  # VaR and CVaR
  var_95 <- quantile(returns_clean, 0.05) * 100
  cvar_95 <- mean(returns_clean[returns_clean <= quantile(returns_clean, 0.05)]) * 100
  
  return(list(
    total_return = total_return,
    annualized_return = annualized_return,
    volatility = volatility,
    sharpe_ratio = sharpe_ratio,
    max_drawdown = max_drawdown,
    win_rate = win_rate,
    skewness = skewness,
    kurtosis = kurtosis,
    var_95 = var_95,
    cvar_95 = cvar_95
  ))
}

#' Generate Analysis Report
#' 
#' @param results Analysis results
generate_analysis_report <- function(results) {
  message("\n" + "="*80)
  message("COMPREHENSIVE ANALYSIS SUMMARY REPORT")
  message("="*80)
  
  for (symbol in names(results)) {
    result <- results[[symbol]]
    
    message(paste("\n📊", toupper(symbol), "ANALYSIS SUMMARY"))
    message(paste("   Data Points:", result$data_points))
    message(paste("   Date Range:", result$date_range))
    message(paste("   Total Return:", round(result$total_return, 2), "%"))
    
    # Technical indicators
    ti_info <- result$technical_indicators
    message(paste("   Technical Indicators:", ti_info$total_indicators, "calculated"))
    
    # Patterns
    pattern_info <- result$patterns
    message(paste("   Candlestick Patterns:", pattern_info$total_patterns, "types detected"))
    message(paste("   Bullish Patterns:", pattern_info$bullish_patterns, "instances"))
    message(paste("   Bearish Patterns:", pattern_info$bearish_patterns, "instances"))
    
    # Clustering
    cluster_info <- result$clustering
    message(paste("   Market Clusters:", cluster_info$kmeans_clusters, 
                 "(Silhouette:", round(cluster_info$kmeans_silhouette, 3), ")"))
    message(paste("   Optimal Clusters:", cluster_info$optimal_clusters))
    
    # Regimes
    regime_info <- result$regimes
    message(paste("   Market Regimes:", regime_info$n_regimes, "detected"))
    message(paste("   Model AIC:", round(regime_info$aic, 2), 
                 "BIC:", round(regime_info$bic, 2)))
    
    # Performance
    perf_info <- result$performance
    message(paste("   Total Return:", round(perf_info$total_return, 2), "%"))
    message(paste("   Annualized Return:", round(perf_info$annualized_return, 2), "%"))
    message(paste("   Volatility:", round(perf_info$volatility, 2), "%"))
    message(paste("   Sharpe Ratio:", round(perf_info$sharpe_ratio, 3)))
    message(paste("   Max Drawdown:", round(perf_info$max_drawdown, 2), "%"))
    message(paste("   Win Rate:", round(perf_info$win_rate, 1), "%"))
  }
  
  message("\n" + "="*80)
  message("Analysis completed successfully!")
  message("="*80)
}

#' Quick Analysis Function
#' 
#' Performs quick analysis on a single symbol
#' 
#' @param symbol Cryptocurrency symbol
#' @param period Time period
#' 
#' @return List with quick analysis results
#' @export
quick_analysis <- function(symbol = "BTC-USD", period = "6mo") {
  message(paste("Running quick analysis for", symbol, "..."))
  
  # Initialize components
  collector <- CryptoDataCollector()
  
  # Get data
  data <- collector$get_ohlcv_data(symbol, period)
  if (is.null(data)) {
    stop(paste("No data available for", symbol))
  }
  
  # Technical indicators
  ti <- TechnicalIndicators(data)
  data_with_indicators <- ti$add_all_indicators()
  features <- ti$get_feature_matrix()
  
  # Patterns
  pattern_detector <- CandlestickPatternDetector(data_with_indicators)
  data_with_patterns <- pattern_detector$detect_all_patterns()
  
  # Clustering
  clusterer <- MarketBehaviorClusterer(data_with_patterns)
  feature_data <- clusterer$prepare_features()
  kmeans_results <- clusterer$kmeans_clustering(feature_data)
  
  # Regimes
  regime_detector <- MarketRegimeDetector(data_with_patterns)
  regime_feature_data <- regime_detector$prepare_features()
  regime_results <- regime_detector$detect_regimes(regime_feature_data)
  
  message(paste("Quick analysis completed for", symbol))
  message(paste("  -", length(ti$get_indicator_list()), "technical indicators calculated"))
  message(paste("  -", length(pattern_detector$patterns), "pattern types detected"))
  message(paste("  -", length(unique(kmeans_results$labels)), "market clusters found"))
  message(paste("  -", length(unique(regime_results$labels)), "market regimes detected"))
  
  return(list(
    technical_indicators = ti,
    patterns = pattern_detector,
    clustering = clusterer,
    regimes = regime_detector,
    data = data_with_patterns
  ))
}

#' Create Interactive Dashboard
#' 
#' Creates an interactive dashboard with all analysis results
#' 
#' @param results Analysis results
#' 
#' @return Plotly dashboard
#' @export
create_dashboard <- function(results) {
  # Create subplots for dashboard
  fig <- plot_ly() %>%
    subplot(
      # Price chart with regimes
      plot_ly(results[[1]]$data, x = ~Date, y = ~Close, type = 'scatter', mode = 'lines',
              name = "Price") %>%
        layout(title = "Price Chart", xaxis = list(title = "Date"), yaxis = list(title = "Price")),
      
      # Technical indicators
      plot_ly(results[[1]]$data, x = ~Date, y = ~RSI_14, type = 'scatter', mode = 'lines',
              name = "RSI") %>%
        layout(title = "RSI", xaxis = list(title = "Date"), yaxis = list(title = "RSI")),
      
      # Pattern signals
      plot_ly(results[[1]]$patterns$pattern_signals, x = ~Date, y = ~Signal_Strength, 
              type = 'scatter', mode = 'lines', name = "Signal Strength") %>%
        layout(title = "Pattern Signals", xaxis = list(title = "Date"), yaxis = list(title = "Strength")),
      
      # Performance metrics
      plot_ly(x = names(results), y = sapply(results, function(x) x$performance$sharpe_ratio),
              type = 'bar', name = "Sharpe Ratio") %>%
        layout(title = "Sharpe Ratios", xaxis = list(title = "Symbol"), yaxis = list(title = "Sharpe Ratio")),
      
      nrows = 2, ncols = 2
    ) %>%
    layout(title = "Crypto Analysis Dashboard", showlegend = TRUE)
  
  return(fig)
}

#' Main execution function
#' 
#' Run the complete analysis
main <- function() {
  tryCatch({
    # Run comprehensive analysis
    results <- comprehensive_crypto_analysis(
      symbols = c("BTC-USD", "ETH-USD", "BNB-USD"),
      period = "1y",
      n_regimes = 4,
      n_clusters = 5
    )
    
    # Create dashboard
    dashboard <- create_dashboard(results)
    
    # Save results
    saveRDS(results, "analysis_results.rds")
    
    message("Analysis completed and saved to 'analysis_results.rds'")
    
    return(list(results = results, dashboard = dashboard))
    
  }, error = function(e) {
    message(paste("Error during analysis:", e$message))
    return(NULL)
  })
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
} 