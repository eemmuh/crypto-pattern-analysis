#' Cryptocurrency Data Collector
#' 
#' A comprehensive data collection module for cryptocurrency market data
#' using quantmod and other R packages for financial data retrieval.
#' 
#' @author Crypto Analyst
#' @keywords data collection, cryptocurrency, OHLCV

library(quantmod)
library(dplyr)
library(lubridate)
library(xts)
library(zoo)

#' CryptoDataCollector Class
#' 
#' Handles data collection for cryptocurrency markets
#' 
#' @field cache_dir Directory for caching data
#' @field symbols List of available cryptocurrency symbols
#' @field data_sources Available data sources
#' 
#' @export
CryptoDataCollector <- setRefClass(
  "CryptoDataCollector",
  fields = list(
    cache_dir = "character",
    symbols = "character",
    data_sources = "character"
  ),
  
  methods = list(
    
    #' Initialize the data collector
    #' 
    #' @param cache_dir Directory for caching data
    initialize = function(cache_dir = "data/cache") {
      cache_dir <<- cache_dir
      symbols <<- c("BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", 
                   "XRP-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD")
      data_sources <<- c("yahoo", "binance", "coinbase")
      
      # Create cache directory if it doesn't exist
      if (!dir.exists(cache_dir)) {
        dir.create(cache_dir, recursive = TRUE)
      }
    },
    
    #' Get OHLCV data for a cryptocurrency
    #' 
    #' @param symbol Cryptocurrency symbol (e.g., "BTC-USD")
    #' @param period Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
    #' @param start_date Start date (optional)
    #' @param end_date End date (optional)
    #' @param interval Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
    #' @param use_cache Whether to use cached data
    #' 
    #' @return xts object with OHLCV data
    get_ohlcv_data = function(symbol, period = "1y", start_date = NULL, 
                             end_date = NULL, interval = "1d", use_cache = TRUE) {
      
      # Check cache first
      cache_file <- file.path(cache_dir, paste0(symbol, "_", period, "_", interval, ".rds"))
      
      if (use_cache && file.exists(cache_file)) {
        message(paste("Loading cached data for", symbol))
        return(readRDS(cache_file))
      }
      
      tryCatch({
        # Get data from Yahoo Finance
        if (is.null(start_date) && is.null(end_date)) {
          data <- getSymbols(symbol, from = Sys.Date() - days(365), 
                           to = Sys.Date(), periodicity = interval, 
                           auto.assign = FALSE)
        } else {
          data <- getSymbols(symbol, from = start_date, to = end_date, 
                           periodicity = interval, auto.assign = FALSE)
        }
        
        # Convert to data frame and standardize column names
        df <- data.frame(
          Date = index(data),
          Open = as.numeric(data[, 1]),
          High = as.numeric(data[, 2]),
          Low = as.numeric(data[, 3]),
          Close = as.numeric(data[, 4]),
          Volume = as.numeric(data[, 5])
        )
        
        # Add technical features
        df <- add_basic_features(df)
        
        # Cache the data
        if (use_cache) {
          saveRDS(df, cache_file)
          message(paste("Downloaded and cached data for", symbol, ":", nrow(df), "records"))
        }
        
        return(df)
        
      }, error = function(e) {
        message(paste("Error fetching data for", symbol, ":", e$message))
        return(NULL)
      })
    },
    
    #' Get data for multiple cryptocurrencies
    #' 
    #' @param symbols Vector of cryptocurrency symbols
    #' @param period Time period
    #' @param interval Data interval
    #' 
    #' @return List of data frames
    get_multiple_cryptos = function(symbols, period = "6mo", interval = "1d") {
      market_data <- list()
      
      for (symbol in symbols) {
        data <- get_ohlcv_data(symbol, period, interval = interval)
        if (!is.null(data)) {
          market_data[[symbol]] <- data
          message(paste("Successfully fetched data for", symbol))
        } else {
          message(paste("Failed to fetch data for", symbol))
        }
      }
      
      return(market_data)
    },
    
    #' Add basic technical features to the data
    #' 
    #' @param df Data frame with OHLCV data
    #' 
    #' @return Data frame with additional features
    add_basic_features = function(df) {
      # Calculate returns
      df$Return <- c(NA, diff(log(df$Close)))
      
      # Calculate volatility (rolling standard deviation of returns)
      df$Volatility <- rollapply(df$Return, width = 20, FUN = sd, fill = NA, align = "right")
      
      # Calculate volume moving average
      df$Volume_MA <- rollapply(df$Volume, width = 20, FUN = mean, fill = NA, align = "right")
      
      # Calculate price moving averages
      df$MA_20 <- rollapply(df$Close, width = 20, FUN = mean, fill = NA, align = "right")
      df$MA_50 <- rollapply(df$Close, width = 50, FUN = mean, fill = NA, align = "right")
      df$MA_200 <- rollapply(df$Close, width = 200, FUN = mean, fill = NA, align = "right")
      
      # Calculate price ranges
      df$High_Low_Range <- df$High - df$Low
      df$Open_Close_Range <- abs(df$Close - df$Open)
      
      # Calculate body size (candlestick body)
      df$Body_Size <- abs(df$Close - df$Open)
      df$Upper_Shadow <- df$High - pmax(df$Open, df$Close)
      df$Lower_Shadow <- pmin(df$Open, df$Close) - df$Low
      
      return(df)
    },
    
    #' Get available symbols
    #' 
    #' @return Vector of available symbols
    get_available_symbols = function() {
      return(symbols)
    },
    
    #' Clear cache
    #' 
    #' @param symbol Specific symbol to clear (optional)
    clear_cache = function(symbol = NULL) {
      if (is.null(symbol)) {
        # Clear all cache
        cache_files <- list.files(cache_dir, pattern = "\\.rds$", full.names = TRUE)
        unlink(cache_files)
        message("Cache cleared")
      } else {
        # Clear specific symbol cache
        cache_files <- list.files(cache_dir, pattern = paste0(symbol, ".*\\.rds$"), 
                                 full.names = TRUE)
        unlink(cache_files)
        message(paste("Cache cleared for", symbol))
      }
    },
    
    #' Get market summary statistics
    #' 
    #' @param symbols Vector of symbols
    #' @param period Time period
    #' 
    #' @return Data frame with summary statistics
    get_market_summary = function(symbols = NULL, period = "1mo") {
      if (is.null(symbols)) {
        symbols <- get_available_symbols()
      }
      
      summary_data <- data.frame()
      
      for (symbol in symbols) {
        data <- get_ohlcv_data(symbol, period)
        if (!is.null(data)) {
          summary <- data.frame(
            Symbol = symbol,
            Start_Date = min(data$Date, na.rm = TRUE),
            End_Date = max(data$Date, na.rm = TRUE),
            Total_Return = (tail(data$Close, 1) / head(data$Close, 1) - 1) * 100,
            Volatility = sd(data$Return, na.rm = TRUE) * sqrt(252) * 100,
            Max_Price = max(data$High, na.rm = TRUE),
            Min_Price = min(data$Low, na.rm = TRUE),
            Avg_Volume = mean(data$Volume, na.rm = TRUE),
            Records = nrow(data)
          )
          summary_data <- rbind(summary_data, summary)
        }
      }
      
      return(summary_data)
    }
  )
)

#' Example usage of the data collector
#' 
#' @examples
#' # Initialize collector
#' collector <- CryptoDataCollector()
#' 
#' # Get Bitcoin data
#' btc_data <- collector$get_ohlcv_data("BTC-USD", period = "1y")
#' 
#' # Get multiple cryptocurrencies
#' symbols <- c("BTC-USD", "ETH-USD", "BNB-USD")
#' market_data <- collector$get_multiple_cryptos(symbols, period = "6mo")
#' 
#' # Get market summary
#' summary <- collector$get_market_summary()
#' print(summary)
example_usage <- function() {
  message("Example usage of the data collector.")
  
  # Initialize collector
  collector <- CryptoDataCollector()
  
  # Get Bitcoin data
  btc_data <- collector$get_ohlcv_data("BTC-USD", period = "1y")
  
  # Get multiple cryptocurrencies
  symbols <- c("BTC-USD", "ETH-USD", "BNB-USD")
  market_data <- collector$get_multiple_cryptos(symbols, period = "6mo")
  
  # Get market summary
  summary <- collector$get_market_summary()
  print(summary)
  
  return(list(btc_data = btc_data, market_data = market_data, summary = summary))
} 