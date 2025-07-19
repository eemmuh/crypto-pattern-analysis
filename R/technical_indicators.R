#' Technical Indicators for Cryptocurrency Analysis
#' 
#' A comprehensive module for calculating technical indicators
#' using TTR and other R packages for financial analysis.
#' 
#' @author Crypto Analyst
#' @keywords technical analysis, indicators, cryptocurrency

library(TTR)
library(dplyr)
library(zoo)
library(xts)

#' TechnicalIndicators Class
#' 
#' Calculates various technical indicators for cryptocurrency analysis
#' 
#' @field data OHLCV data frame
#' @field indicators List of calculated indicators
#' 
#' @export
TechnicalIndicators <- setRefClass(
  "TechnicalIndicators",
  fields = list(
    data = "data.frame",
    indicators = "list"
  ),
  
  methods = list(
    
    #' Initialize the technical indicators calculator
    #' 
    #' @param data OHLCV data frame
    initialize = function(data) {
      data <<- data
      indicators <<- list()
    },
    
    #' Add all technical indicators
    #' 
    #' @return Data frame with all indicators
    add_all_indicators = function() {
      message("Calculating technical indicators...")
      
      # Trend indicators
      add_moving_averages()
      add_bollinger_bands()
      add_parabolic_sar()
      add_ichimoku()
      
      # Momentum indicators
      add_rsi()
      add_stochastic()
      add_williams_r()
      add_macd()
      add_cci()
      add_momentum()
      
      # Volume indicators
      add_volume_indicators()
      add_obv()
      add_vwap()
      
      # Volatility indicators
      add_atr()
      add_keltner_channels()
      
      # Support and resistance
      add_support_resistance()
      
      message(paste("Added", length(indicators), "technical indicators"))
      return(data)
    },
    
    #' Add moving averages
    add_moving_averages = function() {
      # Simple Moving Averages
      data$SMA_5 <<- SMA(data$Close, n = 5)
      data$SMA_10 <<- SMA(data$Close, n = 10)
      data$SMA_20 <<- SMA(data$Close, n = 20)
      data$SMA_50 <<- SMA(data$Close, n = 50)
      data$SMA_100 <<- SMA(data$Close, n = 100)
      data$SMA_200 <<- SMA(data$Close, n = 200)
      
      # Exponential Moving Averages
      data$EMA_12 <<- EMA(data$Close, n = 12)
      data$EMA_26 <<- EMA(data$Close, n = 26)
      data$EMA_50 <<- EMA(data$Close, n = 50)
      
      # Weighted Moving Average
      data$WMA_20 <<- WMA(data$Close, n = 20)
      
      indicators$moving_averages <<- c("SMA_5", "SMA_10", "SMA_20", "SMA_50", 
                                     "SMA_100", "SMA_200", "EMA_12", "EMA_26", 
                                     "EMA_50", "WMA_20")
    },
    
    #' Add Bollinger Bands
    add_bollinger_bands = function() {
      bb <- BBands(data$Close, n = 20, sd = 2)
      data$BB_Upper <<- bb[, "upper"]
      data$BB_Middle <<- bb[, "middle"]
      data$BB_Lower <<- bb[, "lower"]
      data$BB_Width <<- (data$BB_Upper - data$BB_Lower) / data$BB_Middle
      data$BB_Position <<- (data$Close - data$BB_Lower) / (data$BB_Upper - data$BB_Lower)
      
      indicators$bollinger_bands <<- c("BB_Upper", "BB_Middle", "BB_Lower", 
                                     "BB_Width", "BB_Position")
    },
    
    #' Add RSI (Relative Strength Index)
    add_rsi = function() {
      data$RSI_14 <<- RSI(data$Close, n = 14)
      data$RSI_21 <<- RSI(data$Close, n = 21)
      
      # RSI divergences
      data$RSI_Overbought <<- ifelse(data$RSI_14 > 70, 1, 0)
      data$RSI_Oversold <<- ifelse(data$RSI_14 < 30, 1, 0)
      
      indicators$rsi <<- c("RSI_14", "RSI_21", "RSI_Overbought", "RSI_Oversold")
    },
    
    #' Add Stochastic Oscillator
    add_stochastic = function() {
      stoch <- stoch(data$High, data$Low, data$Close, nFastK = 14, nFastD = 3, nSlowD = 3)
      data$Stoch_K <<- stoch[, "fastK"]
      data$Stoch_D <<- stoch[, "fastD"]
      data$Stoch_SlowD <<- stoch[, "slowD"]
      
      # Stochastic signals
      data$Stoch_Overbought <<- ifelse(data$Stoch_K > 80, 1, 0)
      data$Stoch_Oversold <<- ifelse(data$Stoch_K < 20, 1, 0)
      
      indicators$stochastic <<- c("Stoch_K", "Stoch_D", "Stoch_SlowD", 
                                "Stoch_Overbought", "Stoch_Oversold")
    },
    
    #' Add Williams %R
    add_williams_r = function() {
      data$Williams_R <<- WPR(data$High, data$Low, data$Close, n = 14)
      
      # Williams %R signals
      data$Williams_Overbought <<- ifelse(data$Williams_R > -20, 1, 0)
      data$Williams_Oversold <<- ifelse(data$Williams_R < -80, 1, 0)
      
      indicators$williams_r <<- c("Williams_R", "Williams_Overbought", "Williams_Oversold")
    },
    
    #' Add MACD
    add_macd = function() {
      macd_data <- MACD(data$Close, nFast = 12, nSlow = 26, nSig = 9)
      data$MACD_Line <<- macd_data[, "macd"]
      data$MACD_Signal <<- macd_data[, "signal"]
      data$MACD_Histogram <<- macd_data[, "histogram"]
      
      # MACD signals
      data$MACD_Bullish <<- ifelse(data$MACD_Line > data$MACD_Signal, 1, 0)
      data$MACD_Bearish <<- ifelse(data$MACD_Line < data$MACD_Signal, 1, 0)
      
      indicators$macd <<- c("MACD_Line", "MACD_Signal", "MACD_Histogram", 
                           "MACD_Bullish", "MACD_Bearish")
    },
    
    #' Add CCI (Commodity Channel Index)
    add_cci = function() {
      data$CCI <<- CCI(data$High, data$Low, data$Close, n = 20)
      
      # CCI signals
      data$CCI_Overbought <<- ifelse(data$CCI > 100, 1, 0)
      data$CCI_Oversold <<- ifelse(data$CCI < -100, 1, 0)
      
      indicators$cci <<- c("CCI", "CCI_Overbought", "CCI_Oversold")
    },
    
    #' Add Momentum
    add_momentum = function() {
      data$Momentum_10 <<- momentum(data$Close, n = 10)
      data$Momentum_20 <<- momentum(data$Close, n = 20)
      
      indicators$momentum <<- c("Momentum_10", "Momentum_20")
    },
    
    #' Add volume indicators
    add_volume_indicators = function() {
      # Volume moving averages
      data$Volume_SMA_20 <<- SMA(data$Volume, n = 20)
      data$Volume_SMA_50 <<- SMA(data$Volume, n = 50)
      
      # Volume ratio
      data$Volume_Ratio <<- data$Volume / data$Volume_SMA_20
      
      # Volume price trend
      data$VPT <<- VPT(data$Close, data$Volume)
      
      # Money flow index
      data$MFI <<- MFI(data$High, data$Low, data$Close, data$Volume, n = 14)
      
      indicators$volume <<- c("Volume_SMA_20", "Volume_SMA_50", "Volume_Ratio", 
                            "VPT", "MFI")
    },
    
    #' Add OBV (On Balance Volume)
    add_obv = function() {
      data$OBV <<- OBV(data$Close, data$Volume)
      
      # OBV moving average
      data$OBV_MA <<- SMA(data$OBV, n = 20)
      
      indicators$obv <<- c("OBV", "OBV_MA")
    },
    
    #' Add VWAP (Volume Weighted Average Price)
    add_vwap = function() {
      # Simple VWAP calculation
      data$VWAP <<- cumsum(data$Close * data$Volume) / cumsum(data$Volume)
      
      indicators$vwap <<- "VWAP"
    },
    
    #' Add ATR (Average True Range)
    add_atr = function() {
      atr_data <- ATR(data$High, data$Low, data$Close, n = 14)
      data$ATR <<- atr_data[, "atr"]
      data$ATR_Percent <<- data$ATR / data$Close * 100
      
      indicators$atr <<- c("ATR", "ATR_Percent")
    },
    
    #' Add Keltner Channels
    add_keltner_channels = function() {
      # Keltner Channels using ATR
      data$KC_Upper <<- data$EMA_20 + (2 * data$ATR)
      data$KC_Lower <<- data$EMA_20 - (2 * data$ATR)
      data$KC_Middle <<- data$EMA_20
      
      indicators$keltner_channels <<- c("KC_Upper", "KC_Middle", "KC_Lower")
    },
    
    #' Add Parabolic SAR
    add_parabolic_sar = function() {
      data$PSAR <<- SAR(data$High, data$Low, acceleration = 0.02, maximum = 0.2)
      
      # PSAR signals
      data$PSAR_Bullish <<- ifelse(data$Close > data$PSAR, 1, 0)
      data$PSAR_Bearish <<- ifelse(data$Close < data$PSAR, 1, 0)
      
      indicators$psar <<- c("PSAR", "PSAR_Bullish", "PSAR_Bearish")
    },
    
    #' Add Ichimoku Cloud
    add_ichimoku = function() {
      # Ichimoku components
      data$Ichimoku_Tenkan <<- (runMax(data$High, 9) + runMin(data$Low, 9)) / 2
      data$Ichimoku_Kijun <<- (runMax(data$High, 26) + runMin(data$Low, 26)) / 2
      data$Ichimoku_Senkou_A <<- (data$Ichimoku_Tenkan + data$Ichimoku_Kijun) / 2
      data$Ichimoku_Senkou_B <<- (runMax(data$High, 52) + runMin(data$Low, 52)) / 2
      data$Ichimoku_Chikou <<- lag(data$Close, 26)
      
      indicators$ichimoku <<- c("Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_Senkou_A", 
                               "Ichimoku_Senkou_B", "Ichimoku_Chikou")
    },
    
    #' Add support and resistance levels
    add_support_resistance = function() {
      # Rolling support and resistance
      data$Support_20 <<- rollapply(data$Low, width = 20, FUN = min, fill = NA, align = "right")
      data$Resistance_20 <<- rollapply(data$High, width = 20, FUN = max, fill = NA, align = "right")
      
      # Pivot points
      data$Pivot_Point <<- (data$High + data$Low + data$Close) / 3
      data$R1 <<- 2 * data$Pivot_Point - data$Low
      data$S1 <<- 2 * data$Pivot_Point - data$High
      data$R2 <<- data$Pivot_Point + (data$High - data$Low)
      data$S2 <<- data$Pivot_Point - (data$High - data$Low)
      
      indicators$support_resistance <<- c("Support_20", "Resistance_20", "Pivot_Point", 
                                        "R1", "S1", "R2", "S2")
    },
    
    #' Get feature matrix for machine learning
    #' 
    #' @return Matrix of technical indicators
    get_feature_matrix = function() {
      # Get all indicator columns
      all_indicators <- unlist(indicators)
      available_indicators <- all_indicators[all_indicators %in% names(data)]
      
      # Create feature matrix
      feature_matrix <- data[, available_indicators, drop = FALSE]
      
      # Remove rows with NA values
      feature_matrix <- feature_matrix[complete.cases(feature_matrix), , drop = FALSE]
      
      return(feature_matrix)
    },
    
    #' Get list of all indicators
    #' 
    #' @return List of indicator names
    get_indicator_list = function() {
      return(unlist(indicators))
    },
    
    #' Get indicator summary statistics
    #' 
    #' @return Data frame with summary statistics
    get_indicator_summary = function() {
      all_indicators <- unlist(indicators)
      available_indicators <- all_indicators[all_indicators %in% names(data)]
      
      if (length(available_indicators) == 0) {
        return(data.frame())
      }
      
      summary_stats <- data.frame(
        Indicator = available_indicators,
        Mean = sapply(data[available_indicators], mean, na.rm = TRUE),
        SD = sapply(data[available_indicators], sd, na.rm = TRUE),
        Min = sapply(data[available_indicators], min, na.rm = TRUE),
        Max = sapply(data[available_indicators], max, na.rm = TRUE),
        NAs = sapply(data[available_indicators], function(x) sum(is.na(x)))
      )
      
      return(summary_stats)
    }
  )
)

#' Example usage of technical indicators
#' 
#' @examples
#' # Load data
#' collector <- CryptoDataCollector()
#' btc_data <- collector$get_ohlcv_data("BTC-USD", period = "6mo")
#' 
#' # Calculate indicators
#' ti <- TechnicalIndicators(btc_data)
#' data_with_indicators <- ti$add_all_indicators()
#' 
#' # Get feature matrix
#' features <- ti$get_feature_matrix()
#' 
#' # Get summary
#' summary <- ti$get_indicator_summary()
#' print(summary)
example_technical_indicators <- function() {
  message("Example usage of technical indicators.")
  
  # Load data
  collector <- CryptoDataCollector()
  btc_data <- collector$get_ohlcv_data("BTC-USD", period = "6mo")
  
  # Calculate indicators
  ti <- TechnicalIndicators(btc_data)
  data_with_indicators <- ti$add_all_indicators()
  
  # Get feature matrix
  features <- ti$get_feature_matrix()
  
  # Get summary
  summary <- ti$get_indicator_summary()
  print(summary)
  
  return(list(data = data_with_indicators, features = features, summary = summary))
} 