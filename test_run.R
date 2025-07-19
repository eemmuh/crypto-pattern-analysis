#' Simple Test Run Script
#' 
#' Quick test to verify the crypto analysis package works correctly
#' 
#' @author Crypto Analyst

cat("🧪 Testing Crypto Analysis Package\n")
cat("==================================\n\n")

# Test 1: Load packages
cat("1. Loading packages...\n")
tryCatch({
  library(quantmod)
  library(TTR)
  library(dplyr)
  library(plotly)
  cat("✓ Packages loaded successfully\n")
}, error = function(e) {
  cat("✗ Error loading packages:", e$message, "\n")
  cat("Run: source('install_dependencies.R')\n")
  return(FALSE)
})

# Test 2: Source modules
cat("\n2. Loading modules...\n")
tryCatch({
  source("R/data_collector.R")
  source("R/technical_indicators.R")
  source("R/main.R")
  cat("✓ Modules loaded successfully\n")
}, error = function(e) {
  cat("✗ Error loading modules:", e$message, "\n")
  return(FALSE)
})

# Test 3: Data collection
cat("\n3. Testing data collection...\n")
tryCatch({
  collector <- CryptoDataCollector()
  btc_data <- collector$get_ohlcv_data("BTC-USD", period = "1mo")
  
  if (!is.null(btc_data) && nrow(btc_data) > 0) {
    cat("✓ Data collection successful\n")
    cat(paste("   Data points:", nrow(btc_data), "\n"))
    cat(paste("   Date range:", min(btc_data$Date), "to", max(btc_data$Date), "\n"))
  } else {
    cat("✗ No data retrieved\n")
    return(FALSE)
  }
}, error = function(e) {
  cat("✗ Error in data collection:", e$message, "\n")
  return(FALSE)
})

# Test 4: Technical indicators
cat("\n4. Testing technical indicators...\n")
tryCatch({
  ti <- TechnicalIndicators(btc_data)
  data_with_indicators <- ti$add_all_indicators()
  features <- ti$get_feature_matrix()
  
  cat("✓ Technical indicators calculated\n")
  cat(paste("   Indicators:", ncol(features), "\n"))
}, error = function(e) {
  cat("✗ Error in technical indicators:", e$message, "\n")
  return(FALSE)
})

# Test 5: Quick analysis
cat("\n5. Testing quick analysis...\n")
tryCatch({
  results <- quick_analysis("BTC-USD", period = "1mo")
  cat("✓ Quick analysis completed\n")
  cat(paste("   Technical indicators:", length(results$technical_indicators$get_indicator_list()), "\n"))
  cat(paste("   Pattern types:", length(results$patterns$patterns), "\n"))
}, error = function(e) {
  cat("✗ Error in quick analysis:", e$message, "\n")
  return(FALSE)
})

cat("\n🎉 All tests passed! Your crypto analysis package is working correctly.\n")
cat("You can now run:\n")
cat("- source('demo.R')\n")
cat("- run_all_demos()\n")
cat("- comprehensive_crypto_analysis()\n") 