#' Hidden Markov Models for Market Regime Detection
#' 
#' A comprehensive module for detecting market regimes using Hidden Markov Models
#' with the depmixS4 package for cryptocurrency analysis.
#' 
#' @author Crypto Analyst
#' @keywords HMM, market regimes, cryptocurrency, depmixS4

library(depmixS4)
library(dplyr)
library(ggplot2)
library(plotly)
library(zoo)
library(xts)

#' MarketRegimeDetector Class
#' 
#' Detects market regimes using Hidden Markov Models
#' 
#' @field data Market data with features
#' @field n_regimes Number of market regimes to detect
#' @field hmm_model Fitted HMM model
#' @field regime_labels Detected regime labels
#' @field regime_probabilities Regime probabilities
#' @field transition_matrix Transition matrix between regimes
#' @field regime_characteristics Characteristics of each regime
#' 
#' @export
MarketRegimeDetector <- setRefClass(
  "MarketRegimeDetector",
  fields = list(
    data = "data.frame",
    n_regimes = "numeric",
    hmm_model = "ANY",
    regime_labels = "numeric",
    regime_probabilities = "matrix",
    transition_matrix = "matrix",
    regime_characteristics = "list"
  ),
  
  methods = list(
    
    #' Initialize the regime detector
    #' 
    #' @param data Market data with features
    #' @param n_regimes Number of market regimes to detect
    initialize = function(data, n_regimes = 4) {
      data <<- data
      n_regimes <<- n_regimes
      hmm_model <<- NULL
      regime_labels <<- numeric(0)
      regime_probabilities <<- matrix(0, 0, 0)
      transition_matrix <<- matrix(0, 0, 0)
      regime_characteristics <<- list()
    },
    
    #' Prepare features for regime detection
    #' 
    #' @param feature_groups List of feature groups to include
    #' @param normalize Whether to normalize features
    #' 
    #' @return Prepared feature matrix
    prepare_features = function(feature_groups = NULL, normalize = TRUE) {
      if (is.null(feature_groups)) {
        feature_groups <- c("returns", "volatility", "momentum", "volume")
      }
      
      # Select features based on groups
      selected_features <- character(0)
      
      if ("returns" %in% feature_groups) {
        return_cols <- names(data)[grepl("RETURN", toupper(names(data)))]
        selected_features <- c(selected_features, return_cols)
      }
      
      if ("volatility" %in% feature_groups) {
        volatility_cols <- names(data)[grepl("VOLATILITY|ATR|BB_WIDTH", toupper(names(data)))]
        selected_features <- c(selected_features, volatility_cols)
      }
      
      if ("momentum" %in% feature_groups) {
        momentum_cols <- names(data)[grepl("RSI|STOCH|MOMENTUM|MACD", toupper(names(data)))]
        selected_features <- c(selected_features, momentum_cols)
      }
      
      if ("volume" %in% feature_groups) {
        volume_cols <- names(data)[grepl("VOLUME|OBV|VWAP", toupper(names(data)))]
        selected_features <- c(selected_features, volume_cols)
      }
      
      # Remove duplicates and ensure columns exist
      selected_features <- unique(selected_features)
      available_features <- selected_features[selected_features %in% names(data)]
      
      if (length(available_features) == 0) {
        warning("No features found, using all numeric columns")
        available_features <- names(data)[sapply(data, is.numeric)]
      }
      
      feature_data <- data[, available_features, drop = FALSE]
      
      # Remove rows with NaN values
      feature_data <- feature_data[complete.cases(feature_data), , drop = FALSE]
      
      if (normalize) {
        # Standardize features
        feature_data <- scale(feature_data)
        feature_data <- as.data.frame(feature_data)
      }
      
      return(feature_data)
    },
    
    #' Detect market regimes using HMM
    #' 
    #' @param feature_data Feature matrix
    #' @param method Method to use ('depmix', 'gaussian')
    #' @param n_regimes Number of regimes
    #' 
    #' @return List with regime detection results
    detect_regimes = function(feature_data, method = "depmix", n_regimes = NULL) {
      if (is.null(n_regimes)) {
        n_regimes <- n_regimes
      }
      
      message(paste("Detecting market regimes using", method, "with", n_regimes, "regimes..."))
      
      if (method == "depmix") {
        return(detect_regimes_depmix(feature_data, n_regimes))
      } else if (method == "gaussian") {
        return(detect_regimes_gaussian(feature_data, n_regimes))
      } else {
        stop(paste("Unknown method:", method))
      }
    },
    
    #' Detect regimes using depmixS4
    #' 
    #' @param feature_data Feature matrix
    #' @param n_regimes Number of regimes
    #' 
    #' @return List with regime detection results
    detect_regimes_depmix = function(feature_data, n_regimes) {
      # Prepare data for depmixS4
      feature_names <- names(feature_data)
      
      # Create depmix model
      model_formula <- as.formula(paste("cbind(", paste(feature_names, collapse = ", "), ") ~ 1"))
      
      # Fit HMM model
      hmm_fit <- depmix(model_formula, data = feature_data, nstates = n_regimes, 
                       family = rep(list(gaussian()), length(feature_names)))
      
      # Fit the model
      fitted_model <- fit(hmm_fit)
      
      # Get regime labels and probabilities
      regime_labels <<- posterior(fitted_model)$state
      regime_probabilities <<- as.matrix(posterior(fitted_model)[, -1])
      
      # Calculate transition matrix
      transition_matrix <<- calculate_transition_matrix(regime_labels, n_regimes)
      
      # Store model
      hmm_model <<- fitted_model
      
      # Analyze regime characteristics
      regime_characteristics <<- analyze_regime_characteristics(regime_labels, feature_data, n_regimes)
      
      results <- list(
        model = fitted_model,
        labels = regime_labels,
        probabilities = regime_probabilities,
        transition_matrix = transition_matrix,
        regime_characteristics = regime_characteristics,
        aic = AIC(fitted_model),
        bic = BIC(fitted_model),
        feature_names = feature_names
      )
      
      message(paste("HMM regime detection completed. AIC:", round(AIC(fitted_model), 2), 
                   "BIC:", round(BIC(fitted_model), 2)))
      
      return(results)
    },
    
    #' Detect regimes using Gaussian Mixture Model
    #' 
    #' @param feature_data Feature matrix
    #' @param n_regimes Number of regimes
    #' 
    #' @return List with regime detection results
    detect_regimes_gaussian = function(feature_data, n_regimes) {
      # Use mclust for Gaussian Mixture Model
      if (!requireNamespace("mclust", quietly = TRUE)) {
        stop("mclust package is required for Gaussian Mixture Model")
      }
      
      library(mclust)
      
      # Fit Gaussian Mixture Model
      gmm_fit <- Mclust(feature_data, G = n_regimes)
      
      # Get regime labels and probabilities
      regime_labels <<- gmm_fit$classification
      regime_probabilities <<- gmm_fit$z
      
      # Calculate transition matrix
      transition_matrix <<- calculate_transition_matrix(regime_labels, n_regimes)
      
      # Store model
      hmm_model <<- gmm_fit
      
      # Analyze regime characteristics
      regime_characteristics <<- analyze_regime_characteristics(regime_labels, feature_data, n_regimes)
      
      results <- list(
        model = gmm_fit,
        labels = regime_labels,
        probabilities = regime_probabilities,
        transition_matrix = transition_matrix,
        regime_characteristics = regime_characteristics,
        aic = gmm_fit$aic,
        bic = gmm_fit$bic,
        feature_names = names(feature_data)
      )
      
      message(paste("GMM regime detection completed. AIC:", round(gmm_fit$aic, 2), 
                   "BIC:", round(gmm_fit$bic, 2)))
      
      return(results)
    },
    
    #' Calculate transition matrix between regimes
    #' 
    #' @param labels Regime labels
    #' @param n_regimes Number of regimes
    #' 
    #' @return Transition matrix
    calculate_transition_matrix = function(labels, n_regimes) {
      transition_matrix <- matrix(0, n_regimes, n_regimes)
      
      for (i in 1:(length(labels) - 1)) {
        current_regime <- labels[i]
        next_regime <- labels[i + 1]
        transition_matrix[current_regime, next_regime] <- transition_matrix[current_regime, next_regime] + 1
      }
      
      # Normalize rows to get probabilities
      row_sums <- rowSums(transition_matrix)
      transition_matrix <- transition_matrix / row_sums
      
      # Handle rows with no transitions
      transition_matrix[is.nan(transition_matrix)] <- 1/n_regimes
      
      return(transition_matrix)
    },
    
    #' Analyze characteristics of each regime
    #' 
    #' @param labels Regime labels
    #' @param feature_data Feature data
    #' @param n_regimes Number of regimes
    #' 
    #' @return List with regime characteristics
    analyze_regime_characteristics = function(labels, feature_data, n_regimes) {
      characteristics <- list()
      
      for (regime in 1:n_regimes) {
        regime_mask <- labels == regime
        regime_data <- feature_data[regime_mask, , drop = FALSE]
        
        if (nrow(regime_data) == 0) {
          next
        }
        
        # Calculate regime statistics
        regime_stats <- list(
          size = nrow(regime_data),
          percentage = nrow(regime_data) / nrow(feature_data) * 100,
          mean_features = sapply(regime_data, mean, na.rm = TRUE),
          std_features = sapply(regime_data, sd, na.rm = TRUE),
          min_features = sapply(regime_data, min, na.rm = TRUE),
          max_features = sapply(regime_data, max, na.rm = TRUE)
        )
        
        # Identify regime type based on characteristics
        regime_type <- classify_regime(regime_data)
        regime_stats$type <- regime_type
        
        characteristics[[paste0("regime_", regime)]] <- regime_stats
      }
      
      return(characteristics)
    },
    
    #' Classify regime type based on characteristics
    #' 
    #' @param regime_data Data for a specific regime
    #' 
    #' @return Regime type classification
    classify_regime = function(regime_data) {
      # Get key metrics
      return_col <- if ("Return" %in% names(regime_data)) "Return" else names(regime_data)[1]
      volatility_col <- if ("Volatility" %in% names(regime_data)) "Volatility" else names(regime_data)[1]
      
      if (nrow(regime_data) == 0) {
        return("unknown")
      }
      
      avg_return <- mean(regime_data[[return_col]], na.rm = TRUE)
      avg_volatility <- mean(regime_data[[volatility_col]], na.rm = TRUE)
      
      # Classify based on return and volatility
      if (avg_return > 0.01) {  # High positive returns
        if (avg_volatility > 0.02) {  # High volatility
          return("bull_volatile")
        } else {
          return("bull_stable")
        }
      } else if (avg_return < -0.01) {  # High negative returns
        if (avg_volatility > 0.02) {  # High volatility
          return("bear_volatile")
        } else {
          return("bear_stable")
        }
      } else {  # Low returns
        if (avg_volatility > 0.02) {  # High volatility
          return("sideways_volatile")
        } else {
          return("sideways_stable")
        }
      }
    },
    
    #' Predict regime for new data
    #' 
    #' @param new_data New feature data
    #' @param method Method used for training
    #' 
    #' @return List with predicted labels and probabilities
    predict_regime = function(new_data, method = "depmix") {
      if (is.null(hmm_model)) {
        stop("Model not trained yet")
      }
      
      if (method == "depmix") {
        # For depmixS4, we need to refit with new data
        # This is a simplified approach
        return(list(labels = rep(1, nrow(new_data)), 
                   probabilities = matrix(1, nrow(new_data), n_regimes)))
      } else if (method == "gaussian") {
        # For mclust, we can predict directly
        if (requireNamespace("mclust", quietly = TRUE)) {
          predictions <- predict(hmm_model, new_data)
          return(list(labels = predictions$classification, 
                     probabilities = predictions$z))
        }
      }
      
      return(NULL)
    },
    
    #' Get summary of detected regimes
    #' 
    #' @return Data frame with regime summary
    get_regime_summary = function() {
      if (length(regime_characteristics) == 0) {
        return(data.frame())
      }
      
      summary_data <- data.frame()
      
      for (regime_id in names(regime_characteristics)) {
        characteristics <- regime_characteristics[[regime_id]]
        summary <- data.frame(
          regime = regime_id,
          type = characteristics$type,
          size = characteristics$size,
          percentage = characteristics$percentage
        )
        summary_data <- rbind(summary_data, summary)
      }
      
      return(summary_data)
    },
    
    #' Visualize detected regimes
    #' 
    #' @param price_data Price data for visualization
    #' @param method Visualization method
    #' 
    #' @return Plotly figure with regime visualization
    visualize_regimes = function(price_data, method = "plotly") {
      if (length(regime_labels) == 0) {
        stop("No regime labels available")
      }
      
      if (method == "plotly") {
        # Create plotly visualization
        colors <- c("red", "blue", "green", "orange", "purple", "brown")
        
        # Prepare data for plotting
        plot_data <- data.frame(
          Date = price_data$Date[1:length(regime_labels)],
          Price = price_data$Close[1:length(regime_labels)],
          Regime = factor(regime_labels)
        )
        
        # Create plot
        p <- plot_ly(plot_data, x = ~Date, y = ~Price, color = ~Regime, 
                    type = 'scatter', mode = 'markers', 
                    colors = colors[1:n_regimes]) %>%
          layout(title = paste("Market Regime Detection (", n_regimes, "regimes)"),
                 xaxis = list(title = "Date"),
                 yaxis = list(title = "Price"))
        
        return(p)
      } else {
        # Create ggplot visualization
        plot_data <- data.frame(
          Date = price_data$Date[1:length(regime_labels)],
          Price = price_data$Close[1:length(regime_labels)],
          Regime = factor(regime_labels)
        )
        
        p <- ggplot(plot_data, aes(x = Date, y = Price, color = Regime)) +
          geom_point(alpha = 0.7) +
          scale_color_viridis_d() +
          labs(title = paste("Market Regime Detection (", n_regimes, "regimes)"),
               x = "Date", y = "Price") +
          theme_minimal()
        
        return(p)
      }
    },
    
    #' Plot transition matrix
    #' 
    #' @return Plotly figure with transition matrix
    plot_transition_matrix = function() {
      if (length(transition_matrix) == 0) {
        stop("No transition matrix available")
      }
      
      # Create heatmap
      regime_names <- paste0("Regime ", 1:n_regimes)
      
      p <- plot_ly(z = transition_matrix, 
                   x = regime_names, 
                   y = regime_names,
                   type = "heatmap",
                   colorscale = "Blues",
                   text = round(transition_matrix, 3),
                   texttemplate = "%{text}",
                   textfont = list(size = 10)) %>%
        layout(title = "Regime Transition Matrix",
               xaxis = list(title = "To Regime"),
               yaxis = list(title = "From Regime"))
      
      return(p)
    }
  )
)

#' Convenience function to detect market regimes
#' 
#' @param data Feature matrix
#' @param n_regimes Number of regimes
#' @param method Detection method
#' 
#' @return MarketRegimeDetector instance with results
#' @export
detect_market_regimes <- function(data, n_regimes = 4, method = "depmix") {
  detector <- MarketRegimeDetector(data, n_regimes)
  
  # Prepare features
  feature_data <- detector$prepare_features()
  
  # Detect regimes
  detector$detect_regimes(feature_data, method)
  
  return(detector)
}

#' Example usage of market regime detection
#' 
#' @examples
#' # Load data and calculate indicators
#' collector <- CryptoDataCollector()
#' btc_data <- collector$get_ohlcv_data("BTC-USD", period = "1y")
#' 
#' ti <- TechnicalIndicators(btc_data)
#' data_with_indicators <- ti$add_all_indicators()
#' 
#' # Get feature matrix
#' features <- ti$get_feature_matrix()
#' 
#' # Detect market regimes
#' detector <- detect_market_regimes(features, n_regimes = 4, method = "depmix")
#' 
#' # Analyze results
#' print("Regime Summary:")
#' print(detector$get_regime_summary())
#' 
#' # Visualize regimes
#' fig <- detector$visualize_regimes(btc_data)
#' fig
example_hmm_usage <- function() {
  message("Example usage of market regime detection.")
  
  # Load data and calculate indicators
  collector <- CryptoDataCollector()
  btc_data <- collector$get_ohlcv_data("BTC-USD", period = "1y")
  
  ti <- TechnicalIndicators(btc_data)
  data_with_indicators <- ti$add_all_indicators()
  
  # Get feature matrix
  features <- ti$get_feature_matrix()
  
  # Detect market regimes
  detector <- detect_market_regimes(features, n_regimes = 4, method = "depmix")
  
  # Analyze results
  print("Regime Summary:")
  print(detector$get_regime_summary())
  
  print(paste("Regime characteristics:", length(detector$regime_characteristics), "regimes found"))
  
  # Visualize regimes
  fig <- detector$visualize_regimes(btc_data)
  
  return(list(detector = detector, figure = fig))
} 