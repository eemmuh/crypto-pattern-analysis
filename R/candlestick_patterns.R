#' Candlestick Pattern Recognition
#' 
#' A comprehensive module for detecting candlestick patterns in cryptocurrency data
#' using R for technical analysis.
#' 
#' @author Crypto Analyst
#' @keywords candlestick patterns, technical analysis, cryptocurrency

library(dplyr)
library(ggplot2)
library(plotly)
library(zoo)
library(xts)

#' CandlestickPatternDetector Class
#' 
#' Detects various candlestick patterns in OHLCV data
#' 
#' @field data OHLCV data frame
#' @field patterns Detected patterns
#' @field pattern_rules Pattern detection rules
#' 
#' @export
CandlestickPatternDetector <- setRefClass(
  "CandlestickPatternDetector",
  fields = list(
    data = "data.frame",
    patterns = "list",
    pattern_rules = "list"
  ),
  
  methods = list(
    
    #' Initialize the pattern detector
    #' 
    #' @param data OHLCV data frame
    initialize = function(data) {
      data <<- data
      patterns <<- list()
      pattern_rules <<- list()
      
      # Initialize pattern detection rules
      initialize_pattern_rules()
    },
    
    #' Initialize pattern detection rules
    initialize_pattern_rules = function() {
      pattern_rules <<- list(
        doji = list(
          description = "Doji - Open and close are nearly equal",
          bullish = FALSE,
          bearish = FALSE,
          neutral = TRUE
        ),
        hammer = list(
          description = "Hammer - Small body, long lower shadow",
          bullish = TRUE,
          bearish = FALSE,
          neutral = FALSE
        ),
        shooting_star = list(
          description = "Shooting Star - Small body, long upper shadow",
          bullish = FALSE,
          bearish = TRUE,
          neutral = FALSE
        ),
        bullish_engulfing = list(
          description = "Bullish Engulfing - Current candle engulfs previous bearish candle",
          bullish = TRUE,
          bearish = FALSE,
          neutral = FALSE
        ),
        bearish_engulfing = list(
          description = "Bearish Engulfing - Current candle engulfs previous bullish candle",
          bullish = FALSE,
          bearish = TRUE,
          neutral = FALSE
        ),
        morning_star = list(
          description = "Morning Star - Three candle pattern signaling bullish reversal",
          bullish = TRUE,
          bearish = FALSE,
          neutral = FALSE
        ),
        evening_star = list(
          description = "Evening Star - Three candle pattern signaling bearish reversal",
          bullish = FALSE,
          bearish = TRUE,
          neutral = FALSE
        ),
        three_white_soldiers = list(
          description = "Three White Soldiers - Three consecutive bullish candles",
          bullish = TRUE,
          bearish = FALSE,
          neutral = FALSE
        ),
        three_black_crows = list(
          description = "Three Black Crows - Three consecutive bearish candles",
          bullish = FALSE,
          bearish = TRUE,
          neutral = FALSE
        ),
        hanging_man = list(
          description = "Hanging Man - Similar to hammer but in downtrend",
          bullish = FALSE,
          bearish = TRUE,
          neutral = FALSE
        ),
        inverted_hammer = list(
          description = "Inverted Hammer - Similar to shooting star but in uptrend",
          bullish = TRUE,
          bearish = FALSE,
          neutral = FALSE
        )
      )
    },
    
    #' Detect all candlestick patterns
    #' 
    #' @return Data frame with detected patterns
    detect_all_patterns = function() {
      message("Detecting candlestick patterns...")
      
      # Calculate candlestick characteristics
      data$Body_Size <<- abs(data$Close - data$Open)
      data$Upper_Shadow <<- data$High - pmax(data$Open, data$Close)
      data$Lower_Shadow <<- pmin(data$Open, data$Close) - data$Low
      data$Is_Bullish <<- data$Close > data$Open
      data$Is_Bearish <<- data$Close < data$Open
      
      # Calculate average body size for comparison
      avg_body <- rollapply(data$Body_Size, width = 20, FUN = mean, fill = NA, align = "right")
      
      # Detect individual patterns
      detect_doji(avg_body)
      detect_hammer(avg_body)
      detect_shooting_star(avg_body)
      detect_engulfing_patterns()
      detect_star_patterns()
      detect_three_candle_patterns()
      detect_hanging_man(avg_body)
      detect_inverted_hammer(avg_body)
      
      message(paste("Detected", length(patterns), "pattern types"))
      return(data)
    },
    
    #' Detect Doji patterns
    #' 
    #' @param avg_body Average body size
    detect_doji = function(avg_body) {
      # Doji: body size is very small compared to average
      doji_threshold <- 0.1 * avg_body
      data$Doji <<- data$Body_Size < doji_threshold
      
      patterns$doji <<- data$Doji
    },
    
    #' Detect Hammer patterns
    #' 
    #' @param avg_body Average body size
    detect_hammer = function(avg_body) {
      # Hammer: small body, long lower shadow, short upper shadow
      hammer_conditions <- (
        data$Body_Size < 0.3 * avg_body &  # Small body
        data$Lower_Shadow > 2 * data$Body_Size &  # Long lower shadow
        data$Upper_Shadow < 0.1 * data$Body_Size  # Short upper shadow
      )
      
      data$Hammer <<- hammer_conditions
      patterns$hammer <<- data$Hammer
    },
    
    #' Detect Shooting Star patterns
    #' 
    #' @param avg_body Average body size
    detect_shooting_star = function(avg_body) {
      # Shooting Star: small body, long upper shadow, short lower shadow
      shooting_star_conditions <- (
        data$Body_Size < 0.3 * avg_body &  # Small body
        data$Upper_Shadow > 2 * data$Body_Size &  # Long upper shadow
        data$Lower_Shadow < 0.1 * data$Body_Size  # Short lower shadow
      )
      
      data$Shooting_Star <<- shooting_star_conditions
      patterns$shooting_star <<- data$Shooting_Star
    },
    
    #' Detect Engulfing patterns
    detect_engulfing_patterns = function() {
      # Bullish Engulfing: current bullish candle engulfs previous bearish candle
      bullish_engulfing <- logical(nrow(data))
      bearish_engulfing <- logical(nrow(data))
      
      for (i in 2:nrow(data)) {
        current <- data[i, ]
        previous <- data[i-1, ]
        
        # Bullish Engulfing
        if (current$Is_Bullish && previous$Is_Bearish &&
            current$Open < previous$Close && current$Close > previous$Open) {
          bullish_engulfing[i] <- TRUE
        }
        
        # Bearish Engulfing
        if (current$Is_Bearish && previous$Is_Bullish &&
            current$Open > previous$Close && current$Close < previous$Open) {
          bearish_engulfing[i] <- TRUE
        }
      }
      
      data$Bullish_Engulfing <<- bullish_engulfing
      data$Bearish_Engulfing <<- bearish_engulfing
      
      patterns$bullish_engulfing <<- data$Bullish_Engulfing
      patterns$bearish_engulfing <<- data$Bearish_Engulfing
    },
    
    #' Detect Star patterns (Morning Star, Evening Star)
    detect_star_patterns = function() {
      morning_star <- logical(nrow(data))
      evening_star <- logical(nrow(data))
      
      for (i in 3:nrow(data)) {
        first <- data[i-2, ]
        second <- data[i-1, ]
        third <- data[i, ]
        
        # Morning Star: bearish, small body, bullish
        if (first$Is_Bearish && 
            abs(second$Close - second$Open) < 0.3 * mean(c(first$Body_Size, third$Body_Size)) &&
            third$Is_Bullish &&
            third$Close > (first$Open + first$Close) / 2) {
          morning_star[i] <- TRUE
        }
        
        # Evening Star: bullish, small body, bearish
        if (first$Is_Bullish && 
            abs(second$Close - second$Open) < 0.3 * mean(c(first$Body_Size, third$Body_Size)) &&
            third$Is_Bearish &&
            third$Close < (first$Open + first$Close) / 2) {
          evening_star[i] <- TRUE
        }
      }
      
      data$Morning_Star <<- morning_star
      data$Evening_Star <<- evening_star
      
      patterns$morning_star <<- data$Morning_Star
      patterns$evening_star <<- data$Evening_Star
    },
    
    #' Detect Three Candle patterns
    detect_three_candle_patterns = function() {
      three_white_soldiers <- logical(nrow(data))
      three_black_crows <- logical(nrow(data))
      
      for (i in 3:nrow(data)) {
        first <- data[i-2, ]
        second <- data[i-1, ]
        third <- data[i, ]
        
        # Three White Soldiers: three consecutive bullish candles
        if (first$Is_Bullish && second$Is_Bullish && third$Is_Bullish &&
            second$Open > first$Open && third$Open > second$Open) {
          three_white_soldiers[i] <- TRUE
        }
        
        # Three Black Crows: three consecutive bearish candles
        if (first$Is_Bearish && second$Is_Bearish && third$Is_Bearish &&
            second$Open < first$Open && third$Open < second$Open) {
          three_black_crows[i] <- TRUE
        }
      }
      
      data$Three_White_Soldiers <<- three_white_soldiers
      data$Three_Black_Crows <<- three_black_crows
      
      patterns$three_white_soldiers <<- data$Three_White_Soldiers
      patterns$three_black_crows <<- data$Three_Black_Crows
    },
    
    #' Detect Hanging Man patterns
    #' 
    #' @param avg_body Average body size
    detect_hanging_man = function(avg_body) {
      # Hanging Man: similar to hammer but appears in downtrend
      trend <- rollapply(data$Close, width = 20, FUN = mean, fill = NA, align = "right")
      
      hanging_man_conditions <- (
        data$Body_Size < 0.3 * avg_body &  # Small body
        data$Lower_Shadow > 2 * data$Body_Size &  # Long lower shadow
        data$Upper_Shadow < 0.1 * data$Body_Size &  # Short upper shadow
        data$Close < trend  # In downtrend
      )
      
      data$Hanging_Man <<- hanging_man_conditions
      patterns$hanging_man <<- data$Hanging_Man
    },
    
    #' Detect Inverted Hammer patterns
    #' 
    #' @param avg_body Average body size
    detect_inverted_hammer = function(avg_body) {
      # Inverted Hammer: similar to shooting star but appears in uptrend
      trend <- rollapply(data$Close, width = 20, FUN = mean, fill = NA, align = "right")
      
      inverted_hammer_conditions <- (
        data$Body_Size < 0.3 * avg_body &  # Small body
        data$Upper_Shadow > 2 * data$Body_Size &  # Long upper shadow
        data$Lower_Shadow < 0.1 * data$Body_Size &  # Short lower shadow
        data$Close > trend  # In uptrend
      )
      
      data$Inverted_Hammer <<- inverted_hammer_conditions
      patterns$inverted_hammer <<- data$Inverted_Hammer
    },
    
    #' Get pattern summary
    #' 
    #' @return Data frame with pattern summary
    get_pattern_summary = function() {
      if (length(patterns) == 0) {
        return(data.frame())
      }
      
      summary_data <- data.frame()
      
      for (pattern_name in names(patterns)) {
        pattern_data <- patterns[[pattern_name]]
        if (is.logical(pattern_data)) {
          count <- sum(pattern_data, na.rm = TRUE)
          percentage <- count / length(pattern_data) * 100
          
          summary <- data.frame(
            Pattern = pattern_name,
            Description = pattern_rules[[pattern_name]]$description,
            Count = count,
            Percentage = round(percentage, 2),
            Bullish = pattern_rules[[pattern_name]]$bullish,
            Bearish = pattern_rules[[pattern_name]]$bearish,
            Neutral = pattern_rules[[pattern_name]]$neutral
          )
          
          summary_data <- rbind(summary_data, summary)
        }
      }
      
      return(summary_data)
    },
    
    #' Get bullish patterns
    #' 
    #' @return Data frame with bullish patterns
    get_bullish_patterns = function() {
      bullish_patterns <- c("Hammer", "Bullish_Engulfing", "Morning_Star", 
                           "Three_White_Soldiers", "Inverted_Hammer")
      
      available_patterns <- bullish_patterns[bullish_patterns %in% names(data)]
      
      if (length(available_patterns) == 0) {
        return(data.frame())
      }
      
      return(data[, available_patterns, drop = FALSE])
    },
    
    #' Get bearish patterns
    #' 
    #' @return Data frame with bearish patterns
    get_bearish_patterns = function() {
      bearish_patterns <- c("Shooting_Star", "Bearish_Engulfing", "Evening_Star", 
                           "Three_Black_Crows", "Hanging_Man")
      
      available_patterns <- bearish_patterns[bearish_patterns %in% names(data)]
      
      if (length(available_patterns) == 0) {
        return(data.frame())
      }
      
      return(data[, available_patterns, drop = FALSE])
    },
    
    #' Visualize patterns on price chart
    #' 
    #' @param pattern_name Specific pattern to highlight (optional)
    #' @param method Visualization method ('plotly', 'ggplot')
    #' 
    #' @return Plotly figure with pattern visualization
    visualize_patterns = function(pattern_name = NULL, method = "plotly") {
      if (method == "plotly") {
        # Create candlestick chart with plotly
        p <- plot_ly(data, type = "candlestick",
                    x = ~Date, open = ~Open, high = ~High, 
                    low = ~Low, close = ~Close,
                    name = "Price") %>%
          layout(title = "Candlestick Patterns",
                 xaxis = list(title = "Date"),
                 yaxis = list(title = "Price"))
        
        # Add pattern markers if specified
        if (!is.null(pattern_name) && pattern_name %in% names(patterns)) {
          pattern_data <- patterns[[pattern_name]]
          pattern_dates <- data$Date[pattern_data]
          pattern_prices <- data$High[pattern_data]
          
          p <- p %>% add_markers(x = pattern_dates, y = pattern_prices,
                                marker = list(color = "red", size = 8),
                                name = pattern_name)
        }
        
        return(p)
        
      } else {
        # Create ggplot visualization
        p <- ggplot(data, aes(x = Date)) +
          geom_segment(aes(xend = Date, y = Low, yend = High), color = "black") +
          geom_segment(aes(xend = Date, y = Open, yend = Close, 
                          color = Is_Bullish), size = 2) +
          scale_color_manual(values = c("red", "green"), 
                           labels = c("Bearish", "Bullish")) +
          labs(title = "Candlestick Patterns",
               x = "Date", y = "Price", color = "Candle Type") +
          theme_minimal()
        
        return(p)
      }
    },
    
    #' Get pattern signals for trading
    #' 
    #' @return Data frame with pattern signals
    get_pattern_signals = function() {
      bullish_patterns <- get_bullish_patterns()
      bearish_patterns <- get_bearish_patterns()
      
      # Combine all patterns
      all_patterns <- cbind(bullish_patterns, bearish_patterns)
      
      # Create signal columns
      signals <- data.frame(
        Date = data$Date,
        Bullish_Signals = rowSums(bullish_patterns, na.rm = TRUE),
        Bearish_Signals = rowSums(bearish_patterns, na.rm = TRUE),
        Total_Signals = rowSums(all_patterns, na.rm = TRUE)
      )
      
      # Add signal strength
      signals$Signal_Strength <- signals$Bullish_Signals - signals$Bearish_Signals
      signals$Signal_Type <- ifelse(signals$Signal_Strength > 0, "Bullish",
                                   ifelse(signals$Signal_Strength < 0, "Bearish", "Neutral"))
      
      return(signals)
    }
  )
)

#' Example usage of candlestick pattern detection
#' 
#' @examples
#' # Load data
#' collector <- CryptoDataCollector()
#' btc_data <- collector$get_ohlcv_data("BTC-USD", period = "6mo")
#' 
#' # Detect patterns
#' pattern_detector <- CandlestickPatternDetector(btc_data)
#' data_with_patterns <- pattern_detector$detect_all_patterns()
#' 
#' # Get pattern summary
#' summary <- pattern_detector$get_pattern_summary()
#' print(summary)
#' 
#' # Get signals
#' signals <- pattern_detector$get_pattern_signals()
#' 
#' # Visualize patterns
#' fig <- pattern_detector$visualize_patterns("Hammer")
#' fig
example_pattern_detection <- function() {
  message("Example usage of candlestick pattern detection.")
  
  # Load data
  collector <- CryptoDataCollector()
  btc_data <- collector$get_ohlcv_data("BTC-USD", period = "6mo")
  
  # Detect patterns
  pattern_detector <- CandlestickPatternDetector(btc_data)
  data_with_patterns <- pattern_detector$detect_all_patterns()
  
  # Get pattern summary
  summary <- pattern_detector$get_pattern_summary()
  print(summary)
  
  # Get signals
  signals <- pattern_detector$get_pattern_signals()
  
  # Visualize patterns
  fig <- pattern_detector$visualize_patterns("Hammer")
  
  return(list(pattern_detector = pattern_detector, summary = summary, 
              signals = signals, figure = fig))
} 