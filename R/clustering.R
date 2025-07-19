#' Market Behavior Clustering Analysis
#' 
#' A comprehensive module for clustering market behaviors using tsclust
#' and other R clustering packages for cryptocurrency analysis.
#' 
#' @author Crypto Analyst
#' @keywords clustering, market behavior, cryptocurrency, tsclust

library(TSclust)
library(cluster)
library(factoextra)
library(dplyr)
library(ggplot2)
library(plotly)
library(zoo)
library(xts)

#' MarketBehaviorClusterer Class
#' 
#' Performs clustering analysis on market behavior patterns
#' 
#' @field data Market data with features
#' @field n_clusters Number of clusters
#' @field cluster_results Results from clustering algorithms
#' @field optimal_clusters Optimal number of clusters
#' 
#' @export
MarketBehaviorClusterer <- setRefClass(
  "MarketBehaviorClusterer",
  fields = list(
    data = "data.frame",
    n_clusters = "numeric",
    cluster_results = "list",
    optimal_clusters = "numeric"
  ),
  
  methods = list(
    
    #' Initialize the clusterer
    #' 
    #' @param data Market data with features
    #' @param n_clusters Number of clusters
    initialize = function(data, n_clusters = 5) {
      data <<- data
      n_clusters <<- n_clusters
      cluster_results <<- list()
      optimal_clusters <<- 0
    },
    
    #' Prepare features for clustering
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
    
    #' Perform K-means clustering
    #' 
    #' @param feature_data Feature matrix
    #' @param n_clusters Number of clusters
    #' 
    #' @return List with clustering results
    kmeans_clustering = function(feature_data, n_clusters = NULL) {
      if (is.null(n_clusters)) {
        n_clusters <- n_clusters
      }
      
      message(paste("Performing K-means clustering with", n_clusters, "clusters..."))
      
      # Perform K-means clustering
      kmeans_result <- kmeans(feature_data, centers = n_clusters, nstart = 25)
      
      # Calculate silhouette score
      silhouette_score <- silhouette(kmeans_result$cluster, dist(feature_data))
      silhouette_avg <- mean(silhouette_score[, 3])
      
      # Store results
      cluster_results$kmeans <<- list(
        model = kmeans_result,
        labels = kmeans_result$cluster,
        centers = kmeans_result$centers,
        withinss = kmeans_result$withinss,
        tot.withinss = kmeans_result$tot.withinss,
        betweenss = kmeans_result$betweenss,
        silhouette_score = silhouette_avg,
        n_clusters = n_clusters
      )
      
      message(paste("K-means clustering completed. Silhouette score:", round(silhouette_avg, 3)))
      
      return(cluster_results$kmeans)
    },
    
    #' Perform DBSCAN clustering
    #' 
    #' @param feature_data Feature matrix
    #' @param eps Epsilon parameter for DBSCAN
    #' @param minPts Minimum points parameter for DBSCAN
    #' 
    #' @return List with clustering results
    dbscan_clustering = function(feature_data, eps = 0.5, minPts = 5) {
      message(paste("Performing DBSCAN clustering with eps =", eps, ", minPts =", minPts, "..."))
      
      # Perform DBSCAN clustering
      dbscan_result <- dbscan(feature_data, eps = eps, minPts = minPts)
      
      # Calculate number of clusters (excluding noise)
      n_clusters <- length(unique(dbscan_result$cluster[dbscan_result$cluster != 0]))
      
      # Calculate silhouette score (excluding noise points)
      non_noise_mask <- dbscan_result$cluster != 0
      if (sum(non_noise_mask) > 1 && n_clusters > 1) {
        silhouette_score <- silhouette(dbscan_result$cluster[non_noise_mask], 
                                     dist(feature_data[non_noise_mask, ]))
        silhouette_avg <- mean(silhouette_score[, 3])
      } else {
        silhouette_avg <- 0
      }
      
      # Store results
      cluster_results$dbscan <<- list(
        model = dbscan_result,
        labels = dbscan_result$cluster,
        eps = eps,
        minPts = minPts,
        n_clusters = n_clusters,
        silhouette_score = silhouette_avg
      )
      
      message(paste("DBSCAN clustering completed. Clusters:", n_clusters, 
                   "Silhouette score:", round(silhouette_avg, 3)))
      
      return(cluster_results$dbscan)
    },
    
    #' Perform time series clustering
    #' 
    #' @param feature_data Feature matrix
    #' @param n_clusters Number of clusters
    #' @param window_size Window size for time series segmentation
    #' 
    #' @return List with clustering results
    time_series_clustering = function(feature_data, n_clusters = NULL, window_size = 20) {
      if (is.null(n_clusters)) {
        n_clusters <- n_clusters
      }
      
      message(paste("Performing time series clustering with", n_clusters, 
                   "clusters, window =", window_size, "..."))
      
      # Create time series segments
      segments <- create_time_series_segments(feature_data, window_size)
      
      if (length(segments) < n_clusters) {
        warning(paste("Not enough segments (", length(segments), ") for", n_clusters, "clusters"))
        return(NULL)
      }
      
      # Perform time series clustering using tsclust
      tsclust_result <- tsclust(segments, k = n_clusters, distance = "dtw", 
                               centroid = "mean", seed = 42)
      
      # Calculate silhouette score
      silhouette_score <- silhouette(tsclust_result@cluster, dist(tsclust_result@distmat))
      silhouette_avg <- mean(silhouette_score[, 3])
      
      # Store results
      cluster_results$tsclust <<- list(
        model = tsclust_result,
        labels = tsclust_result@cluster,
        segments = segments,
        window_size = window_size,
        n_clusters = n_clusters,
        silhouette_score = silhouette_avg
      )
      
      message(paste("Time series clustering completed. Silhouette score:", round(silhouette_avg, 3)))
      
      return(cluster_results$tsclust)
    },
    
    #' Create time series segments
    #' 
    #' @param feature_data Feature matrix
    #' @param window_size Window size
    #' 
    #' @return List of time series segments
    create_time_series_segments = function(feature_data, window_size) {
      segments <- list()
      
      for (i in 1:(nrow(feature_data) - window_size + 1)) {
        segment <- feature_data[i:(i + window_size - 1), , drop = FALSE]
        segments[[i]] <- as.matrix(segment)
      }
      
      return(segments)
    },
    
    #' Find optimal number of clusters
    #' 
    #' @param feature_data Feature matrix
    #' @param max_clusters Maximum number of clusters to test
    #' @param method Method to use ('elbow', 'silhouette', 'gap')
    #' 
    #' @return Optimal number of clusters
    find_optimal_clusters = function(feature_data, max_clusters = 10, method = "silhouette") {
      message("Finding optimal number of clusters...")
      
      if (method == "elbow") {
        # Elbow method
        wcss <- numeric(max_clusters)
        for (i in 1:max_clusters) {
          kmeans_result <- kmeans(feature_data, centers = i, nstart = 25)
          wcss[i] <- kmeans_result$tot.withinss
        }
        
        # Find elbow point
        optimal_k <- find_elbow_point(wcss)
        
      } else if (method == "silhouette") {
        # Silhouette method
        silhouette_scores <- numeric(max_clusters)
        for (i in 2:max_clusters) {
          kmeans_result <- kmeans(feature_data, centers = i, nstart = 25)
          silhouette_score <- silhouette(kmeans_result$cluster, dist(feature_data))
          silhouette_scores[i] <- mean(silhouette_score[, 3])
        }
        
        optimal_k <- which.max(silhouette_scores)
        
      } else if (method == "gap") {
        # Gap statistic method
        gap_result <- cluster::clusGap(feature_data, FUN = kmeans, nstart = 25, 
                                      K.max = max_clusters, B = 50)
        optimal_k <- maxSE(gap_result$Tab[, "gap"], gap_result$Tab[, "SE.sim"])
        
      } else {
        stop("Unknown method. Use 'elbow', 'silhouette', or 'gap'")
      }
      
      optimal_clusters <<- optimal_k
      message(paste("Optimal number of clusters:", optimal_k))
      
      return(optimal_k)
    },
    
    #' Find elbow point in WCSS curve
    #' 
    #' @param wcss Within-cluster sum of squares
    #' 
    #' @return Elbow point
    find_elbow_point = function(wcss) {
      # Simple elbow detection
      diffs <- diff(wcss)
      diffs2 <- diff(diffs)
      elbow_point <- which.max(abs(diffs2)) + 1
      
      return(elbow_point)
    },
    
    #' Get cluster summary
    #' 
    #' @param cluster_type Type of clustering ('kmeans', 'dbscan', 'tsclust')
    #' 
    #' @return Data frame with cluster summary
    get_cluster_summary = function(cluster_type = "kmeans") {
      if (!(cluster_type %in% names(cluster_results))) {
        stop(paste("Clustering", cluster_type, "not found"))
      }
      
      result <- cluster_results[[cluster_type]]
      labels <- result$labels
      
      # Create summary
      summary_data <- data.frame(
        Cluster = 1:result$n_clusters,
        Size = sapply(1:result$n_clusters, function(i) sum(labels == i)),
        Percentage = sapply(1:result$n_clusters, function(i) 
          sum(labels == i) / length(labels) * 100)
      )
      
      return(summary_data)
    },
    
    #' Analyze clusters
    #' 
    #' @param cluster_type Type of clustering
    #' @param feature_data Feature data
    #' 
    #' @return List with cluster analysis
    analyze_clusters = function(cluster_type = "kmeans", feature_data) {
      if (!(cluster_type %in% names(cluster_results))) {
        stop(paste("Clustering", cluster_type, "not found"))
      }
      
      result <- cluster_results[[cluster_type]]
      labels <- result$labels
      
      # Analyze each cluster
      cluster_analysis <- list()
      
      for (i in 1:result$n_clusters) {
        cluster_mask <- labels == i
        cluster_data <- feature_data[cluster_mask, , drop = FALSE]
        
        if (nrow(cluster_data) > 0) {
          cluster_analysis[[paste0("cluster_", i)]] <- list(
            size = nrow(cluster_data),
            mean_features = sapply(cluster_data, mean, na.rm = TRUE),
            std_features = sapply(cluster_data, sd, na.rm = TRUE)
          )
        }
      }
      
      return(cluster_analysis)
    },
    
    #' Visualize clusters
    #' 
    #' @param cluster_type Type of clustering
    #' @param feature_data Feature data
    #' @param method Visualization method ('pca', 'tsne', 'umap')
    #' 
    #' @return Plotly figure with cluster visualization
    visualize_clusters = function(cluster_type = "kmeans", feature_data, method = "pca") {
      if (!(cluster_type %in% names(cluster_results))) {
        stop(paste("Clustering", cluster_type, "not found"))
      }
      
      result <- cluster_results[[cluster_type]]
      labels <- result$labels
      
      # Reduce dimensionality for visualization
      if (method == "pca") {
        pca_result <- prcomp(feature_data, scale. = TRUE)
        coords <- pca_result$x[, 1:2]
        method_name <- "PCA"
      } else if (method == "tsne") {
        if (!requireNamespace("Rtsne", quietly = TRUE)) {
          stop("Rtsne package is required for t-SNE visualization")
        }
        library(Rtsne)
        tsne_result <- Rtsne(feature_data, perplexity = 30, theta = 0.5, dims = 2)
        coords <- tsne_result$Y
        method_name <- "t-SNE"
      } else if (method == "umap") {
        if (!requireNamespace("umap", quietly = TRUE)) {
          stop("umap package is required for UMAP visualization")
        }
        library(umap)
        umap_result <- umap(feature_data)
        coords <- umap_result$layout
        method_name <- "UMAP"
      } else {
        stop(paste("Unknown visualization method:", method))
      }
      
      # Create visualization
      plot_data <- data.frame(
        X = coords[, 1],
        Y = coords[, 2],
        Cluster = factor(labels)
      )
      
      p <- plot_ly(plot_data, x = ~X, y = ~Y, color = ~Cluster, 
                  type = 'scatter', mode = 'markers') %>%
        layout(title = paste("Cluster Visualization using", method_name),
               xaxis = list(title = paste(method_name, "1")),
               yaxis = list(title = paste(method_name, "2")))
      
      return(p)
    }
  )
)

#' Example usage of market behavior clustering
#' 
#' @examples
#' # Load data and calculate indicators
#' collector <- CryptoDataCollector()
#' btc_data <- collector$get_ohlcv_data("BTC-USD", period = "6mo")
#' 
#' ti <- TechnicalIndicators(btc_data)
#' data_with_indicators <- ti$add_all_indicators()
#' 
#' # Get feature matrix
#' features <- ti$get_feature_matrix()
#' 
#' # Perform clustering
#' clusterer <- MarketBehaviorClusterer(btc_data)
#' feature_data <- clusterer$prepare_features()
#' 
#' # K-means clustering
#' kmeans_results <- clusterer$kmeans_clustering(feature_data)
#' 
#' # DBSCAN clustering
#' dbscan_results <- clusterer$dbscan_clustering(feature_data)
#' 
#' # Time series clustering
#' tsclust_results <- clusterer$time_series_clustering(feature_data)
#' 
#' # Find optimal clusters
#' optimal <- clusterer$find_optimal_clusters(feature_data)
#' 
#' # Get summary
#' print(clusterer$get_cluster_summary())
#' 
#' # Analyze clusters
#' kmeans_analysis <- clusterer$analyze_clusters("kmeans", features)
#' 
#' # Visualize clusters
#' fig <- clusterer$visualize_clusters("kmeans", features, method = "pca")
#' fig
example_clustering_usage <- function() {
  message("Example usage of market behavior clustering.")
  
  # Load data and calculate indicators
  collector <- CryptoDataCollector()
  btc_data <- collector$get_ohlcv_data("BTC-USD", period = "6mo")
  
  ti <- TechnicalIndicators(btc_data)
  data_with_indicators <- ti$add_all_indicators()
  
  # Get feature matrix
  features <- ti$get_feature_matrix()
  
  # Perform clustering
  clusterer <- MarketBehaviorClusterer(btc_data)
  feature_data <- clusterer$prepare_features()
  
  # K-means clustering
  kmeans_results <- clusterer$kmeans_clustering(feature_data)
  
  # DBSCAN clustering
  dbscan_results <- clusterer$dbscan_clustering(feature_data)
  
  # Time series clustering
  tsclust_results <- clusterer$time_series_clustering(feature_data)
  
  # Find optimal clusters
  optimal <- clusterer$find_optimal_clusters(feature_data)
  
  # Get summary
  print(clusterer$get_cluster_summary())
  
  # Analyze clusters
  kmeans_analysis <- clusterer$analyze_clusters("kmeans", features)
  
  # Visualize clusters
  fig <- clusterer$visualize_clusters("kmeans", features, method = "pca")
  
  return(list(clusterer = clusterer, kmeans_results = kmeans_results, 
              dbscan_results = dbscan_results, tsclust_results = tsclust_results,
              optimal = optimal, figure = fig))
} 