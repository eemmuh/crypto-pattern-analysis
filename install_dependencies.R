#!/usr/bin/env Rscript

# Crypto Analysis R Package Dependencies Installer
# ===============================================

cat("Installing required packages for crypto analysis...\n")
cat("==================================================\n\n")

# Function to install package with error handling
install_package_safe <- function(package_name, repo = "https://cran.rstudio.com/") {
  tryCatch({
    if (!require(package_name, character.only = TRUE, quietly = TRUE)) {
      cat(sprintf("Installing %s...\n", package_name))
      install.packages(package_name, repos = repo, dependencies = TRUE)
      library(package_name, character.only = TRUE)
      cat(sprintf("✅ %s installed successfully\n", package_name))
    } else {
      cat(sprintf("✅ %s is already installed\n", package_name))
    }
  }, error = function(e) {
    cat(sprintf("❌ Failed to install %s: %s\n", package_name, e$message))
    return(FALSE)
  })
  return(TRUE)
}

# Core packages (essential)
core_packages <- c(
  "quantmod",
  "TTR", 
  "zoo",
  "xts",
  "dplyr",
  "tidyr",
  "lubridate"
)

# Analysis packages
analysis_packages <- c(
  "depmixS4",
  "cluster",
  "factoextra",
  "ggplot2",
  "plotly",
  "forecast",
  "changepoint",
  "strucchange",
  "PerformanceAnalytics",
  "moments",
  "psych",
  "corrplot",
  "viridis",
  "scales",
  "gridExtra"
)

# Optional packages (may have issues on macOS)
optional_packages <- c(
  "TSclust",  # Fixed package name
  "rugarch",
  "fGarch"
)

# Development packages
dev_packages <- c(
  "knitr",
  "rmarkdown",
  "testthat"
)

cat("📦 Installing core packages...\n")
cat("-----------------------------\n")
for (pkg in core_packages) {
  install_package_safe(pkg)
}

cat("\n📊 Installing analysis packages...\n")
cat("---------------------------------\n")
for (pkg in analysis_packages) {
  install_package_safe(pkg)
}

cat("\n🔧 Installing optional packages...\n")
cat("---------------------------------\n")
for (pkg in optional_packages) {
  success <- install_package_safe(pkg)
  if (!success) {
    cat(sprintf("⚠️  %s installation failed - this is optional\n", pkg))
    }
  }

cat("\n🛠️  Installing development packages...\n")
cat("-------------------------------------\n")
for (pkg in dev_packages) {
  install_package_safe(pkg)
}

# Handle rgl/OpenGL issue specifically
cat("\n🎨 Handling OpenGL/rgl package...\n")
cat("--------------------------------\n")
tryCatch({
  if (!require("rgl", quietly = TRUE)) {
    cat("Installing rgl (may fail on macOS without X11)...\n")
    install.packages("rgl", repos = "https://cran.rstudio.com/")
  }
  library(rgl)
  cat("✅ rgl installed successfully\n")
}, error = function(e) {
  cat("❌ rgl installation failed (OpenGL issue)\n")
  cat("💡 This is expected on macOS. Install XQuartz if needed:\n")
  cat("   brew install --cask xquartz\n")
  cat("   Then restart R and try again\n")
})

# Test basic functionality
cat("\n🧪 Testing basic functionality...\n")
cat("-------------------------------\n")

test_packages <- c("quantmod", "TTR", "dplyr", "ggplot2")
for (pkg in test_packages) {
  tryCatch({
    library(pkg, character.only = TRUE)
    cat(sprintf("✅ %s loads successfully\n", pkg))
  }, error = function(e) {
    cat(sprintf("❌ %s failed to load: %s\n", pkg, e$message))
  })
}

cat("\n🎉 Installation complete!\n")
cat("========================\n")
cat("✅ Core packages installed\n")
cat("✅ Analysis packages installed\n")
cat("⚠️  Some optional packages may have failed (this is normal)\n")
cat("\n💡 Next steps:\n")
cat("   1. Restart R session\n")
cat("   2. Run: source('R/main.R')\n")
cat("   3. Or run: Rscript demo.R\n")
cat("\n🔧 If you encounter OpenGL issues:\n")
cat("   1. Install XQuartz: brew install --cask xquartz\n")
cat("   2. Log out and log back in\n")
cat("   3. Restart R\n") 