#!/usr/bin/env python3
"""
Project management script for crypto trading analysis.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import get_logger, setup_logging
from src.data.data_collector import CryptoDataCollector
from src.data.database import get_database
from src.models.model_registry import get_model_registry

logger = get_logger(__name__)


def run_api():
    """Run the FastAPI application."""
    try:
        logger.info("Starting FastAPI application...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api.app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        logger.info("API server stopped")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        sys.exit(1)


def collect_data():
    """Collect data for all supported symbols."""
    try:
        logger.info("Starting data collection...")
        collector = CryptoDataCollector()
        symbols = collector.get_available_symbols()
        
        logger.info(f"Collecting data for {len(symbols)} symbols...")
        data_dict = collector.get_multiple_cryptos(symbols)
        
        # Store in database
        db = get_database()
        for symbol, data in data_dict.items():
            try:
                db.store_ohlcv_data(symbol, data)
                logger.info(f"Stored data for {symbol}: {len(data)} records")
            except Exception as e:
                logger.warning(f"Failed to store data for {symbol}: {e}")
        
        logger.info("Data collection completed")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)


def clean_cache():
    """Clean old cache files."""
    try:
        logger.info("Cleaning cache...")
        collector = CryptoDataCollector()
        cache_info = collector.get_cache_info()
        
        if cache_info['total_files'] > 0:
            collector.clear_cache()
            logger.info(f"Cleaned {cache_info['total_files']} cache files")
        else:
            logger.info("No cache files to clean")
            
    except Exception as e:
        logger.error(f"Cache cleaning failed: {e}")
        sys.exit(1)


def show_stats():
    """Show project statistics."""
    try:
        logger.info("Gathering project statistics...")
        
        # Data collector stats
        collector = CryptoDataCollector()
        symbols = collector.get_available_symbols()
        cache_info = collector.get_cache_info()
        
        print("\n=== DATA COLLECTOR STATS ===")
        print(f"Supported symbols: {len(symbols)}")
        print(f"Cache files: {cache_info['total_files']}")
        print(f"Cache size: {cache_info['total_size_mb']:.2f} MB")
        
        # Database stats
        try:
            db = get_database()
            db_stats = db.get_database_stats()
            
            print("\n=== DATABASE STATS ===")
            print(f"OHLCV records: {db_stats['ohlcv_records']}")
            print(f"Feature records: {db_stats['feature_records']}")
            print(f"Metadata records: {db_stats['metadata_records']}")
            print(f"Unique symbols: {len(db_stats['unique_symbols'])}")
            print(f"Database size: {db_stats['database_size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"Database stats unavailable: {e}")
        
        # Model registry stats
        try:
            registry = get_model_registry()
            registry_stats = registry.get_registry_stats()
            
            print("\n=== MODEL REGISTRY STATS ===")
            print(f"Total models: {registry_stats['total_models']}")
            print(f"Registry size: {registry_stats['total_size_mb']:.2f} MB")
            print(f"Model types: {registry_stats['model_types']}")
            print(f"Unique model names: {registry_stats['unique_model_names']}")
            
        except Exception as e:
            print(f"Model registry stats unavailable: {e}")
        
        print(f"\nLast updated: {datetime.now().isoformat()}")
        
    except Exception as e:
        logger.error(f"Failed to show stats: {e}")
        sys.exit(1)


def run_tests():
    """Run project tests."""
    try:
        logger.info("Running tests...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)


def setup_project():
    """Setup project directories and initial configuration."""
    try:
        logger.info("Setting up project...")
        
        # Create necessary directories
        directories = [
            "data/cache",
            "data/database",
            "logs",
            "models/registry",
            "notebooks",
            "reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Initialize logging
        setup_logging()
        
        # Test data collector
        collector = CryptoDataCollector()
        symbols = collector.get_available_symbols()
        logger.info(f"Data collector initialized with {len(symbols)} symbols")
        
        # Test database
        db = get_database()
        stats = db.get_database_stats()
        logger.info("Database initialized successfully")
        
        # Test model registry
        registry = get_model_registry()
        registry_stats = registry.get_registry_stats()
        logger.info("Model registry initialized successfully")
        
        logger.info("Project setup completed successfully")
        
    except Exception as e:
        logger.error(f"Project setup failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Crypto Trading Analysis Project Manager")
    parser.add_argument("command", choices=[
        "api", "collect", "clean", "stats", "test", "setup"
    ], help="Command to execute")
    
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Execute command
    if args.command == "api":
        run_api()
    elif args.command == "collect":
        collect_data()
    elif args.command == "clean":
        clean_cache()
    elif args.command == "stats":
        show_stats()
    elif args.command == "test":
        run_tests()
    elif args.command == "setup":
        setup_project()
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
