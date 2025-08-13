"""
Database layer for cryptocurrency data persistence.
"""

import sqlite3
import pandas as pd
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging

from src.utils.logging_config import get_logger
from src.utils.error_handling import DataCollectionError, ValidationError

logger = get_logger(__name__)


class CryptoDatabase:
    """
    SQLite database for storing cryptocurrency market data.
    """
    
    def __init__(self, db_path: str = "data/crypto_data.db"):
        """
        Initialize the database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create OHLCV data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    period TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    start_date DATETIME,
                    end_date DATETIME,
                    data_points INTEGER,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    quality_score REAL,
                    UNIQUE(symbol, period, interval)
                )
            """)
            
            # Create features table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, feature_name)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON ohlcv_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON data_metadata(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol ON technical_features(symbol, timestamp)")
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def store_ohlcv_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store OHLCV data in the database.
        
        Args:
            symbol: Cryptocurrency symbol
            data: DataFrame with OHLCV data
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare data for insertion
                data_to_insert = []
                for timestamp, row in data.iterrows():
                    data_to_insert.append((
                        symbol.upper(),
                        timestamp,
                        row['OPEN'],
                        row['HIGH'],
                        row['LOW'],
                        row['CLOSE'],
                        row['VOLUME']
                    ))
                
                # Insert data with conflict resolution
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO ohlcv_data 
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, data_to_insert)
                
                # Update metadata
                self._update_metadata(symbol, data, conn)
                
                conn.commit()
                logger.info(f"Stored {len(data_to_insert)} records for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store data for {symbol}: {e}")
            raise DataCollectionError(f"Database storage failed for {symbol}: {e}")
    
    def _update_metadata(self, symbol: str, data: pd.DataFrame, conn: sqlite3.Connection):
        """Update metadata for stored data."""
        cursor = conn.cursor()
        
        metadata = {
            'symbol': symbol.upper(),
            'period': 'custom',  # Could be enhanced to detect period
            'interval': '1d',    # Could be enhanced to detect interval
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'data_points': len(data),
            'quality_score': self._calculate_quality_score(data)
        }
        
        cursor.execute("""
            INSERT OR REPLACE INTO data_metadata 
            (symbol, period, interval, start_date, end_date, data_points, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata['symbol'], metadata['period'], metadata['interval'],
            metadata['start_date'], metadata['end_date'], 
            metadata['data_points'], metadata['quality_score']
        ))
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        if data.empty:
            return 0.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        # Check for zero volumes
        zero_volume_ratio = (data['VOLUME'] == 0).sum() / len(data)
        
        # Check for price anomalies
        price_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        price_anomalies = 0
        for col in price_columns:
            if col in data.columns:
                # Check for negative prices
                price_anomalies += (data[col] < 0).sum()
                # Check for extreme price changes (>50% in one day)
                if col == 'CLOSE':
                    returns = data[col].pct_change().abs()
                    price_anomalies += (returns > 0.5).sum()
        
        anomaly_ratio = price_anomalies / (len(data) * len(price_columns))
        
        # Calculate quality score (0-1, higher is better)
        quality_score = 1.0 - (missing_ratio + zero_volume_ratio + anomaly_ratio)
        return max(0.0, min(1.0, quality_score))
    
    def get_ohlcv_data(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve OHLCV data from database.
        
        Args:
            symbol: Cryptocurrency symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE symbol = ?
            """
            params = [symbol.upper()]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
                
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                    df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
                
                logger.info(f"Retrieved {len(df)} records for {symbol}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to retrieve data for {symbol}: {e}")
            raise DataCollectionError(f"Database retrieval failed for {symbol}: {e}")
    
    def get_metadata(self, symbol: str = None) -> pd.DataFrame:
        """
        Get metadata for stored data.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            DataFrame with metadata
        """
        try:
            query = "SELECT * FROM data_metadata"
            params = []
            
            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol.upper())
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['start_date', 'end_date', 'last_updated'])
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to retrieve metadata: {e}")
            raise DataCollectionError(f"Metadata retrieval failed: {e}")
    
    def store_features(self, symbol: str, features: pd.DataFrame) -> bool:
        """
        Store technical features in the database.
        
        Args:
            symbol: Cryptocurrency symbol
            features: DataFrame with technical features
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for timestamp, row in features.iterrows():
                    for feature_name, value in row.items():
                        if pd.notna(value):
                            cursor.execute("""
                                INSERT OR REPLACE INTO technical_features 
                                (symbol, timestamp, feature_name, feature_value)
                                VALUES (?, ?, ?, ?)
                            """, (symbol.upper(), timestamp, feature_name, value))
                
                conn.commit()
                logger.info(f"Stored {len(features.columns)} features for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store features for {symbol}: {e}")
            raise DataCollectionError(f"Feature storage failed for {symbol}: {e}")
    
    def get_features(self, symbol: str, feature_names: List[str] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve technical features from database.
        
        Args:
            symbol: Cryptocurrency symbol
            feature_names: List of feature names to retrieve
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with features
        """
        try:
            query = """
                SELECT timestamp, feature_name, feature_value
                FROM technical_features 
                WHERE symbol = ?
            """
            params = [symbol.upper()]
            
            if feature_names:
                placeholders = ','.join(['?' for _ in feature_names])
                query += f" AND feature_name IN ({placeholders})"
                params.extend(feature_names)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
                
                if not df.empty:
                    # Pivot to wide format
                    df = df.pivot(index='timestamp', columns='feature_name', values='feature_value')
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to retrieve features for {symbol}: {e}")
            raise DataCollectionError(f"Feature retrieval failed for {symbol}: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """
        Clean up old data from the database.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old OHLCV data
                cursor.execute("DELETE FROM ohlcv_data WHERE timestamp < ?", (cutoff_date,))
                ohlcv_deleted = cursor.rowcount
                
                # Delete old features
                cursor.execute("DELETE FROM technical_features WHERE timestamp < ?", (cutoff_date,))
                features_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up {ohlcv_deleted} OHLCV records and {features_deleted} feature records")
                return ohlcv_deleted + features_deleted
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            raise DataCollectionError(f"Database cleanup failed: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table sizes
                cursor.execute("SELECT COUNT(*) FROM ohlcv_data")
                ohlcv_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM technical_features")
                features_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM data_metadata")
                metadata_count = cursor.fetchone()[0]
                
                # Get unique symbols
                cursor.execute("SELECT DISTINCT symbol FROM ohlcv_data")
                symbols = [row[0] for row in cursor.fetchall()]
                
                # Get database size
                db_size = self.db_path.stat().st_size / (1024 * 1024)  # MB
                
                return {
                    'ohlcv_records': ohlcv_count,
                    'feature_records': features_count,
                    'metadata_records': metadata_count,
                    'unique_symbols': symbols,
                    'database_size_mb': db_size,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise DataCollectionError(f"Database stats failed: {e}")


# Singleton instance
_db_instance = None

def get_database() -> CryptoDatabase:
    """Get database singleton instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = CryptoDatabase()
    return _db_instance
