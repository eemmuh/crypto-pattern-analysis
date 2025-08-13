"""
FastAPI web application for crypto trading analysis.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.data.data_collector import CryptoDataCollector
from src.data.database import get_database
from src.models.model_registry import get_model_registry
from src.features.technical_indicators import TechnicalIndicators
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Trading Analysis API",
    description="API for cryptocurrency market analysis and trading insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class DataRequest(BaseModel):
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    period: str = Field("1y", description="Data period")
    interval: str = Field("1d", description="Data interval")

class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Cryptocurrency symbol")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    indicators: List[str] = Field(default=["rsi", "macd", "bollinger"], description="Technical indicators to calculate")

class ModelRequest(BaseModel):
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    hyperparameters: Dict[str, Any] = Field(default={}, description="Model hyperparameters")

# Dependency injection
def get_data_collector():
    return CryptoDataCollector()

def get_database_instance():
    return get_database()

def get_model_registry_instance():
    return get_model_registry()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Data endpoints
@app.get("/api/v1/symbols")
async def get_available_symbols(
    collector: CryptoDataCollector = Depends(get_data_collector)
):
    """Get list of available cryptocurrency symbols."""
    try:
        symbols = collector.get_available_symbols()
        return {
            "symbols": symbols,
            "count": len(symbols),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/ohlcv")
async def get_ohlcv_data(
    request: DataRequest,
    collector: CryptoDataCollector = Depends(get_data_collector)
):
    """Get OHLCV data for a cryptocurrency."""
    try:
        data = collector.get_ohlcv_data(
            symbol=request.symbol,
            period=request.period,
            interval=request.interval
        )
        
        # Convert to JSON-serializable format
        data_dict = {
            "symbol": request.symbol,
            "period": request.period,
            "interval": request.interval,
            "data_points": len(data),
            "start_date": data.index[0].isoformat() if not data.empty else None,
            "end_date": data.index[-1].isoformat() if not data.empty else None,
            "columns": data.columns.tolist(),
            "data": data.tail(100).to_dict('records')  # Last 100 records
        }
        
        return data_dict
        
    except Exception as e:
        logger.error(f"Failed to get OHLCV data for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/analysis")
async def get_technical_analysis(
    request: AnalysisRequest,
    collector: CryptoDataCollector = Depends(get_data_collector)
):
    """Get technical analysis for a cryptocurrency."""
    try:
        # Get data
        if request.start_date and request.end_date:
            data = collector.get_ohlcv_data(
                symbol=request.symbol,
                start_date=request.start_date,
                end_date=request.end_date
            )
        else:
            data = collector.get_ohlcv_data(symbol=request.symbol)
        
        # Calculate technical indicators
        indicators = TechnicalIndicators()
        analysis_results = {}
        
        for indicator in request.indicators:
            try:
                if indicator == "rsi":
                    analysis_results["rsi"] = indicators.calculate_rsi(data['CLOSE']).tail(20).to_dict()
                elif indicator == "macd":
                    macd_data = indicators.calculate_macd(data['CLOSE'])
                    analysis_results["macd"] = {
                        "macd": macd_data['MACD'].tail(20).to_dict(),
                        "signal": macd_data['MACD_SIGNAL'].tail(20).to_dict(),
                        "histogram": macd_data['MACD_HISTOGRAM'].tail(20).to_dict()
                    }
                elif indicator == "bollinger":
                    bb_data = indicators.calculate_bollinger_bands(data['CLOSE'])
                    analysis_results["bollinger_bands"] = {
                        "upper": bb_data['BB_UPPER'].tail(20).to_dict(),
                        "middle": bb_data['BB_MIDDLE'].tail(20).to_dict(),
                        "lower": bb_data['BB_LOWER'].tail(20).to_dict()
                    }
                elif indicator == "sma":
                    analysis_results["sma"] = {
                        "sma_20": indicators.calculate_sma(data['CLOSE'], 20).tail(20).to_dict(),
                        "sma_50": indicators.calculate_sma(data['CLOSE'], 50).tail(20).to_dict()
                    }
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator}: {e}")
                analysis_results[indicator] = {"error": str(e)}
        
        return {
            "symbol": request.symbol,
            "analysis": analysis_results,
            "data_points": len(data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get technical analysis for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Database endpoints
@app.get("/api/v1/database/stats")
async def get_database_stats(
    db: Any = Depends(get_database_instance)
):
    """Get database statistics."""
    try:
        stats = db.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/database/metadata")
async def get_database_metadata(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    db: Any = Depends(get_database_instance)
):
    """Get database metadata."""
    try:
        metadata = db.get_metadata(symbol)
        return {
            "metadata": metadata.to_dict('records') if not metadata.empty else [],
            "count": len(metadata),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get database metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model registry endpoints
@app.get("/api/v1/models")
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    registry: Any = Depends(get_model_registry_instance)
):
    """List all models in the registry."""
    try:
        models = registry.list_models(model_type)
        return {
            "models": models,
            "count": len(models),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/{model_name}")
async def get_model_info(
    model_name: str,
    version: Optional[str] = Query(None, description="Model version"),
    registry: Any = Depends(get_model_registry_instance)
):
    """Get model information."""
    try:
        info = registry.get_model_info(model_name, version)
        return info
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/{model_name}/best")
async def get_best_model(
    model_name: str,
    metric: Optional[str] = Query(None, description="Metric to optimize"),
    higher_is_better: bool = Query(True, description="Whether higher metric values are better"),
    registry: Any = Depends(get_model_registry_instance)
):
    """Get the best performing model."""
    try:
        model, metadata = registry.get_best_model(model_name, metric, higher_is_better)
        return {
            "model_name": model_name,
            "version": metadata['version'],
            "metric": metric,
            "score": metadata['metrics'].get(metric, None),
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get best model for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/registry/stats")
async def get_registry_stats(
    registry: Any = Depends(get_model_registry_instance)
):
    """Get model registry statistics."""
    try:
        stats = registry.get_registry_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Market analysis endpoints
@app.get("/api/v1/market/overview")
async def get_market_overview(
    symbols: List[str] = Query(["BTC", "ETH", "ADA"], description="Symbols to analyze"),
    period: str = Query("1mo", description="Analysis period"),
    collector: CryptoDataCollector = Depends(get_data_collector)
):
    """Get market overview for multiple cryptocurrencies."""
    try:
        market_data = collector.get_multiple_cryptos(symbols, period)
        
        overview = {}
        for symbol, data in market_data.items():
            if not data.empty:
                latest = data.iloc[-1]
                overview[symbol] = {
                    "current_price": float(latest['CLOSE']),
                    "price_change_24h": float(data['CLOSE'].pct_change().iloc[-1] * 100),
                    "volume_24h": float(latest['VOLUME']),
                    "high_24h": float(latest['HIGH']),
                    "low_24h": float(latest['LOW']),
                    "data_points": len(data)
                }
        
        return {
            "market_overview": overview,
            "period": period,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/market/correlation")
async def get_market_correlation(
    symbols: List[str] = Query(["BTC", "ETH", "ADA"], description="Symbols to analyze"),
    period: str = Query("1mo", description="Analysis period"),
    collector: CryptoDataCollector = Depends(get_data_collector)
):
    """Get correlation matrix for multiple cryptocurrencies."""
    try:
        market_data = collector.get_multiple_cryptos(symbols, period)
        
        # Extract close prices
        close_prices = {}
        for symbol, data in market_data.items():
            if not data.empty:
                close_prices[symbol] = data['CLOSE']
        
        if not close_prices:
            raise HTTPException(status_code=400, detail="No valid data found")
        
        # Calculate correlation matrix
        df = pd.DataFrame(close_prices)
        correlation_matrix = df.corr()
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "symbols": symbols,
            "period": period,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get market correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Crypto Trading Analysis API starting up")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Crypto Trading Analysis API shutting down")

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
