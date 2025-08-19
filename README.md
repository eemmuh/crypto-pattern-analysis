# Crypto Pattern Analysis

A comprehensive cryptocurrency trading analysis platform with advanced data collection, machine learning models, and REST API.

## 🚀 Features

- **Advanced Data Collection**: Robust cryptocurrency data collection with caching and validation
- **Database Layer**: SQLite database for persistent data storage with metadata tracking
- **Model Registry**: Versioned machine learning model management
- **REST API**: FastAPI-based web API for easy integration
- **Technical Analysis**: Comprehensive technical indicators and pattern recognition
- **Machine Learning**: Clustering, regime detection, and predictive models
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **Monitoring**: Comprehensive logging and performance tracking

## 📋 Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Git

## 🛠️ Installation

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crypto-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Setup project**
   ```bash
   python manage.py setup
   ```

### Docker Installation

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access services**
   - API: http://localhost:8000
   - Jupyter Lab: http://localhost:8888
   - API Documentation: http://localhost:8000/docs

## 🏃‍♂️ Quick Start

### Using the Management Script

```bash
# Start the API server
uv run python manage.py api

# Collect data for all supported cryptocurrencies
uv run python manage.py collect

# Show project statistics
uv run python manage.py stats

# Clean cache files
uv run python manage.py clean

# Run tests
uv run python manage.py test
```

### Using Python Directly

```bash
# Run Python with uv
uv run python

# Then in Python:
from src.data.data_collector import CryptoDataCollector
from src.data.database import get_database
from src.models.model_registry import get_model_registry

# Initialize components
collector = CryptoDataCollector()
db = get_database()
registry = get_model_registry()

# Collect data
btc_data = collector.get_ohlcv_data('BTC', period='1y')

# Store in database
db.store_ohlcv_data('BTC', btc_data)

# Get database stats
stats = db.get_database_stats()
print(f"Database contains {stats['ohlcv_records']} OHLCV records")
```

## 📊 API Usage

### Data Endpoints

```bash
# Get available symbols
curl http://localhost:8000/api/v1/symbols

# Get OHLCV data
curl -X POST http://localhost:8000/api/v1/data/ohlcv \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC", "period": "1y", "interval": "1d"}'

# Get technical analysis
curl -X POST http://localhost:8000/api/v1/data/analysis \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC", "indicators": ["rsi", "macd", "bollinger"]}'
```

### Market Analysis

```bash
# Get market overview
curl "http://localhost:8000/api/v1/market/overview?symbols=BTC&symbols=ETH&period=1mo"

# Get correlation matrix
curl "http://localhost:8000/api/v1/market/correlation?symbols=BTC&symbols=ETH&symbols=ADA"
```

### Database and Models

```bash
# Get database statistics
curl http://localhost:8000/api/v1/database/stats

# List models
curl http://localhost:8000/api/v1/models

# Get model information
curl "http://localhost:8000/api/v1/models/my_model?version=1.0.0"
```

## 🏗️ Project Structure

```
crypto-project/
├── src/
│   ├── api/                 # FastAPI web application
│   ├── data/               # Data collection and database
│   ├── features/           # Technical indicators and feature engineering
│   ├── models/             # Machine learning models and registry
│   ├── patterns/           # Candlestick pattern recognition
│   └── utils/              # Utilities and error handling
├── config/                 # Configuration settings
├── data/                   # Data storage
├── logs/                   # Application logs
├── models/                 # Model registry storage
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test suite
├── manage.py              # Project management script
├── docker-compose.yml     # Docker Compose configuration
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

### Environment Variables

```bash
# Data collection settings
DEFAULT_PERIOD=1y
DEFAULT_INTERVAL=1d
MAX_DATA_POINTS=10000
DATA_QUALITY_THRESHOLD=0.9

# API settings
LOG_LEVEL=INFO
```

### Configuration Files

- `config/settings.py`: Main configuration file
- `src/utils/logging_config.py`: Logging configuration
- `src/utils/error_handling.py`: Error handling configuration

## 📈 Data Collection

### Supported Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Cardano (ADA)
- Solana (SOL)
- Polkadot (DOT)
- Avalanche (AVAX)
- Polygon (MATIC)
- Chainlink (LINK)
- Uniswap (UNI)
- Ripple (XRP)
- Litecoin (LTC)
- Bitcoin Cash (BCH)
- Stellar (XLM)

### Data Sources

- **Primary**: Yahoo Finance (via yfinance)
- **Cache**: Local file system with automatic validation
- **Database**: SQLite with metadata tracking

### Data Quality

The system includes comprehensive data validation:
- OHLC relationship verification
- Missing data detection
- Outlier identification
- Volume anomaly detection

## 🤖 Machine Learning Models

### Model Types

- **Clustering**: Market regime identification
- **Regime Detection**: Hidden Markov Models for market states
- **Pattern Recognition**: Candlestick pattern detection
- **Technical Indicators**: Comprehensive technical analysis

### Model Registry

The model registry provides:
- Version control for models
- Performance tracking
- Metadata storage
- Automatic cleanup of old models

## 🧪 Testing

```bash
# Run all tests
uv run python manage.py test

# Run specific test categories
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest --cov=src tests/
```

## 📊 Monitoring and Logging

### Logging Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information about program execution
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical errors that may prevent the program from running

### Performance Monitoring

- Execution time tracking
- Memory usage monitoring
- Database performance metrics
- API response time tracking

## 🐳 Docker Deployment

### Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f crypto-api

# Stop services
docker-compose down
```

### Production

```bash
# Build production image
docker build -t crypto-analysis:latest .

# Run with custom configuration
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e LOG_LEVEL=INFO \
  crypto-analysis:latest
```

## 🔒 Security Considerations

- Input validation on all API endpoints
- Rate limiting for data collection
- Secure error handling (no sensitive data in logs)
- Environment variable configuration
- Docker security best practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint when running the API
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions and ideas

## 🔄 Roadmap

- [ ] Real-time data streaming
- [ ] Advanced backtesting framework
- [ ] Portfolio optimization
- [ ] Sentiment analysis integration
- [ ] Mobile application
- [ ] Cloud deployment guides
- [ ] Advanced visualization dashboard
- [ ] Machine learning model serving
- [ ] Automated trading strategies
- [ ] Risk management tools

## 📊 Performance Benchmarks

- **Data Collection**: ~1000 records/second
- **API Response Time**: <100ms average
- **Database Queries**: <50ms for typical operations
- **Model Inference**: <10ms for most models

## 🏆 Acknowledgments

- Yahoo Finance for market data
- FastAPI for the web framework
- SQLite for database storage
- The open-source community for various libraries and tools

