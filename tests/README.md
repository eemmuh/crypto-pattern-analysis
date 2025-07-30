# Testing Infrastructure

This directory contains the comprehensive testing infrastructure for the Crypto Trading Analysis project.

## 🏗️ Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── README.md               # This file
├── unit/                   # Unit tests
│   ├── test_data_collector.py
│   ├── test_technical_indicators.py
│   ├── test_patterns.py
│   └── test_models.py
├── integration/            # Integration tests
│   ├── test_end_to_end.py
│   └── test_api_integration.py
├── performance/            # Performance tests
│   └── test_benchmarks.py
└── test_data/             # Test data files
    └── sample_data.csv
```

## 🚀 Quick Start

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with coverage
make test-coverage

# Run fast tests (skip slow ones)
make test-fast

# Run performance benchmarks
make benchmark
```

### Using the Test Runner

```bash
# Run specific test types
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type coverage --html

# Run with options
python run_tests.py --type all --verbose --parallel --fast
```

### Direct Pytest Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_data_collector.py

# Run tests with markers
pytest -m unit
pytest -m "not slow"
pytest -m "unit or integration"

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📋 Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions and classes in isolation
- **Scope**: Single module or class
- **Speed**: Fast (< 1 second per test)
- **Dependencies**: Minimal, use mocks for external dependencies

### Integration Tests (`tests/integration/`)
- **Purpose**: Test interactions between multiple components
- **Scope**: Multiple modules working together
- **Speed**: Medium (1-10 seconds per test)
- **Dependencies**: May use real data or external services

### Performance Tests (`tests/performance/`)
- **Purpose**: Test performance characteristics and benchmarks
- **Scope**: End-to-end performance scenarios
- **Speed**: Slow (> 10 seconds per test)
- **Dependencies**: Real data, performance monitoring

## 🎯 Test Markers

Use these markers to categorize and run specific tests:

```python
import pytest

@pytest.mark.unit
def test_data_collector():
    """Unit test for data collector."""
    pass

@pytest.mark.integration
def test_end_to_end_analysis():
    """Integration test for complete analysis pipeline."""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Slow test with large dataset."""
    pass

@pytest.mark.performance
def test_clustering_performance():
    """Performance test for clustering algorithm."""
    pass
```

## 🔧 Test Fixtures

### Available Fixtures

```python
def test_with_sample_data(sample_crypto_data):
    """Test using sample cryptocurrency data."""
    assert len(sample_crypto_data) > 0

def test_with_market_data(sample_market_data):
    """Test using multiple cryptocurrency data."""
    assert 'BTC' in sample_market_data

def test_with_temp_cache(temp_cache_dir):
    """Test with temporary cache directory."""
    assert os.path.exists(temp_cache_dir)

def test_with_mock_api(mock_yfinance):
    """Test with mocked yfinance API."""
    # API calls are mocked
    pass
```

### Creating Custom Fixtures

```python
@pytest.fixture
def custom_data():
    """Custom fixture for specific test data."""
    return pd.DataFrame({
        'OPEN': [100, 101, 102],
        'CLOSE': [101, 102, 103],
        'HIGH': [102, 103, 104],
        'LOW': [99, 100, 101],
        'VOLUME': [1000, 1100, 1200]
    })
```

## 📊 Coverage Reporting

### HTML Coverage Report
```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Terminal Coverage Report
```bash
pytest --cov=src --cov-report=term-missing
```

### XML Coverage Report (for CI)
```bash
pytest --cov=src --cov-report=xml
```

## 🧪 Test Data Management

### Sample Data Generation
The test fixtures automatically generate realistic sample data:

```python
@pytest.fixture(scope="session")
def sample_crypto_data():
    """Generate 100 days of realistic Bitcoin data."""
    # Generates OHLCV data with realistic price movements
    # Uses reproducible random seed for consistent tests
```

### Mocking External APIs
```python
@pytest.fixture
def mock_yfinance(monkeypatch):
    """Mock yfinance to avoid actual API calls."""
    # Returns sample data instead of making real API calls
    # Ensures tests are fast and reliable
```

## 🔍 Debugging Tests

### Verbose Output
```bash
pytest -v -s  # Verbose with print statements
pytest -vv    # More verbose
pytest -vvv   # Maximum verbosity
```

### Debugging Specific Tests
```bash
# Run single test
pytest tests/unit/test_data_collector.py::TestCryptoDataCollector::test_initialization

# Run tests matching pattern
pytest -k "test_data_collector"

# Stop on first failure
pytest -x

# Drop into debugger on failures
pytest --pdb
```

### Test Isolation
```bash
# Run tests in parallel
pytest -n auto

# Run tests in random order
pytest --random-order

# Run tests with specific seed
pytest --random-order-seed=42
```

## 📈 Performance Testing

### Benchmark Tests
```python
import pytest
import time

@pytest.mark.performance
def test_data_collection_performance(benchmark):
    """Benchmark data collection performance."""
    def collect_data():
        collector = CryptoDataCollector()
        return collector.get_ohlcv_data('BTC', period='1mo')
    
    result = benchmark(collect_data)
    assert result is not None
```

### Performance Assertions
```python
def test_fast_operation():
    """Test that operation completes within time limit."""
    import time
    
    start_time = time.time()
    # Perform operation
    result = some_operation()
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < 1.0  # Should complete within 1 second
```

## 🚨 Error Handling Tests

### Testing Exceptions
```python
def test_invalid_symbol():
    """Test handling of invalid cryptocurrency symbol."""
    collector = CryptoDataCollector()
    
    with pytest.raises(ValueError, match="Invalid symbol"):
        collector.get_ohlcv_data('INVALID', period='1mo')

def test_network_failure():
    """Test handling of network failures."""
    collector = CryptoDataCollector()
    
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.side_effect = Exception("Network error")
        
        # Should fall back to sample data
        data = collector.get_ohlcv_data('BTC', period='1mo')
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
```

## 🔄 Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          make test-coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 📝 Best Practices

### Writing Good Tests
1. **Test one thing at a time** - Each test should verify one specific behavior
2. **Use descriptive names** - Test names should clearly describe what they test
3. **Arrange-Act-Assert** - Structure tests with clear sections
4. **Test edge cases** - Include tests for boundary conditions and error cases
5. **Keep tests fast** - Unit tests should run quickly
6. **Use fixtures** - Reuse test data and setup code
7. **Mock external dependencies** - Don't rely on external services in unit tests

### Test Organization
```python
class TestCryptoDataCollector:
    """Test cases for CryptoDataCollector class."""
    
    def test_initialization(self):
        """Test collector initialization."""
        # Arrange
        cache_dir = "test_cache"
        
        # Act
        collector = CryptoDataCollector(cache_dir)
        
        # Assert
        assert collector.cache_dir == cache_dir
        assert os.path.exists(cache_dir)
    
    def test_get_ohlcv_data_success(self):
        """Test successful data retrieval."""
        # Arrange
        collector = CryptoDataCollector()
        
        # Act
        data = collector.get_ohlcv_data('BTC', period='1mo')
        
        # Assert
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
```

## 🛠️ Troubleshooting

### Common Issues

**Tests failing due to missing dependencies:**
```bash
pip install -r requirements.txt
```

**Tests failing due to cache issues:**
```bash
make clean
```

**Tests running slowly:**
```bash
pytest -m "not slow"  # Skip slow tests
pytest -n auto        # Run in parallel
```

**Coverage not working:**
```bash
pip install pytest-cov
pytest --cov=src --cov-report=html
```

### Getting Help
- Check the test output for specific error messages
- Use `pytest -v` for verbose output
- Use `pytest --pdb` to debug failures
- Check the main project README for setup instructions 