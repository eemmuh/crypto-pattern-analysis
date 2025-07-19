# Crypto Trading Analysis Makefile

.PHONY: help install clean test lint format docs run-demo run-notebook

# Default target
help:
	@echo "Crypto Trading Analysis - Available Commands:"
	@echo ""
	@echo "  install      - Install Python dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  clean        - Clean cache and temporary files"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  docs         - Generate documentation"
	@echo "  run-demo     - Run Python demo"
	@echo "  run-notebook - Start Jupyter notebook"
	@echo "  setup-r      - Install R dependencies"
	@echo "  run-r-demo   - Run R demo"

# Install Python dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Clean cache and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf data/cache/*.csv 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	@echo "✅ Cleanup completed"

# Run tests
test:
	python -m pytest tests/ -v --cov=src --cov-report=html

# Run linting
lint:
	flake8 src/ --max-line-length=88 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ --line-length=88
	black demo.py --line-length=88

# Generate documentation
docs:
	python -c "import src; help(src)"

# Run Python demo
run-demo:
	python demo.py

# Start Jupyter notebook
run-notebook:
	jupyter notebook notebooks/crypto_analysis_demo.ipynb

# Install R dependencies
setup-r:
	Rscript install_dependencies.R

# Run R demo
run-r-demo:
	Rscript demo.R

# Full setup
setup: install install-dev setup-r
	@echo "✅ Full setup completed"

# Quick start
quick-start: setup run-demo
	@echo "✅ Quick start completed" 