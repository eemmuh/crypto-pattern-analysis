# Crypto Trading Analysis Makefile

.PHONY: help install test test-unit test-integration test-coverage clean lint format docs demo

# Default target
help:
	@echo "Crypto Trading Analysis - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install all dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-fast        Run fast tests (skip slow ones)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black"
	@echo "  clean            Clean up generated files"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Generate documentation"
	@echo ""
	@echo "Demo:"
	@echo "  demo             Run the demo script"
	@echo "  demo-r           Run the R demo script"
	@echo ""

# Installation
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Installing R dependencies..."
	Rscript install_dependencies.R
	@echo "✅ Installation complete!"

install-dev: install
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	@echo "✅ Development installation complete!"

# Testing
test:
	@echo "Running all tests..."
	python run_tests.py --type all --verbose

test-unit:
	@echo "Running unit tests..."
	python run_tests.py --type unit --verbose

test-integration:
	@echo "Running integration tests..."
	python run_tests.py --type integration --verbose

test-coverage:
	@echo "Running tests with coverage..."
	python run_tests.py --type coverage --html

test-fast:
	@echo "Running fast tests..."
	python run_tests.py --type all --fast --verbose

# Code Quality
lint:
	@echo "Running linting checks..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503; \
	else \
		echo "flake8 not found. Install with: pip install flake8"; \
	fi
	@if command -v pylint >/dev/null 2>&1; then \
		pylint src/ --disable=C0114,C0116; \
	else \
		echo "pylint not found. Install with: pip install pylint"; \
	fi

format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black src/ tests/ --line-length=100; \
	else \
		echo "black not found. Install with: pip install black"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		isort src/ tests/; \
	else \
		echo "isort not found. Install with: pip install isort"; \
	fi

clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf tests/*/__pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf *.pyc
	rm -rf data/cache/*
	rm -rf logs/*
	@echo "✅ Cleanup complete!"

# Documentation
docs:
	@echo "Generating documentation..."
	@if command -v pdoc >/dev/null 2>&1; then \
		pdoc --html --output-dir docs src/; \
	else \
		echo "pdoc not found. Install with: pip install pdoc3"; \
	fi

# Demo
demo:
	@echo "Running Python demo..."
	python demo.py

demo-r:
	@echo "Running R demo..."
	Rscript demo.R

# Development workflow
dev-setup: install-dev
	@echo "Setting up development environment..."
	mkdir -p logs
	mkdir -p data/cache
	mkdir -p tests/test_data
	mkdir -p tests/test_cache
	@echo "✅ Development setup complete!"

# Quick check
check: lint test-unit
	@echo "✅ All checks passed!"

# Full CI pipeline
ci: clean install-dev lint test-coverage
	@echo "✅ CI pipeline complete!"

# Performance testing
benchmark:
	@echo "Running performance benchmarks..."
	python run_tests.py --type performance

# Security check
security:
	@echo "Running security checks..."
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r src/ -f json -o bandit-report.json; \
	else \
		echo "bandit not found. Install with: pip install bandit"; \
	fi

# Package building
build:
	@echo "Building package..."
	python setup.py sdist bdist_wheel

# Installation from source
install-source: build
	@echo "Installing from source..."
	pip install dist/*.whl

# Docker commands (if using Docker)
docker-build:
	@echo "Building Docker image..."
	docker build -t crypto-analysis .

docker-run:
	@echo "Running Docker container..."
	docker run -it crypto-analysis

# Help for specific targets
test-help:
	@echo "Test Commands:"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-coverage    - Run tests with coverage"
	@echo "  make test-fast        - Run fast tests only"
	@echo "  make benchmark        - Run performance benchmarks"

lint-help:
	@echo "Linting Commands:"
	@echo "  make lint   - Run flake8 and pylint"
	@echo "  make format - Format code with black and isort"
	@echo "  make clean  - Clean up generated files" 