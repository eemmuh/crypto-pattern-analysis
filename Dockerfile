# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy pyproject.toml, uv.lock, and README for dependency management
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies with uv
RUN uv sync --frozen

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/cache data/database logs models/registry notebooks reports

# Set permissions
RUN chmod +x manage.py

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uv", "run", "python", "manage.py", "api"]
