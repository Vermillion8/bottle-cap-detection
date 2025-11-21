# Use a lightweight Python image
FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set work directory
WORKDIR /app

# Copy configuration files first (for caching)
COPY pyproject.toml uv.lock README.md ./

# Copy source code directory (needed for package build)
COPY bsort/ ./bsort/

# Install dependencies
RUN uv sync --frozen

# Copy the rest of the application
COPY . .

# Verify installation
RUN uv run bsort --help

# Default command
CMD ["uv", "run", "bsort", "--help"]