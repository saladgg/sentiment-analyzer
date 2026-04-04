# ==========================================================
# sentiment-analyzer - Dockerfile
# ----------------------------------------------------------
# Multi-stage build for a lean production image.
#   Stage 1 (builder): installs dependencies via uv.
#   Stage 2 (runtime): copies only the venv and source code.
# ==========================================================

# ----------------------------------------------------------
# Stage 1: Builder
# ----------------------------------------------------------
FROM python:3.13-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock* ./

# Install production dependencies into a venv
RUN uv venv /build/.venv && \
    uv sync --no-dev --no-editable

# Copy application source
COPY sentiment_analyzer/ sentiment_analyzer/

# ----------------------------------------------------------
# Stage 2: Runtime
# ----------------------------------------------------------
FROM python:3.13-slim AS runtime

LABEL maintainer="Salad Guyo <saladguyo60@gmail.com>"
LABEL description="Input-agnostic, explainable sentiment analysis platform"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the virtual environment and source from the builder
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/sentiment_analyzer /app/sentiment_analyzer

# Expose the default FastAPI port
EXPOSE 8000

# Health check against the FastAPI docs endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')"]

# Start the application with uvicorn
CMD ["uvicorn", "sentiment_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]
