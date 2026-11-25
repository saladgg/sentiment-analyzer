# ==========================================================
# sentiment-analyzer - Dockerfile
# ----------------------------------------------------------
# Multi-stage build.
#   builder  : resolves dependencies from uv.lock (fails if stale)
#   api      : FastAPI service (default target)
#   frontend : Streamlit UI
#
#   docker build -t sentiment-analyzer .                    # api
#   docker build --target frontend -t sentiment-analyzer-ui .
#
# torch comes from the CPU-only index on Linux (see pyproject.toml).
# ==========================================================

ARG PYTHON_VERSION=3.14

# ----------------------------------------------------------
# Stage 1: Builder
# ----------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# Pinned uv (bump deliberately; Dependabot tracks it)
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv

# The venv must live at the same path in every stage: console scripts
# (uvicorn, streamlit) hard-code their interpreter path in the shebang.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached layer); the project itself is installed after
# its source is copied in.
COPY pyproject.toml uv.lock README.md ./

ARG EXTRAS="--extra ml"
RUN uv sync --locked --no-dev --no-install-project ${EXTRAS}

COPY sentiment_analyzer/ sentiment_analyzer/
RUN uv sync --locked --no-dev --no-editable ${EXTRAS}

# ----------------------------------------------------------
# Shared runtime base: non-root user, state on /app/data
# ----------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime-base

LABEL org.opencontainers.image.title="sentiment-analyzer" \
      org.opencontainers.image.description="Input-agnostic, explainable sentiment analysis platform" \
      org.opencontainers.image.source="https://github.com/saladgg/sentiment-analyzer" \
      org.opencontainers.image.licenses="MIT"

# All mutable state (run history, vector store, model caches) lives under
# /app/data, which docker-compose mounts as a named volume. HOME points
# there too because Chroma caches its ONNX embedder under ~/.cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/app/data/home \
    HF_HOME=/app/data/hf \
    LITELLM_MODE=PRODUCTION \
    ASA_RUNS_DB_PATH=/app/data/runs.duckdb \
    ASA_VECTOR_STORE_PATH=/app/data/vector_store

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app/data/home --no-create-home app \
    && mkdir -p /app/data/home \
    && chown -R app:app /app/data

WORKDIR /app

# ----------------------------------------------------------
# API image (default)
# ----------------------------------------------------------
FROM runtime-base AS api

COPY --from=builder /app/.venv /app/.venv

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"]

# One worker: DuckDB allows a single writing process per database file.
CMD ["uvicorn", "sentiment_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ----------------------------------------------------------
# Frontend image
# ----------------------------------------------------------
FROM builder AS frontend-builder
RUN uv sync --locked --no-dev --no-editable --no-install-project --extra frontend

FROM runtime-base AS frontend

COPY --from=frontend-builder /app/.venv /app/.venv
COPY frontend/ /app/frontend/

ENV ASA_API_URL=http://api:8000/api/v1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# Keep `docker build .` producing the API image.
FROM api
