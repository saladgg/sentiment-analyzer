"""
FastAPI entrypoint for SentimentAnalyzer.

``create_app`` builds the application: settings, middleware, error
handlers and routers. Expensive, long-lived objects (run store, vector
store, engine registry, analysis service) are created once per process
in ``lifespan`` and shared through ``app.state``.

Routes are served under ``/api/v1``. The unversioned ``/api`` prefix is
kept as a deprecated alias (hidden from the OpenAPI schema).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sentiment_analyzer.api.engine_routes import router as engine_router
from sentiment_analyzer.api.evaluation_routes import router as evaluation_router
from sentiment_analyzer.api.health_routes import router as health_router
from sentiment_analyzer.api.rag_routes import router as rag_router
from sentiment_analyzer.api.routes import router as analysis_router
from sentiment_analyzer.api.run_routes import router as run_router
from sentiment_analyzer.core.config import Settings, get_settings
from sentiment_analyzer.core.errors import register_exception_handlers
from sentiment_analyzer.core.logging import configure_logging, logger
from sentiment_analyzer.core.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from sentiment_analyzer.core.security import require_api_key
from sentiment_analyzer.core.version import app_version
from sentiment_analyzer.rag.context_enrichment import RAGContextEnricher
from sentiment_analyzer.rag.vector_store import VectorStore
from sentiment_analyzer.sentiment.registry import EngineRegistry
from sentiment_analyzer.services.analysis_service import AnalysisService
from sentiment_analyzer.storage.runs import RunStore

API_ROUTERS = (analysis_router, run_router, rag_router, evaluation_router, engine_router)


def create_app(
    settings: Settings | None = None, *, embedding_function: Any | None = None
) -> FastAPI:
    """
    Build the FastAPI application.

    ``embedding_function`` overrides Chroma's default embedder (tests use
    an offline stub).
    """
    settings = settings or get_settings()
    settings.validate_for_startup()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        run_store = RunStore(settings.runs_db_path)
        vector_store = VectorStore(
            settings.vector_store_path, embedding_function=embedding_function
        )
        engines = EngineRegistry(settings)
        enricher = (
            RAGContextEnricher(
                vector_store, top_k=settings.rag_top_k, namespace=settings.rag_namespace
            )
            if settings.rag_enabled
            else None
        )
        app.state.run_store = run_store
        app.state.vector_store = vector_store
        app.state.engines = engines
        app.state.service = AnalysisService(settings, run_store, engines, enricher)

        if settings.warmup_default_engine:
            try:
                engines.get(settings.default_sentiment_engine)
            except Exception:  # noqa: BLE001 - /readyz reports it; don't crash the process
                logger.exception("engine_warmup_failed", engine=settings.default_sentiment_engine)

        logger.info(
            "app_started",
            version=app_version(),
            environment=settings.environment,
            auth_enabled=settings.auth_enabled,
            rag_enabled=settings.rag_enabled,
        )
        try:
            yield
        finally:
            run_store.close()

    docs_enabled = settings.environment != "prod"
    app = FastAPI(
        title="SentimentAnalyzer",
        description="Input-agnostic, explainable sentiment analysis platform",
        version=app_version(),
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.settings = settings

    # Middleware runs outermost-last: request context wraps everything.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(health_router)
    auth = [Depends(require_api_key)]
    for router in API_ROUTERS:
        app.include_router(router, prefix="/api/v1", dependencies=auth)
        app.include_router(router, prefix="/api", dependencies=auth, include_in_schema=False)

    return app


app = create_app()
