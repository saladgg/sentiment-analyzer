"""
Liveness and readiness probes. Unauthenticated, outside ``/api``.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from sentiment_analyzer.api.schemas import Health, Readiness
from sentiment_analyzer.core.version import app_version

router = APIRouter(tags=["Health"])


@router.get("/healthz", response_model=Health)
def healthz():
    """Liveness: the process is up. Performs no I/O."""
    return {"status": "ok", "version": app_version()}


@router.get("/readyz", response_model=Readiness, responses={503: {"model": Readiness}})
def readyz(request: Request):
    """Readiness: storage, vector store and the default engine are usable."""
    state = request.app.state
    checks: dict[str, str] = {}

    for name, probe in (
        ("runs_db", lambda: state.run_store.ping()),
        ("vector_store", lambda: state.vector_store.heartbeat()),
        ("default_engine", lambda: state.engines.get(state.settings.default_sentiment_engine)),
    ):
        try:
            probe()
            checks[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            checks[name] = f"error: {type(exc).__name__}"

    ready = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )
