"""
FastAPI dependencies.

Long-lived objects are built once in ``main.lifespan`` and stored on
``app.state``; routes receive them through these functions, and tests
can swap them with ``app.dependency_overrides``.
"""

from fastapi import Request, UploadFile

from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.core.errors import InputTooLargeError
from sentiment_analyzer.rag.vector_store import VectorStore
from sentiment_analyzer.sentiment.registry import EngineRegistry
from sentiment_analyzer.services.analysis_service import AnalysisService
from sentiment_analyzer.services.jobs import JobManager
from sentiment_analyzer.storage.runs import RunStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_service(request: Request) -> AnalysisService:
    return request.app.state.service


def get_run_store(request: Request) -> RunStore:
    return request.app.state.run_store


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_engines(request: Request) -> EngineRegistry:
    return request.app.state.engines


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, refusing anything over ``max_bytes`` without reading it all."""
    if file.size is not None and file.size > max_bytes:
        raise InputTooLargeError(f"File exceeds the {max_bytes} byte limit (ASA_MAX_UPLOAD_BYTES)")
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise InputTooLargeError(f"File exceeds the {max_bytes} byte limit (ASA_MAX_UPLOAD_BYTES)")
    return data


def split_csv_param(value: str | None) -> list[str] | None:
    """Parse a comma-separated query parameter (``a,b``) into a list."""
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None
