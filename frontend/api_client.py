"""
Thin HTTP client for the SentimentAnalyzer FastAPI backend.

Every function returns raw Python objects (dicts / lists / bytes) so that
Streamlit pages never deal with ``requests`` directly.

Configuration (environment variables):
    ASA_API_URL   Base URL of the API (default: http://localhost:8000/api/v1)
    ASA_API_KEY   API key sent as ``X-API-Key`` when the API requires one
"""

from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = os.environ.get("ASA_API_URL", "http://localhost:8000/api/v1").rstrip("/")
API_KEY = os.environ.get("ASA_API_KEY", "")
_TIMEOUT = 300  # seconds — LLM runs over large files can be slow


class APIError(Exception):
    """An error response from the API, carrying its ``detail`` message."""

    def __init__(self, status_code: int, detail: str, request_id: str | None = None):
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id
        suffix = f" (request {request_id})" if request_id else ""
        super().__init__(f"{detail}{suffix}")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _detail(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300] or resp.reason
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', [])[1:])}: {err.get('msg')}"
            for err in detail
        )
    return str(detail or body)


def _request(method: str, path: str, *, raw: bool = False, **kwargs: Any) -> Any:
    try:
        resp = requests.request(
            method, f"{BASE_URL}{path}", headers=_headers(), timeout=_TIMEOUT, **kwargs
        )
    except requests.ConnectionError as exc:
        raise APIError(0, f"Cannot reach the API at {BASE_URL}") from exc
    if not resp.ok:
        raise APIError(resp.status_code, _detail(resp), resp.headers.get("X-Request-ID"))
    return resp.content if raw else resp.json()


def _clean(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v not in (None, "", [])}


# ------------------------------------------------------------------
# Engines
# ------------------------------------------------------------------


def list_engines() -> dict:
    """GET /engines — engines, their status, and input limits."""
    return _request("GET", "/engines")


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------


def analyze_json(
    data: list[dict[str, Any]] | str,
    engine: str | None = None,
    *,
    source_type: str = "json",
    text_fields: list[str] | None = None,
    timestamp_field: str | None = None,
    granularity: str = "day",
    use_rag: bool | None = None,
    path: str = "/analyze",
) -> dict:
    """
    POST a JSON list of records, or plain text (source_type='text').

    ``path`` selects the endpoint: ``/analyze`` (synchronous),
    ``/jobs/analyze`` (background job) or ``/runs/ab`` (RAG A/B test).
    """
    body = _clean(
        {
            "data": data,
            "source_type": source_type,
            "sentiment_engine": engine,
            "text_fields": text_fields,
            "timestamp_field": timestamp_field,
        }
    )
    body["granularity"] = granularity
    if use_rag is not None:
        body["use_rag"] = use_rag
    return _request("POST", path, json=body)


def _upload(path: str, file, mime: str, engine: str | None, options: dict[str, Any]) -> dict:
    text_fields = options.get("text_fields")
    params = _clean(
        {
            "sentiment_engine": engine,
            "text_fields": ",".join(text_fields) if text_fields else None,
            "timestamp_field": options.get("timestamp_field"),
            "granularity": options.get("granularity", "day"),
        }
    )
    if options.get("use_rag") is not None:
        params["use_rag"] = str(options["use_rag"]).lower()
    return _request("POST", path, files={"file": (file.name, file, mime)}, params=params)


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def analyze_csv(
    file, engine: str | None = None, *, background: bool = False, **options: Any
) -> dict:
    """POST /analyze/csv (or /jobs/analyze/csv when ``background``)."""
    path = "/jobs/analyze/csv" if background else "/analyze/csv"
    return _upload(path, file, "text/csv", engine, options)


def analyze_excel(
    file, engine: str | None = None, *, background: bool = False, **options: Any
) -> dict:
    """POST /analyze/excel (or /jobs/analyze/excel when ``background``)."""
    path = "/jobs/analyze/excel" if background else "/analyze/excel"
    return _upload(path, file, XLSX_MIME, engine, options)


# ------------------------------------------------------------------
# Background jobs
# ------------------------------------------------------------------


def get_job(job_id: str) -> dict:
    """GET /jobs/{id} — status, progress and run_id once completed."""
    return _request("GET", f"/jobs/{job_id}")


def list_jobs(limit: int = 20, status: str | None = None) -> dict:
    """GET /jobs — one page: {items, total, limit, offset}."""
    return _request("GET", "/jobs", params=_clean({"limit": limit, "status": status}))


def cancel_job(job_id: str) -> dict:
    """DELETE /jobs/{id} — cancel a queued job."""
    return _request("DELETE", f"/jobs/{job_id}")


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------


def evaluate_engine(items: list[dict[str, str]], engine: str | None = None) -> dict:
    """POST /evaluation/engine with [{text, label}, ...]."""
    return _request(
        "POST", "/evaluation/engine", json=_clean({"items": items, "sentiment_engine": engine})
    )


# ------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------


def list_runs(limit: int = 50, offset: int = 0) -> dict:
    """GET /runs — one page: {items, total, limit, offset}."""
    return _request("GET", "/runs", params={"limit": limit, "offset": offset})


def get_run(run_id: str, records_limit: int = 100, records_offset: int = 0) -> dict:
    """GET /runs/{id} — metadata, config and a page of per-record results."""
    return _request(
        "GET",
        f"/runs/{run_id}",
        params={"records_limit": records_limit, "records_offset": records_offset},
    )


def export_run(run_id: str, fmt: str = "csv") -> bytes:
    """GET /runs/{id}/export — file contents."""
    return _request("GET", f"/runs/{run_id}/export", params={"format": fmt}, raw=True)


def compare_runs(base_run_id: str, rag_run_id: str) -> dict:
    """GET /runs/compare."""
    return _request(
        "GET", "/runs/compare", params={"base_run_id": base_run_id, "rag_run_id": rag_run_id}
    )


# ------------------------------------------------------------------
# RAG
# ------------------------------------------------------------------


def upload_rag_document(file, namespace: str = "default") -> dict:
    """POST /rag/documents."""
    return _request(
        "POST",
        "/rag/documents",
        files={"file": (file.name, file, "text/plain")},
        params={"namespace": namespace},
    )


def list_rag_documents(namespace: str | None = None) -> list[dict]:
    """GET /rag/documents, optionally within one namespace."""
    return _request("GET", "/rag/documents", params=_clean({"namespace": namespace}))


def delete_rag_document(document_id: str) -> dict:
    """DELETE /rag/documents/{document_id}."""
    return _request("DELETE", f"/rag/documents/{document_id}")


def rag_retrieve(query: str, top_k: int = 5, namespace: str = "default") -> list[dict]:
    """POST /rag/retrieve."""
    return _request(
        "POST", "/rag/retrieve", json={"query": query, "top_k": top_k, "namespace": namespace}
    )
