"""
Thin HTTP client for the SentimentAnalyzer FastAPI backend.

Every function returns raw Python objects (dicts / lists) so that
Streamlit pages never deal with ``requests`` directly.
"""

from __future__ import annotations

from typing import Any

import requests

BASE_URL = "http://localhost:8000/api"
_TIMEOUT = 120  # seconds – LLM engine can be slow


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------


def analyze_json(
    data: list[dict[str, Any]],
    engine: str = "rule_based",
) -> dict:
    """POST /api/analyze with JSON payload."""
    resp = requests.post(
        f"{BASE_URL}/analyze",
        json={"data": data, "source_type": "json", "sentiment_engine": engine},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def analyze_csv(file, engine: str = "rule_based") -> dict:
    """POST /api/analyze/csv with file upload."""
    resp = requests.post(
        f"{BASE_URL}/analyze/csv",
        files={"file": (file.name, file, "text/csv")},
        params={"sentiment_engine": engine},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def analyze_excel(file, engine: str = "rule_based") -> dict:
    """POST /api/analyze/excel with file upload."""
    resp = requests.post(
        f"{BASE_URL}/analyze/excel",
        files={
            "file": (
                file.name,
                file,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        params={"sentiment_engine": engine},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------


def list_runs() -> list[dict]:
    """GET /api/runs."""
    resp = requests.get(f"{BASE_URL}/runs", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def compare_runs(base_run_id: str, rag_run_id: str) -> dict:
    """GET /api/runs/compare."""
    resp = requests.get(
        f"{BASE_URL}/runs/compare",
        params={"base_run_id": base_run_id, "rag_run_id": rag_run_id},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# RAG
# ------------------------------------------------------------------


def upload_rag_document(file, namespace: str = "default") -> dict:
    """POST /api/rag/documents."""
    resp = requests.post(
        f"{BASE_URL}/rag/documents",
        files={"file": (file.name, file, "text/plain")},
        params={"namespace": namespace},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def list_rag_documents() -> list[dict]:
    """GET /api/rag/documents."""
    resp = requests.get(f"{BASE_URL}/rag/documents", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def delete_rag_document(document_id: str) -> dict:
    """DELETE /api/rag/documents/{document_id}."""
    resp = requests.delete(
        f"{BASE_URL}/rag/documents/{document_id}",
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def rag_retrieve(query: str, top_k: int = 5, namespace: str = "default") -> list[dict]:
    """POST /api/rag/retrieve."""
    resp = requests.post(
        f"{BASE_URL}/rag/retrieve",
        json={"query": query, "top_k": top_k, "namespace": namespace},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()
