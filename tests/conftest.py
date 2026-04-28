"""
Shared pytest fixtures for the SentimentAnalyzer test suite.

Provides:
- Isolated per-test DuckDB and Chroma directories so tests never touch
  the developer's local ``runs.duckdb`` or vector store.
- Stubs for the HuggingFace transformer engine so tests don't need to
  download real model weights.
- Sample records / payloads reused across ingestion, analytics, and
  API tests.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Environment isolation — applied before any sentiment_analyzer import that
# reads ``settings`` at module load time.
# ---------------------------------------------------------------------------
os.environ.setdefault("ASA_RAG_ENABLED", "false")
os.environ.setdefault("ASA_LLM_API_KEY", "test-key")


@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    """Return a writable DuckDB path scoped to a single test."""
    return str(tmp_path / "runs.duckdb")


@pytest.fixture
def tmp_vector_path(tmp_path) -> str:
    """Return a writable directory for an isolated Chroma store."""
    path = tmp_path / "vector_store"
    path.mkdir()
    return str(path)


@pytest.fixture(autouse=True)
def stub_hf_transformer(monkeypatch) -> None:
    """
    Replace the HuggingFace pipeline with a deterministic stub so the
    ``AnalysisService`` can be instantiated in tests without downloading
    model weights.
    """
    from sentiment_analyzer.sentiment import hf_transformer

    def _fake_pipeline(*_args, **_kwargs):
        def _run(text: str):
            label = "POSITIVE" if "good" in text.lower() else "NEGATIVE"
            return [{"label": label, "score": 0.99}]

        return _run

    monkeypatch.setattr(hf_transformer, "pipeline", _fake_pipeline)


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    """A small mixed-sentiment record set used across analytics tests."""
    return [
        {"record_id": 0, "sentiment": "positive", "score": 0.9},
        {"record_id": 1, "sentiment": "negative", "score": -0.8},
        {"record_id": 2, "sentiment": "neutral", "score": 0.0},
    ]


@pytest.fixture
def analyze_payload() -> dict[str, Any]:
    """Default JSON payload accepted by ``POST /api/analyze``."""
    return {
        "source_type": "json",
        "sentiment_engine": "rule_based",
        "data": [
            {"text": "good product"},
            {"text": "bad experience"},
            {"text": "an average everyday item"},
        ],
    }


@pytest.fixture
def client(monkeypatch, tmp_db_path) -> Iterator[Any]:
    """
    Return a FastAPI ``TestClient`` against an isolated DuckDB so API
    tests never write to the repository's ``runs.duckdb``.
    """
    from fastapi.testclient import TestClient
    from sentiment_analyzer.core import config as config_module

    monkeypatch.setattr(config_module.settings, "rag_db_path", tmp_db_path)
    monkeypatch.setattr(config_module.settings, "rag_enabled", False)

    from sentiment_analyzer.main import app

    with TestClient(app) as test_client:
        yield test_client
