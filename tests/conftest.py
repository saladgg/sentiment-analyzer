"""
Shared pytest fixtures for the SentimentAnalyzer test suite.

Provides:
- An explicit test ``Settings`` (never read from the developer's ``.env``)
  with per-test DuckDB and Chroma paths.
- An app factory that builds a fresh FastAPI app per test.
- An offline embedding function for Chroma.
- A stub for the HuggingFace pipeline so no model weights are downloaded.
- A fake ``litellm.completion`` for the LLM engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from chromadb import Documents, EmbeddingFunction
from chromadb.api.types import Embeddings, Space

# ---------------------------------------------------------------------------
# Importing ``sentiment_analyzer.main`` builds a module-level app from the
# process environment. Force safe values so a developer's .env (e.g.
# ASA_ENVIRONMENT=prod without keys) can't break collection. Tests build
# their own apps from explicit Settings via ``make_app``.
# ---------------------------------------------------------------------------
os.environ["LITELLM_MODE"] = "PRODUCTION"  # stop litellm loading ./.env into os.environ
os.environ["ASA_ENVIRONMENT"] = "dev"
os.environ["ASA_API_KEYS"] = ""
os.environ["ASA_VECTOR_STORE_PATH"] = tempfile.mkdtemp(prefix="asa-test-vectors-")
os.environ["ASA_RUNS_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="asa-test-runs-"), "r.db")


class FakeEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Deterministic, offline embedding function for Chroma.

    Avoids downloading the default ONNX model in tests. Texts sharing
    words get similar vectors, so retrieval ordering is meaningful.
    """

    _DIM = 32

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma API
        vectors: Embeddings = []
        for text in input:
            vec = np.zeros(self._DIM, dtype=np.float32)
            for word in text.lower().split():
                digest = hashlib.sha256(word.strip(".,!?").encode()).digest()
                vec[digest[0] % self._DIM] += 1.0
            vectors.append(vec)
        return vectors

    @staticmethod
    def name() -> str:
        return "asa-test-fake"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(_config: dict) -> FakeEmbeddingFunction:
        return FakeEmbeddingFunction()

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "l2", "ip"]


@pytest.fixture
def fake_embedding() -> FakeEmbeddingFunction:
    return FakeEmbeddingFunction()


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


@pytest.fixture
def make_settings(tmp_db_path, tmp_vector_path) -> Callable[..., Any]:
    """Build an explicit Settings object; keyword overrides win."""
    from sentiment_analyzer.core.config import Settings

    def _make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "runs_db_path": tmp_db_path,
            "vector_store_path": tmp_vector_path,
            "rag_enabled": False,
            "warmup_default_engine": False,
            "llm_model": "test-model",
            "api_keys": [],
            "environment": "dev",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)  # type: ignore[call-arg]

    return _make


@pytest.fixture
def settings(make_settings):
    return make_settings()


# ---------------------------------------------------------------------------
# Engine stubs
# ---------------------------------------------------------------------------
class FakePipeline:
    """
    Mimics a binary transformers text-classification pipeline.

    ``p(positive)`` is 0.99 when the text contains "good", 0.55 when it
    contains "meh", else 0.01. Text containing "explode" raises. Records
    every call for assertions.
    """

    def __init__(self, labels: tuple[str, ...] = ("POSITIVE", "NEGATIVE")):
        self.labels = labels
        self.calls: list[dict[str, Any]] = []

    def _scores(self, text: str) -> list[dict[str, Any]]:
        lowered = text.lower()
        if "explode" in lowered:
            raise RuntimeError("model failure")
        p_pos = 0.99 if "good" in lowered else 0.55 if "meh" in lowered else 0.01
        return [
            {"label": self.labels[0], "score": p_pos},
            {"label": self.labels[1], "score": 1 - p_pos},
        ]

    def __call__(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        if isinstance(texts, str):
            return self._scores(texts)
        return [self._scores(t) for t in texts]


@pytest.fixture(autouse=True)
def stub_hf_transformer(monkeypatch) -> dict[str, Any]:
    """
    Replace the HuggingFace pipeline loader with a deterministic stub so
    no model weights are downloaded. Returns a dict with load ``count``
    and the last ``pipeline`` built.
    """
    from sentiment_analyzer.sentiment import hf_transformer

    state: dict[str, Any] = {"count": 0, "pipeline": None}

    def _fake_loader(model_name: str, revision: str | None):
        state["count"] += 1
        state["pipeline"] = FakePipeline()
        return state["pipeline"]

    monkeypatch.setattr(hf_transformer, "_load_pipeline", _fake_loader)
    return state


def llm_response(payload: Any) -> SimpleNamespace:
    content = payload if isinstance(payload, str) or payload is None else json.dumps(payload)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake_llm(monkeypatch) -> dict[str, Any]:
    """
    Replace ``litellm.completion``. Labels are derived from the text:
    "good" -> positive, "bad" -> negative, else neutral; any context
    mentioning "refund" forces negative. Calls are recorded.
    """
    from sentiment_analyzer.sentiment import llm

    state: dict[str, Any] = {"calls": []}

    def _completion(**kwargs):
        state["calls"].append(kwargs)
        user = kwargs["messages"][1]["content"]
        text = user.split("</text>")[0].lower()
        if "good" in text:
            label, score = "positive", 0.8
        elif "bad" in text:
            label, score = "negative", -0.7
        else:
            label, score = "neutral", 0.0
        if "<context>" in user and "refund" in user.lower():
            label, score = "negative", -0.9
        return llm_response(
            {"label": label, "score": score, "confidence": 0.9, "drivers": {"tone": 0.5}}
        )

    monkeypatch.setattr(llm.litellm, "completion", _completion)
    return state


# ---------------------------------------------------------------------------
# App / client
# ---------------------------------------------------------------------------
@pytest.fixture
def make_app(make_settings, fake_embedding) -> Callable[..., Any]:
    """Build a FastAPI app from explicit settings overrides."""
    from sentiment_analyzer.main import create_app

    def _make(**overrides: Any):
        return create_app(make_settings(**overrides), embedding_function=fake_embedding)

    return _make


@pytest.fixture
def client(make_app) -> Iterator[Any]:
    """A ``TestClient`` against a fresh, isolated app (RAG disabled, no auth)."""
    from fastapi.testclient import TestClient

    with TestClient(make_app()) as test_client:
        yield test_client


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    """A small mixed-sentiment record set used across analytics tests."""
    return [
        {"record_id": 0, "sentiment": "positive", "score": 0.9, "confidence": 0.9},
        {"record_id": 1, "sentiment": "negative", "score": -0.8, "confidence": 0.8},
        {"record_id": 2, "sentiment": "neutral", "score": 0.0, "confidence": 0.0},
    ]


@pytest.fixture
def analyze_payload() -> dict[str, Any]:
    """Default JSON payload accepted by ``POST /api/v1/analyze``."""
    return {
        "source_type": "json",
        "sentiment_engine": "rule_based",
        "data": [
            {"text": "good product overall"},
            {"text": "bad experience today"},
            {"text": "an average everyday item"},
        ],
    }
