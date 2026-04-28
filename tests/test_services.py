"""
Tests for the orchestration layer: ``AnalysisService`` and the
``SentimentWorkflow`` wrappers used by the API and batch entry points.
"""

from __future__ import annotations

import pytest
from sentiment_analyzer.core import config as config_module
from sentiment_analyzer.services.analysis_service import AnalysisService
from sentiment_analyzer.services.workflow import SentimentWorkflow


@pytest.fixture
def isolated_settings(monkeypatch, tmp_db_path):
    """Disable RAG and pin the DuckDB path so services run hermetically."""
    monkeypatch.setattr(config_module.settings, "rag_enabled", False)
    monkeypatch.setattr(config_module.settings, "rag_db_path", tmp_db_path)


@pytest.fixture
def records():
    return [
        {"comment": "this product is great and excellent"},
        {"comment": "absolutely terrible and disappointing"},
        {"comment": "  "},  # invalid — should be filtered
    ]


class TestAnalysisService:
    def test_runs_full_pipeline_and_persists(self, isolated_settings, records):
        service = AnalysisService()
        result = service.run(data=records, source_type="json", sentiment_engine="rule_based")

        assert "run_id" in result
        assert result["summary"]["total_records"] == 2  # blank record dropped
        assert len(result["records"]) == 2
        assert result["rag"] is None  # RAG disabled in this fixture

    def test_unknown_engine_raises_keyerror(self, isolated_settings, records):
        with pytest.raises(KeyError):
            AnalysisService().run(data=records, source_type="json", sentiment_engine="???")


class TestAnalysisServiceWithRAG:
    def test_enricher_path_populates_rag_metadata(self, monkeypatch, tmp_db_path, records):
        """
        Exercise the ``self.enricher`` branch by replacing the real
        ``RAGContextEnricher`` with a deterministic stub and flipping
        ``rag_enabled`` on the settings object. This avoids spinning up
        a real Chroma client and guarantees the RAG-aware response shape
        is exercised.
        """
        from sentiment_analyzer.rag.models import RAGChunk, RAGResult
        from sentiment_analyzer.services import analysis_service as svc_module

        class _StubEnricher:
            def __init__(self):
                pass

            def enrich(self, _text: str) -> RAGResult:
                chunk = RAGChunk(content="ctx", source="doc-x", score=0.9)
                return RAGResult(
                    enabled=True,
                    context="ctx",
                    sources=["doc-x"],
                    chunks=[chunk],
                )

        monkeypatch.setattr(config_module.settings, "rag_enabled", True)
        monkeypatch.setattr(config_module.settings, "rag_db_path", tmp_db_path)
        monkeypatch.setattr(svc_module, "RAGContextEnricher", _StubEnricher)

        result = AnalysisService().run(
            data=records,
            source_type="json",
            sentiment_engine="rule_based",
        )

        assert result["rag"] is not None
        assert result["rag"]["enabled"] is True
        assert result["rag"]["used"] is True
        assert result["rag"]["source_count"] >= 1


class TestSentimentWorkflow:
    def test_run_batch_delegates_to_service(self, isolated_settings, records):
        out = SentimentWorkflow().run_batch(
            records,
            source_type="json",
            sentiment_engine="rule_based",
        )
        assert out["summary"]["total_records"] == 2

    def test_run_stream_returns_one_result_per_record(self, isolated_settings, records):
        valid = [r for r in records if r["comment"].strip()]
        outputs = SentimentWorkflow().run_stream(
            iter(valid),
            source_type="json",
            sentiment_engine="rule_based",
        )
        assert len(outputs) == len(valid)
        assert all("run_id" in o for o in outputs)
