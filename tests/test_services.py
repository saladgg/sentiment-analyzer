"""
Tests for the orchestration layer: ``AnalysisService``.
"""

from __future__ import annotations

import pytest

from sentiment_analyzer.core.errors import (
    EngineUnavailableError,
    InputTooLargeError,
    InvalidInputError,
)
from sentiment_analyzer.rag.context_enrichment import RAGContextEnricher
from sentiment_analyzer.rag.vector_store import VectorStore
from sentiment_analyzer.sentiment.registry import EngineRegistry
from sentiment_analyzer.services.analysis_service import AnalysisService, fingerprint
from sentiment_analyzer.storage.runs import RunStore


@pytest.fixture
def make_service(make_settings, tmp_db_path, tmp_vector_path, fake_embedding):
    def _make(rag: bool = False, **overrides):
        settings = make_settings(rag_enabled=rag, **overrides)
        enricher = None
        if rag:
            store = VectorStore(tmp_vector_path, embedding_function=fake_embedding)
            store.add(["refund requests take thirty days"], document_id="d1", source="policy")
            enricher = RAGContextEnricher(store, top_k=1)
        return AnalysisService(settings, RunStore(tmp_db_path), EngineRegistry(settings), enricher)

    return _make


@pytest.fixture
def records():
    return [
        {"comment": "this product is great and excellent"},
        {"comment": "absolutely terrible and disappointing"},
        {"comment": "  "},  # no text — skipped
    ]


class TestAnalysisService:
    def test_runs_full_pipeline_and_persists(self, make_service, records):
        service = make_service()
        result = service.run(data=records, source_type="json", sentiment_engine="rule_based")

        assert result["summary"]["total_records"] == 2
        assert [r["record_id"] for r in result["records"]] == [0, 1]
        assert result["rag"] is None

        run = service.run_store.get_run(result["run_id"])
        assert run["records_in"] == 3
        assert run["records_skipped"] == 1
        assert run["input_sha256"] == fingerprint(records)
        assert run["config"]["app_version"]
        assert len(service.run_store.get_records(result["run_id"])) == 2

    def test_unknown_engine_raises(self, make_service, records):
        with pytest.raises(EngineUnavailableError):
            make_service().run(data=records, source_type="json", sentiment_engine="???")

    def test_non_list_json_is_invalid(self, make_service):
        with pytest.raises(InvalidInputError):
            make_service().run(data={"a": 1}, source_type="json")

    def test_record_limit(self, make_service, records):
        with pytest.raises(InputTooLargeError):
            make_service(max_records=2).run(data=records, source_type="json")

    def test_empty_input(self, make_service):
        result = make_service().run(data=[], source_type="json")
        assert result["records"] == []
        assert result["summary"]["dominant_sentiment"] is None

    def test_per_field_contributions(self, make_service):
        result = make_service().run(
            data=[{"title": "great value overall", "body": "but the support was terrible"}],
            source_type="json",
        )
        assert result["text_fields"] == ["title", "body"]
        assert result["records"][0]["contributing_fields"] == {"title": 1.0, "body": -1.0}
        assert "strongest field" in result["explanations"][0]

    def test_paid_engines_skip_per_field_scoring(self, make_service, fake_llm):
        result = make_service().run(
            data=[{"title": "good stuff here", "body": "bad stuff there"}],
            source_type="json",
            sentiment_engine="llm",
        )
        assert result["records"][0]["contributing_fields"] == {}
        assert len(fake_llm["calls"]) == 1

    def test_engine_errors_are_per_record(self, make_service):
        result = make_service().run(
            data="good line\nexplode now\nanother good one",
            source_type="text",
            sentiment_engine="hf_transformer",
        )
        assert result["summary"]["failed_records"] == 1
        assert result["records"][1]["error"].startswith("RuntimeError")
        run = make_service().run_store.get_run(result["run_id"])
        assert run["status"] == "completed_with_errors"
        assert run["records_failed"] == 1


class TestRAGIntegration:
    def test_rag_is_skipped_for_engines_that_ignore_context(self, make_service):
        """BUG-11: rule_based ignores context, so it must not be queried or reported."""
        result = make_service(rag=True).run(data="refund was slow", source_type="text")
        assert result["rag"] == {"enabled": False, "used": False, "source_count": 0}
        assert result["records"][0]["rag"]["chunks_used"] == 0

    def test_rag_feeds_context_to_llm(self, make_service, fake_llm):
        service = make_service(rag=True)
        result = service.run(data="the refund arrived", source_type="text", sentiment_engine="llm")
        assert result["rag"] == {"enabled": True, "used": True, "source_count": 1}
        assert "<context>" in fake_llm["calls"][0]["messages"][1]["content"]
        assert result["records"][0]["rag"]["sources"] == ["policy"]
        stored = service.run_store.get_records(result["run_id"])
        assert stored[0]["rag_chunk_ids"] == ["d1:0"]

    def test_use_rag_false_overrides(self, make_service, fake_llm):
        result = make_service(rag=True).run(
            data="the refund arrived", source_type="text", sentiment_engine="llm", use_rag=False
        )
        assert result["rag"]["enabled"] is False
        assert "<context>" not in fake_llm["calls"][0]["messages"][1]["content"]

    def test_use_rag_true_requires_rag_enabled(self, make_service, fake_llm):
        with pytest.raises(InvalidInputError):
            make_service(rag=False).run(
                data="x", source_type="text", sentiment_engine="llm", use_rag=True
            )


def test_fingerprint_is_order_and_key_order_sensitive_only_where_it_matters():
    assert fingerprint([{"a": 1, "b": 2}]) == fingerprint([{"b": 2, "a": 1}])
    assert fingerprint([{"a": 1}, {"a": 2}]) != fingerprint([{"a": 2}, {"a": 1}])
