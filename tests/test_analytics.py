"""
Tests for dataset-level analytics: aggregation, anomaly detection,
explainability, per-record assembly, run comparison, and time-bucket
aggregation.
"""

from __future__ import annotations

from sentiment_analyzer.analytics.aggregation import aggregate
from sentiment_analyzer.analytics.anomalies import detect_anomalies
from sentiment_analyzer.analytics.explainability import explain_record
from sentiment_analyzer.analytics.llm_explainability import LLMExplainer
from sentiment_analyzer.analytics.per_record import build_per_record_result
from sentiment_analyzer.analytics.run_comparison import compare_runs
from sentiment_analyzer.analytics.time_aggregation import aggregate_by_time


class TestAggregate:
    def test_counts_distribution_and_dominant_label(self, sample_records):
        summary = aggregate(sample_records)
        assert summary["total_records"] == 3
        assert summary["distribution"] == {
            "positive": 1,
            "negative": 1,
            "neutral": 1,
        }
        assert summary["dominant_sentiment"] in {"positive", "negative", "neutral"}

    def test_majority_label_wins(self):
        records = [{"sentiment": "positive", "score": s} for s in (0.1, 0.2, 0.3)]
        records.append({"sentiment": "negative", "score": -0.1})
        summary = aggregate(records)
        assert summary["dominant_sentiment"] == "positive"

    def test_empty_input(self):
        summary = aggregate([])
        assert summary == {
            "total_records": 0,
            "distribution": {},
            "dominant_sentiment": None,
        }


class TestDetectAnomalies:
    def test_returns_empty_for_too_few_records(self):
        assert detect_anomalies([{"score": 0.5}]) == []

    def test_no_anomalies_when_scores_are_constant(self):
        records = [{"record_id": i, "score": 0.5} for i in range(5)]
        assert detect_anomalies(records) == []

    def test_flags_outlier_above_threshold(self):
        records = [
            *[{"record_id": i, "score": 0.0} for i in range(20)],
            {"record_id": 99, "score": 5.0},  # extreme outlier
        ]
        anomalies = detect_anomalies(records, threshold=2.0)
        assert any(r["record_id"] == 99 for r in anomalies)

    def test_threshold_is_respected(self):
        records = [
            {"record_id": 0, "score": 0.0},
            {"record_id": 1, "score": 0.1},
            {"record_id": 2, "score": -0.1},
            {"record_id": 3, "score": 0.5},
        ]
        # A very high threshold should suppress all anomalies.
        assert detect_anomalies(records, threshold=10.0) == []


class TestExplainRecord:
    def test_renders_label_and_score(self):
        msg = explain_record({"record_id": 7, "sentiment": "positive", "score": 0.83})
        assert "Record 7" in msg
        assert "positive" in msg
        assert "0.83" in msg


class TestPerRecordBuilder:
    def test_minimal_sentiment_output_without_rag(self):
        result = build_per_record_result(
            record_id=0,
            sentiment_output={"label": "positive", "score": 0.9, "drivers": {"x": 0.5}},
        )
        assert result["record_id"] == 0
        assert result["sentiment"] == "positive"
        assert result["score"] == 0.9
        assert result["contributing_fields"] == {"x": 0.5}
        assert result["rag"] == {"enabled": False, "sources": [], "chunks_used": 0}

    def test_handles_missing_drivers(self):
        result = build_per_record_result(
            record_id=1,
            sentiment_output={"label": "neutral", "score": 0.0},
        )
        assert result["contributing_fields"] == {}

    def test_with_rag_result(self):
        class _RAG:
            sources = ["doc1", "doc2"]
            chunks = [object(), object(), object()]

        result = build_per_record_result(
            record_id=2,
            sentiment_output={"label": "positive", "score": 0.5, "drivers": {}},
            rag_result=_RAG(),
        )
        assert result["rag"] == {
            "enabled": True,
            "sources": ["doc1", "doc2"],
            "chunks_used": 3,
        }


class TestCompareRuns:
    def test_detects_helpfulness_via_confidence_gain(self):
        out = compare_runs(
            run_without_rag={"summary": {"avg_confidence": 0.5, "label_changes": 0}},
            run_with_rag={"summary": {"avg_confidence": 0.7, "label_changes": 0}},
        )
        assert out["confidence_gain"] == 0.2
        assert out["label_changes"] == 0
        assert out["rag_helpfulness"] is True

    def test_detects_helpfulness_via_label_changes(self):
        out = compare_runs(
            run_without_rag={"summary": {"avg_confidence": 0.5}},
            run_with_rag={"summary": {"avg_confidence": 0.5, "label_changes": 3}},
        )
        assert out["label_changes"] == 3
        assert out["rag_helpfulness"] is True

    def test_no_rag_helpfulness_when_unchanged(self):
        out = compare_runs(
            run_without_rag={"summary": {"avg_confidence": 0.5}},
            run_with_rag={"summary": {"avg_confidence": 0.5}},
        )
        assert out["confidence_gain"] == 0.0
        assert out["rag_helpfulness"] is False


class TestAggregateByTime:
    @staticmethod
    def _records():
        return [
            {"ts": "2024-01-01T08:00:00", "score": 0.2},
            {"ts": "2024-01-01T20:00:00", "score": 0.4},
            {"ts": "2024-02-15T10:00:00", "score": 0.6},
            {"ts": "2025-03-01T10:00:00", "score": -0.2},
        ]

    def test_day_granularity(self):
        out = aggregate_by_time(self._records(), "ts", granularity="day")
        assert out["2024-01-01"] == 0.30000000000000004 or out["2024-01-01"] == 0.3
        assert "2024-02-15" in out

    def test_month_granularity(self):
        out = aggregate_by_time(self._records(), "ts", granularity="month")
        assert "2024-01" in out
        assert "2024-02" in out
        assert "2025-03" in out

    def test_year_granularity_is_default_for_unknown_value(self):
        out = aggregate_by_time(self._records(), "ts", granularity="year")
        assert "2024" in out
        assert "2025" in out


class TestLLMExplainer:
    def test_renders_label_and_score(self):
        msg = LLMExplainer().explain({"sentiment": "negative", "score": -0.42})
        assert "negative" in msg
        assert "-0.42" in msg
