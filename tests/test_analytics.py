"""
Unit tests for the analytics layer: aggregation, anomalies,
explanations, run comparison and time-based aggregation.
"""

from __future__ import annotations

import pytest
from sentiment_analyzer.analytics.aggregation import aggregate
from sentiment_analyzer.analytics.anomalies import detect_anomalies
from sentiment_analyzer.analytics.explainability import explain_record
from sentiment_analyzer.analytics.run_comparison import compare_runs
from sentiment_analyzer.analytics.time_aggregation import aggregate_by_time
from sentiment_analyzer.core.errors import ConflictError


class TestAggregate:
    def test_counts_distribution_and_averages(self, sample_records):
        summary = aggregate(sample_records)
        assert summary["total_records"] == 3
        assert summary["distribution"] == {"positive": 1, "negative": 1, "neutral": 1}
        assert summary["avg_score"] == pytest.approx(0.0333, abs=1e-4)
        assert summary["avg_confidence"] == pytest.approx(0.5667, abs=1e-4)

    def test_ties_break_in_fixed_order(self, sample_records):
        assert aggregate(sample_records)["dominant_sentiment"] == "positive"
        assert aggregate(sample_records[1:])["dominant_sentiment"] == "neutral"

    def test_majority_label_wins(self):
        records = [{"sentiment": s, "score": 0} for s in ["negative", "negative", "positive"]]
        assert aggregate(records)["dominant_sentiment"] == "negative"

    def test_failed_records_are_excluded(self, sample_records):
        summary = aggregate([*sample_records, {"error": "boom", "sentiment": None}])
        assert summary["total_records"] == 3
        assert summary["failed_records"] == 1

    def test_empty_input(self):
        summary = aggregate([])
        assert summary["total_records"] == 0
        assert summary["dominant_sentiment"] is None
        assert summary["avg_score"] is None


class TestDetectAnomalies:
    def test_returns_empty_for_too_few_records(self):
        assert detect_anomalies([{"score": 0.5}]) == []

    def test_no_anomalies_when_scores_are_constant(self):
        assert detect_anomalies([{"score": 0.1}] * 5) == []

    def test_flags_outlier_with_z_score(self):
        records = [{"record_id": i, "score": 0.1} for i in range(10)]
        records.append({"record_id": 99, "score": -0.9})
        anomalies = detect_anomalies(records)
        assert [a["record_id"] for a in anomalies] == [99]
        assert anomalies[0]["z_score"] < -2

    def test_ignores_failed_records(self):
        records = [{"score": 0.1}] * 10 + [{"score": None, "error": "x"}]
        assert detect_anomalies(records) == []


class TestExplainRecord:
    def test_renders_label_score_and_confidence(self):
        text = explain_record(
            {"record_id": 7, "sentiment": "positive", "score": 0.834, "confidence": 0.9}
        )
        assert text == "Record 7 classified as positive with score 0.83 (confidence 0.90)."

    def test_names_the_strongest_field(self):
        text = explain_record(
            {
                "record_id": 1,
                "sentiment": "negative",
                "score": -0.2,
                "contributing_fields": {"title": 0.5, "body": -1.0},
            }
        )
        assert "strongest field: 'body' (-1.00)" in text

    def test_failed_record(self):
        assert "could not be analyzed: boom" in explain_record({"record_id": 1, "error": "boom"})


class TestCompareRuns:
    def _run(self, sha="h"):
        return {"input_sha256": sha}

    def test_counts_label_changes_and_deltas(self):
        base = [
            {"record_id": 0, "sentiment": "neutral", "score": 0.0, "confidence": 0.5},
            {"record_id": 1, "sentiment": "positive", "score": 0.5, "confidence": 0.5},
            {"record_id": 2, "sentiment": "positive", "score": 0.5, "error": "x"},
        ]
        other = [
            {"record_id": 0, "sentiment": "negative", "score": -0.6, "confidence": 0.9},
            {"record_id": 1, "sentiment": "positive", "score": 0.7, "confidence": 0.7},
            {"record_id": 2, "sentiment": "positive", "score": 0.5},
        ]
        result = compare_runs(self._run(), base, self._run(), other)
        assert result["records_compared"] == 2
        assert result["label_changes"] == 1
        assert result["label_change_rate"] == 0.5
        assert result["mean_score_delta"] == pytest.approx(-0.2)
        assert result["mean_confidence_delta"] == pytest.approx(0.3)
        assert result["changed_records"][0]["record_id"] == 0

    def test_no_overlap(self):
        result = compare_runs(self._run(), [], self._run(), [])
        assert result["records_compared"] == 0
        assert result["mean_score_delta"] is None

    @pytest.mark.parametrize(("a", "b"), [("h1", "h2"), (None, "h"), ("h", None)])
    def test_requires_matching_inputs(self, a, b):
        with pytest.raises(ConflictError):
            compare_runs(self._run(a), [], self._run(b), [])


class TestAggregateByTime:
    records = [
        {"sentiment": "positive", "score": 0.5},
        {"sentiment": "negative", "score": -0.5},
        {"sentiment": "positive", "score": 1.0},
    ]
    timestamps = ["2026-01-05T10:00:00", "2026-01-05", "2026-03-01"]

    @pytest.mark.parametrize(
        ("granularity", "buckets"),
        [
            ("day", ["2026-01-05", "2026-03-01"]),
            ("week", ["2026-W02", "2026-W09"]),
            ("month", ["2026-01", "2026-03"]),
            ("year", ["2026"]),
        ],
    )
    def test_granularities(self, granularity, buckets):
        trend = aggregate_by_time(self.records, self.timestamps, granularity)
        assert [b["bucket"] for b in trend["buckets"]] == buckets

    def test_bucket_stats(self):
        trend = aggregate_by_time(self.records, self.timestamps, "month")
        jan = trend["buckets"][0]
        assert jan["count"] == 2
        assert jan["avg_score"] == 0.0
        assert jan["distribution"] == {"positive": 1, "negative": 1}

    def test_skips_unparseable_and_failed(self):
        records = [*self.records, {"error": "x"}, {"sentiment": "neutral", "score": 0}]
        trend = aggregate_by_time(records, [*self.timestamps, "2026-01-01", None], "year")
        assert trend["unparseable"] == 1
        assert trend["buckets"][0]["count"] == 3
