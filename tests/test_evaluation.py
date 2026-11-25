"""
Unit tests for the evaluation harness.
"""

from __future__ import annotations

import pytest
from sentiment_analyzer.evaluation.harness import EvaluationHarness


@pytest.fixture
def harness() -> EvaluationHarness:
    return EvaluationHarness()


class TestAccuracy:
    def test_partial_match(self, harness):
        assert harness.accuracy(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx(2 / 3)

    def test_empty_inputs_return_zero(self, harness):
        assert harness.accuracy([], []) == 0.0

    def test_misaligned_lengths_raise(self, harness):
        with pytest.raises(ValueError):
            harness.accuracy(["a"], ["a", "b"])


class TestReport:
    def test_per_class_metrics_and_confusion_matrix(self, harness):
        predictions = ["positive", "positive", "negative", "neutral"]
        labels = ["positive", "negative", "negative", "negative"]
        report = harness.report(predictions, labels)

        assert report["accuracy"] == 0.5
        assert report["per_class"]["positive"] == {
            "precision": 0.5,
            "recall": 1.0,
            "f1": 0.6667,
            "support": 1,
        }
        assert report["per_class"]["negative"]["recall"] == pytest.approx(0.3333, abs=1e-4)
        # neutral has no support, so it doesn't count towards macro-F1
        assert report["macro_f1"] == pytest.approx((0.6667 + 0.5) / 2, abs=1e-4)
        assert report["confusion_matrix"] == {
            "labels": ["negative", "neutral", "positive"],
            "matrix": [[1, 1, 1], [0, 0, 0], [0, 0, 1]],
        }

    def test_empty(self, harness):
        report = harness.report([], [])
        assert report["total_samples"] == 0
        assert report["macro_f1"] == 0.0
