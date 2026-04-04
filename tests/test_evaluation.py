"""
Tests for the evaluation harness metrics.
"""

from sentiment_analyzer.evaluation.harness import EvaluationHarness


def test_precision_metric():
    """Verify that precision returns a value between 0 and 1."""
    harness = EvaluationHarness()

    preds = ["positive", "negative", "positive"]
    labels = ["positive", "negative", "negative"]

    accuracy = harness.precision(preds, labels)

    assert 0.0 <= accuracy <= 1.0
