"""
Evaluation endpoints for SentimentAnalyzer.

These endpoints are intentionally separated from inference
to keep evaluation explicit and auditable.
"""

from fastapi import APIRouter

from sentiment_analyzer.evaluation.harness import EvaluationHarness

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/sentiment")
def evaluate_sentiment(
    predictions: list[str],
    labels: list[str],
):
    """
    Evaluate sentiment predictions against ground-truth labels.

    Intended for offline evaluation and benchmarking.
    """
    harness = EvaluationHarness()
    accuracy = harness.precision(predictions, labels)

    return {
        "metric": "accuracy",
        "value": accuracy,
        "total_samples": len(labels),
    }
