"""
Evaluation endpoints for SentimentAnalyzer.

These endpoints are intentionally separated from inference
to keep evaluation explicit and auditable.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from sentiment_analyzer.api.deps import get_service
from sentiment_analyzer.api.schemas import (
    AccuracyRequest,
    AccuracyResponse,
    EngineEvaluationRequest,
    EngineEvaluationResponse,
)
from sentiment_analyzer.core.errors import InputTooLargeError
from sentiment_analyzer.evaluation.harness import EvaluationHarness
from sentiment_analyzer.sentiment.base import SentimentResult
from sentiment_analyzer.services.analysis_service import AnalysisService

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/sentiment", response_model=AccuracyResponse)
def evaluate_sentiment(payload: AccuracyRequest):
    """
    Score existing predictions against ground-truth labels.

    Intended for offline evaluation and benchmarking.
    """
    report = EvaluationHarness().report(payload.predictions, payload.labels)
    return {
        "metric": "accuracy",
        "value": report["accuracy"],
        "total_samples": report["total_samples"],
        "report": report,
    }


@router.post("/engine", response_model=EngineEvaluationResponse)
def evaluate_engine(
    payload: EngineEvaluationRequest,
    service: Annotated[AnalysisService, Depends(get_service)],
):
    """
    Run an engine over labeled texts and report accuracy, per-class
    precision/recall/F1, macro-F1 and the confusion matrix.

    Texts the engine fails on are counted in ``failed`` and excluded from
    the metrics. Evaluations are not persisted as runs.
    """
    settings = service.settings
    engine_name = str(payload.sentiment_engine or settings.default_sentiment_engine)
    engine = service.engines.get(engine_name)

    limit = settings.max_llm_records if engine.is_paid else settings.max_records
    if len(payload.items) > limit:
        raise InputTooLargeError(f"Engine {engine_name!r} accepts at most {limit} items per call")

    texts = [item.text[: settings.max_text_chars] for item in payload.items]
    outputs = engine.analyze_batch(texts)
    pairs = [
        (output.label, item.label)
        for output, item in zip(outputs, payload.items, strict=True)
        if isinstance(output, SentimentResult)
    ]
    predictions = [p for p, _ in pairs]
    labels = [label for _, label in pairs]
    report = (
        EvaluationHarness().report(predictions, labels)
        if pairs
        else {
            "total_samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": {"labels": [], "matrix": []},
        }
    )
    return {"engine": engine_name, "failed": len(payload.items) - len(pairs), "report": report}
