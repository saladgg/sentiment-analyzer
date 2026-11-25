"""
Dataset-level sentiment aggregation logic.

Transforms per-record sentiment into dataset-level insights.
"""

import statistics
from collections import Counter

LABEL_ORDER = ("positive", "neutral", "negative")


def aggregate(sentiments: list[dict]) -> dict:
    """
    Compute dataset-level statistics over successfully analyzed records.

    Records with an ``error`` are counted in ``failed_records`` and
    excluded from every other figure. Ties for the dominant label are
    broken by the fixed order positive > neutral > negative.
    """
    ok = [s for s in sentiments if not s.get("error")]
    distribution = Counter(s["sentiment"] for s in ok)

    dominant = None
    if distribution:
        top = max(distribution.values())
        dominant = next(label for label in LABEL_ORDER if distribution.get(label) == top)

    scores = [float(s["score"]) for s in ok]
    confidences = [float(s["confidence"]) for s in ok if s.get("confidence") is not None]

    return {
        "total_records": len(ok),
        "failed_records": len(sentiments) - len(ok),
        "distribution": dict(distribution),
        "dominant_sentiment": dominant,
        "avg_score": round(statistics.fmean(scores), 4) if scores else None,
        "avg_confidence": round(statistics.fmean(confidences), 4) if confidences else None,
    }
