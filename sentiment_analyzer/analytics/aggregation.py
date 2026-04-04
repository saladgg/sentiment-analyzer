"""
Dataset-level sentiment aggregation logic.

Transforms per-record sentiment into engineering-grade insights.
"""

from collections import Counter


def aggregate(sentiments: list[dict]) -> dict:
    """Compute dataset-level sentiment distribution and dominant label."""
    labels = [s["sentiment"] for s in sentiments]
    distribution = Counter(labels)

    return {
        "total_records": len(sentiments),
        "distribution": dict(distribution),
        "dominant_sentiment": distribution.most_common(1)[0][0] if distribution else None,
    }
