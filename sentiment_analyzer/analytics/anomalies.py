"""
Sentiment anomaly detection.

Identifies records whose sentiment deviates significantly
from the dataset distribution.
"""

import statistics


def detect_anomalies(records: list[dict], threshold: float = 2.0):
    """
    Detect sentiment score outliers using z-score.
    """
    scores = [r["score"] for r in records]

    if len(scores) < 2:
        return []

    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)

    anomalies = []

    for r in records:
        if stdev > 0 and abs(r["score"] - mean) / stdev > threshold:
            anomalies.append(r)

    return anomalies
