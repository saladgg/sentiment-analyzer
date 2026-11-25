"""
Sentiment anomaly detection.

Identifies records whose sentiment deviates significantly
from the dataset distribution.
"""

import statistics


def detect_anomalies(records: list[dict], threshold: float = 2.0) -> list[dict]:
    """
    Detect sentiment score outliers using a z-score.

    Only successfully analyzed records from a single run are considered,
    so every score comes from the same engine and is on the same scale.
    """
    ok = [r for r in records if not r.get("error") and r.get("score") is not None]
    scores = [float(r["score"]) for r in ok]

    if len(scores) < 2:
        return []

    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    if stdev == 0:
        return []

    return [
        {**r, "z_score": round((float(r["score"]) - mean) / stdev, 4)}
        for r in ok
        if abs(float(r["score"]) - mean) / stdev > threshold
    ]
