"""
Time-based sentiment aggregation.

Enables trend analysis when datasets contain timestamps,
a common requirement for monitoring sentiment drift over time.
"""

from collections import defaultdict
from datetime import datetime


def aggregate_by_time(
    records: list[dict],
    timestamp_field: str,
    granularity: str = "day",
) -> dict:
    """
    Aggregate sentiment scores over time.

    granularity:
        - 'day'
        - 'month'
        - 'year'
    """
    buckets = defaultdict(list)

    for r in records:
        ts = datetime.fromisoformat(r[timestamp_field])

        if granularity == "day":
            key = ts.date().isoformat()
        elif granularity == "month":
            key = f"{ts.year}-{ts.month:02d}"
        else:
            key = str(ts.year)

        buckets[key].append(r["score"])

    return {k: sum(v) / len(v) for k, v in buckets.items()}
