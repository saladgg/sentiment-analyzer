"""
Time-based sentiment aggregation.

Enables trend analysis when datasets contain timestamps,
a common requirement for monitoring sentiment drift over time.
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import pandas as pd

Granularity = Literal["day", "week", "month", "year"]


def _bucket(ts: pd.Timestamp, granularity: Granularity) -> str:
    if granularity == "day":
        return ts.date().isoformat()
    if granularity == "week":
        iso = ts.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if granularity == "month":
        return f"{ts.year}-{ts.month:02d}"
    return str(ts.year)


def aggregate_by_time(
    records: list[dict],
    timestamps: Sequence[object],
    granularity: Granularity = "day",
) -> dict:
    """
    Average sentiment score and label counts per time bucket.

    ``timestamps[i]`` belongs to ``records[i]``. Values that can't be
    parsed as dates, and failed records, are skipped and counted.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    unparseable = 0

    for record, raw in zip(records, timestamps, strict=True):
        if record.get("error"):
            continue
        ts = pd.to_datetime(raw, errors="coerce", utc=True) if raw not in (None, "") else pd.NaT
        if pd.isna(ts):
            unparseable += 1
            continue
        buckets[_bucket(ts, granularity)].append(record)

    series = []
    for key in sorted(buckets):
        items = buckets[key]
        counts: dict[str, int] = defaultdict(int)
        for item in items:
            counts[item["sentiment"]] += 1
        series.append(
            {
                "bucket": key,
                "count": len(items),
                "avg_score": round(sum(float(i["score"]) for i in items) / len(items), 4),
                "distribution": dict(counts),
            }
        )

    return {"granularity": granularity, "buckets": series, "unparseable": unparseable}
