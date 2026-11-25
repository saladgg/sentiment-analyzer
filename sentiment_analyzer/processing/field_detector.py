"""
Automatically detects text-bearing fields in records.

This enables the system to work with heterogeneous schemas
without requiring manual field configuration.
"""

import statistics
from typing import Any

from sentiment_analyzer.processing.normalizer import is_missing

SAMPLE_SIZE = 200
MIN_STRING_SHARE = 0.5
MIN_MEDIAN_CHARS = 8
MIN_DISTINCT_SHARE = 0.2


def _looks_like_free_text(values: list[str]) -> bool:
    """Free text has spaces or letters; ids and codes usually don't."""
    return any(" " in v for v in values) or any(any(c.isalpha() for c in v) for v in values)


def detect_text_fields(records: list[dict[str, Any]], sample_size: int = SAMPLE_SIZE) -> list[str]:
    """
    Identify fields that hold free text worth analyzing.

    Looks at up to ``sample_size`` records (not just the first), and keeps
    a field when, among its non-missing values:
    - at least half are non-numeric strings,
    - the median string length is at least ``MIN_MEDIAN_CHARS``,
    - values are reasonably varied (not a categorical column), and
    - values look like words rather than codes.

    Field order follows first appearance in the records.
    """
    sample = records[:sample_size]
    fields: dict[str, list[Any]] = {}
    for record in sample:
        for key, value in record.items():
            fields.setdefault(str(key), [])
            if not is_missing(value) and str(value).strip():
                fields[str(key)].append(value)

    detected = []
    for key, values in fields.items():
        if not values:
            continue
        strings = [v.strip() for v in values if isinstance(v, str)]
        if len(strings) / len(values) < MIN_STRING_SHARE or not strings:
            continue
        if statistics.median(len(s) for s in strings) < MIN_MEDIAN_CHARS:
            continue
        if len(strings) >= 5 and len(set(strings)) / len(strings) < MIN_DISTINCT_SHARE:
            continue
        if not _looks_like_free_text(strings):
            continue
        detected.append(key)
    return detected
