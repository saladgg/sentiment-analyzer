"""
Input normalization utilities.

Ensures all records conform to a predictable structure
before sentiment analysis.
"""

from typing import Any

import pandas as pd


def is_missing(value: Any) -> bool:
    """True for ``None`` and pandas missing markers (NaN, NaT, NA)."""
    if value is None:
        return True
    if isinstance(value, list | dict | tuple | set):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_record(record: dict) -> dict:
    """
    Normalize a single record.

    - Missing values (None, NaN, NaT) become empty strings
    - Converts all other values to stripped strings
    """
    return {str(k): "" if is_missing(v) else str(v).strip() for k, v in record.items()}
