"""
Base interface for all ingestion adapters.

Adapters are responsible for converting arbitrary input formats
into a normalized list of records (dicts).
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class InputAdapter(ABC):
    @abstractmethod
    def load(self, raw_input: Any) -> list[dict[str, Any]]:
        """
        Load raw input into a list of records.

        Each record must be a flat dictionary.
        """
        raise NotImplementedError


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to records, mapping NaN/NaT/NA to ``None``."""
    df.columns = [str(c) for c in df.columns]
    cleaned = df.astype(object).where(pd.notna(df), None)
    return cleaned.to_dict(orient="records")  # type: ignore[return-value]
