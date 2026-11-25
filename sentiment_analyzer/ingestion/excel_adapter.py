"""
Excel input adapter.

Supports ingestion of Excel (.xlsx) files while remaining schema-agnostic.
"""

import zipfile
from typing import Any

import pandas as pd

from sentiment_analyzer.core.errors import InvalidInputError
from sentiment_analyzer.ingestion.base import InputAdapter, dataframe_to_records


class ExcelAdapter(InputAdapter):
    def __init__(self, max_records: int | None = None):
        self.max_records = max_records

    def load(self, raw_input: Any) -> list[dict]:
        """
        Load Excel input into a list of records.

        Reads the first sheet. Missing cells become ``None``. At most
        ``max_records + 1`` rows are parsed.
        """
        nrows = self.max_records + 1 if self.max_records else None
        try:
            df = pd.read_excel(raw_input, nrows=nrows)
        except (ValueError, zipfile.BadZipFile, KeyError) as exc:
            raise InvalidInputError(f"Could not parse Excel file: {exc}") from exc
        return dataframe_to_records(df)
