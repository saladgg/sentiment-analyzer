"""
CSV input adapter for SentimentAnalyzer.

This adapter enables ingestion of tabular CSV data and converts it
into a normalized list of records (dicts), preserving column names.

Why this exists:
- Enables analysts and non-technical users to upload CSV files
- Keeps ingestion logic isolated from processing & sentiment logic
"""

from typing import Any

import pandas as pd
from sentiment_analyzer.core.errors import InvalidInputError, UnsupportedInputError
from sentiment_analyzer.ingestion.base import InputAdapter, dataframe_to_records


class CSVAdapter(InputAdapter):
    def __init__(self, max_records: int | None = None):
        self.max_records = max_records

    def load(self, raw_input: Any) -> list[dict]:
        """
        Load CSV input into a list of dictionaries.

        raw_input:
            - File path (str)
            - File-like object

        Missing cells become ``None``. At most ``max_records + 1`` rows are
        parsed so oversized files are rejected without loading them fully.
        """
        nrows = self.max_records + 1 if self.max_records else None
        try:
            df = pd.read_csv(raw_input, nrows=nrows)
        except UnicodeDecodeError as exc:
            raise UnsupportedInputError("CSV file must be UTF-8 encoded") from exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise InvalidInputError(f"Could not parse CSV: {exc}") from exc
        return dataframe_to_records(df)
