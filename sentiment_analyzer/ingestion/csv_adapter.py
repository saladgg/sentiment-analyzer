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
from sentiment_analyzer.ingestion.base import InputAdapter


class CSVAdapter(InputAdapter):
    def load(self, raw_input: Any) -> list[dict]:
        """
        Load CSV input into a list of dictionaries.

        raw_input:
            - File path (str)
            - File-like object
        """
        df = pd.read_csv(raw_input)
        return df.to_dict(orient="records")
