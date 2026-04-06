"""
Excel input adapter.

Supports ingestion of Excel (.xlsx) files while remaining schema-agnostic.
"""

from typing import Any

import pandas as pd
from sentiment_analyzer.ingestion.base import InputAdapter


class ExcelAdapter(InputAdapter):
    def load(self, raw_input: Any) -> list[dict]:
        """
        Load Excel input into a list of records.

        By default, reads the first sheet.
        """
        df = pd.read_excel(raw_input)
        return df.to_dict(orient="records")
