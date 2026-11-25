"""
JSON input adapter for list-of-dict payloads.
"""

from typing import Any

from sentiment_analyzer.ingestion.base import InputAdapter


class JSONAdapter(InputAdapter):
    """
    Adapter for JSON-like inputs (list of dictionaries).
    """

    def load(self, raw_input: Any) -> list[dict[str, Any]]:
        """Load a list of dictionaries, validating the input type."""
        if not isinstance(raw_input, list) or not all(isinstance(r, dict) for r in raw_input):
            raise ValueError("JSON input must be a list of objects")
        return raw_input
