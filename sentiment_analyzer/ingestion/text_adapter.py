"""Plain-text input adapter for raw string ingestion."""

from typing import Any

from sentiment_analyzer.ingestion.base import InputAdapter

TEXT_FIELD = "text"


class TextAdapter(InputAdapter):
    """
    Adapter for plain-text input.

    Each non-empty line becomes one record ``{"text": <line>}``.
    """

    def load(self, raw_input: Any) -> list[dict[str, Any]]:
        """Split a string into one record per non-empty line."""
        if not isinstance(raw_input, str):
            raise ValueError("Text input must be a string")
        return [{TEXT_FIELD: line.strip()} for line in raw_input.splitlines() if line.strip()]
