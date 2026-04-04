"""
Base interface for all ingestion adapters.

Adapters are responsible for converting arbitrary input formats
into a normalized list of records (dicts).
"""

from abc import ABC, abstractmethod
from typing import Any


class InputAdapter(ABC):
    @abstractmethod
    def load(self, raw_input: Any) -> list[dict[str, Any]]:
        """
        Load raw input into a list of records.

        Each record must be a flat dictionary.
        """
        raise NotImplementedError
