"""
Sentiment engine interface.

Each engine must produce:
- sentiment label
- confidence score
- per-field contribution
"""

from abc import ABC, abstractmethod


class SentimentEngine(ABC):
    @abstractmethod
    def analyze(self, text: str) -> dict:
        """
        Analyze text and return sentiment metadata.

        Returns:
            {
                "label": "positive|neutral|negative",
                "score": float,
                "drivers": dict
            }
        """
        raise NotImplementedError
