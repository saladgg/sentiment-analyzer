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
    def analyze(self, text: str, context: str | None = None) -> dict:
        """
        Analyze text and return sentiment metadata.

        Parameters
        ----------
        text : str
            Input text to analyze.
        context : str | None
            Optional RAG context for enrichment.

        Returns
        -------
        dict
            {
                "label": "positive|neutral|negative",
                "score": float,
                "drivers": dict
            }
        """
        raise NotImplementedError
