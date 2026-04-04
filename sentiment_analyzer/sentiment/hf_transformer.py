"""
Transformer-based sentiment engine using HuggingFace.

This engine provides:
- Strong baseline accuracy
- Deterministic inference (fixed model version)
- Explainable confidence scores

Used when higher semantic understanding is required
compared to rule-based sentiment.
"""

from transformers import pipeline


class HuggingFaceSentimentEngine:
    """
    Transformer-based sentiment engine.

    Uses pretrained HuggingFace models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
        )

    def analyze(self, text: str, context: str | None = None) -> dict:
        """
        Analyze sentiment using a transformer model.

        Parameters
        ----------
        text : str
            Input text to analyze.
        context : str | None
            Optional RAG context (currently ignored).

        Returns
        -------
        dict
            Sentiment output.
        """
        result = self.pipeline(text)[0]

        label = result["label"].lower()
        score = result["score"]

        return {
            "label": label,
            "score": score if label == "positive" else -score,
            "drivers": {
                "model_confidence": score,
            },
        }
