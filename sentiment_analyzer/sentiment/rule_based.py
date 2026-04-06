"""
Simple deterministic rule-based sentiment engine.

Used as:
- Baseline
- Debugging reference
- Deterministic audit-safe engine
"""

from sentiment_analyzer.sentiment.base import SentimentEngine

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "happy",
    "love",
    "wonderful",
    "fantastic",
    "amazing",
    "best",
    "like",
}
NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "hate",
    "poor",
    "worst",
    "awful",
    "horrible",
    "disappointing",
    "dislike",
    "ugly",
}


class RuleBasedSentimentEngine(SentimentEngine):
    """
    Deterministic rule-based sentiment engine.

    Ignores contextual enrichment by design.
    """

    def analyze(self, text: str, context: str | None = None) -> dict:
        """
        Analyze sentiment using simple lexical rules.

        Parameters
        ----------
        text : str
            Input text to analyze.
        context : str | None
            Optional RAG context (ignored by this engine).

        Returns
        -------
        dict
            Sentiment label, score, and contributing drivers.
        """
        words = text.lower().split()
        pos_hits = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_hits = sum(1 for w in words if w in NEGATIVE_WORDS)
        total_hits = pos_hits + neg_hits

        if total_hits == 0:
            return {
                "label": "neutral",
                "score": 0.0,
                "drivers": {"lexical_rules": 0.0},
            }

        score = (pos_hits - neg_hits) / total_hits
        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": score,
            "drivers": {
                "lexical_rules": abs(score),
                "positive_hits": pos_hits,
                "negative_hits": neg_hits,
            },
        }
