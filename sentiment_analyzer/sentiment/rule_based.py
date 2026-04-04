"""
Simple deterministic rule-based sentiment engine.

Used as:
- Baseline
- Debugging reference
- Deterministic audit-safe engine
"""


# POSITIVE_WORDS = {"good", "excellent", "happy", "love"}
# NEGATIVE_WORDS = {"bad", "terrible", "hate", "poor"}


class RuleBasedSentimentEngine:
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
        score = 0.0
        label = "neutral"

        if "good" in text.lower():
            score = 0.7
            label = "positive"
        elif "bad" in text.lower():
            score = -0.7
            label = "negative"

        return {
            "label": label,
            "score": score,
            "drivers": {
                "lexical_rules": abs(score),
            },
        }
