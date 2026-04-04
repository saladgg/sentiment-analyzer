"""
LLM-backed sentiment engine (stub).

This module defines the contract for LLM-based sentiment analysis
without binding the system to a specific provider.

Why this is a stub:
- Keeps the system deterministic by default
- Allows safe, controlled introduction of LLMs later
- Prevents accidental dependency on non-auditable outputs
"""


class LLMSentimentEngine:
    """
    LLM-backed sentiment engine.

    Uses both text and optional RAG context.
    """

    def analyze(self, text: str, context: str | None = None) -> dict:
        prompt = f"""
        TEXT:
        {text}

        CONTEXT:
        {context or "No additional context"}

        Determine sentiment and explain why.
        """

        # Stubbed response
        return {
            "label": "positive",
            "score": 0.85,
            "drivers": {
                "text": 0.6,
                "context": 0.25,
            },
        }
