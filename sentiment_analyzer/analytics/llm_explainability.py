"""
LLM-backed explainability layer.

This module is OPTIONAL and gated.
It should never replace deterministic sentiment engines,
only explain their outputs in human-readable terms.
"""


class LLMExplainer:
    """Optional LLM-backed explainer for sentiment outputs."""

    def explain(self, sentiment_result: dict) -> str:
        """
        Convert sentiment outputs into a human-readable explanation.

        In production, this would call an LLM.
        Here, we keep it deterministic for safety.
        """
        label = sentiment_result["sentiment"]
        score = sentiment_result["score"]

        return (
            f"The record was classified as '{label}' "
            f"with an overall sentiment score of {score:.2f}. "
            "This score reflects the combined emotional signals "
            "detected in the text fields."
        )
