"""
Per-record sentiment analytics.

This module is responsible for converting raw sentiment engine
outputs into a stable, explainable per-record representation.

Why this exists:
- Separates inference from interpretation
- Ensures consistent record-level schema
- Enables downstream aggregation, anomalies, and audits
"""


def build_per_record_result(
    record_id: int,
    sentiment_output: dict,
    rag_result=None,
):
    """
    Construct a standardized per-record sentiment result.

    This structure MUST match the API response schema exactly.
    """
    return {
        "record_id": record_id,
        "sentiment": sentiment_output["label"],
        "score": sentiment_output["score"],
        "contributing_fields": {
            k: float(v) for k, v in sentiment_output.get("drivers", {}).items()
        },
        "rag": {
            "enabled": bool(rag_result),
            "sources": rag_result.sources if rag_result else [],
            "chunks_used": len(rag_result.chunks) if rag_result else 0,
        },
    }
