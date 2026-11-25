"""
Explainability utilities for sentiment results.
"""


def explain_record(record: dict) -> str:
    """
    Generate a deterministic, human-readable explanation.
    """
    if record.get("error"):
        return f"Record {record['record_id']} could not be analyzed: {record['error']}"

    explanation = (
        f"Record {record['record_id']} classified as "
        f"{record['sentiment']} with score {record['score']:.2f}"
    )
    if record.get("confidence") is not None:
        explanation += f" (confidence {record['confidence']:.2f})"
    fields = record.get("contributing_fields") or {}
    if len(fields) > 1:
        strongest = max(fields.items(), key=lambda kv: abs(kv[1]))
        explanation += f"; strongest field: '{strongest[0]}' ({strongest[1]:+.2f})"
    return explanation + "."
