"""
Explainability utilities for sentiment results.
"""


def explain_record(record: dict) -> str:
    """
    Generate a deterministic, human-readable explanation.
    """
    return (
        f"Record {record['record_id']} classified as "
        f"{record['sentiment']} with score {record['score']:.2f}."
    )
