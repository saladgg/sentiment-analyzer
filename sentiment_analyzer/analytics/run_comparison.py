"""
Utilities for comparing two analysis runs.

Used to evaluate RAG impact.
"""


def compare_runs(run_without_rag: dict, run_with_rag: dict) -> dict:
    """
    Compare two runs and quantify RAG impact.
    """

    summary_a = run_without_rag["summary"]
    summary_b = run_with_rag["summary"]

    confidence_gain = summary_b.get("avg_confidence", 0.0) - summary_a.get("avg_confidence", 0.0)

    label_changes = summary_b.get("label_changes", 0)

    rag_helpfulness = confidence_gain > 0.05 or label_changes > 0

    return {
        "confidence_gain": round(confidence_gain, 4),
        "label_changes": label_changes,
        "rag_helpfulness": rag_helpfulness,
    }
