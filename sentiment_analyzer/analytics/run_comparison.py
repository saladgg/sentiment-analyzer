"""
Compare two analysis runs over the same input.

Used to measure the effect of RAG (or of switching engines): which
records changed label, and how scores and confidence moved.
"""

import statistics

from sentiment_analyzer.core.errors import ConflictError


def compare_runs(
    base_run: dict,
    base_records: list[dict],
    other_run: dict,
    other_records: list[dict],
) -> dict:
    """
    Compare two runs record by record.

    Both runs must have analyzed the same input (matching
    ``input_sha256``); otherwise a record-level comparison is meaningless
    and :class:`ConflictError` is raised.
    """
    base_hash, other_hash = base_run.get("input_sha256"), other_run.get("input_sha256")
    if not base_hash or not other_hash:
        raise ConflictError("One or both runs predate input fingerprinting and cannot be compared")
    if base_hash != other_hash:
        raise ConflictError("Runs analyzed different inputs and cannot be compared")

    other_by_id = {r["record_id"]: r for r in other_records}
    pairs = [
        (b, other_by_id[b["record_id"]])
        for b in base_records
        if b["record_id"] in other_by_id
        and not b.get("error")
        and not other_by_id[b["record_id"]].get("error")
    ]

    changed = [
        {
            "record_id": b["record_id"],
            "base_sentiment": b["sentiment"],
            "other_sentiment": o["sentiment"],
            "base_score": b["score"],
            "other_score": o["score"],
        }
        for b, o in pairs
        if b["sentiment"] != o["sentiment"]
    ]

    def _mean_delta(key: str) -> float | None:
        deltas = [
            float(o[key]) - float(b[key])
            for b, o in pairs
            if b.get(key) is not None and o.get(key) is not None
        ]
        return round(statistics.fmean(deltas), 4) if deltas else None

    return {
        "records_compared": len(pairs),
        "label_changes": len(changed),
        "label_change_rate": round(len(changed) / len(pairs), 4) if pairs else 0.0,
        "mean_score_delta": _mean_delta("score"),
        "mean_confidence_delta": _mean_delta("confidence"),
        "changed_records": changed,
    }
