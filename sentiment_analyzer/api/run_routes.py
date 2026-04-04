"""
API routes for inspecting and comparing analysis runs.
"""

from fastapi import APIRouter, HTTPException

from sentiment_analyzer.analytics.run_comparison import compare_runs
from sentiment_analyzer.core.config import settings
from sentiment_analyzer.storage.runs import RunStore

router = APIRouter(prefix="/runs", tags=["Runs"])

store = RunStore(settings.rag_db_path)


@router.get("/compare")
def compare_runs_endpoint(
    base_run_id: str,
    rag_run_id: str,
):
    """
    Compare two completed analysis runs.

    Typically:
    - base_run_id: run without RAG
    - rag_run_id: run with RAG
    """
    base = store.get_run(base_run_id)
    rag = store.get_run(rag_run_id)

    if not base or not rag:
        raise HTTPException(
            status_code=404,
            detail="One or both runs not found",
        )

    comparison = compare_runs(
        run_without_rag=base,
        run_with_rag=rag,
    )

    return {
        "base_run_id": base_run_id,
        "rag_run_id": rag_run_id,
        "comparison": comparison,
    }
