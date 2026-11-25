"""
API routes for listing, inspecting, exporting and comparing analysis runs.
"""

import csv
import io
import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from sentiment_analyzer.analytics.run_comparison import compare_runs
from sentiment_analyzer.api.deps import get_engines, get_run_store, get_service
from sentiment_analyzer.api.schemas import (
    ABTestResponse,
    AnalysisRequest,
    EngineName,
    RunComparison,
    RunDetail,
    RunPage,
)
from sentiment_analyzer.core.errors import InvalidInputError, NotFoundError
from sentiment_analyzer.sentiment.registry import EngineRegistry
from sentiment_analyzer.services.analysis_service import AnalysisService
from sentiment_analyzer.storage.runs import RunStore

router = APIRouter(prefix="/runs", tags=["Runs"])

StoreDep = Annotated[RunStore, Depends(get_run_store)]


def _load_run(store: RunStore, run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise NotFoundError(f"Run {run_id!r} not found")
    return run


def _compare(store: RunStore, base_run_id: str, other_run_id: str) -> dict:
    base = store.get_run(base_run_id)
    other = store.get_run(other_run_id)
    if not base or not other:
        raise NotFoundError("One or both runs not found")
    result = compare_runs(
        base, store.get_records(base_run_id), other, store.get_records(other_run_id)
    )
    changed = result.pop("changed_records")
    return {
        "base_run_id": base_run_id,
        "rag_run_id": other_run_id,
        "comparison": result,
        "changed_records": changed,
    }


@router.get("", response_model=RunPage)
def list_runs(
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    engine: EngineName | None = None,
    rag_enabled: bool | None = None,
) -> dict[str, Any]:
    """List past analysis runs, newest first."""
    items, total = store.list_runs(
        limit=limit,
        offset=offset,
        engine=str(engine) if engine else None,
        rag_enabled=rag_enabled,
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/compare", response_model=RunComparison)
def compare_runs_endpoint(base_run_id: str, rag_run_id: str, store: StoreDep) -> dict[str, Any]:
    """
    Compare two runs over the same input, record by record.

    Typically ``base_run_id`` is a run without RAG and ``rag_run_id`` the
    same input with RAG, but any two runs of identical input (e.g. two
    engines) can be compared. Runs of different inputs return 409.
    """
    return _compare(store, base_run_id, rag_run_id)


@router.post("/ab", response_model=ABTestResponse)
def rag_ab_test(
    request: AnalysisRequest,
    store: StoreDep,
    service: Annotated[AnalysisService, Depends(get_service)],
    engines: Annotated[EngineRegistry, Depends(get_engines)],
) -> dict[str, Any]:
    """
    Analyze the same input without and with RAG, and compare the two runs.

    Only meaningful for engines that read context (currently ``llm``).
    """
    engine_name = str(request.sentiment_engine or service.settings.default_sentiment_engine)
    if service.enricher is None:
        raise InvalidInputError("RAG is disabled (ASA_RAG_ENABLED=false); nothing to compare")
    if not engines.engine_class(engine_name).uses_context:
        raise InvalidInputError(
            f"Engine {engine_name!r} ignores RAG context; an A/B test would show no difference"
        )

    def _run(use_rag: bool) -> dict:
        return service.run(
            data=request.data,
            source_type=request.source_type,
            sentiment_engine=engine_name,
            text_fields=request.text_fields,
            use_rag=use_rag,
            timestamp_field=request.timestamp_field,
            granularity=request.granularity,
        )

    base = _run(use_rag=False)
    rag = _run(use_rag=True)
    return {"base": base, "rag": rag, "comparison": _compare(store, base["run_id"], rag["run_id"])}


@router.get("/{run_id}", response_model=RunDetail)
def get_run(
    run_id: str,
    store: StoreDep,
    records_limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    records_offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Run metadata, summary, config snapshot and a page of per-record results."""
    run = _load_run(store, run_id)
    return {
        **run,
        "records": store.get_records(run_id, limit=records_limit, offset=records_offset),
        "records_total": store.count_records(run_id),
        "records_limit": records_limit,
        "records_offset": records_offset,
    }


@router.get("/{run_id}/export")
def export_run(run_id: str, store: StoreDep, format: Literal["csv", "json"] = "csv") -> Response:
    """Download a run's per-record results as CSV, or the whole run as JSON."""
    run = _load_run(store, run_id)
    records = store.get_records(run_id)
    filename = f"run-{run_id}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    if format == "json":
        body = json.dumps({**run, "records": records}, default=str, indent=2)
        return Response(body, media_type="application/json", headers=headers)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "record_id",
            "sentiment",
            "score",
            "confidence",
            "truncated",
            "error",
            "contributing_fields",
            "drivers",
        ]
    )
    for r in records:
        writer.writerow(
            [
                r["record_id"],
                r["sentiment"],
                r["score"],
                r["confidence"],
                r["truncated"],
                r["error"],
                json.dumps(r["contributing_fields"]),
                json.dumps(r["drivers"]),
            ]
        )
    return Response(buffer.getvalue(), media_type="text/csv", headers=headers)
