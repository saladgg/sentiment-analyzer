"""
Background analysis jobs: submit large inputs, poll for progress, cancel.

Submissions are validated synchronously (engine, limits, text fields), so
bad input is rejected with a 4xx before anything is queued. Job limits
(``ASA_MAX_JOB_RECORDS``, ``ASA_MAX_LLM_JOB_RECORDS``,
``ASA_MAX_JOB_UPLOAD_BYTES``) are larger than the synchronous ones.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile, status

from sentiment_analyzer.api.deps import get_jobs, get_service, get_settings, split_csv_param
from sentiment_analyzer.api.routes import load_upload
from sentiment_analyzer.api.schemas import (
    AnalysisRequest,
    EngineName,
    Granularity,
    Job,
    JobPage,
    JobStatus,
)
from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.services.analysis_service import AnalysisService
from sentiment_analyzer.services.jobs import JobManager

router = APIRouter(prefix="/jobs", tags=["Jobs"])

JobsDep = Annotated[JobManager, Depends(get_jobs)]
ServiceDep = Annotated[AnalysisService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
TextFieldsParam = Annotated[
    str | None,
    Query(description="Comma-separated columns to analyze; auto-detected when omitted."),
]


def _with_progress(job: dict[str, Any]) -> dict[str, Any]:
    total = job.get("records_total") or 0
    done = job.get("records_done") or 0
    progress = 1.0 if job["status"] == "completed" else (done / total if total else 0.0)
    return {**job, "progress": round(progress, 4)}


def _client(request: Request) -> str | None:
    return getattr(request.state, "client_id", None)


@router.post("/analyze", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def submit_analysis(
    payload: AnalysisRequest, request: Request, jobs: JobsDep, service: ServiceDep
) -> dict[str, Any]:
    """Queue a JSON or plain-text analysis. Returns ``202`` with the job to poll."""
    prepared = service.prepare(
        payload.data,
        payload.source_type,
        payload.sentiment_engine,
        text_fields=payload.text_fields,
        timestamp_field=payload.timestamp_field,
        background=True,
    )
    job = jobs.submit(
        prepared,
        use_rag=payload.use_rag,
        granularity=payload.granularity,
        client_id=_client(request),
    )
    return _with_progress(job)


def _submit_upload(
    *,
    file: UploadFile,
    extension: str,
    source_type: str,
    request: Request,
    jobs: JobManager,
    service: AnalysisService,
    settings: Settings,
    sentiment_engine: EngineName | None,
    text_fields: str | None,
    use_rag: bool | None,
    timestamp_field: str | None,
    granularity: Granularity,
) -> dict[str, Any]:
    records = load_upload(
        file,
        extension=extension,
        source_type=source_type,
        max_bytes=settings.max_job_upload_bytes,
        max_records=settings.max_job_records,
    )
    prepared = service.prepare(
        records,
        source_type,
        sentiment_engine,
        text_fields=split_csv_param(text_fields),
        timestamp_field=timestamp_field,
        background=True,
    )
    job = jobs.submit(
        prepared, use_rag=use_rag, granularity=granularity, client_id=_client(request)
    )
    return _with_progress(job)


@router.post("/analyze/csv", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def submit_csv(
    file: Annotated[UploadFile, File(...)],
    request: Request,
    jobs: JobsDep,
    service: ServiceDep,
    settings: SettingsDep,
    sentiment_engine: EngineName | None = None,
    text_fields: TextFieldsParam = None,
    use_rag: bool | None = None,
    timestamp_field: str | None = None,
    granularity: Granularity = "day",
) -> dict[str, Any]:
    """Queue a CSV analysis (up to ``ASA_MAX_JOB_UPLOAD_BYTES``)."""
    return _submit_upload(
        file=file,
        extension=".csv",
        source_type="csv",
        request=request,
        jobs=jobs,
        service=service,
        settings=settings,
        sentiment_engine=sentiment_engine,
        text_fields=text_fields,
        use_rag=use_rag,
        timestamp_field=timestamp_field,
        granularity=granularity,
    )


@router.post("/analyze/excel", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
def submit_excel(
    file: Annotated[UploadFile, File(...)],
    request: Request,
    jobs: JobsDep,
    service: ServiceDep,
    settings: SettingsDep,
    sentiment_engine: EngineName | None = None,
    text_fields: TextFieldsParam = None,
    use_rag: bool | None = None,
    timestamp_field: str | None = None,
    granularity: Granularity = "day",
) -> dict[str, Any]:
    """Queue an Excel (.xlsx) analysis (up to ``ASA_MAX_JOB_UPLOAD_BYTES``)."""
    return _submit_upload(
        file=file,
        extension=".xlsx",
        source_type="excel",
        request=request,
        jobs=jobs,
        service=service,
        settings=settings,
        sentiment_engine=sentiment_engine,
        text_fields=text_fields,
        use_rag=use_rag,
        timestamp_field=timestamp_field,
        granularity=granularity,
    )


@router.get("", response_model=JobPage)
def list_jobs(
    jobs: JobsDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[JobStatus | None, Query(alias="status")] = None,
) -> dict[str, Any]:
    """List jobs, newest first."""
    items, total = jobs.store.list_jobs(limit=limit, offset=offset, status=status_filter)
    return {
        "items": [_with_progress(j) for j in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str, jobs: JobsDep) -> dict[str, Any]:
    """Job status and progress. When ``completed``, fetch ``/runs/{run_id}``."""
    return _with_progress(jobs.get(job_id))


@router.delete("/{job_id}", response_model=Job)
def cancel_job(job_id: str, jobs: JobsDep) -> dict[str, Any]:
    """Cancel a queued job. Running or finished jobs return ``409``."""
    return _with_progress(jobs.cancel(job_id))
