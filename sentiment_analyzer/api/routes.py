"""
Analysis endpoints: JSON / text bodies and CSV / Excel uploads.

Routes are plain ``def`` so FastAPI runs the blocking pipeline in its
threadpool instead of on the event loop.
"""

import io
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, UploadFile

from sentiment_analyzer.api.deps import get_service, get_settings, read_upload, split_csv_param
from sentiment_analyzer.api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    EngineName,
    Granularity,
)
from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.core.errors import UnsupportedInputError
from sentiment_analyzer.ingestion.csv_adapter import CSVAdapter
from sentiment_analyzer.ingestion.excel_adapter import ExcelAdapter
from sentiment_analyzer.services.analysis_service import AnalysisService

router = APIRouter(tags=["Analysis"])

ServiceDep = Annotated[AnalysisService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
TextFieldsParam = Annotated[
    str | None,
    Query(description="Comma-separated columns to analyze; auto-detected when omitted."),
]


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest, service: ServiceDep) -> dict[str, Any]:
    """
    Analyze a JSON list of records, or plain text (``source_type: "text"``).

    This is the primary endpoint used by programmatic clients.
    """
    return service.run(
        data=request.data,
        source_type=request.source_type,
        sentiment_engine=request.sentiment_engine,
        text_fields=request.text_fields,
        use_rag=request.use_rag,
        timestamp_field=request.timestamp_field,
        granularity=request.granularity,
    )


def load_upload(
    file: UploadFile,
    *,
    extension: str,
    source_type: str,
    max_bytes: int,
    max_records: int,
) -> list[dict[str, Any]]:
    """Check the extension, read within ``max_bytes``, parse into records."""
    if not file.filename or not file.filename.lower().endswith(extension):
        raise UnsupportedInputError(f"Only {extension} files are supported")
    payload = io.BytesIO(read_upload(file, max_bytes))
    adapter = (
        CSVAdapter(max_records=max_records)
        if source_type == "csv"
        else ExcelAdapter(max_records=max_records)
    )
    return adapter.load(payload)


def _analyze_upload(
    *,
    file: UploadFile,
    extension: str,
    source_type: str,
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
        max_bytes=settings.max_upload_bytes,
        max_records=settings.max_records,
    )
    return service.run(
        data=records,
        source_type=source_type,
        sentiment_engine=sentiment_engine,
        text_fields=split_csv_param(text_fields),
        use_rag=use_rag,
        timestamp_field=timestamp_field,
        granularity=granularity,
    )


@router.post("/analyze/csv", response_model=AnalysisResponse)
def analyze_csv(
    file: Annotated[UploadFile, File(...)],
    service: ServiceDep,
    settings: SettingsDep,
    sentiment_engine: EngineName | None = None,
    text_fields: TextFieldsParam = None,
    use_rag: bool | None = None,
    timestamp_field: str | None = None,
    granularity: Granularity = "day",
) -> dict[str, Any]:
    """Analyze sentiment from a UTF-8 CSV file upload."""
    return _analyze_upload(
        file=file,
        extension=".csv",
        source_type="csv",
        service=service,
        settings=settings,
        sentiment_engine=sentiment_engine,
        text_fields=text_fields,
        use_rag=use_rag,
        timestamp_field=timestamp_field,
        granularity=granularity,
    )


@router.post("/analyze/excel", response_model=AnalysisResponse)
def analyze_excel(
    file: Annotated[UploadFile, File(...)],
    service: ServiceDep,
    settings: SettingsDep,
    sentiment_engine: EngineName | None = None,
    text_fields: TextFieldsParam = None,
    use_rag: bool | None = None,
    timestamp_field: str | None = None,
    granularity: Granularity = "day",
) -> dict[str, Any]:
    """Analyze sentiment from an Excel (.xlsx) file upload (first sheet)."""
    return _analyze_upload(
        file=file,
        extension=".xlsx",
        source_type="excel",
        service=service,
        settings=settings,
        sentiment_engine=sentiment_engine,
        text_fields=text_fields,
        use_rag=use_rag,
        timestamp_field=timestamp_field,
        granularity=granularity,
    )
