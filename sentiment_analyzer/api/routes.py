"""
HTTP routes exposed to frontend clients.
"""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from sentiment_analyzer.api.schemas import AnalysisRequest, AnalysisResponse
from sentiment_analyzer.core.config import settings
from sentiment_analyzer.core.logging import logger
from sentiment_analyzer.ingestion.csv_adapter import CSVAdapter
from sentiment_analyzer.ingestion.excel_adapter import ExcelAdapter
from sentiment_analyzer.services.analysis_service import AnalysisService
from sentiment_analyzer.storage.runs import RunStore

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    """
    Analyze JSON-based input data.

    This is the primary endpoint used by programmatic clients.
    """
    try:
        service = AnalysisService()
        return service.run(
            data=request.data,
            source_type=request.source_type,
            sentiment_engine=request.sentiment_engine,
        )
    except Exception as e:
        logger.error("default_analysis_failed", error=str(e))
        raise


@router.post("/analyze/csv", response_model=AnalysisResponse)
async def analyze_csv(
    file: Annotated[UploadFile, File(...)],
    sentiment_engine: str = "rule_based",
):
    """
    Analyze sentiment from a CSV file upload.
    """
    try:
        if not file.filename or not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        adapter = CSVAdapter()
        records = adapter.load(file.file)

        service = AnalysisService()
        return service.run(
            data=records,
            source_type="csv",
            sentiment_engine=sentiment_engine,
        )
    except Exception as e:
        logger.error("csv_analysis_failed", error=str(e))
        raise


@router.post("/analyze/excel", response_model=AnalysisResponse)
async def analyze_excel(
    file: Annotated[UploadFile, File(...)],
    sentiment_engine: str = "rule_based",
):
    """
    Analyze sentiment from an Excel (.xlsx) file upload.
    """
    try:
        if not file.filename or not file.filename.endswith(".xlsx"):
            raise HTTPException(status_code=400, detail="Only .xlsx files are supported")

        adapter = ExcelAdapter()
        records = adapter.load(file.file)

        service = AnalysisService()
        return service.run(
            data=records,
            source_type="excel",
            sentiment_engine=sentiment_engine,
        )
    except Exception as e:
        logger.error("excel_analysis_failed", error=str(e))
        raise


@router.get("/runs")
def list_runs():
    """
    List past analysis runs.
    """
    try:
        store = RunStore(settings.rag_db_path)
        return store._load_all_runs()
    except Exception as e:
        logger.error("runs_store_failed", error=str(e))
        raise
