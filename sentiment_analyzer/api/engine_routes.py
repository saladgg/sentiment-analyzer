"""Engine discovery endpoint."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from sentiment_analyzer.api.deps import get_engines, get_settings
from sentiment_analyzer.api.schemas import EngineList
from sentiment_analyzer.core.config import Settings
from sentiment_analyzer.sentiment.registry import EngineRegistry

router = APIRouter(tags=["Engines"])


@router.get("/engines", response_model=EngineList)
def list_engines(
    engines: Annotated[EngineRegistry, Depends(get_engines)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Available engines, whether each is configured/loaded, and input limits."""
    return {
        "engines": engines.describe(),
        "limits": {
            "max_upload_bytes": settings.max_upload_bytes,
            "max_records": settings.max_records,
            "max_text_chars": settings.max_text_chars,
            "max_llm_records": settings.max_llm_records,
        },
        "rag_enabled": settings.rag_enabled,
    }
