"""
Pydantic models defining the API contract.

These schemas are intentionally explicit to:
- Support frontend integration (React)
- Enable validation
- Guarantee deterministic responses
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, model_validator


class EngineName(StrEnum):
    """Sentiment engines selectable by clients."""

    RULE_BASED = "rule_based"
    HF_TRANSFORMER = "hf_transformer"
    LLM = "llm"


SourceType = Literal["json", "csv", "excel", "text"]


class AnalysisRequest(BaseModel):
    """
    Represents a sentiment analysis request.

    data:
        Raw input payload:
        - List of dicts (JSON rows) for ``json`` / ``csv`` / ``excel``
        - Plain text string for ``text`` (one record per non-empty line)
    source_type:
        Explicit source identifier: 'csv', 'excel', 'json', 'text'
    sentiment_engine:
        Which sentiment engine to use. Defaults to
        ``ASA_DEFAULT_SENTIMENT_ENGINE``.
    """

    data: list[dict[str, Any]] | str
    source_type: SourceType
    sentiment_engine: EngineName | None = None

    @model_validator(mode="after")
    def _data_matches_source_type(self) -> "AnalysisRequest":
        if self.source_type == "text" and not isinstance(self.data, str):
            raise ValueError("source_type 'text' requires 'data' to be a string")
        if self.source_type != "text" and isinstance(self.data, str):
            raise ValueError(
                f"source_type '{self.source_type}' requires 'data' to be a list of objects"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_type": "json",
                    "sentiment_engine": "hf_transformer",
                    "data": [{"comment": "I love this product"}, {"comment": "This is terrible"}],
                }
            ]
        }
    }


class RecordSentiment(BaseModel):
    """Sentiment result for a single record."""

    record_id: int
    sentiment: str
    score: float
    contributing_fields: dict[str, float]


class RAGMetadata(BaseModel):
    """
    Metadata describing RAG usage for an analysis run.
    """

    enabled: bool
    used: bool
    source_count: int = 0


class AnalysisResponse(BaseModel):
    """Complete response payload for an analysis run."""

    run_id: str
    summary: dict
    records: list[RecordSentiment]
    anomalies: list[dict]
    explanations: list[str]
    rag: RAGMetadata | None = None
