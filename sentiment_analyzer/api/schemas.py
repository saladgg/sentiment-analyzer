"""
Pydantic models defining the API contract.

These schemas are intentionally explicit to:
- Support frontend integration (React)
- Enable validation
- Guarantee deterministic responses
"""

from typing import Any

from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    """
    Represents a sentiment analysis request.

    data:
        Raw input payload. Can be:
        - List of dicts (JSON rows)
        - Plain text string
    source_type:
        Explicit source identifier: 'csv', 'excel', 'json', 'text'
    sentiment_engine:
        Which sentiment engine to use.
    """

    data: Any
    source_type: str
    sentiment_engine: str = "rule_based"

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
