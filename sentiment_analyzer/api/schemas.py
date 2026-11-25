"""
Pydantic models defining the API contract.

These schemas are intentionally explicit to:
- Support frontend integration
- Validate input at the edge (4xx, never 500, for bad input)
- Document every response in the generated OpenAPI schema
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SentimentLabel = Literal["positive", "neutral", "negative"]
SourceType = Literal["json", "csv", "excel", "text"]
Granularity = Literal["day", "week", "month", "year"]

NAMESPACE_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class EngineName(StrEnum):
    """Sentiment engines selectable by clients."""

    RULE_BASED = "rule_based"
    HF_TRANSFORMER = "hf_transformer"
    LLM = "llm"


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
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
    text_fields:
        Columns to analyze. Auto-detected when omitted.
    use_rag:
        Force RAG enrichment on/off for this run. Defaults to
        ``ASA_RAG_ENABLED``. Only engines that read context use it.
    timestamp_field / granularity:
        When set, the response includes a sentiment trend over time.
    """

    data: list[dict[str, Any]] | str
    source_type: SourceType
    sentiment_engine: EngineName | None = None
    text_fields: list[str] | None = Field(default=None, min_length=1)
    use_rag: bool | None = None
    timestamp_field: str | None = None
    granularity: Granularity = "day"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_type": "json",
                    "sentiment_engine": "rule_based",
                    "data": [{"comment": "I love this product"}, {"comment": "This is terrible"}],
                },
                {"source_type": "text", "data": "Great service!\nThe delivery was late."},
            ]
        }
    }

    @model_validator(mode="after")
    def _data_matches_source_type(self) -> "AnalysisRequest":
        if self.source_type == "text" and not isinstance(self.data, str):
            raise ValueError("source_type 'text' requires 'data' to be a string")
        if self.source_type != "text" and isinstance(self.data, str):
            raise ValueError(
                f"source_type '{self.source_type}' requires 'data' to be a list of objects"
            )
        return self


class RecordRAG(BaseModel):
    """RAG context used for one record."""

    enabled: bool
    sources: list[str] = []
    chunks_used: int = 0


class RecordSentiment(BaseModel):
    """
    Sentiment result for a single record.

    ``score`` is the polarity in [-1, 1]; ``confidence`` is in [0, 1].
    ``contributing_fields`` maps each text field to its own polarity;
    ``drivers`` holds engine-specific signals. On failure, ``error`` is
    set and the sentiment fields are null.
    """

    record_id: int
    sentiment: SentimentLabel | None
    score: float | None
    confidence: float | None = None
    contributing_fields: dict[str, float] = {}
    drivers: dict[str, float] = {}
    rag: RecordRAG | None = None
    truncated: bool = False
    error: str | None = None


class Anomaly(RecordSentiment):
    z_score: float


class LLMUsage(BaseModel):
    """Totals for paid-engine calls in one run."""

    requests: int
    cached: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_complete: bool


class Summary(BaseModel):
    total_records: int
    failed_records: int = 0
    distribution: dict[str, int]
    dominant_sentiment: SentimentLabel | None
    avg_score: float | None = None
    avg_confidence: float | None = None
    llm_usage: LLMUsage | None = None


class RAGMetadata(BaseModel):
    """
    Metadata describing RAG usage for an analysis run.
    """

    enabled: bool
    used: bool
    source_count: int = 0


class TrendBucket(BaseModel):
    bucket: str
    count: int
    avg_score: float
    distribution: dict[str, int]


class Trend(BaseModel):
    granularity: Granularity
    buckets: list[TrendBucket]
    unparseable: int


class AnalysisResponse(BaseModel):
    """Complete response payload for an analysis run."""

    run_id: str
    engine: str
    input_sha256: str
    text_fields: list[str]
    summary: Summary
    records: list[RecordSentiment]
    anomalies: list[Anomaly]
    explanations: list[str]
    rag: RAGMetadata | None = None
    trend: Trend | None = None


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------
class RunSummary(BaseModel):
    run_id: str
    engine: str
    summary: dict[str, Any]
    rag_enabled: bool
    created_at: datetime | None
    source_type: str | None = None
    input_sha256: str | None = None
    config: dict[str, Any] = {}
    records_in: int | None = None
    records_skipped: int | None = None
    records_failed: int | None = None
    duration_ms: int | None = None
    status: str = "completed"
    error: str | None = None


class RunPage(BaseModel):
    items: list[RunSummary]
    total: int
    limit: int
    offset: int


class StoredRecord(BaseModel):
    record_id: int
    sentiment: SentimentLabel | None
    score: float | None
    confidence: float | None = None
    contributing_fields: dict[str, float] = {}
    drivers: dict[str, float] = {}
    rag_chunk_ids: list[str] = []
    truncated: bool = False
    error: str | None = None


class RunDetail(RunSummary):
    records: list[StoredRecord]
    records_total: int
    records_limit: int
    records_offset: int


class ChangedRecord(BaseModel):
    record_id: int
    base_sentiment: SentimentLabel
    other_sentiment: SentimentLabel
    base_score: float
    other_score: float


class ComparisonStats(BaseModel):
    records_compared: int
    label_changes: int
    label_change_rate: float
    mean_score_delta: float | None
    mean_confidence_delta: float | None


class RunComparison(BaseModel):
    base_run_id: str
    rag_run_id: str
    comparison: ComparisonStats
    changed_records: list[ChangedRecord]


class ABTestResponse(BaseModel):
    """The same input analyzed without and with RAG, plus the diff."""

    base: AnalysisResponse
    rag: AnalysisResponse
    comparison: RunComparison


# ----------------------------------------------------------------------
# Background jobs
# ----------------------------------------------------------------------
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class Job(BaseModel):
    """A background analysis job. ``run_id`` is set once it completes."""

    job_id: str
    status: JobStatus
    engine: str | None
    source_type: str | None
    records_total: int | None
    records_done: int | None
    progress: float
    run_id: str | None
    error: str | None
    created_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None


class JobPage(BaseModel):
    items: list[Job]
    total: int
    limit: int
    offset: int


# ----------------------------------------------------------------------
# RAG
# ----------------------------------------------------------------------
class RagDocument(BaseModel):
    document_id: str
    namespace: str | None = None
    source: str | None = None


class RagUploadResponse(BaseModel):
    document_id: str
    filename: str | None
    namespace: str
    chunks_added: int


class RagDeleteResponse(BaseModel):
    status: Literal["deleted"]
    document_id: str


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    namespace: str | None = Field(default=None, pattern=NAMESPACE_PATTERN)


class RetrieveHit(BaseModel):
    content: str
    source: str | None
    document_id: str | None
    namespace: str | None
    score: float


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
class AccuracyRequest(BaseModel):
    predictions: list[str] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _same_length(self) -> "AccuracyRequest":
        if len(self.predictions) != len(self.labels):
            raise ValueError("predictions and labels must have the same length")
        return self


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class ConfusionMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class EvaluationReport(BaseModel):
    total_samples: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: ConfusionMatrix


class AccuracyResponse(BaseModel):
    metric: Literal["accuracy"]
    value: float
    total_samples: int
    report: EvaluationReport


class LabeledText(BaseModel):
    text: str = Field(min_length=1)
    label: SentimentLabel


class EngineEvaluationRequest(BaseModel):
    sentiment_engine: EngineName | None = None
    items: list[LabeledText] = Field(min_length=1)


class EngineEvaluationResponse(BaseModel):
    engine: str
    failed: int
    report: EvaluationReport


# ----------------------------------------------------------------------
# Engines & health
# ----------------------------------------------------------------------
class EngineInfo(BaseModel):
    name: str
    display_name: str
    description: str
    uses_context: bool
    is_paid: bool
    configured: bool
    loaded: bool
    default: bool


class Limits(BaseModel):
    max_upload_bytes: int
    max_records: int
    max_text_chars: int
    max_llm_records: int


class EngineList(BaseModel):
    engines: list[EngineInfo]
    limits: Limits
    rag_enabled: bool


class Health(BaseModel):
    status: str
    version: str


class Readiness(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, str]
