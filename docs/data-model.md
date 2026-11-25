# Data model

This document describes the shapes that cross layer boundaries: the API, the
engines, RAG, and persistence. The authoritative source for the API is
[`api/schemas.py`](../sentiment_analyzer/api/schemas.py).

## API contract

Shared types:

```python
SentimentLabel = Literal["positive", "neutral", "negative"]
SourceType     = Literal["json", "csv", "excel", "text"]
Granularity    = Literal["day", "week", "month", "year"]

class EngineName(StrEnum):
    RULE_BASED = "rule_based"
    HF_TRANSFORMER = "hf_transformer"
    LLM = "llm"
```

### `AnalysisRequest`

```python
class AnalysisRequest(BaseModel):
    data: list[dict[str, Any]] | str
    source_type: SourceType
    sentiment_engine: EngineName | None = None       # ASA_DEFAULT_SENTIMENT_ENGINE
    text_fields: list[str] | None = None             # min 1 item when given
    use_rag: bool | None = None                      # ASA_RAG_ENABLED
    timestamp_field: str | None = None
    granularity: Granularity = "day"
```

A validator requires `data` to be a string when `source_type` is `text` and a
list otherwise. Field-by-field semantics are in
[api-reference.md](api-reference.md#post-analyze).

### `RecordSentiment`

```python
class RecordRAG(BaseModel):
    enabled: bool
    sources: list[str] = []
    chunks_used: int = 0

class RecordSentiment(BaseModel):
    record_id: int
    sentiment: SentimentLabel | None
    score: float | None                      # polarity in [-1, 1], rounded to 4 dp
    confidence: float | None = None          # in [0, 1], rounded to 4 dp
    contributing_fields: dict[str, float] = {}
    drivers: dict[str, float] = {}
    rag: RecordRAG | None = None
    truncated: bool = False
    error: str | None = None
```

- `record_id` is the record's index in the original input (CSV/Excel: the
  data row, starting at 0). Skipped records leave gaps.
- `score` and `confidence` follow the engine contract below.
- `contributing_fields` maps each text field to its own polarity. With one
  text field it is `{field: score}`. With several, each field is scored
  separately, except for paid engines (`llm`), where it is `{}`.
- `drivers` are engine-specific numeric signals (see
  [engines.md](engines.md)).
- `rag` describes the context used for this record; `enabled` is whether RAG
  was active for the run.
- `truncated` is `true` when the text was cut to `ASA_MAX_TEXT_CHARS`.
- On failure, `error` holds `"<ExceptionType>: <message>"` (at most 500
  characters), `sentiment`/`score`/`confidence` are `null`, and
  `contributing_fields`/`drivers` are empty.

### `Summary`

```python
class Summary(BaseModel):
    total_records: int                    # successfully analyzed records
    failed_records: int = 0
    distribution: dict[str, int]          # label -> count, successful records only
    dominant_sentiment: SentimentLabel | None
    avg_score: float | None = None
    avg_confidence: float | None = None
```

Computed by [`aggregate`](../sentiment_analyzer/analytics/aggregation.py) over
successful records only. Ties for the dominant label are broken in the order
positive > neutral > negative. `dominant_sentiment` and the averages are
`null` when nothing succeeded.

### `RAGMetadata`

```python
class RAGMetadata(BaseModel):
    enabled: bool          # RAG was active: requested and the engine uses context
    used: bool             # at least one record received at least one chunk
    source_count: int = 0  # distinct sources across all records
```

Present only when `ASA_RAG_ENABLED=true`; otherwise `AnalysisResponse.rag` is
`null`. See [rag.md](rag.md#enrichment-in-the-analysis-pipeline).

### `Trend`

```python
class TrendBucket(BaseModel):
    bucket: str                   # "2026-01-05", "2026-W02", "2026-01" or "2026"
    count: int
    avg_score: float
    distribution: dict[str, int]

class Trend(BaseModel):
    granularity: Granularity
    buckets: list[TrendBucket]    # sorted by bucket
    unparseable: int              # records whose timestamp could not be parsed
```

Built by
[`aggregate_by_time`](../sentiment_analyzer/analytics/time_aggregation.py)
when the request sets `timestamp_field`. Timestamps are parsed with pandas
and bucketed in UTC; weeks are ISO weeks. Failed records are excluded and not
counted as unparseable.

### `AnalysisResponse`

```python
class AnalysisResponse(BaseModel):
    run_id: str                   # UUID
    engine: str
    input_sha256: str             # fingerprint of the input records
    text_fields: list[str]        # fields that were analyzed
    summary: Summary
    records: list[RecordSentiment]
    anomalies: list[Anomaly]      # RecordSentiment + z_score
    explanations: list[str]       # one per record, same order as `records`
    rag: RAGMetadata | None = None
    trend: Trend | None = None
```

- `input_sha256` is the SHA-256 of the input records serialized as canonical
  JSON (sorted keys). Two runs with the same value analyzed the same input,
  which is what run comparison checks.
- `anomalies` are successful records whose score lies more than 2 standard
  deviations from the run's mean, each with its `z_score`
  ([`analytics/anomalies.py`](../sentiment_analyzer/analytics/anomalies.py)).
  Empty when fewer than two records succeeded or all scores are equal.
- `explanations` are deterministic one-liners from
  [`analytics/explainability.py`](../sentiment_analyzer/analytics/explainability.py),
  for example `Record 0 classified as positive with score 1.00 (confidence
  1.00); strongest field: 'comment' (+1.00).` Failed records get `Record N
  could not be analyzed: <error>`.

### Run, evaluation and RAG schemas

| Schema                                   | Used by                                     |
| ---------------------------------------- | ------------------------------------------- |
| `RunSummary`, `RunPage`                  | `GET /runs`                                 |
| `RunDetail`, `StoredRecord`              | `GET /runs/{run_id}`                        |
| `RunComparison`, `ComparisonStats`, `ChangedRecord` | `GET /runs/compare`              |
| `ABTestResponse`                         | `POST /runs/ab`                             |
| `RagUploadResponse`, `RagDocument`, `RagDeleteResponse`, `RetrieveRequest`, `RetrieveHit` | `/rag/*` |
| `AccuracyRequest`, `AccuracyResponse`, `EvaluationReport`, `ClassMetrics`, `ConfusionMatrix` | `POST /evaluation/sentiment` |
| `EngineEvaluationRequest`, `LabeledText`, `EngineEvaluationResponse` | `POST /evaluation/engine` |
| `EngineList`, `EngineInfo`, `Limits`     | `GET /engines`                              |
| `Health`, `Readiness`                    | `/healthz`, `/readyz`                       |

Examples of each are in [api-reference.md](api-reference.md).

## Engine contract

Every engine returns a `SentimentResult` from
[`sentiment/base.py`](../sentiment_analyzer/sentiment/base.py):

```python
@dataclass(frozen=True)
class SentimentResult:
    label: Literal["positive", "neutral", "negative"]
    polarity: float           # clamped to [-1, 1]
    confidence: float         # clamped to [0, 1]
    drivers: dict[str, float] = {}
```

`analyze_batch` returns `list[SentimentResult | Exception]`, one item per
input text. The service maps `polarity` to `score` and an exception to the
record's `error`. See [engines.md](engines.md).

## RAG types

Defined in [`rag/models.py`](../sentiment_analyzer/rag/models.py):

```python
@dataclass(frozen=True)
class RAGChunk:
    content: str
    source: str
    score: float           # 1 - distance; higher = more similar
    document_id: str = ""
    namespace: str = ""
    chunk_id: str = ""     # "{document_id}:{chunk_index}"

@dataclass
class RAGResult:
    chunks: list[RAGChunk] = []
    # properties
    context: str           # "\n".join(c.content for c in chunks)
    sources: list[str]     # sorted unique non-empty sources
    chunk_ids: list[str]
```

## Vector store records

Each chunk stored in Chroma (collection `rag_context`, via
[`rag/vector_store.py`](../sentiment_analyzer/rag/vector_store.py)) has the
id `{document_id}:{chunk_index}` and the metadata:

```python
{
    "document_id": str,
    "namespace":   str,   # "" if none was given
    "source":      str,   # filename or label; "" if none
    "chunk_index": int,
}
```

## Run persistence

Runs are stored in DuckDB at `ASA_RUNS_DB_PATH` (default `runs.duckdb`) by
[`RunStore`](../sentiment_analyzer/storage/runs.py).

### Migrations

The schema is versioned. A `schema_version` table records the applied
migrations; on startup `RunStore` applies any missing ones in order, each in
its own transaction. Migrations are append-only:

| Version | Migration                       | Change                                                        |
| ------- | ------------------------------- | ------------------------------------------------------------- |
| 1       | `_m1_initial`                   | `runs` table (original five columns).                         |
| 2       | `_m2_run_details_and_records`   | New `runs` columns and the `run_records` table.               |

Databases created by older versions are upgraded in place; `CREATE ... IF NOT
EXISTS` and `ADD COLUMN IF NOT EXISTS` make the first migration safe on them.

### `runs`

```sql
CREATE TABLE runs (
    run_id          TEXT PRIMARY KEY,
    engine          TEXT,
    summary         JSON,       -- Summary
    rag_enabled     BOOLEAN,    -- RAG was active for the run
    created_at      TIMESTAMP,  -- naive, always UTC
    source_type     TEXT,       -- json | csv | excel | text
    input_sha256    TEXT,
    config          JSON,       -- config snapshot
    records_in      INTEGER,    -- records received
    records_skipped INTEGER,    -- records with no text in any text field
    records_failed  INTEGER,    -- records the engine failed on
    duration_ms     INTEGER,
    status          TEXT,       -- completed | completed_with_errors
    error           TEXT        -- reserved; currently always NULL
);
```

The `config` snapshot holds the engine's `config_snapshot()` (for example
`model` and `revision` for `hf_transformer`) plus `app_version`,
`text_fields`, `rag` (`requested`, `active`, `top_k`, `namespace`) and
`max_text_chars`.

`created_at` is written by the application as UTC and returned by the API with
a `Z` suffix. **Legacy rows** written before this change got `created_at`
from DuckDB's `CURRENT_TIMESTAMP` default, which was local time; they are
still reported as UTC, so they may be off by the server's UTC offset. Legacy
rows also have `NULL` for the columns added in migration 2 (the API reports
`status` as `completed` and `config` as `{}`), and no per-record results.

### `run_records`

```sql
CREATE TABLE run_records (
    run_id              TEXT NOT NULL,
    record_id           INTEGER NOT NULL,
    sentiment           TEXT,
    score               DOUBLE,
    confidence          DOUBLE,
    contributing_fields JSON,
    drivers             JSON,
    rag_chunk_ids       JSON,     -- ids of the chunks used as context
    truncated           BOOLEAN,
    error               TEXT,
    PRIMARY KEY (run_id, record_id)
);
```

The run row and all its records are written in one transaction. Records are
read back ordered by `record_id`.

### Concurrency

DuckDB allows one writing process per database file. Run the API as a single
process (one uvicorn worker). Inside it, `RunStore` shares one connection,
opens a cursor per operation, and serializes writes with a lock, so it is
safe across FastAPI's threadpool. Do not point two API processes at the same
file.

## Field detection

Unless the request lists `text_fields`,
[`detect_text_fields`](../sentiment_analyzer/processing/field_detector.py)
inspects up to the first 200 records and keeps a field when, among its
non-missing values:

- at least half are strings,
- the median string length is at least 8 characters,
- values are varied: with 5 or more strings, at least 20% are distinct (this
  rules out categorical columns),
- values look like words: at least one contains a space or a letter (this
  rules out ids and numeric codes).

Fields keep the order in which they first appear. If none qualify, the request
fails with `422` listing the available fields; pass `text_fields` explicitly.

## Missing values

CSV and Excel adapters turn empty cells (NaN/NaT/NA) into `None`.
[`normalize_record`](../sentiment_analyzer/processing/normalizer.py) then maps
`None` and any pandas missing marker to `""`, and every other value to a
stripped string. A record with no text in any text field is skipped
(`records_skipped`); it is not an error.
