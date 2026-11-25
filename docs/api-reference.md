# HTTP API reference

All application endpoints are served under `/api/v1`, set up in
[`main.py`](../sentiment_analyzer/main.py). The same routes are also served
under the unversioned `/api` prefix as a **deprecated** alias, hidden from the
OpenAPI schema; new clients should use `/api/v1`. The health probes
`/healthz` and `/readyz` sit outside both prefixes.

The schemas referenced below live in
[`api/schemas.py`](../sentiment_analyzer/api/schemas.py). Every JSON endpoint
declares a `response_model`, so the generated OpenAPI schema is complete.
Outside `prod` you can browse it at:

- Swagger UI — `http://localhost:8000/docs`
- ReDoc — `http://localhost:8000/redoc`
- OpenAPI JSON — `http://localhost:8000/openapi.json`

With `ASA_ENVIRONMENT=prod` these three URLs are disabled.

## Conventions

### Authentication

When `ASA_API_KEYS` is set, every `/api/v1/*` (and `/api/*`) request must send
one of the keys:

```
X-API-Key: <key>
```

A missing or wrong key returns `401` (`code: "unauthorized"`, the standard
[error envelope](#errors)) and a `WWW-Authenticate: X-API-Key` header. When no keys are configured
the API is open; this is only allowed with `ASA_ENVIRONMENT=dev`, and the app
refuses to start otherwise. `/healthz` and `/readyz` never require a key.

The `curl` examples below assume:

```bash
API=http://localhost:8000/api/v1
KEY=your-key   # omit -H "X-API-Key: $KEY" when auth is disabled
```

### Request IDs

Responses carry an `X-Request-ID` header. If the request sends a printable
`X-Request-ID` of at most 128 characters it is reused; otherwise a new one is
generated. The same ID appears in error bodies and on every log line for the
request. (Unhandled-error `500` responses are produced outside the request-ID
middleware, so they have the ID in the body but not in the header.)

### Limits

| Setting                | Default | Effect                                                                   |
| ---------------------- | ------- | ------------------------------------------------------------------------ |
| `ASA_MAX_UPLOAD_BYTES` | 10 MB   | Larger request bodies or uploaded files get `413`.                       |
| `ASA_MAX_RECORDS`      | 10,000  | More records in one run get `413`.                                       |
| `ASA_MAX_LLM_RECORDS`  | 500     | More records in one run of a paid engine (`llm`) get `413`.              |
| `ASA_MAX_TEXT_CHARS`   | 5,000   | Longer texts are truncated and the record is flagged `truncated: true`. |

`GET /engines` returns the current values.

### CORS

Allowed origins come from `ASA_CORS_ORIGINS` (default
`http://localhost:5173,http://localhost:8501`). Allowed methods are `GET`,
`POST`, `DELETE`; allowed headers are `Content-Type`, `X-API-Key`,
`X-Request-ID`; `X-Request-ID` is exposed to browsers. Credentials are not
allowed.

## Health

### `GET /healthz`

Liveness. Performs no I/O.

```json
{ "status": "ok", "version": "1.0.0" }
```

### `GET /readyz`

Readiness. Checks the runs database, the vector store and the default engine
(building it if it is not loaded yet). Returns `200` when all pass, `503`
otherwise:

```json
{
  "status": "ready",
  "checks": { "runs_db": "ok", "vector_store": "ok", "default_engine": "ok" }
}
```

A failed check reads `"error: <ExceptionType>"` and `status` is `"not_ready"`.

## Engines

### `GET /engines`

Available engines, their capabilities and status, the input limits, and
whether RAG is enabled. Does not load any engine.

```json
{
  "engines": [
    {
      "name": "rule_based",
      "display_name": "Rule-Based (Lexical)",
      "description": "Fast, deterministic lexical matching. Best for auditable results.",
      "uses_context": false,
      "is_paid": false,
      "configured": true,
      "loaded": false,
      "default": true
    },
    {
      "name": "llm",
      "display_name": "LLM (any litellm provider)",
      "description": "Provider-agnostic LLM via litellm. Most nuanced, slowest, and billed per call.",
      "uses_context": true,
      "is_paid": true,
      "configured": false,
      "loaded": false,
      "default": false
    }
  ],
  "limits": {
    "max_upload_bytes": 10485760,
    "max_records": 10000,
    "max_text_chars": 5000,
    "max_llm_records": 500
  },
  "rag_enabled": true
}
```

(The `hf_transformer` entry is omitted above for brevity.) `configured` is
`false` for `llm` until `ASA_LLM_MODEL` is set. See [engines.md](engines.md).

## Analysis

### `POST /analyze`

Analyze a JSON list of records, or plain text.

**Request body** — `AnalysisRequest`:

| Field              | Type                     | Default                        | Notes |
| ------------------ | ------------------------ | ------------------------------ | ----- |
| `data`             | `list[object]` \| string | required                       | A list of objects for `json` (also accepted for `csv` / `excel`). A string for `text`, analyzed one record per non-empty line. |
| `source_type`      | string                   | required                       | `json` \| `csv` \| `excel` \| `text`. Must match the type of `data` (`422` otherwise). |
| `sentiment_engine` | string \| null           | `ASA_DEFAULT_SENTIMENT_ENGINE` | `rule_based` \| `hf_transformer` \| `llm`. |
| `text_fields`      | `list[string]` \| null   | auto-detect                    | Fields to analyze; at least one if given. Unknown fields give `422`. Ignored for `text`. |
| `use_rag`          | bool \| null             | `ASA_RAG_ENABLED`              | Force RAG on or off for this run. Only engines that use context are affected. `true` while RAG is disabled gives `422`. |
| `timestamp_field`  | string \| null           | null                           | Field holding dates. When set, the response includes `trend`. `422` if no record has it. |
| `granularity`      | string                   | `day`                          | `day` \| `week` \| `month` \| `year`. Trend bucket size. |

If `text_fields` is omitted and no free-text field is detected, the request
fails with `422` and lists the available fields.

**Example**:

```bash
curl -s -X POST "$API/analyze" \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{
    "source_type": "json",
    "sentiment_engine": "rule_based",
    "text_fields": ["comment", "title"],
    "timestamp_field": "date",
    "granularity": "month",
    "data": [
      {"id": 1, "comment": "I love this product, great value", "title": "Great buy overall", "date": "2026-01-05"},
      {"id": 2, "comment": "Terrible support and the delivery was late", "title": "Not good at all", "date": "2026-01-12"},
      {"id": 3, "comment": "It arrived on Tuesday in a box", "title": "Delivery happened", "date": "2026-02-02"}
    ]
  }'
```

Plain text:

```bash
curl -s -X POST "$API/analyze" \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"source_type": "text", "data": "Great service!\nThe delivery was late."}'
```

**Response** — `AnalysisResponse` (records 1 and 2 elided):

```json
{
  "run_id": "9916aa47-538e-454d-8c63-ad6949092432",
  "engine": "rule_based",
  "input_sha256": "a04adc49f4f12ca087fcafc073707c95beddf2886e16290647061918eeffcbea",
  "text_fields": ["comment", "title"],
  "summary": {
    "total_records": 3,
    "failed_records": 0,
    "distribution": {"positive": 1, "negative": 1, "neutral": 1},
    "dominant_sentiment": "positive",
    "avg_score": 0.0,
    "avg_confidence": 0.6667
  },
  "records": [
    {
      "record_id": 0,
      "sentiment": "positive",
      "score": 1.0,
      "confidence": 1.0,
      "contributing_fields": {"comment": 1.0, "title": 1.0},
      "drivers": {"positive_hits": 3.0, "negative_hits": 0.0, "negated_hits": 0.0},
      "rag": {"enabled": false, "sources": [], "chunks_used": 0},
      "truncated": false,
      "error": null
    }
  ],
  "anomalies": [],
  "explanations": [
    "Record 0 classified as positive with score 1.00 (confidence 1.00); strongest field: 'comment' (+1.00).",
    "Record 1 classified as negative with score -1.00 (confidence 1.00); strongest field: 'comment' (-1.00).",
    "Record 2 classified as neutral with score 0.00 (confidence 0.00); strongest field: 'comment' (+0.00)."
  ],
  "rag": {"enabled": false, "used": false, "source_count": 0},
  "trend": {
    "granularity": "month",
    "buckets": [
      {"bucket": "2026-01", "count": 2, "avg_score": 0.0, "distribution": {"positive": 1, "negative": 1}},
      {"bucket": "2026-02", "count": 1, "avg_score": 0.0, "distribution": {"neutral": 1}}
    ],
    "unparseable": 0
  }
}
```

Field semantics are in [data-model.md](data-model.md#analysisresponse). In
short:

- `record_id` is the record's index in the input. Records with no text in any
  text field are skipped, so ids can have gaps.
- A record the engine failed on has `error` set and `sentiment`, `score`,
  `confidence` null. It is counted in `summary.failed_records` and excluded
  from every other summary figure. The request still succeeds.
- `rag` is `null` when `ASA_RAG_ENABLED=false`; `trend` is `null` without
  `timestamp_field`.

### `POST /analyze/csv`

Multipart upload with a `file` field holding a UTF-8 `.csv`. Parsed with pandas
([`ingestion/csv_adapter.py`](../sentiment_analyzer/ingestion/csv_adapter.py));
empty cells become missing values. At most `ASA_MAX_RECORDS + 1` rows are
parsed, so oversized files are rejected without being loaded fully.

| Query param        | Default                        | Description                                         |
| ------------------ | ------------------------------ | --------------------------------------------------- |
| `sentiment_engine` | `ASA_DEFAULT_SENTIMENT_ENGINE` | Engine name.                                        |
| `text_fields`      | auto-detect                    | Comma-separated column names, e.g. `review,title`.  |
| `use_rag`          | `ASA_RAG_ENABLED`              | `true` / `false`.                                   |
| `timestamp_field`  | none                           | Column with dates for the trend.                    |
| `granularity`      | `day`                          | `day` \| `week` \| `month` \| `year`.               |

```bash
curl -s -X POST "$API/analyze/csv?sentiment_engine=hf_transformer&text_fields=review" \
  -H "X-API-Key: $KEY" -F "file=@reviews.csv"
```

Returns `AnalysisResponse` with a run `source_type` of `csv`. A filename not
ending in `.csv` gives `415`; a file that is not UTF-8 gives `415`; an
unparseable file gives `422`.

### `POST /analyze/excel`

Same as `/analyze/csv` for `.xlsx` files (first sheet only,
[`ingestion/excel_adapter.py`](../sentiment_analyzer/ingestion/excel_adapter.py)).
A filename not ending in `.xlsx` gives `415`; an unreadable workbook gives
`422`.

```bash
curl -s -X POST "$API/analyze/excel?sentiment_engine=llm" \
  -H "X-API-Key: $KEY" -F "file=@feedback.xlsx"
```

## Runs

Every analysis is persisted. See [data-model.md](data-model.md#run-persistence).

### `GET /runs`

One page of runs, newest first.

| Query param   | Default | Description                                   |
| ------------- | ------- | --------------------------------------------- |
| `limit`       | `50`    | 1-500.                                        |
| `offset`      | `0`     | Runs to skip.                                 |
| `engine`      | -       | Filter by engine name.                        |
| `rag_enabled` | -       | Filter by whether RAG was active.             |

Response — `RunPage`:

```json
{
  "items": [
    {
      "run_id": "a232dc25-9fb1-4d5c-afaa-bc2d49499634",
      "engine": "rule_based",
      "summary": {
        "total_records": 2,
        "failed_records": 0,
        "distribution": {"positive": 1, "negative": 1},
        "dominant_sentiment": "positive",
        "avg_score": 0.0,
        "avg_confidence": 1.0
      },
      "rag_enabled": false,
      "created_at": "2026-11-25T08:25:48.208622Z",
      "source_type": "csv",
      "input_sha256": "ddf14c6e504ef693d3aefd1a49d18a96a65c497872ab8ce2992abf44514d05d3",
      "config": {
        "engine": "rule_based",
        "app_version": "1.0.0",
        "text_fields": ["review"],
        "rag": {"requested": true, "active": false, "top_k": 5, "namespace": "default"},
        "max_text_chars": 5000
      },
      "records_in": 2,
      "records_skipped": 0,
      "records_failed": 0,
      "duration_ms": 0,
      "status": "completed",
      "error": null
    }
  ],
  "total": 3,
  "limit": 1,
  "offset": 0
}
```

`created_at` is UTC. Runs written before the schema migration have `null`
for the newer fields and `{}` for `config`.

### `GET /runs/{run_id}`

Run metadata (same fields as a `RunPage` item) plus one page of per-record
results. `404` if the run does not exist.

| Query param      | Default | Description           |
| ---------------- | ------- | --------------------- |
| `records_limit`  | `100`   | 1-1000.               |
| `records_offset` | `0`     | Records to skip.      |

Response — `RunDetail`:

```json
{
  "run_id": "9916aa47-538e-454d-8c63-ad6949092432",
  "engine": "rule_based",
  "summary": {
    "total_records": 3,
    "failed_records": 0,
    "distribution": {"positive": 1, "negative": 1, "neutral": 1},
    "dominant_sentiment": "positive",
    "avg_score": 0.0,
    "avg_confidence": 0.6667
  },
  "rag_enabled": false,
  "created_at": "2026-11-25T08:25:48.186882Z",
  "source_type": "json",
  "input_sha256": "a04adc49f4f12ca087fcafc073707c95beddf2886e16290647061918eeffcbea",
  "config": {
    "engine": "rule_based",
    "app_version": "1.0.0",
    "text_fields": ["comment", "title"],
    "rag": {"requested": true, "active": false, "top_k": 5, "namespace": "default"},
    "max_text_chars": 5000
  },
  "records_in": 3,
  "records_skipped": 0,
  "records_failed": 0,
  "duration_ms": 2,
  "status": "completed",
  "error": null,
  "records": [
    {
      "record_id": 0,
      "sentiment": "positive",
      "score": 1.0,
      "confidence": 1.0,
      "contributing_fields": {"comment": 1.0, "title": 1.0},
      "drivers": {"positive_hits": 3.0, "negative_hits": 0.0, "negated_hits": 0.0},
      "rag_chunk_ids": [],
      "truncated": false,
      "error": null
    }
  ],
  "records_total": 3,
  "records_limit": 1,
  "records_offset": 0
}
```

Stored records carry `rag_chunk_ids` (the vector-store chunks used) instead
of the response's `rag` block.

### `GET /runs/{run_id}/export`

Download a run. `404` if the run does not exist.

| Query param | Default | Description      |
| ----------- | ------- | ---------------- |
| `format`    | `csv`   | `csv` \| `json`. |

- `csv` — one row per record with columns `record_id, sentiment, score,
  confidence, truncated, error, contributing_fields, drivers` (the last two
  as JSON strings).
- `json` — the run metadata with a `records` list of all stored records.

Both are sent with `Content-Disposition: attachment; filename="run-<run_id>.<format>"`.

```bash
curl -s -H "X-API-Key: $KEY" "$API/runs/$RUN_ID/export?format=csv" -o run.csv
```

### `GET /runs/compare`

Compare two runs over the **same input** record by record — for example a run
without RAG and one with RAG, or two engines.

| Query param   | Required | Description                                             |
| ------------- | -------- | ------------------------------------------------------- |
| `base_run_id` | yes      | The baseline run.                                       |
| `rag_run_id`  | yes      | The run to compare against it (any run, despite the name). |

- `404` if either run does not exist.
- `409` if the runs have different `input_sha256`, or either predates input
  fingerprinting.

Response — `RunComparison`:

```json
{
  "base_run_id": "9916aa47-538e-454d-8c63-ad6949092432",
  "rag_run_id": "43c76c8b-7835-487f-82d9-7325995ec2df",
  "comparison": {
    "records_compared": 3,
    "label_changes": 1,
    "label_change_rate": 0.3333,
    "mean_score_delta": -0.1,
    "mean_confidence_delta": 0.05
  },
  "changed_records": [
    {
      "record_id": 2,
      "base_sentiment": "neutral",
      "other_sentiment": "negative",
      "base_score": 0.0,
      "other_score": -0.3
    }
  ]
}
```

Records are paired by `record_id`; records that failed in either run are
skipped. Deltas are `other - base`; they are `null` when no pair has the
value. Logic in
[`analytics/run_comparison.py`](../sentiment_analyzer/analytics/run_comparison.py).

### `POST /runs/ab`

RAG A/B test: analyze the same input once without and once with RAG, persist
both runs, and compare them.

**Request body** — `AnalysisRequest` (as for `POST /analyze`; `use_rag` is
ignored). The engine must use context (`llm`) and RAG must be enabled
(`ASA_RAG_ENABLED=true`); otherwise `422` before anything runs or is saved.

Response — `ABTestResponse`:

```python
class ABTestResponse(BaseModel):
    base: AnalysisResponse  # run without RAG
    rag: AnalysisResponse  # run with RAG
    comparison: RunComparison  # base_run_id = base.run_id, rag_run_id = rag.run_id
```

Both runs are persisted and appear in `GET /runs`.

## RAG

Namespaces must match `^[A-Za-z0-9_-]{1,64}$` (`422` otherwise). A namespace is
a grouping label, not an access boundary; see [rag.md](rag.md).

### `POST /rag/documents`

Upload a UTF-8 text document to use as RAG context.

| Query param | Default   | Description          |
| ----------- | --------- | -------------------- |
| `namespace` | `default` | Namespace to use.    |

Body: `multipart/form-data` with a `file` field.

```bash
curl -s -X POST "$API/rag/documents?namespace=policies" \
  -H "X-API-Key: $KEY" -F "file=@policy.txt"
```

Response — `RagUploadResponse`:

```json
{
  "document_id": "87a8a501-776b-4a80-81e4-d9f4088741c8",
  "filename": "policy.txt",
  "namespace": "policies",
  "chunks_added": 1
}
```

The text is split into overlapping chunks (`ASA_RAG_CHUNK_SIZE`,
`ASA_RAG_CHUNK_OVERLAP`), embedded, and persisted. Non-UTF-8 content gives
`415`; a document with no text gives `422`; a file over
`ASA_MAX_UPLOAD_BYTES` gives `413`.

### `GET /rag/documents`

One entry per document, optionally filtered by `?namespace=...`. Response —
`list[RagDocument]`:

```json
[
  { "document_id": "87a8a501-776b-4a80-81e4-d9f4088741c8", "namespace": "policies", "source": "policy.txt" }
]
```

### `DELETE /rag/documents/{document_id}`

Delete all chunks of a document. Returns the same body whether or not the
document existed. Response — `RagDeleteResponse`:

```json
{ "status": "deleted", "document_id": "87a8a501-776b-4a80-81e4-d9f4088741c8" }
```

### `POST /rag/retrieve`

Preview semantic search. Does **not** persist anything or affect runs.

**Request body** — `RetrieveRequest`:

| Field       | Type           | Default             | Notes                     |
| ----------- | -------------- | ------------------- | ------------------------- |
| `query`     | string         | required            | 1-2000 characters.        |
| `top_k`     | int \| null    | `ASA_RAG_TOP_K`     | 1-50.                     |
| `namespace` | string \| null | `ASA_RAG_NAMESPACE` | Namespace to search.      |

```bash
curl -s -X POST "$API/rag/retrieve" \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"query": "refund delays", "top_k": 5, "namespace": "policies"}'
```

Response — `list[RetrieveHit]`:

```json
[
  {
    "content": "Refunds are processed within 30 days. Late deliveries are refunded.",
    "source": "policy.txt",
    "document_id": "87a8a501-776b-4a80-81e4-d9f4088741c8",
    "namespace": "policies",
    "score": 0.62
  }
]
```

`score` is `1 - distance`; higher is more similar.

## Evaluation

Evaluation results are returned, not persisted as runs.

### `POST /evaluation/sentiment`

Score existing predictions against ground-truth labels.

**Request body** — `AccuracyRequest`:

| Field         | Type           | Notes                                       |
| ------------- | -------------- | ------------------------------------------- |
| `predictions` | `list[string]` | At least one.                               |
| `labels`      | `list[string]` | Same length as `predictions` (`422` if not). |

```bash
curl -s -X POST "$API/evaluation/sentiment" \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"predictions": ["positive", "negative", "neutral"], "labels": ["positive", "negative", "positive"]}'
```

Response — `AccuracyResponse`:

```json
{
  "metric": "accuracy",
  "value": 0.6667,
  "total_samples": 3,
  "report": {
    "total_samples": 3,
    "accuracy": 0.6667,
    "macro_f1": 0.8334,
    "per_class": {
      "negative": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
      "neutral": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
      "positive": {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    },
    "confusion_matrix": {
      "labels": ["negative", "neutral", "positive"],
      "matrix": [[1, 0, 0], [0, 0, 0], [0, 1, 1]]
    }
  }
}
```

- `per_class` covers every label seen in either list; `support` is the number
  of true instances.
- `macro_f1` averages F1 over classes with `support > 0`.
- `confusion_matrix.matrix[i][j]` counts records whose true label is
  `labels[i]` and predicted label is `labels[j]`.

Logic in [`evaluation/harness.py`](../sentiment_analyzer/evaluation/harness.py).

### `POST /evaluation/engine`

Run an engine over labelled texts and report the same metrics.

**Request body** — `EngineEvaluationRequest`:

| Field              | Type                  | Default                        | Notes |
| ------------------ | --------------------- | ------------------------------ | ----- |
| `sentiment_engine` | string \| null        | `ASA_DEFAULT_SENTIMENT_ENGINE` | Engine name. |
| `items`            | `list[LabeledText]`   | required                       | At least one `{"text": str, "label": "positive" \| "neutral" \| "negative"}`. At most `ASA_MAX_RECORDS` items (`ASA_MAX_LLM_RECORDS` for paid engines), else `413`. |

Texts are truncated to `ASA_MAX_TEXT_CHARS`. RAG is not used.

```bash
curl -s -X POST "$API/evaluation/engine" \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{
    "sentiment_engine": "rule_based",
    "items": [
      {"text": "I love it", "label": "positive"},
      {"text": "awful service", "label": "negative"},
      {"text": "it is a chair", "label": "neutral"}
    ]
  }'
```

Response — `EngineEvaluationResponse`:

```json
{
  "engine": "rule_based",
  "failed": 0,
  "report": {
    "total_samples": 3,
    "accuracy": 1.0,
    "macro_f1": 1.0,
    "per_class": {
      "negative": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
      "neutral": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
      "positive": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
    },
    "confusion_matrix": {
      "labels": ["negative", "neutral", "positive"],
      "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    }
  }
}
```

Texts the engine fails on are counted in `failed` and excluded from the
report.

## Errors

Every error response shares one JSON envelope, and its `request_id` matches
the `X-Request-ID` response header:

```json
{
  "detail": "Runs analyzed different inputs and cannot be compared",
  "code": "conflict",
  "request_id": "a5808f8c4c9e4694a542373ede9892d7"
}
```

| Status | `code`                  | Typical causes |
| ------ | ----------------------- | -------------- |
| `401`  | `unauthorized`          | Missing or wrong `X-API-Key`. |
| `413`  | `input_too_large`       | Body or file over `ASA_MAX_UPLOAD_BYTES`; more than `ASA_MAX_RECORDS` / `ASA_MAX_LLM_RECORDS` records or evaluation items. |
| `415`  | `unsupported_input`     | Wrong upload extension; CSV or RAG document not UTF-8. |
| `422`  | `invalid_input`         | No text fields detected; unknown `text_fields` or `timestamp_field`; unparseable CSV/Excel; empty RAG document; `use_rag: true` while RAG is disabled; `/runs/ab` with an engine that ignores context. |
| `422`  | `validation_error`      | Request failed schema validation (wrong types, missing fields, bad enum or namespace values). `detail` is FastAPI's list of `{type, loc, msg, input, ...}` objects rather than a string. |
| `404`  | `not_found`             | Unknown run id or unknown path. |
| `409`  | `conflict`              | Comparing runs of different inputs. |
| `503`  | `engine_unavailable`    | Unknown engine, `llm` without `ASA_LLM_MODEL`, `hf_transformer` without the `ml` extra, or an engine that failed to load. |
| `502`  | `engine_output_invalid` | An engine returned output that breaks the engine contract (e.g. invalid LLM JSON). During analysis and engine evaluation this is caught per record, so it normally shows up as a record's `error` (`EngineOutputError: ...`) rather than as the response status. |
| `500`  | `internal_error`        | Unexpected error. Details are logged with the request ID, not returned. |

`detail` is a string in every case except `validation_error`.
