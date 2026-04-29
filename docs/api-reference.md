# HTTP API reference

All endpoints are mounted under the `/api` prefix in
[`main.py`](../sentiment_analyzer/main.py). For an interactive view of these
schemas, run the API and open:

- Swagger UI — `http://localhost:8000/docs`
- ReDoc — `http://localhost:8000/redoc`
- OpenAPI JSON — `http://localhost:8000/openapi.json`

CORS is permissive for `http://localhost:5173` and `http://localhost:8501`
(React dev server and Streamlit). Adjust `app.add_middleware(...)` in
`main.py` for production.

The Pydantic schemas referenced below live in
[`api/schemas.py`](../sentiment_analyzer/api/schemas.py).

## Analysis

### `POST /api/analyze`

Run sentiment analysis on a JSON list-of-records.

**Request body** — `AnalysisRequest`:

| Field              | Type     | Default      | Notes                                              |
| ------------------ | -------- | ------------ | -------------------------------------------------- |
| `data`             | any      | required     | List of dicts, or any payload accepted by adapters.|
| `source_type`      | string   | required     | `json` \| `csv` \| `excel` \| `text`. Informational; the route only uses the JSON adapter at this stage. |
| `sentiment_engine` | string   | `rule_based` | `rule_based` \| `hf_transformer` \| `llm`.         |

**Example**:

```bash
curl -s -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "data": [
      {"comment": "I love this product"},
      {"comment": "Terrible experience"}
    ],
    "source_type": "json",
    "sentiment_engine": "rule_based"
  }' | jq
```

**Response** — `AnalysisResponse`. See
[data-model.md](data-model.md#analysisresponse) for the full shape.

### `POST /api/analyze/csv`

Multipart upload (`file` field) of a `.csv`. The CSV is parsed via pandas
([`ingestion/csv_adapter.py`](../sentiment_analyzer/ingestion/csv_adapter.py))
and routed through the same orchestration as `POST /api/analyze`.

| Query param        | Default      | Description                |
| ------------------ | ------------ | -------------------------- |
| `sentiment_engine` | `rule_based` | One of the engine values.  |

```bash
curl -s -X POST "http://localhost:8000/api/analyze/csv?sentiment_engine=hf_transformer" \
  -F "file=@reviews.csv"
```

Returns `400` if the filename does not end in `.csv`.

### `POST /api/analyze/excel`

As above, but for `.xlsx`. Reads the first sheet only by default
([`ingestion/excel_adapter.py`](../sentiment_analyzer/ingestion/excel_adapter.py)).
Returns `400` if the filename does not end in `.xlsx`.

```bash
curl -s -X POST "http://localhost:8000/api/analyze/excel?sentiment_engine=llm" \
  -F "file=@feedback.xlsx"
```

## Runs

### `GET /api/runs`

List all persisted runs in DuckDB, newest first. Each item:

```json
{
  "run_id": "abc123",
  "engine": "rule_based",
  "summary": { ... },
  "rag_enabled": true,
  "created_at": "2026-04-29T12:34:56"
}
```

Backed by
[`RunStore._load_all_runs`](../sentiment_analyzer/storage/runs.py).

### `GET /api/runs/compare`

Compare two runs — typically baseline vs. RAG-enriched.

| Query param    | Required | Description                |
| -------------- | -------- | -------------------------- |
| `base_run_id`  | yes      | Run id (without RAG).      |
| `rag_run_id`   | yes      | Run id (with RAG).         |

`404` if either id is unknown.

Response:

```json
{
  "base_run_id": "...",
  "rag_run_id": "...",
  "comparison": {
    "confidence_gain": 0.12,
    "label_changes": 3,
    "rag_helpfulness": true
  }
}
```

Logic in
[`analytics/run_comparison.py`](../sentiment_analyzer/analytics/run_comparison.py).

## RAG

### `POST /api/rag/documents`

Upload a text document for RAG enrichment.

| Query param | Default     | Description                                |
| ----------- | ----------- | ------------------------------------------ |
| `namespace` | `default`   | Logical namespace (tenant / project / etc.)|

Body: `multipart/form-data` with a `file` field.

```bash
curl -s -X POST "http://localhost:8000/api/rag/documents?namespace=policies" \
  -F "file=@policy.txt"
```

Response:

```json
{
  "document_id": "uuid",
  "filename": "policy.txt",
  "namespace": "policies",
  "chunks_added": 5
}
```

The body is decoded as UTF-8, split into 500-char chunks, embedded with
`sentence-transformers/all-MiniLM-L6-v2`, and persisted in Chroma. See
[rag.md](rag.md).

### `GET /api/rag/documents`

Returns one entry per unique document seen in the vector store:

```json
[
  { "document_id": "uuid", "namespace": "default", "source": "policy.txt" }
]
```

### `DELETE /api/rag/documents/{document_id}`

Removes all chunks belonging to that `document_id`.

```json
{ "status": "deleted", "document_id": "uuid" }
```

### `POST /api/rag/retrieve`

Preview semantic search results. Does **not** persist or affect a run.

Body:

```json
{
  "query": "refund delays",
  "top_k": 5,
  "namespace": "default"
}
```

| Field       | Default                    | Notes                                |
| ----------- | -------------------------- | ------------------------------------ |
| `query`     | required                   |                                      |
| `top_k`     | `ASA_RAG_TOP_K` (5)        |                                      |
| `namespace` | `ASA_RAG_NAMESPACE`        | Pass `null` to ignore namespace.     |

Response:

```json
[
  {
    "content": "chunk text",
    "source": "policy.txt",
    "document_id": "uuid",
    "namespace": "default",
    "score": 0.83
  }
]
```

`score` is `1 - distance` from Chroma — higher is more similar.

## Evaluation

### `POST /api/evaluation/sentiment`

Compute simple accuracy against ground-truth labels. Predictions and labels
are passed as **query parameters** (lists), not as a JSON body — see
[`api/evaluation_routes.py`](../sentiment_analyzer/api/evaluation_routes.py).

```bash
curl -s -X POST "http://localhost:8000/api/evaluation/sentiment\
?predictions=positive&predictions=negative&predictions=neutral\
&labels=positive&labels=negative&labels=positive"
```

Response:

```json
{
  "metric": "accuracy",
  "value": 0.6667,
  "total_samples": 3
}
```

Accuracy = `correct / total`. Internally this uses
`EvaluationHarness.precision`, which is the misnomer for "accuracy" today.
See [`evaluation/harness.py`](../sentiment_analyzer/evaluation/harness.py)
for additional metrics (`drift`, `stability`) that are not yet exposed via
HTTP.

## Errors

The API does not currently impose a uniform error envelope; FastAPI's default
JSON error shape is used. Common cases:

| Endpoint                          | Status | Cause                                    |
| --------------------------------- | ------ | ---------------------------------------- |
| `POST /api/analyze/csv`           | 400    | File extension is not `.csv`.            |
| `POST /api/analyze/excel`         | 400    | File extension is not `.xlsx`.           |
| `GET /api/runs/compare`           | 404    | Either run id was not found.             |
| Any analysis route                | 5xx    | Engine failure (e.g. LLM provider down). Errors are logged via structlog; the original exception is re-raised. |
