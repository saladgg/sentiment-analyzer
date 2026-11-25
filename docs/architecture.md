# Architecture

SentimentAnalyzer is a layered Python service that turns heterogeneous input
(JSON records, plain text, CSV, Excel) into explainable, persisted sentiment
results. This document describes the layers, what each is responsible for,
and how a request flows through them.

## High-level layout

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit frontend  (frontend/)                             │
│  Analyze, Run History, RAG Management pages                  │
└──────────────────────────────────────────────────────────────┘
                          │ HTTP + X-API-Key
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI app  (main.py: create_app)                          │
│  Middleware: request ID + access log, body-size limit, CORS  │
│  /healthz, /readyz (open) · /api/v1/* (API key)              │
└──────────────────────────────────────────────────────────────┘
     │ analyze │ runs │ rag │ evaluation │ engines
     ▼         ▼      ▼     ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│  app.state singletons, built once in lifespan:               │
│  RunStore · VectorStore · EngineRegistry · AnalysisService   │
└──────────────────────────────────────────────────────────────┘
   │           │            │             │            │
   ▼           ▼            ▼             ▼            ▼
ingestion/  processing/  rag/        sentiment/   analytics/
                                                       │
                                                       ▼
                                                  storage/
                                                  (DuckDB)
```

## Application wiring

[`main.py`](../sentiment_analyzer/main.py) exposes
`create_app(settings=None, *, embedding_function=None)` and a module-level
`app = create_app()` for uvicorn.

`create_app`:

1. Loads `Settings` (or uses the ones passed in) and calls
   `validate_for_startup()`, which refuses to start outside `dev` without
   API keys.
2. Configures JSON logging.
3. Registers middleware: `RequestContextMiddleware` (outermost),
   `BodySizeLimitMiddleware`, then CORS.
4. Registers the error handlers from
   [`core/errors.py`](../sentiment_analyzer/core/errors.py).
5. Mounts `/healthz` and `/readyz` without auth, and every API router under
   `/api/v1` with the `require_api_key` dependency. The same routers are also
   mounted under `/api` as a deprecated alias that is hidden from the OpenAPI
   schema.
6. Disables `/docs`, `/redoc` and `/openapi.json` when
   `ASA_ENVIRONMENT=prod`.

The `lifespan` handler builds the long-lived objects once per process and
stores them on `app.state`:

| `app.state` attribute | Object                                                                 |
| --------------------- | ---------------------------------------------------------------------- |
| `settings`            | The `Settings` instance.                                               |
| `run_store`           | `RunStore(settings.runs_db_path)` (DuckDB). Closed on shutdown.        |
| `vector_store`        | `VectorStore(settings.vector_store_path)` (persistent Chroma).         |
| `engines`             | `EngineRegistry(settings)`; engines are built lazily on first use.     |
| `service`             | `AnalysisService(settings, run_store, engines, enricher)`. The enricher is `None` when `ASA_RAG_ENABLED=false`. |

If `ASA_WARMUP_DEFAULT_ENGINE=true`, the default engine is built during
startup; a failure is logged and reported by `/readyz` rather than crashing
the process.

Routes get these objects through the dependency functions in
[`api/deps.py`](../sentiment_analyzer/api/deps.py), which tests can override.
`embedding_function` lets tests replace Chroma's default embedder with an
offline stub.

## Module responsibilities

| Layer / package                                 | Responsibility                                                                                   |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [api/](../sentiment_analyzer/api)               | FastAPI routers, dependencies and Pydantic request/response schemas. No business logic.          |
| [services/](../sentiment_analyzer/services)     | `AnalysisService`: ingestion → processing → RAG → engine → analytics → storage.                  |
| [ingestion/](../sentiment_analyzer/ingestion)   | Adapters that turn raw input (records, text, CSV, Excel) into `list[dict]`.                      |
| [processing/](../sentiment_analyzer/processing) | Missing-value normalization, text-field detection, record validation.                            |
| [sentiment/](../sentiment_analyzer/sentiment)   | `SentimentEngine` interface, `SentimentResult`, the three engines and the `EngineRegistry`.      |
| [rag/](../sentiment_analyzer/rag)               | Persistent Chroma vector store, chunking, and context enrichment.                                |
| [analytics/](../sentiment_analyzer/analytics)   | Aggregation, anomaly detection, explanations, time trends, run comparison.                       |
| [evaluation/](../sentiment_analyzer/evaluation) | Classification metrics (accuracy, per-class precision/recall/F1, macro-F1, confusion matrix).   |
| [storage/](../sentiment_analyzer/storage)       | DuckDB run persistence with schema migrations.                                                   |
| [core/](../sentiment_analyzer/core)             | Settings, API-key auth, middleware, error mapping, JSON logging, app version.                    |
| [scripts/](../sentiment_analyzer/scripts)       | `bootstrap_rag_store`: seeds the vector store with starter documents.                            |
| [frontend/](../frontend)                        | Streamlit UI. Talks to the API over HTTP via `api_client.py`.                                    |

## Request lifecycle (`POST /api/v1/analyze`)

Every request first passes through the middleware:

- **`RequestContextMiddleware`** takes a valid incoming `X-Request-ID`
  (printable, at most 128 characters) or generates one, binds it to the log
  context, returns it in the `X-Request-ID` response header, and logs one
  `http_request` line (method, path, status, duration). `/healthz` and
  `/readyz` are not logged.
- **`BodySizeLimitMiddleware`** rejects bodies larger than
  `ASA_MAX_UPLOAD_BYTES` with `413`, using `Content-Length` when present and
  counting streamed chunks otherwise.
- **`require_api_key`** (a router dependency) checks `X-API-Key` against
  `ASA_API_KEYS` with a constant-time comparison. Skipped when no keys are
  configured.

Route handlers are plain `def` functions, so FastAPI runs the blocking
pipeline in its threadpool. The orchestration lives in
[`AnalysisService.run`](../sentiment_analyzer/services/analysis_service.py):

1. **Engine lookup** — `EngineRegistry.get(name)` builds the engine on first
   use. Unknown or unavailable engines raise `EngineUnavailableError` (`503`).
2. **Ingestion** — `JSONAdapter` for `json` records, `TextAdapter` for
   `source_type: "text"` (one record per non-empty line, field `text`). CSV
   and Excel uploads are parsed by their adapters in the route, reading at
   most `ASA_MAX_RECORDS + 1` rows. Inputs over `ASA_MAX_RECORDS` (or
   `ASA_MAX_LLM_RECORDS` for paid engines) raise `413`.
3. **Text-field resolution** — explicit `text_fields` are validated against
   the input (`422` if missing); otherwise
   [`detect_text_fields`](../sentiment_analyzer/processing/field_detector.py)
   picks them. If none are detected, the request fails with `422` listing the
   available fields. A `timestamp_field` absent from every record is also
   `422`.
4. **Normalization and validation** —
   [`normalize_record`](../sentiment_analyzer/processing/normalizer.py) turns
   `None`/NaN/NaT into `""` and other values into stripped strings;
   [`validate_record`](../sentiment_analyzer/processing/validator.py) skips
   records with no text in any of the text fields. Skipped records keep their
   position: `record_id` is always the index in the original input.
5. **Text assembly** — the record's text fields are joined with a space and
   truncated to `ASA_MAX_TEXT_CHARS` (the record is flagged `truncated`).
6. **Optional RAG enrichment** — only when RAG is requested (`use_rag`, or
   `ASA_RAG_ENABLED` by default) **and** the engine has `uses_context = True`
   (currently only `llm`). All texts are retrieved in one batched vector-store
   query.
7. **Inference** — `engine.analyze_batch(texts, contexts)` returns one
   `SentimentResult` or one exception per text. A failing record gets an
   `error` and null sentiment; the rest of the run continues.
8. **Per-field contributions** — for free engines with more than one text
   field, each field is also scored separately. With a single field, its
   contribution is the record score.
9. **Analytics** — [`aggregate`](../sentiment_analyzer/analytics/aggregation.py)
   (summary over successful records),
   [`detect_anomalies`](../sentiment_analyzer/analytics/anomalies.py) (z-score
   > 2), [`explain_record`](../sentiment_analyzer/analytics/explainability.py),
   and, with `timestamp_field`,
   [`aggregate_by_time`](../sentiment_analyzer/analytics/time_aggregation.py).
10. **Persistence** — [`RunStore.save_run`](../sentiment_analyzer/storage/runs.py)
    writes the run (metadata, input SHA-256, config snapshot, counts,
    duration, status) and every per-record result in one transaction.
11. **Response** — `AnalysisResponse` (see
    [data-model.md](data-model.md#analysisresponse)).

## Errors

Services raise domain exceptions from
[`core/errors.py`](../sentiment_analyzer/core/errors.py); a single handler
maps them, along with FastAPI's own HTTP (`401`, `404`, ...) and request-validation
(`422`) errors, to one JSON envelope `{"detail", "code", "request_id"}` with a
specific status (`413`, `415`, `422`, `404`, `409`, `503`, `502`). Any other exception becomes
`500` with `code: "internal_error"` and is logged with its traceback. Routes do
not contain their own `try/except` blocks. See
[api-reference.md](api-reference.md#errors).

## Data contracts

- **API contract** — [`api/schemas.py`](../sentiment_analyzer/api/schemas.py).
  Every JSON route declares a `response_model`, so responses are validated and
  documented in the OpenAPI schema.
- **Engine contract** — [`sentiment/base.py`](../sentiment_analyzer/sentiment/base.py).
  Every engine returns a `SentimentResult(label, polarity, confidence,
  drivers)` with the same semantics. See [engines.md](engines.md).

## Determinism and auditability

- Every run stores the SHA-256 of its input records and a config snapshot
  (engine settings such as model and revision, app version, text fields, RAG
  settings, `max_text_chars`).
- Every per-record result is persisted and can be exported as CSV or JSON.
- `ASA_HF_MODEL_REVISION` pins the transformer model. The LLM engine calls
  the model with `temperature=0`, but provider output is not guaranteed to be
  identical across calls.
- The rule-based engine is fully deterministic.
- Logs are JSON, one object per line, with the request ID on every line
  emitted during a request.

## Deployment constraints

- **Single process.** DuckDB allows one writing process per database file,
  and Chroma's on-disk store should have one writer. Run the API with one
  uvicorn worker (the Docker image does). Within that process, the run store
  uses one connection with a cursor per operation and a write lock, which is
  safe across the threadpool.
- Stop the API before running the RAG bootstrap script against the same
  store.

## Why these choices

- **DuckDB** — no external service, supports analytical queries. See
  [`storage/runs.py`](../sentiment_analyzer/storage/runs.py).
- **ChromaDB** — persistent local vector store with a built-in ONNX
  `all-MiniLM-L6-v2` embedder, so retrieval needs no external API (the
  embedder model is downloaded once on first use).
- **litellm** — provider-agnostic LLM access; switching providers is a config
  change. See [engines.md](engines.md#llm).
- **Pydantic + pydantic-settings** — validates requests, responses and
  configuration with the same library.
- **Streamlit** — keeps the frontend in Python.

## Extension points

| To add...                 | Where                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| A new input format        | Subclass `InputAdapter` in [`ingestion/`](../sentiment_analyzer/ingestion) and call it from a route or `AnalysisService._load_records`. |
| A new sentiment engine    | Subclass `SentimentEngine` in [`sentiment/`](../sentiment_analyzer/sentiment) and register it in [`registry.py`](../sentiment_analyzer/sentiment/registry.py). See [engines.md](engines.md#adding-a-new-engine). |
| A new analytics signal    | Add a function in [`analytics/`](../sentiment_analyzer/analytics), call it from `AnalysisService.run`, and add the field to `AnalysisResponse`. |
| A new evaluation metric   | Extend `EvaluationHarness.report` in [`evaluation/harness.py`](../sentiment_analyzer/evaluation/harness.py) and `EvaluationReport` in the schemas. |
| A schema change in storage| Append a migration to `MIGRATIONS` in [`storage/runs.py`](../sentiment_analyzer/storage/runs.py). Never edit a released one. |
