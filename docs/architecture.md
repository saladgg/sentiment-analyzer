# Architecture

SentimentAnalyzer is a layered Python service that turns heterogeneous
tabular input (CSV / Excel / JSON) into explainable, persisted sentiment
results. This document describes the layers, what each is responsible for,
and how a request flows through them.

## High-level layout

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit frontend  (frontend/)                             │
│  - Analyze, Run History, RAG Management pages                │
└──────────────────────────────────────────────────────────────┘
                          │ HTTP (requests)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI app  (sentiment_analyzer/main.py)                   │
│  - CORS, routers under /api                                  │
└──────────────────────────────────────────────────────────────┘
            │ analysis  │ runs  │ rag  │ evaluation
            ▼           ▼       ▼      ▼
┌──────────────────────────────────────────────────────────────┐
│  Service layer (services/)                                   │
│  AnalysisService.run() — single orchestration entry point    │
└──────────────────────────────────────────────────────────────┘
   │           │            │             │            │
   ▼           ▼            ▼             ▼            ▼
ingestion/  processing/  rag/        sentiment/   analytics/
                                                       │
                                                       ▼
                                                  storage/
                                                  (DuckDB)
```

## Module responsibilities

| Layer / package          | Responsibility                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| [api/](../sentiment_analyzer/api)             | FastAPI routers and Pydantic request/response schemas. No business logic. |
| [services/](../sentiment_analyzer/services)   | Orchestration: wires ingestion → processing → engines → analytics → storage. |
| [ingestion/](../sentiment_analyzer/ingestion) | Adapters that turn raw input (file / list / string) into `list[dict]`.        |
| [processing/](../sentiment_analyzer/processing) | Field detection, normalization, validation. Schema-agnostic.               |
| [sentiment/](../sentiment_analyzer/sentiment) | Pluggable inference engines, all implementing the same `SentimentEngine` ABC.  |
| [rag/](../sentiment_analyzer/rag)             | Optional vector-store-backed context retrieval (Chroma + sentence-transformers). |
| [analytics/](../sentiment_analyzer/analytics) | Aggregation, anomaly detection, explainability, per-record shaping, run comparison. |
| [evaluation/](../sentiment_analyzer/evaluation) | Offline harness for accuracy / drift / stability metrics.                    |
| [storage/](../sentiment_analyzer/storage)     | DuckDB-backed run persistence.                                                 |
| [core/](../sentiment_analyzer/core)           | `Settings` (pydantic-settings) and structured `logger` (structlog).            |
| [scripts/](../sentiment_analyzer/scripts)     | One-off utilities (e.g. seeding the RAG store).                                |
| [frontend/](../frontend)                      | Streamlit UI. Talks to the API via `requests` (`api_client.py`).               |

## Request lifecycle (`POST /api/analyze`)

The orchestration logic lives entirely in
[`AnalysisService.run`](../sentiment_analyzer/services/analysis_service.py).
A request flows through the following stages:

1. **Route handler** — [`api/routes.py`](../sentiment_analyzer/api/routes.py)
   parses the request body into `AnalysisRequest` and instantiates
   `AnalysisService`. CSV/Excel uploads are first decoded by their respective
   adapters before being passed in.
2. **Run ID assigned** — a fresh UUID is generated and threaded through logs.
3. **Ingestion** — `JSONAdapter.load(data)` returns a `list[dict]`. (CSV/Excel
   adapters do this earlier in the route handler.)
4. **Field detection** —
   [`detect_text_fields`](../sentiment_analyzer/processing/field_detector.py)
   inspects the first record and returns string fields with length > 10 as
   text-bearing candidates.
5. **Per-record loop**:
   - [`normalize_record`](../sentiment_analyzer/processing/normalizer.py) —
     coerces values to stripped strings, replaces `None` with `""`.
   - [`validate_record`](../sentiment_analyzer/processing/validator.py) —
     drops records with no non-empty values.
   - **Text assembly** — text-bearing fields are joined with a single space.
   - **Optional RAG enrichment** —
     [`RAGContextEnricher.enrich`](../sentiment_analyzer/rag/context_enrichment.py)
     queries Chroma for the top-K similar chunks. Skipped when
     `ASA_RAG_ENABLED=false`.
   - **Inference** — the chosen `SentimentEngine` returns
     `{label, score, drivers}`. RAG context is passed in for engines that use
     it (currently only the LLM engine).
   - **Per-record shaping** —
     [`build_per_record_result`](../sentiment_analyzer/analytics/per_record.py)
     produces the canonical record schema; it is then validated through
     `RecordSentiment`.
6. **Dataset-level analytics**:
   - [`aggregate`](../sentiment_analyzer/analytics/aggregation.py) — total
     count, label distribution, dominant label.
   - [`detect_anomalies`](../sentiment_analyzer/analytics/anomalies.py) —
     z-score outliers (threshold = 2.0).
   - [`explain_record`](../sentiment_analyzer/analytics/explainability.py) —
     deterministic per-record one-liner.
7. **Persistence** —
   [`RunStore.save_run`](../sentiment_analyzer/storage/runs.py) writes
   `(run_id, engine, summary, rag_enabled, created_at)` to DuckDB.
8. **Response** — `AnalysisResponse` is returned, including a `rag` metadata
   block when RAG is enabled.

## Data contracts

The two boundaries that matter most are the API contract and the engine
contract. Both are enforced at runtime.

- **API contract** — defined in
  [`api/schemas.py`](../sentiment_analyzer/api/schemas.py). Outputs flow
  through `AnalysisResponse`/`RecordSentiment` so the response is always
  shaped consistently.
- **Engine contract** — defined in
  [`sentiment/base.py`](../sentiment_analyzer/sentiment/base.py). Every engine
  must return `{"label", "score", "drivers"}`. See
  [engines.md](engines.md) for details and how to add a new one.

## Determinism and auditability

The platform is designed so the same input + config produces the same output.
This is enforced by:

- Fixed model versions (`ASA_HF_SENTIMENT_MODEL`, `ASA_LLM_MODEL`).
- A deterministic rule-based baseline engine for audits.
- Persisting every run (id, engine, summary, RAG flag, timestamp) in DuckDB.
- Structured JSON logs via
  [`core/logging.py`](../sentiment_analyzer/core/logging.py).
- Optional `RAGMetadata` on every response (`enabled`, `used`,
  `source_count`) so RAG influence is observable.

## Why these choices

- **DuckDB** — zero-ops, supports analytical queries, perfect for local /
  prototype deployments. See
  [`storage/runs.py`](../sentiment_analyzer/storage/runs.py).
- **ChromaDB + sentence-transformers** — embeddings happen locally; no
  third-party API call is required just to retrieve context.
- **litellm** — provider-agnostic LLM access. Switching from Anthropic to
  OpenAI or a local Ollama model is a one-line config change. See
  [engines.md](engines.md#llm-engine).
- **Pydantic + pydantic-settings** — validates both incoming requests and
  environment-driven configuration with the same library.
- **Streamlit** — keeps the frontend single-language so the same engineer can
  iterate on the UI and the API without context switching.

## Extension points

| To add…                  | Where                                                   |
| ------------------------ | ------------------------------------------------------- |
| A new input format       | Create `IngestionAdapter` subclass under [`ingestion/`](../sentiment_analyzer/ingestion). |
| A new sentiment engine   | Subclass `SentimentEngine` under [`sentiment/`](../sentiment_analyzer/sentiment); register it in `AnalysisService.__init__`. |
| A new analytics signal   | Add a function under [`analytics/`](../sentiment_analyzer/analytics) and call it from `AnalysisService.run`. |
| A new evaluation metric  | Add a method on `EvaluationHarness` in [`evaluation/harness.py`](../sentiment_analyzer/evaluation/harness.py). |
| A new persistence backend| Implement a new store with the same interface as `RunStore` and swap it in `AnalysisService.__init__`. |
