# SentimentAnalyzer

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![LiteLLM](https://img.shields.io/badge/LiteLLM-Multi--Provider-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple)
![DuckDB](https://img.shields.io/badge/DuckDB-Persistence-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**SentimentAnalyzer** is an input-agnostic, explainable sentiment analysis
service for structured and semi-structured data. Send it JSON records, plain
text, CSV or Excel; it finds the free-text fields, scores every record with
the engine you choose, and returns per-record results with explanations, a
dataset summary, anomalies and an optional trend over time. Every run is
stored so it can be inspected, exported and compared later.

## Features

- Input: JSON records, plain text (one record per line), CSV and Excel
  uploads.
- Automatic text-field detection, or explicit `text_fields`.
- Three engines behind one contract (label, polarity, confidence, drivers):
  - `rule_based` — deterministic lexicon with negation handling.
  - `hf_transformer` — local HuggingFace model (optional `ml` extra).
  - `llm` — any provider supported by [litellm](https://docs.litellm.ai/)
    (OpenAI, Anthropic, Ollama, Groq, ...), with prompt-injection hardening
    and validated JSON output.
- Per-record results with per-field contributions, failures reported per
  record instead of failing the run.
- Dataset summary, z-score anomalies, explanations, and sentiment trends by
  day/week/month/year.
- Optional RAG context from a persistent Chroma store, with an A/B endpoint to
  measure its effect.
- Run history in DuckDB: input fingerprint, config snapshot, CSV/JSON export,
  record-level run comparison.
- Evaluation endpoints: accuracy, per-class precision/recall/F1, macro-F1 and
  confusion matrix, for given predictions or for an engine on labelled texts.
- Background jobs for large inputs, with progress polling and cancellation.
- API-key auth (full and read-only keys), per-client rate limiting, input
  limits, JSON errors with request IDs, JSON logs, `/healthz`, `/readyz` and
  Prometheus `/metrics` (including LLM tokens and estimated cost).
- Streamlit UI for analysis (including background jobs and RAG A/B tests),
  engine evaluation, run history and RAG management.

## Quick start

Prerequisites: Python 3.13+ (the repo pins 3.14 in `.python-version`) and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/saladgg/sentiment-analyzer.git
cd sentiment-analyzer
make install-dev        # all extras (ml, frontend) + dev tools
cp .env.example .env    # adjust as needed
make run                # API on http://localhost:8000
make frontend           # UI on http://localhost:8501 (separate terminal)
```

Try it:

```bash
curl -s -X POST http://localhost:8000/api/v1/analyze \
  -H 'Content-Type: application/json' \
  -d '{"source_type": "text", "data": "Great service!\nThe delivery was late."}'
```

With the default `ASA_ENVIRONMENT=dev` and no `ASA_API_KEYS`, the API is
open. Outside `dev` it requires keys and every request must send
`X-API-Key`; set `ASA_API_KEY` for the frontend to match.

Interactive API docs (disabled in `prod`): `http://localhost:8000/docs`.

To use the LLM engine, set `ASA_LLM_MODEL` (and `ASA_LLM_API_KEY` if your
provider needs one). To seed the RAG store with starter documents, run
`make rag-bootstrap` while the API is stopped.

### Docker

```bash
make docker-up     # API on :8000, UI on :8501; data on the run-data volume
make docker-logs
make docker-down
```

### Tests and checks

```bash
make test          # pytest, fails under 90% coverage
make lint          # ruff + mypy
make format        # ruff format
make audit         # pip-audit on locked dependencies
```

Run `make help` for all targets.

## API at a glance

All routes are under `/api/v1` (the old `/api` prefix still works but is
deprecated).

| Endpoint                                  | Purpose                                         |
| ----------------------------------------- | ----------------------------------------------- |
| `POST /analyze`                           | Analyze JSON records or plain text.             |
| `POST /analyze/csv`, `/analyze/excel`     | Analyze an uploaded file.                       |
| `GET /engines`                            | Engines, their status, and input limits.        |
| `GET /runs`, `GET /runs/{id}`             | Run history and run detail.                     |
| `GET /runs/{id}/export`                   | Download a run as CSV or JSON.                  |
| `GET /runs/compare`                       | Compare two runs of the same input.             |
| `POST /runs/ab`                           | Run the same input without and with RAG.        |
| `POST /evaluation/sentiment`, `/evaluation/engine` | Classification metrics.                |
| `/rag/documents`, `POST /rag/retrieve`    | Manage and query RAG context documents.         |
| `GET /healthz`, `GET /readyz`             | Liveness and readiness (no auth, no prefix).    |

See the [API reference](docs/api-reference.md) for schemas, examples and
error codes.

## Documentation

Detailed docs live in [docs/](docs/README.md):

| Doc                                     | Topic                                                         |
| --------------------------------------- | ------------------------------------------------------------- |
| [Architecture](docs/architecture.md)    | App wiring, request lifecycle, module responsibilities        |
| [Configuration](docs/configuration.md)  | All `ASA_*` settings and defaults                             |
| [Engines](docs/engines.md)              | Engine contract, the three engines, adding your own           |
| [RAG](docs/rag.md)                      | Vector store, chunking, enrichment, measuring RAG impact      |
| [API reference](docs/api-reference.md)  | HTTP API with schemas and `curl` examples                     |
| [Data model](docs/data-model.md)        | Schemas, persistence layout and migrations                    |
| [Frontend](docs/frontend.md)            | Streamlit pages and how they call the backend                 |
| [Development](docs/development.md)      | Setup, testing, lint, Docker, CI, releases                    |

## Project structure

```
sentiment_analyzer/
  main.py           # App factory (create_app) and lifespan
  api/              # Routes, dependencies, Pydantic schemas
  analytics/        # Aggregation, anomalies, explanations, trends, run comparison
  core/             # Settings, auth, middleware, errors, logging
  evaluation/       # Classification metrics
  ingestion/        # Input adapters (CSV, Excel, JSON, text)
  processing/       # Field detection, normalization, validation
  rag/              # Vector store, chunking, context enrichment
  scripts/          # bootstrap_rag_store
  sentiment/        # Engines and engine registry
  services/         # AnalysisService (pipeline orchestration)
  storage/          # DuckDB run store and migrations
frontend/           # Streamlit UI (app.py, api_client.py, pages/)
tests/              # Test suite
assets/             # Sample data files
docs/               # Reference documentation
```
