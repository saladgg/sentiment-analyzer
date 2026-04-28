# SentimentAnalyzer

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![LiteLLM](https://img.shields.io/badge/LiteLLM-Multi--Provider-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple)
![DuckDB](https://img.shields.io/badge/DuckDB-Persistence-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
[![Coverage Status](https://coveralls.io/repos/github/saladgg/sentiment-analyzer/badge.svg)](https://coveralls.io/github/saladgg/sentiment-analyzer)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

**SentimentAnalyzer** is an input-agnostic, explainable sentiment analysis platform designed for structured and semi-structured datasets.

Unlike traditional sentiment APIs that return raw labels, this system focuses on:

- Transparency
- Aggregation
- Evaluation
- Deterministic re-runs

It is built as a production-oriented platform suitable for batch analysis, experimentation, and interactive exploration via its Streamlit frontend.

---

## Key Features

- Input-agnostic ingestion (JSON, CSV, Excel)
- Automatic detection of text-bearing fields
- Pluggable sentiment engines:
  - Rule-based
  - Transformer-based (HuggingFace)
  - LLM-backed (any provider via [litellm](https://docs.litellm.ai/) — OpenAI, Anthropic, Ollama, Groq, etc.)
- Explainable per-record sentiment outputs
- Dataset-level aggregation and summaries
- Optional RAG-based sentiment context enrichment
- Offline evaluation harness (accuracy, drift-ready)
- Deterministic, auditable runs
- Streamlit frontend with interactive analysis, run history, and RAG management

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/saladgg/sentiment-analyzer.git
cd sentiment-analyzer

# Install dependencies
uv sync
```

### Configuration

Copy the example environment file and adjust values as needed:

```bash
cp .env.example .env
```

### Running the Server

```bash
make run
```

The API will be available at `http://localhost:8000`.

### Running the Frontend

```bash
make frontend
```

The Streamlit UI will be available at `http://localhost:8501`.

The frontend provides three pages:

- **Analyze** — Upload CSV/Excel or paste JSON, pick a sentiment engine, and view interactive results with charts, per-record scores, anomalies, and explanations.
- **Run History** — Browse past analysis runs and compare two runs side-by-side (e.g. base vs. RAG-enriched).
- **RAG Management** — Upload context documents, manage the vector store, and preview semantic retrieval results.

### Running Tests

```bash
make test
```

### Code Quality

```bash
make lint          # ruff + mypy
make format        # auto-format
make fix           # auto-fix lint issues
```

### Docker

```bash
make docker-up     # build and start
make docker-down   # stop
make docker-logs   # tail logs
```

Run `make help` to see all available targets.

---

## API Documentation

FastAPI provides built-in interactive API documentation:

| Docs UI    | URL                              |
| ---------- | -------------------------------- |
| Swagger UI | `http://localhost:8000/docs`     |
| ReDoc      | `http://localhost:8000/redoc`    |
| OpenAPI    | `http://localhost:8000/openapi.json` |

All application endpoints are mounted under the `/api` prefix.

---

### Analysis Endpoints

#### `POST /api/analyze`

Run sentiment analysis on JSON or text data.

**Request body:**

```json
{
  "data": [
    {"comment": "I love this product, it works great"},
    {"comment": "Terrible experience, very disappointing"}
  ],
  "source_type": "json",
  "sentiment_engine": "rule_based"
}
```

| Field              | Type     | Default        | Description                                      |
| ------------------ | -------- | -------------- | ------------------------------------------------ |
| `data`             | any      | required       | List of dicts (JSON rows) or a plain text string |
| `source_type`      | string   | required       | One of `json`, `csv`, `excel`, `text`            |
| `sentiment_engine` | string   | `"rule_based"` | One of `rule_based`, `hf_transformer`, `llm`     |

**Response:**

```json
{
  "run_id": "abc123",
  "summary": { ... },
  "records": [
    {
      "record_id": 0,
      "sentiment": "positive",
      "score": 0.85,
      "contributing_fields": {"lexical_rules": 1.0, "positive_hits": 1, "negative_hits": 0}
    }
  ],
  "anomalies": [],
  "explanations": [],
  "rag": null
}
```

---

#### `POST /api/analyze/csv`

Upload a CSV file for sentiment analysis.

| Query Param        | Type   | Default        | Description            |
| ------------------ | ------ | -------------- | ---------------------- |
| `sentiment_engine` | string | `"rule_based"` | Sentiment engine to use |

**Request:** `multipart/form-data` with a `file` field containing a `.csv` file.

**Response:** Same as `POST /api/analyze`.

---

#### `POST /api/analyze/excel`

Upload an Excel file for sentiment analysis.

| Query Param        | Type   | Default        | Description            |
| ------------------ | ------ | -------------- | ---------------------- |
| `sentiment_engine` | string | `"rule_based"` | Sentiment engine to use |

**Request:** `multipart/form-data` with a `file` field containing a `.xlsx` file.

**Response:** Same as `POST /api/analyze`.

---

#### `GET /api/runs`

List all past analysis runs.

**Response:** Array of run objects.

---

### Run Comparison Endpoints

#### `GET /api/runs/compare`

Compare two analysis runs (typically a base run vs. a RAG-enriched run).

| Query Param  | Type   | Required | Description                |
| ------------ | ------ | -------- | -------------------------- |
| `base_run_id`| string | yes      | ID of the base run         |
| `rag_run_id` | string | yes      | ID of the RAG-enriched run |

**Response:**

```json
{
  "base_run_id": "abc123",
  "rag_run_id": "def456",
  "comparison": { ... }
}
```

---

### Evaluation Endpoints

#### `POST /api/evaluation/sentiment`

Evaluate sentiment predictions against ground-truth labels.

| Query Param   | Type       | Required | Description               |
| ------------- | ---------- | -------- | ------------------------- |
| `predictions` | list[str]  | yes      | Predicted sentiment labels |
| `labels`      | list[str]  | yes      | Ground-truth labels        |

**Response:**

```json
{
  "metric": "accuracy",
  "value": 0.667,
  "total_samples": 3
}
```

---

### RAG Endpoints

#### `POST /api/rag/documents`

Upload a document to the RAG vector store.

| Query Param | Type   | Default     | Description              |
| ----------- | ------ | ----------- | ------------------------ |
| `namespace` | string | `"default"` | Namespace for the document |

**Request:** `multipart/form-data` with a `file` field containing a text file.

**Response:**

```json
{
  "document_id": "doc_abc123",
  "filename": "context.txt",
  "namespace": "default",
  "chunks_added": 5
}
```

---

#### `GET /api/rag/documents`

List all uploaded RAG documents.

**Response:** Array of document objects.

---

#### `DELETE /api/rag/documents/{document_id}`

Delete a document from the RAG store.

**Response:**

```json
{
  "status": "deleted",
  "document_id": "doc_abc123"
}
```

---

#### `POST /api/rag/retrieve`

Retrieve relevant context chunks via semantic search.

**Request body:**

```json
{
  "query": "product quality issues",
  "top_k": 5,
  "namespace": "default"
}
```

| Field       | Type   | Default     | Description                     |
| ----------- | ------ | ----------- | ------------------------------- |
| `query`     | string | required    | Search query                    |
| `top_k`     | int    | optional    | Number of results to return     |
| `namespace` | string | optional    | Namespace to search within      |

**Response:**

```json
[
  {
    "content": "chunk text...",
    "source": "context.txt",
    "document_id": "doc_abc123",
    "namespace": "default",
    "score": 0.92
  }
]
```

---

## CI

GitHub Actions runs lint, type checking, and tests on every push to `main` and on pull requests. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Project Structure

```
sentiment_analyzer/
  api/              # FastAPI route handlers and schemas
  analytics/        # Aggregation, anomaly detection, explainability
  core/             # Configuration and logging
  evaluation/       # Evaluation harness (accuracy metrics)
  ingestion/        # Input adapters (CSV, Excel, JSON, text)
  processing/       # Field detection, normalization, validation
  rag/              # RAG vector store and context enrichment
  scripts/          # Utility scripts (e.g., bootstrap RAG store)
  sentiment/        # Sentiment engines (rule-based, transformer, LLM)
  services/         # Orchestration and workflow logic
  storage/          # Run persistence
tests/              # Test suite
assets/             # Sample data files
frontend/
  app.py            # Streamlit entry point
  api_client.py     # HTTP client for the FastAPI backend
  theme.py          # Shared constants (colors, engine labels)
  pages/            # Streamlit pages (analyze, history, rag)
```
