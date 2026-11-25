# Streamlit frontend

A small Python UI lives in [`frontend/`](../frontend). It talks to the
FastAPI backend over HTTP only; there is no shared state between the two
processes. It needs the `frontend` extra (`streamlit`, `plotly`, `requests`),
which `make install` and `make install-dev` include.

```bash
make frontend                # uv run streamlit run frontend/app.py
# UI:  http://localhost:8501
# API: $ASA_API_URL (default http://localhost:8000/api/v1)
```

## Configuration

The client reads two process environment variables (it does not read
`.env`):

| Variable      | Default                        | Description                                                        |
| ------------- | ------------------------------ | ------------------------------------------------------------------ |
| `ASA_API_URL` | `http://localhost:8000/api/v1` | Base URL of the API, including the `/api/v1` prefix.              |
| `ASA_API_KEY` | empty                          | Sent as `X-API-Key` on every request. Must be one of the API's `ASA_API_KEYS` when auth is enabled. |

```bash
ASA_API_URL=http://localhost:8000/api/v1 ASA_API_KEY=key-one make frontend
```

In Docker Compose the frontend container gets `ASA_API_URL=http://api:8000/api/v1`
and `ASA_API_KEY` from the shell environment (see
[development.md](development.md#docker)).

The API's CORS settings (`ASA_CORS_ORIGINS`) do not affect the Streamlit UI,
because requests are made server-side by the Streamlit process, not by the
browser.

## Layout

| File                                                        | Role                                                         |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| [frontend/app.py](../frontend/app.py)                       | Page config, navigation, sidebar.                            |
| [frontend/api_client.py](../frontend/api_client.py)         | Thin `requests` wrapper around the API.                      |
| [frontend/theme.py](../frontend/theme.py)                   | Sentiment colours and fallback engine labels/descriptions.   |
| [frontend/pages/analyze.py](../frontend/pages/analyze.py)   | Run an analysis and explore the result.                      |
| [frontend/pages/history.py](../frontend/pages/history.py)   | Browse, inspect, export and compare past runs.               |
| [frontend/pages/rag.py](../frontend/pages/rag.py)           | Manage RAG documents and preview retrieval.                  |

## Navigation

`app.py` calls `st.set_page_config` first (wide layout, expanded sidebar),
then registers four pages with `st.navigation`, grouped as "Analysis"
(Analyze, Evaluate) and "Platform" (Run History, RAG Management).

## Pages

### Analyze (`pages/analyze.py`)

- **Engine selector** — populated from `GET /engines` (cached for 60 s), with
  the server's default engine preselected. Engines that are not configured
  are marked "(not configured)". If the API is unreachable it falls back to
  the labels in `theme.py`.
- **Limits** — shows the record, file-size and text-length limits returned by
  `GET /engines`.
- **Input** — three methods: file upload (CSV / Excel), pasted JSON array,
  or plain text (one record per line).
- **Options** — comma-separated text fields (empty = auto-detect), a
  timestamp field, the trend granularity, and RAG context (server default /
  on / off; enabled only for engines that use context). Text fields and
  timestamp are disabled for plain text.
- **Mode** — *Run now* (`POST /analyze`), *Background job*
  (`POST /jobs/analyze...`; the page polls the job every second with a
  progress bar, then loads the run, showing up to 1,000 records), or
  *RAG A/B test* (`POST /runs/ab`; pasted JSON/text with the `llm` engine
  only). The A/B result shows label changes and score/confidence deltas above
  the RAG run's results.
- **Results** — counts of analyzed / positive / neutral / negative / failed
  records, dominant label, average score and confidence; RAG metadata when
  present; LLM usage (requests, cached, tokens, estimated cost) for paid
  engines; run id, engine and text fields used; CSV and JSON download
  buttons; a list of failed records with their errors.
- **Charts** — label distribution donut, score histogram, sentiment over time
  (when a timestamp field was given, with a note on unparseable timestamps),
  per-field scores (when more than one text field was analyzed), anomalies,
  and per-record explanations.

### Evaluate (`pages/evaluate.py`)

Pick an engine and upload a CSV with `text` and `label` columns (labels
`positive` / `neutral` / `negative`), or use the built-in six-example sample.
Calls `POST /evaluation/engine` and shows accuracy, macro-F1, evaluated and
failed counts, the per-class precision/recall/F1 table, and a confusion
matrix heatmap. Paid engines show a billing warning. Evaluations are not
saved.

### Run History (`pages/history.py`)

- **Background jobs** — the 20 most recent jobs with status, progress and
  run id; queued jobs can be cancelled.
- **Run list** — paginated, 25 runs per page, with Newer/Older buttons and
  the total count. Columns: run id, engine, status, records in, RAG flag,
  created at.
- **Run detail** — summary metrics, failed-record count, distribution chart,
  up to 200 per-record results, the configuration snapshot (config, source
  type, input hash, duration), and CSV/JSON downloads. Runs that predate
  per-record storage show only their summary.
- **Compare runs** — pick two runs of the same input and see records
  compared, label changes and rate, mean score and confidence deltas, and the
  changed records. The API's `409` for different inputs is shown as an error.

### RAG Management (`pages/rag.py`)

- **Upload** — `.txt`, `.md`, `.csv` or `.json` files (UTF-8) to a namespace.
- **Stored documents** — lists documents, optionally filtered by namespace;
  select one to delete it.
- **Semantic search preview** — query, top-k (1-20) and namespace; shows each
  hit's relevance score, source and first 500 characters.

## Backend calls

All HTTP calls go through [`frontend/api_client.py`](../frontend/api_client.py).
Paths are relative to `ASA_API_URL`.

| Function                                          | Endpoint                          |
| ------------------------------------------------- | --------------------------------- |
| `list_engines()`                                  | `GET /engines`                    |
| `analyze_json(data, engine, source_type=..., text_fields=..., timestamp_field=..., granularity=...)` | `POST /analyze` |
| `analyze_csv(file, engine, **options)`            | `POST /analyze/csv`               |
| `analyze_excel(file, engine, **options)`          | `POST /analyze/excel`             |
| `list_runs(limit, offset)`                        | `GET /runs`                       |
| `get_run(run_id, records_limit, records_offset)`  | `GET /runs/{run_id}`              |
| `export_run(run_id, fmt)`                         | `GET /runs/{run_id}/export`       |
| `compare_runs(base_run_id, rag_run_id)`           | `GET /runs/compare`               |
| `upload_rag_document(file, namespace)`            | `POST /rag/documents`             |
| `list_rag_documents()`                            | `GET /rag/documents`              |
| `delete_rag_document(document_id)`                | `DELETE /rag/documents/{id}`      |
| `rag_retrieve(query, top_k, namespace)`           | `POST /rag/retrieve`              |

Conventions:

- Every request sends `X-API-Key` when `ASA_API_KEY` is set.
- Timeout is 300 seconds, to allow for LLM runs over large files.
- Non-2xx responses raise `APIError(status_code, detail, request_id)`. The
  message is the API's `detail`; FastAPI validation errors are flattened to
  `field: message` pairs, and the `X-Request-ID` is appended so a failure can
  be matched to the server logs. A connection failure raises `APIError` with
  status `0`.
- Pages catch `APIError` and show it with `st.error`; they never use
  `requests` directly.

## Theming

[`frontend/theme.py`](../frontend/theme.py) holds `SENTIMENT_COLORS` (used by
every chart and table) and `ENGINE_LABELS` / `ENGINE_DESCRIPTIONS`. The engine
maps are only a fallback for when `GET /engines` fails; a new engine shows up
in the selector without changing them (see
[engines.md](engines.md#adding-a-new-engine)).
