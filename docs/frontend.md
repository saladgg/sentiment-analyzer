# Streamlit frontend

A small, single-language UI lives in [`frontend/`](../frontend) and is the
primary interactive surface for SentimentAnalyzer. It speaks to the FastAPI
backend over HTTP — there is no shared state between the two processes.

```bash
make frontend                # uv run streamlit run frontend/app.py
# UI: http://localhost:8501
# API expected at: http://localhost:8000/api
```

If you change the backend port, edit `BASE_URL` in
[`frontend/api_client.py`](../frontend/api_client.py).

## Layout

| File                                                        | Role                                                 |
| ----------------------------------------------------------- | ---------------------------------------------------- |
| [frontend/app.py](../frontend/app.py)                       | Page registration, sidebar, navigation.              |
| [frontend/api_client.py](../frontend/api_client.py)         | Thin `requests`-based wrapper around `/api/*`.       |
| [frontend/theme.py](../frontend/theme.py)                   | Color map and engine label/description constants.    |
| [frontend/pages/analyze.py](../frontend/pages/analyze.py)   | Run a sentiment analysis interactively.              |
| [frontend/pages/history.py](../frontend/pages/history.py)   | Browse and compare past runs.                        |
| [frontend/pages/rag.py](../frontend/pages/rag.py)           | Manage RAG documents and preview retrieval.          |

## Navigation

`app.py` registers three pages with `st.navigation`:

```python
analyze_page = st.Page("pages/analyze.py", title="Analyze", default=True)
history_page = st.Page("pages/history.py", title="Run History")
rag_page     = st.Page("pages/rag.py", title="RAG Management")
```

The grouping is "Analysis" (just Analyze) and "Platform" (history + RAG).
`st.set_page_config` sets a wide layout and an open sidebar — that call
happens before any other Streamlit call, which is required by Streamlit.

## Pages

### Analyze (`pages/analyze.py`)

The primary workflow page. Lets the user:

- Pick an engine via a `selectbox` driven by
  `theme.ENGINE_LABELS` / `ENGINE_DESCRIPTIONS`.
- Choose between file upload (CSV / Excel) and pasted JSON.
- Run the analysis and visualize the response with `plotly.express` and
  `plotly.graph_objects` — distribution charts, per-record scores, and
  anomaly callouts.

Backend calls used:

- `api_client.analyze_csv(file, engine)` → `POST /api/analyze/csv`
- `api_client.analyze_excel(file, engine)` → `POST /api/analyze/excel`
- `api_client.analyze_json(data, engine)` → `POST /api/analyze`

### Run History (`pages/history.py`)

Lists past runs and supports side-by-side comparison of two runs (for
example, baseline vs. RAG-enriched). The comparison surface is intentionally
minimal — confidence gain, label changes, and a `rag_helpfulness` flag —
mirroring what the `/api/runs/compare` endpoint returns.

Backend calls used:

- `api_client.list_runs()` → `GET /api/runs`
- `api_client.compare_runs(base_run_id, rag_run_id)` → `GET /api/runs/compare`

### RAG Management (`pages/rag.py`)

Handles three RAG operations:

- Upload context documents to a chosen namespace.
- Preview semantic retrieval for a query, with similarity scores.
- List documents and delete them.

Backend calls used:

- `api_client.upload_rag_document(file, namespace)` → `POST /api/rag/documents`
- `api_client.list_rag_documents()` → `GET /api/rag/documents`
- `api_client.delete_rag_document(document_id)` → `DELETE /api/rag/documents/{id}`
- `api_client.rag_retrieve(query, top_k, namespace)` → `POST /api/rag/retrieve`

## Theming

`frontend/theme.py` is the single place to change look-and-feel constants:

```python
SENTIMENT_COLORS = {
    "positive": "#10b981",  # emerald
    "neutral":  "#6366f1",  # indigo
    "negative": "#ef4444",  # red
}

ENGINE_LABELS = {
    "rule_based":    "Rule-Based (Lexical)",
    "hf_transformer":"HuggingFace Transformer",
    "llm":           "LLM (Claude / OpenAI / Ollama)",
}
```

When you add a new engine (see [engines.md](engines.md#adding-a-new-engine)),
update both `ENGINE_LABELS` and `ENGINE_DESCRIPTIONS` so it is selectable
from the UI.

## HTTP client conventions

All HTTP work is centralized in
[`frontend/api_client.py`](../frontend/api_client.py):

- `BASE_URL = "http://localhost:8000/api"` — change this for non-local
  deployments.
- `_TIMEOUT = 120` — accommodates the slower LLM engine.
- Each function `raise_for_status()`s and returns parsed JSON; pages do not
  handle `requests` directly.

## CORS

The backend currently allows the Streamlit origin via:

```python
allow_origins=["http://localhost:5173", "http://localhost:8501"]
```

(See [`main.py`](../sentiment_analyzer/main.py).) Add additional origins
there for non-local Streamlit deployments.
