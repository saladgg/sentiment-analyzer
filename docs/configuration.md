# Configuration

All runtime configuration is centralized in
[`core/config.py`](../sentiment_analyzer/core/config.py) using
[pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
Values are read from a `.env` file in the working directory and from process
environment variables. Environment variables win over `.env`.

## The `ASA_` prefix

All variables are prefixed with `ASA_` (AllSentimentAnalyzer):

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_by_name=True,
    )
```

The field `default_sentiment_engine` is therefore set with
`ASA_DEFAULT_SENTIMENT_ENGINE`. Unknown `ASA_*` keys in `.env` (for example
settings that have since been removed) are ignored rather than failing startup.

List settings (`ASA_API_KEYS`, `ASA_CORS_ORIGINS`) are comma-separated
strings: `ASA_API_KEYS=key-one,key-two`.

## Workflow

```bash
cp .env.example .env
# edit .env
make run
```

[`.env.example`](../.env.example) lists the commonly changed variables. Keep it
and the tables below in sync with `Settings` when you add a setting.

## Startup validation

`create_app` calls `Settings.validate_for_startup()` before building the app:

- When `ASA_ENVIRONMENT` is `staging` or `prod` and `ASA_API_KEYS` is empty,
  the process refuses to start (`RuntimeError`).
- Field constraints (ranges, allowed values) are enforced by pydantic when
  `Settings` is constructed; an invalid value stops startup with a validation
  error.

## Reference

### Application

| Variable                    | Default             | Description                                                                                     |
| --------------------------- | ------------------- | ----------------------------------------------------------------------------------------------- |
| `ASA_APP_NAME`              | `SentimentAnalyzer` | API title shown in the OpenAPI schema and `/docs`.                                              |
| `ASA_ENVIRONMENT`           | `dev`               | `dev`, `staging` or `prod`. Outside `dev`, API keys are required. In `prod`, `/docs`, `/redoc` and `/openapi.json` are disabled. |
| `ASA_LOG_LEVEL`             | `INFO`              | `DEBUG`, `INFO`, `WARNING` or `ERROR`. Applies to the app's and all stdlib loggers.             |
| `ASA_WARMUP_DEFAULT_ENGINE` | `true`              | Build the default engine at startup so the first request is not slow. A failure is logged, not fatal; `/readyz` reports it. |

### HTTP and security

| Variable           | Default                                         | Description                                                                                       |
| ------------------ | ----------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `ASA_API_KEYS`     | empty                                           | Comma-separated keys accepted in the `X-API-Key` header. Empty disables auth (allowed only in `dev`). |
| `ASA_CORS_ORIGINS` | `http://localhost:5173,http://localhost:8501`   | Comma-separated browser origins allowed by CORS.                                                  |

### Input limits

| Variable               | Default            | Description                                                                                         |
| ---------------------- | ------------------ | --------------------------------------------------------------------------------------------------- |
| `ASA_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Maximum request body and uploaded file size. Larger requests get `413`.                            |
| `ASA_MAX_RECORDS`      | `10000`            | Maximum records per run (and items per `/evaluation/engine` call for free engines). Larger inputs get `413`. |
| `ASA_MAX_TEXT_CHARS`   | `5000`             | Text longer than this is truncated before analysis and the record is flagged `truncated: true`.    |
| `ASA_MAX_LLM_RECORDS`  | `500`              | Maximum records per run for paid engines (`llm`). Larger inputs get `413`.                         |

All four must be greater than zero.

### Sentiment engines

| Variable                       | Default                                           | Description                                                                                  |
| ------------------------------ | ------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `ASA_DEFAULT_SENTIMENT_ENGINE` | `rule_based`                                      | `rule_based`, `hf_transformer` or `llm`. Used when a request omits `sentiment_engine`.       |
| `ASA_HF_SENTIMENT_MODEL`       | `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace model id for the `hf_transformer` engine.                                        |
| `ASA_HF_MODEL_REVISION`        | unset                                             | Model revision (commit hash or tag) to pin. Blank means the latest revision.                 |
| `ASA_HF_BATCH_SIZE`            | `16`                                              | Batch size passed to the transformers pipeline.                                              |
| `ASA_HF_NEUTRAL_BAND`          | `0.0`                                             | `0 <= x < 0.5`. For binary models, predictions with `p(positive)` within `0.5 ± x` are labelled `neutral`. |

### LLM engine (litellm)

| Variable              | Default | Description                                                                                   |
| --------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `ASA_LLM_MODEL`       | empty   | Any litellm model id (`gpt-4o`, `anthropic/claude-sonnet-4-5`, `ollama/llama3`, ...). There is no default: the `llm` engine is unavailable (`503`) until this is set. |
| `ASA_LLM_API_KEY`     | empty   | Provider API key passed to litellm. Leave empty to use the provider's own environment variable, or for local models. |
| `ASA_LLM_TIMEOUT`     | `60`    | Seconds per completion call.                                                                  |
| `ASA_LLM_NUM_RETRIES` | `2`     | Retries per completion call (litellm).                                                        |
| `ASA_LLM_MAX_TOKENS`  | `512`   | `max_tokens` per completion call.                                                             |
| `ASA_LLM_CONCURRENCY` | `8`     | Maximum parallel completion calls within one batch.                                           |

See [engines.md](engines.md) for engine behaviour.

### RAG

| Variable                | Default          | Description                                                                                       |
| ----------------------- | ---------------- | ------------------------------------------------------------------------------------------------- |
| `ASA_RAG_ENABLED`       | `true`           | When `false`, no enricher is built, `use_rag: true` requests get `422`, and `AnalysisResponse.rag` is `null`. |
| `ASA_RAG_TOP_K`         | `5`              | Chunks retrieved per record, and the default for `POST /rag/retrieve`. Range 1-50.               |
| `ASA_RAG_NAMESPACE`     | `default`        | Namespace used by enrichment, the default for `POST /rag/retrieve`, and the bootstrap script's default. |
| `ASA_VECTOR_STORE_PATH` | `./vector_store` | Directory of the persistent Chroma store. Used by both the API and the bootstrap script.          |
| `ASA_RAG_CHUNK_SIZE`    | `500`            | Maximum characters per chunk when a document is uploaded.                                         |
| `ASA_RAG_CHUNK_OVERLAP` | `50`             | Characters of overlap between consecutive chunks. Must be smaller than `ASA_RAG_CHUNK_SIZE`.     |

See [rag.md](rag.md).

### Storage

| Variable           | Default       | Description                                                                                  |
| ------------------ | ------------- | -------------------------------------------------------------------------------------------- |
| `ASA_RUNS_DB_PATH` | `runs.duckdb` | DuckDB file holding run history. Its parent directory is created if missing.                 |
| `ASA_RAG_DB_PATH`  | -             | Deprecated alias for `ASA_RUNS_DB_PATH`. Still accepted; a `DeprecationWarning` is emitted when it is set without `ASA_RUNS_DB_PATH`. |

### Frontend

These are read by the Streamlit client ([`frontend/api_client.py`](../frontend/api_client.py)),
not by `Settings`. They are plain process environment variables; the frontend
does not read `.env`.

| Variable      | Default                        | Description                                                       |
| ------------- | ------------------------------ | ----------------------------------------------------------------- |
| `ASA_API_URL` | `http://localhost:8000/api/v1` | Base URL of the API.                                              |
| `ASA_API_KEY` | empty                          | Sent as `X-API-Key`. Must be one of the API's `ASA_API_KEYS` when auth is enabled. |

### Removed settings

`ASA_ENABLE_EVALUATION` and `ASA_EVALUATION_SAMPLE_RATE` no longer exist. If
they remain in an old `.env` they are ignored.

## Docker

The images set `ASA_RUNS_DB_PATH=/app/data/runs.duckdb` and
`ASA_VECTOR_STORE_PATH=/app/data/vector_store`, and
[`docker-compose.yml`](../docker-compose.yml) sets them again so they take
precedence over relative paths in `.env`. See
[development.md](development.md#docker).

## Reading configuration in code

```python
from sentiment_analyzer.core.config import get_settings

settings = get_settings()
settings.rag_top_k                 # 5
settings.default_sentiment_engine  # "rule_based"
```

Inside request handlers, use the `get_settings` dependency from
[`api/deps.py`](../sentiment_analyzer/api/deps.py), which returns the
`Settings` the app was built with (`app.state.settings`). Tests build their
own `Settings` and pass them to `create_app(settings)`.

Read configuration through `Settings` rather than `os.environ`, so values are
validated in one place.

## Adding a new setting

1. Add a field to `Settings` in [`core/config.py`](../sentiment_analyzer/core/config.py),
   with constraints (`Field(gt=0)`, `Literal[...]`) where they apply.
2. Add the `ASA_*` line to [`.env.example`](../.env.example) with a comment.
3. Update the tables above.
4. For secrets, use `SecretStr` with an empty default, and never log the value.
