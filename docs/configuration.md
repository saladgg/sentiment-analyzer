# Configuration

All runtime configuration is centralized in
[`core/config.py`](../sentiment_analyzer/core/config.py) using
[pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
Values are read from a `.env` file at the repo root and from process
environment variables. Environment variables always win.

## The `ASA_` prefix

All variables are prefixed with `ASA_` (AllSentimentAnalyzer) to keep them
namespaced and easy to grep:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASA_",
        env_file=".env",
        env_file_encoding="utf-8",
    )
```

This means the field `default_sentiment_engine` is set via the env var
`ASA_DEFAULT_SENTIMENT_ENGINE`.

## Workflow

```bash
cp .env.example .env
# edit .env to taste
make run
```

`.env.example` is the source of truth for the variable list and defaults — keep
it up to date when you add a new setting.

## Reference

### Application

| Variable          | Default            | Description                          |
| ----------------- | ------------------ | ------------------------------------ |
| `ASA_APP_NAME`    | `SentimentAnalyzer`| Display name (cosmetic).             |
| `ASA_ENVIRONMENT` | `dev`              | One of `dev`, `staging`, `prod`.     |

### Sentiment engines

| Variable                       | Default                                          | Description                                                      |
| ------------------------------ | ------------------------------------------------ | ---------------------------------------------------------------- |
| `ASA_DEFAULT_SENTIMENT_ENGINE` | `rule_based`                                     | Default engine when the request omits `sentiment_engine`.        |
| `ASA_HF_SENTIMENT_MODEL`       | `distilbert-base-uncased-finetuned-sst-2-english`| HuggingFace model id loaded by `HuggingFaceSentimentEngine`.     |

See [engines.md](engines.md) for engine-specific behavior.

### LLM engine (litellm)

| Variable          | Default                       | Description                                                       |
| ----------------- | ----------------------------- | ----------------------------------------------------------------- |
| `ASA_LLM_API_KEY` | `""`                          | Provider API key. Forwarded to litellm at engine init.            |
| `ASA_LLM_MODEL`   | `claude-sonnet-4-20250514`    | Any litellm-compatible model id (`gpt-4o`, `ollama/llama3`, etc.).|

The LLM engine is the only one that consumes RAG context — see
[rag.md](rag.md).

### RAG

| Variable                | Default          | Description                                                                |
| ----------------------- | ---------------- | -------------------------------------------------------------------------- |
| `ASA_RAG_ENABLED`       | `true`           | When false, `AnalysisService` skips the enricher entirely.                 |
| `ASA_RAG_TOP_K`         | `5`              | Default number of chunks retrieved per record.                             |
| `ASA_RAG_NAMESPACE`     | `default`        | Default namespace used by the enricher and the `/api/rag/retrieve` route.  |
| `ASA_VECTOR_STORE_PATH` | `./vector_store` | Filesystem path used by the bootstrap script.                              |

Note: the API uses `ASA_RAG_DB_PATH` (below) as the Chroma persist directory in
[`rag/vector_store.py`](../sentiment_analyzer/rag/vector_store.py); the
bootstrap script in [`scripts/bootstrap_rag_store.py`](../sentiment_analyzer/scripts/bootstrap_rag_store.py)
uses `ASA_VECTOR_STORE_PATH`. If you seed via the script, set both to the same
location.

### Evaluation

| Variable                     | Default | Description                                                                |
| ---------------------------- | ------- | -------------------------------------------------------------------------- |
| `ASA_ENABLE_EVALUATION`      | `true`  | Reserved for gating evaluation features (currently informational).         |
| `ASA_EVALUATION_SAMPLE_RATE` | `1.0`   | Reserved for future sampled evaluation.                                    |

### Storage

| Variable          | Default        | Description                                                                  |
| ----------------- | -------------- | ---------------------------------------------------------------------------- |
| `ASA_RAG_DB_PATH` | `runs.duckdb`  | DuckDB file used by `RunStore` *and* the Chroma persist directory used by the API's `VectorStore`. |

## Reading configuration in code

```python
from sentiment_analyzer.core.config import settings

settings.rag_top_k          # 5
settings.default_sentiment_engine  # "rule_based"
```

Always read through `settings` rather than calling `os.environ` directly —
this keeps configuration centralized and validated.

## Adding a new setting

1. Add a field to `Settings` in [`core/config.py`](../sentiment_analyzer/core/config.py).
2. Add the corresponding `ASA_*` line to [.env.example](../.env.example) with
   a comment describing it.
3. Update the table above.
4. If it has security implications (API keys, secrets), make sure it is
   `""` by default and never logged.
