# SentimentAnalyzer Documentation

Detailed reference docs for the SentimentAnalyzer platform.

For installation and a quick tour, see the [project README](../README.md). The
documents in this folder go deeper into how the system is composed, configured,
and extended.

## Table of contents

| Doc                                        | Topic                                                                                  |
| ------------------------------------------ | -------------------------------------------------------------------------------------- |
| [architecture.md](architecture.md)         | System architecture, request lifecycle, module responsibilities, data flow.            |
| [configuration.md](configuration.md)       | All `ASA_*` environment variables, defaults, `.env` workflow.                          |
| [engines.md](engines.md)                   | The three sentiment engines (rule-based, HF transformer, LLM) and how to add another.  |
| [rag.md](rag.md)                           | RAG pipeline — vector store, ingestion, enrichment, run comparison.                    |
| [api-reference.md](api-reference.md)       | Full HTTP API reference with request/response schemas and examples.                    |
| [data-model.md](data-model.md)             | Pydantic schemas, internal record shape, persistence layout.                           |
| [frontend.md](frontend.md)                 | Streamlit pages (Analyze, Run History, RAG Management) and how they call the backend. |
| [development.md](development.md)           | Local setup, testing, linting, type checking, Docker, releases, CI.                    |

## Conventions used in these docs

- File and folder references use the `path/to/file.py` form so they remain
  navigable from any markdown viewer in the repo.
- All environment variables are prefixed with `ASA_` (AllSentimentAnalyzer).
- All HTTP routes are mounted under the `/api` prefix in
  [main.py](../sentiment_analyzer/main.py).
- "Run" refers to a single execution of the analysis pipeline against a
  dataset, identified by a UUID `run_id`.
