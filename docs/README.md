# SentimentAnalyzer Documentation

Detailed reference docs for the SentimentAnalyzer platform.

For installation and a quick tour, see the [project README](../README.md). The
documents in this folder go deeper into how the system is composed,
configured, and extended.

## Table of contents

| Doc                                  | Topic                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| [architecture.md](architecture.md)   | App factory and lifespan, middleware, request lifecycle, module responsibilities.         |
| [configuration.md](configuration.md) | Every `ASA_*` setting with its default, startup validation, `.env` workflow.              |
| [engines.md](engines.md)             | Engine contract and registry, the three engines, and how to add another.                  |
| [rag.md](rag.md)                     | Vector store, chunking, enrichment, bootstrap script, measuring RAG impact.               |
| [api-reference.md](api-reference.md) | Full HTTP API: auth, limits, errors, request/response schemas, `curl` examples.           |
| [data-model.md](data-model.md)       | Pydantic schemas, engine and RAG types, DuckDB schema and migrations, field detection.    |
| [frontend.md](frontend.md)           | Streamlit pages and how they call the backend.                                            |
| [development.md](development.md)     | Setup, extras, tests and coverage, lint, pre-commit, Docker, CI, releases.                |
| [todos/](todos/README.md)            | Audit backlogs: findings plus execution checklists.                                       |

## Conventions used in these docs

- File and folder references are relative links, so they work from any
  markdown viewer in the repo.
- All settings are environment variables prefixed with `ASA_`
  (AllSentimentAnalyzer).
- HTTP routes are served under `/api/v1` (see
  [main.py](../sentiment_analyzer/main.py)); the unversioned `/api` prefix is a
  deprecated alias. Paths in [api-reference.md](api-reference.md) are relative
  to `/api/v1` unless shown in full.
- A "run" is one execution of the analysis pipeline against a dataset,
  identified by a UUID `run_id` and persisted with its per-record results.
