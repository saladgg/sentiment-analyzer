# Development

Conventions for working on SentimentAnalyzer locally — environment,
testing, lint, type checking, Docker, and the release flow.

## Prerequisites

- Python `>=3.13` (the project pins to 3.13 in `pyproject.toml`; the Docker
  base image is `python:3.13-slim`).
- [`uv`](https://docs.astral.sh/uv/) is the canonical package manager. The
  Makefile and CI both assume it.

## First-time setup

```bash
git clone https://github.com/saladgg/sentiment-analyzer.git
cd sentiment-analyzer
make install-dev          # installs runtime + dev/test deps
cp .env.example .env      # then edit
```

To create a managed-Python venv up front (Makefile uses 3.14 for the venv,
but the project requires `>=3.13`):

```bash
make venv
```

## Make targets

| Target          | Description                                                    |
| --------------- | -------------------------------------------------------------- |
| `make help`     | List all targets.                                              |
| `make install`  | Production deps via `uv sync`.                                 |
| `make install-dev` | Production + dev/test deps.                                  |
| `make install-test`| Test deps only.                                              |
| `make update`   | `uv lock --upgrade` — bump pins.                               |
| `make format`   | `ruff format` (write).                                         |
| `make format-check` | `ruff format --check` (no writes — used in CI).            |
| `make lint`     | `ruff check` + `mypy`.                                         |
| `make fix`      | `ruff check --fix`.                                            |
| `make test`     | `pytest` with coverage.                                        |
| `make run`      | FastAPI dev server on `:8000` (auto-reload).                   |
| `make frontend` | Streamlit on `:8501`.                                          |
| `make docker-up`| `docker compose up --build -d`.                                |
| `make docker-down` | `docker compose down`.                                      |
| `make docker-logs` | `docker compose logs -f`.                                   |

## Running tests

```bash
make test
```

Configuration for pytest lives in
[`pyproject.toml`](../pyproject.toml) under `[tool.pytest.ini_options]`:

- Tests are discovered under `tests/`.
- `asyncio_mode = "auto"` — async test functions are run automatically.
- Coverage is computed against the `sentiment_analyzer` package and reported
  with `term-missing`.

The chromadb deprecation warning is intentionally suppressed via
`filterwarnings`.

Test files are organized per module:

| File                                  | Covers                                  |
| ------------------------------------- | --------------------------------------- |
| `tests/test_api.py`                   | FastAPI route handlers                  |
| `tests/test_services.py`              | `AnalysisService.run` orchestration     |
| `tests/test_sentiment_engines.py`     | Each engine's contract                  |
| `tests/test_processing.py`            | Field detection / normalization / validation |
| `tests/test_ingestion.py`             | CSV / Excel / JSON adapters             |
| `tests/test_analytics.py`             | Aggregation, anomalies, comparison      |
| `tests/test_storage.py`               | DuckDB run persistence                  |
| `tests/test_rag.py`                   | Vector store + enricher                 |
| `tests/test_evaluation.py`            | `EvaluationHarness`                     |

## Linting and types

`make lint` runs both:

- **ruff** — selectors `E, F, I, UP, B, C4, SIM`; line length 100;
  `target-version = "py313"`. Configured in `pyproject.toml`.
- **mypy** — `strict = false`, `ignore_missing_imports = true`. Targets
  `sentiment_analyzer` and `tests`.

`make format-check` is the formatter check used in CI; `make format` is the
local equivalent that writes.

## Pre-commit (optional)

`pre-commit>=4.0` is in the dev group. There is no checked-in
`.pre-commit-config.yaml` yet — install one if your team workflow needs it.

## Docker

[`Dockerfile`](../Dockerfile) is a two-stage build:

1. **Builder** — Python 3.13-slim with `uv` from the official Astral image.
   Installs production deps into a venv at `/build/.venv` and copies the
   source.
2. **Runtime** — Python 3.13-slim with the venv and source copied across.
   Exposes port 8000 and runs `uvicorn sentiment_analyzer.main:app`.

A healthcheck hits `/docs` every 30 seconds.

[`docker-compose.yml`](../docker-compose.yml) wires it up locally:

- Mounts a named volume `run-data` at `/app/data` for persistence.
- Loads env vars from `.env`.
- `restart: unless-stopped`.

To get a fresh build and tail logs:

```bash
make docker-up
make docker-logs
```

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on push to
`main`:

1. `make install-dev`
2. `make format`
3. `make lint`
4. `make test`

Note: the workflow runs `make format` (write mode), not `make format-check`.
On a clean tree this is a no-op; if the formatter would have made changes,
the subsequent steps still pass — consider switching this step to
`format-check` if you want CI to enforce formatting.

## Releases

[`.github/workflows/release.yml`](../.github/workflows/release.yml) runs on
push to `main` and uses
[python-semantic-release](https://python-semantic-release.readthedocs.io/) to
cut tags and GitHub releases automatically.

Configuration lives in `pyproject.toml` under `[tool.semantic_release]`:

```toml
commit_parser = "conventional"
version_toml = ["pyproject.toml:project.version:tf"]
branch = "main"
build_command = "python -m build"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "perf", "refactor", "docs", "chore", "build", "ci", "test", "style"]
minor_tags  = ["feat"]
patch_tags  = ["fix", "perf", "refactor"]
```

This means commit messages drive versioning. Use Conventional Commits:

- `feat: …` — minor bump
- `fix: …` / `perf: …` / `refactor: …` — patch bump
- `chore: …`, `docs: …`, `test: …`, `ci: …`, `style: …`, `build: …` — no
  version change

The release job also generates `CHANGELOG.md` and uploads release artifacts.

## Logging

Structured JSON logs via
[`core/logging.py`](../sentiment_analyzer/core/logging.py). Use the shared
`logger`:

```python
from sentiment_analyzer.core.logging import logger

logger.info("analysis_started", run_id=run_id)
logger.error("csv_analysis_failed", error=str(e))
```

Each log line is a JSON object with `timestamp`, `level`, the event name as
the first positional argument, and any keyword arguments as fields. This is
deliberately analytics-friendly — pipe stdout into your log aggregator and
filter by event name.

## File and folder layout

```
sentiment_analyzer/
  api/              FastAPI route handlers and Pydantic schemas
  analytics/        Aggregation, anomaly detection, explainability, run comparison
  core/             Configuration (pydantic-settings) and structured logging
  evaluation/       Offline evaluation harness
  ingestion/        CSV / Excel / JSON / text adapters
  processing/       Field detection, normalization, validation
  rag/              Vector store, retrieval, RAG dataclasses
  scripts/          Bootstrap / utility scripts
  sentiment/        Sentiment engines + base interface
  services/         Analysis orchestration and workflow abstractions
  storage/          DuckDB-backed run persistence
tests/              Test suite (one file per module)
assets/             Sample data files
frontend/
  app.py            Streamlit entrypoint with navigation
  api_client.py     Thin requests-based HTTP client
  theme.py          Color and label constants
  pages/            Analyze, Run History, RAG Management
docs/               This documentation
.github/workflows/  CI and release workflows
```
