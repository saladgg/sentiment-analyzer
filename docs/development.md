# Development

Conventions for working on SentimentAnalyzer locally: environment, testing,
lint, type checking, Docker, CI, and releases.

## Prerequisites

- Python `>=3.13` (`requires-python` in `pyproject.toml`). The repo's
  [`.python-version`](../.python-version) selects **3.14**, which `uv` uses
  for the local venv, and the Docker images and release workflow use 3.14 too.
- [`uv`](https://docs.astral.sh/uv/). The Makefile and CI assume it.
- Docker, for the container targets.

## Dependencies

Runtime dependencies are in `[project.dependencies]`. Optional parts are
extras:

| Extra      | Contents                                  | Needed for                         |
| ---------- | ----------------------------------------- | ---------------------------------- |
| `ml`       | `torch`, `transformers`                   | The `hf_transformer` engine.       |
| `frontend` | `streamlit`, `plotly`, `requests`         | The Streamlit UI.                  |

Without `ml`, the API runs normally and requests for `hf_transformer` return
`503` with an install hint.

On Linux, `torch` is resolved from the CPU-only PyTorch index
(`[tool.uv.sources]` in `pyproject.toml`), which keeps installs and images
several GB smaller. For a GPU build, remove that source or point it at a CUDA
index.

Development tooling is in dependency groups: `test` (pytest, pytest-cov,
pytest-asyncio, Faker, numpy, requests) and `dev` (`test` plus ruff, mypy,
httpx, types-requests, pre-commit).

All `make install*` targets run `uv sync --locked`, so they fail if
`uv.lock` is out of date with `pyproject.toml`. Run `uv lock` (or
`make update` to upgrade everything) after changing dependencies.

## First-time setup

```bash
git clone https://github.com/saladgg/sentiment-analyzer.git
cd sentiment-analyzer
make install-dev          # all extras + dev and test groups
cp .env.example .env      # then edit
uv run pre-commit install # optional, see below
```

## Make targets

| Target              | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `make help`         | List all targets.                                                           |
| `make venv`         | Create `.venv` with a uv-managed Python (version from `.python-version`).   |
| `make install`      | Runtime deps with all extras (`ml`, `frontend`), no dev tools.              |
| `make install-dev`  | All extras plus the `dev` group.                                            |
| `make install-test` | All extras plus the `test` group.                                           |
| `make update`       | `uv lock --upgrade`.                                                        |
| `make format`       | `ruff format` (writes).                                                     |
| `make format-check` | `ruff format --check` (no writes; used in CI).                              |
| `make lint`         | `ruff check` and `mypy`.                                                    |
| `make fix`          | `ruff check --fix`.                                                         |
| `make test`         | `pytest` with coverage; fails below the coverage threshold.                 |
| `make audit`        | Scan the locked runtime dependencies (all extras) with `pip-audit`.         |
| `make run`          | FastAPI dev server on `127.0.0.1:8000` with auto-reload.                    |
| `make frontend`     | Streamlit on `:8501` (uses `ASA_API_URL` / `ASA_API_KEY`).                  |
| `make rag-bootstrap`| Seed the RAG vector store with starter documents. Stop the API first.       |
| `make docker-up`    | `docker compose up --build -d` (API on `:8000`, UI on `:8501`).             |
| `make docker-down`  | `docker compose down`.                                                      |
| `make docker-logs`  | `docker compose logs -f`.                                                   |

## Running tests

```bash
make test
```

pytest is configured in [`pyproject.toml`](../pyproject.toml) under
`[tool.pytest.ini_options]`:

- Tests are discovered under `tests/`; `asyncio_mode = "auto"`.
- Coverage (with branch coverage) is measured on `sentiment_analyzer`. The run
  fails if total coverage is below **90%** (`[tool.coverage.report]
  fail_under`).
- A known chromadb deprecation warning is filtered.

The suite runs offline. [`tests/conftest.py`](../tests/conftest.py) provides:

- `make_settings` / `settings` — explicit `Settings` built with
  `_env_file=None` (your `.env` is never read), per-test DuckDB and Chroma
  paths, RAG off, warm-up off, no API keys.
- `make_app` / `client` — a fresh app per test via `create_app(settings,
  embedding_function=...)`, wrapped in a `TestClient`.
- `FakeEmbeddingFunction` — a deterministic, offline Chroma embedder.
- `stub_hf_transformer` (autouse) — replaces the transformers pipeline with a
  fake, so no model is downloaded.
- `fake_llm` — replaces `litellm.completion`.

It also forces safe environment values (`ASA_ENVIRONMENT=dev`, empty keys,
temp paths, `LITELLM_MODE=PRODUCTION`) before `sentiment_analyzer.main` is
imported, because importing it builds the module-level app.

Test files:

| File                              | Covers                                                    |
| --------------------------------- | --------------------------------------------------------- |
| `tests/test_api.py`               | HTTP routes, auth, errors, limits, request IDs, health    |
| `tests/test_config.py`            | Settings parsing, validation, deprecated alias            |
| `tests/test_services.py`          | `AnalysisService.run` orchestration                       |
| `tests/test_sentiment_engines.py` | `SentimentResult`, each engine, the registry              |
| `tests/test_processing.py`        | Field detection, normalization, validation                |
| `tests/test_ingestion.py`         | CSV / Excel / JSON / text adapters                        |
| `tests/test_analytics.py`         | Aggregation, anomalies, explanations, trends, comparison  |
| `tests/test_storage.py`           | DuckDB run store and migrations                           |
| `tests/test_rag.py`               | Chunking, enrichment, bootstrap script, RAG endpoints     |
| `tests/test_vector_store.py`      | Chroma vector store                                       |
| `tests/test_evaluation.py`        | `EvaluationHarness`                                       |
| `tests/test_frontend_client.py`   | Frontend `api_client`                                     |

## Linting and types

`make lint` runs:

- **ruff** — rule sets `E, F, I, UP, B, C4, SIM, S, ASYNC, RUF, BLE` (pyflakes,
  pycodestyle, isort, pyupgrade, bugbear, comprehensions, simplify, bandit
  security checks, async, ruff-specific, blind `except`). Line length 100,
  `target-version = "py313"`. Tests may use `assert` and test secrets/paths.
- **mypy** — over `sentiment_analyzer`, `tests` and `frontend`, with
  `check_untyped_defs = true`; inside `sentiment_analyzer`,
  `disallow_incomplete_defs = true` (a partly annotated function must be fully
  annotated). `ignore_missing_imports = true`.

`make format-check` is what CI runs; `make format` writes the changes.

## Pre-commit

[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) runs basic file
checks (end-of-file, trailing whitespace, YAML/TOML syntax, files over 1 MB,
private keys) and then `ruff format`, `ruff check --fix` and `mypy` through
`uv run`, so it uses the project's own tool versions. Enable it with:

```bash
uv run pre-commit install
```

## Docker

[`Dockerfile`](../Dockerfile) is a multi-stage build with two runnable
targets:

| Target     | Image contents                                              | Port | Health check                 |
| ---------- | ----------------------------------------------------------- | ---- | ---------------------------- |
| `api` (default) | The package with the `ml` extra (CPU torch).           | 8000 | `GET /healthz`               |
| `frontend` | The `frontend` extra and `frontend/`.                       | 8501 | `GET /_stcore/health`        |

```bash
docker build -t sentiment-analyzer .                      # api
docker build --target frontend -t sentiment-analyzer-ui .
```

Details:

- The builder stage uses a pinned `uv` and installs from `uv.lock` with
  `--locked`, so a stale lockfile fails the build. `EXTRAS` (build arg,
  default `--extra ml`) controls the API image's extras.
- Both images use `python:3.14-slim` and run as a non-root user `app`
  (uid/gid 10001).
- All mutable state lives under `/app/data`: `ASA_RUNS_DB_PATH=/app/data/runs.duckdb`,
  `ASA_VECTOR_STORE_PATH=/app/data/vector_store`, `HF_HOME=/app/data/hf`, and
  `HOME=/app/data/home` (Chroma caches its embedder under `~/.cache`).
- `LITELLM_MODE=PRODUCTION` is set.
- The API runs `uvicorn sentiment_analyzer.main:app --workers 1`; DuckDB needs
  a single writing process.
- The frontend image sets `ASA_API_URL=http://api:8000/api/v1` and headless
  Streamlit settings.

[`docker-compose.yml`](../docker-compose.yml) runs both services:

- **api** — port 8000, optional `.env` (`env_file` with `required: false`),
  named volume `run-data` at `/app/data`, `mem_limit: 4g`,
  `restart: unless-stopped`. It re-sets `ASA_RUNS_DB_PATH`,
  `ASA_VECTOR_STORE_PATH` and `HF_HOME` so relative paths in `.env` cannot
  move state off the volume.
- **frontend** — port 8501, `ASA_API_URL=http://api:8000/api/v1`,
  `ASA_API_KEY` taken from your shell (`${ASA_API_KEY:-}`), starts once the API
  is healthy, `mem_limit: 1g`.

```bash
make docker-up
make docker-logs
```

If `.env` sets `ASA_ENVIRONMENT` to `staging` or `prod`, it must also set
`ASA_API_KEYS`, or the API container will not start. Export a matching
`ASA_API_KEY` before `make docker-up` so the UI can authenticate.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pushes to
any branch except `main`, on pull requests, and when called by the release
workflow. Actions are pinned to commit SHAs. Jobs:

1. **Lint & Test** — `make install-dev` (fails on a stale lockfile),
   `make format-check`, `make lint`, `make test` (coverage gate).
2. **Dependency audit** — `make audit` (`pip-audit` over the locked runtime
   dependencies). `torch`'s local `+cpu` build is checked under its upstream
   PyPI version. Accepted advisories are listed in `AUDIT_IGNORES` in the
   `Makefile`, each with a reason and a re-check trigger. Add to that list
   only after confirming the advisory can't be reached in this app; upgrade
   instead whenever a fix exists (`make update`).
3. **Docker build & smoke test** — builds the `api` target, starts it, waits
   for `/healthz`, then checks `/readyz` and a `POST /api/v1/analyze` with
   plain text.

## Dependabot

[`.github/dependabot.yml`](../.github/dependabot.yml) opens weekly update PRs
for `uv` dependencies (minor and patch updates grouped), GitHub Actions, and
the Docker base images.

## Releases

[`.github/workflows/release.yml`](../.github/workflows/release.yml) runs on
push to `main`:

1. **CI gate** — calls `ci.yml`; nothing is released if it fails.
2. **Semantic release** — runs
   [python-semantic-release](https://python-semantic-release.readthedocs.io/)
   10.x on Python 3.14 to bump the version, tag, update `CHANGELOG.md`, and
   create a GitHub release with the built distributions.
3. **Publish container images** — only if a new version was released. Builds
   the `api` and `frontend` targets at the new tag and pushes them to GHCR as
   `ghcr.io/<owner>/sentiment-analyzer` and
   `ghcr.io/<owner>/sentiment-analyzer-ui`, tagged with the version and
   `latest`, with an SBOM and provenance attestation.

Semantic-release is configured in `pyproject.toml` under
`[tool.semantic_release]`:

```toml
commit_parser = "conventional"
version_toml = ["pyproject.toml:project.version:nf"]
branch = "main"
build_command = "python -m build"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "perf", "refactor", "docs", "chore", "build", "ci", "test", "style"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf", "refactor"]
```

Commit messages drive versioning (Conventional Commits):

- `feat: ...` — minor bump
- `fix: ...` / `perf: ...` / `refactor: ...` — patch bump
- `chore`, `docs`, `test`, `ci`, `style`, `build` — no version change

## Logging

[`core/logging.py`](../sentiment_analyzer/core/logging.py) configures
structlog and routes the standard `logging` module (uvicorn, libraries)
through the same JSON renderer, so every line on stdout is one JSON object
with `timestamp` (ISO 8601, UTC), `level`, `logger`, `event`, any keyword
fields, and `request_id` when emitted during a request. uvicorn's access log
is silenced in favour of the app's own `http_request` line.

```python
from sentiment_analyzer.core.logging import logger

logger.info("analysis_started", run_id=run_id)
```

The level comes from `ASA_LOG_LEVEL`.

## File and folder layout

```
sentiment_analyzer/
  main.py           create_app, lifespan, router mounting
  api/              Routes, dependencies and Pydantic schemas
  analytics/        Aggregation, anomalies, explanations, trends, run comparison
  core/             Settings, auth, middleware, errors, logging, version
  evaluation/       Classification metrics
  ingestion/        CSV / Excel / JSON / text adapters
  processing/       Field detection, normalization, validation
  rag/              Vector store, chunking, enrichment, RAG dataclasses
  scripts/          bootstrap_rag_store
  sentiment/        Engine interface, engines, registry
  services/         AnalysisService
  storage/          DuckDB run store and migrations
tests/              Test suite
assets/             Sample data files
frontend/
  app.py            Streamlit entrypoint and navigation
  api_client.py     HTTP client
  theme.py          Colours and fallback engine labels
  pages/            Analyze, Run History, RAG Management
docs/               This documentation
.github/            CI, release and Dependabot configuration
```
