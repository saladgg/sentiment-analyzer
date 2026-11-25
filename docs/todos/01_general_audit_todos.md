# SentimentAnalyzer — General Audit Backlog

Execution checklist for the whole-repo audit. Findings, evidence and code
references are in [`01_general_audit.md`](01_general_audit.md), and the IDs
match that file.

> **Status (2026-11-25): all items shipped.** Tiers 0–6 landed in PR #1
> (`v1.1.0`). Tier 7 and the remaining partial items (SEC-1 read-only
> keys, QUAL-5, ARCH-3, OPS-4 `/metrics`) are on branch `audit-tier7`.
> Totals: 320+ tests, ~98% coverage with a 90% gate; ruff and mypy
> (`disallow_untyped_defs` for the package) are clean.
>
> **Release follow-up (REL-1, new):** the `v1.1.0` image publish failed.
> semantic-release bumped `pyproject.toml` but not `uv.lock`, so the
> tag's `uv sync --locked` failed. Fixed in PR #8 (the build command
> refreshes and commits the lock); its `v1.2.0` release republishes the
> images.
>
> **New findings during implementation:** SEC-9 (litellm loading `.env`
> into `os.environ`), fixed. SEC-10 (263 vulnerable locked dependencies,
> found by the first CI audit run), fixed by upgrading the lock; 4
> unfixable Chroma-server CVEs are accepted with a reason.

Effort key: **S** ≤ 0.5d · **M** ≤ 2d · **L** > 2d.

---

## Tier 0 — Make the headline features work; make CI honest ✅

- [x] **BUG-1 — Persist the RAG vector store.** `chromadb.PersistentClient`
  on `ASA_VECTOR_STORE_PATH` for the API, the enricher and the bootstrap
  script. The cross-process round-trip test is in
  `tests/test_vector_store.py`.
- [x] **BUG-2a — Fix the compare crash.** `get_run` decodes JSON, and the
  run store is resolved through dependency injection, not a module global.
- [x] **OPS-1 — CI.** `format-check`; runs on every branch and PR; locked
  sync; concurrency group; release only after the CI gate (`needs: ci`).
- [x] **OPS-2 — Data paths onto the volume.** Set in the image `ENV` *and*
  compose `environment:`, which overrides `.env`'s relative defaults.
- [x] **BUG-5 — Engine name validation.** `EngineName` enum gives a 422;
  the default comes from `ASA_DEFAULT_SENTIMENT_ENGINE`.
- [x] **BUG-6 — Plain-text input.** `TextAdapter` (one record per
  non-empty line); a model validator rejects mismatched
  `source_type`/`data` with a 422.
- [x] **TEST-2 — Fix tests that locked in bugs.**

## Tier 1 — Harden before the API is reachable by anyone else ✅

- [x] **SEC-1 — API-key auth.** `X-API-Key` on every `/api` route, using
  `secrets.compare_digest`. Boot is refused outside `dev` without keys.
  Health probes stay open. Read-only keys (`ASA_API_READ_KEYS`, GET/HEAD
  only) were added in Tier 7.
- [x] **SEC-2 — Input limits.** Upload/body bytes are checked by ASGI
  middleware (which also counts streamed chunks) and in `read_upload`;
  also enforced: records (CSV/Excel parse at most N+1 rows), text chars
  (truncated and flagged `truncated: true`), and paid-engine records.
- [x] **SEC-3 — Prompt-injection mitigations.** `<text>`/`<context>`
  delimiters with tag breakouts stripped, a "data, not instructions"
  system prompt, JSON mode where litellm reports support, a strict
  Pydantic output schema, and RAG chunk IDs stored per record.
- [x] **SEC-4 — CORS.** `ASA_CORS_ORIGINS`, no credentials, explicit
  methods and headers.
- [x] **SEC-5 — Hide docs in prod.**
- [x] **SEC-9 (new) — litellm `load_dotenv()` on import.** Copied every
  key in `./.env` into `os.environ`. Now `LITELLM_MODE=PRODUCTION` is set
  before import, and in the image.
- [x] **BUG-9 — LLM engine robustness.** Fence/prose-tolerant parsing,
  validation and clamping, timeout and retries, and per-record `error`
  instead of failing the run.
- [x] **BUG-12 — Evaluation endpoint.** Length validator (422);
  `accuracy` naming; a full report.
- [x] **BUG-13 — RAG route validation.** `RetrieveRequest`, a namespace
  pattern, and 415 for non-UTF-8 input.

## Tier 2 — Lifespan, DI and core performance ✅

- [x] **ARCH-2 — Config cleanup.** `runs_db_path` (with a deprecated
  `ASA_RAG_DB_PATH` alias), `SecretStr`, typed `environment`/`log_level`,
  dead settings removed, `extra="ignore"` so old `.env` files still boot.
  *`data_dir` was not added; the Docker env pins the paths instead.*
- [x] **ARCH-1 + PERF-1 — Lifespan singletons and lazy engines.**
  `create_app(settings)`, `app.state`, `Depends`; `EngineRegistry` builds
  each engine once, on first use. A test asserts that `rule_based` never
  loads HF.
- [x] **ARCH-6a — No blocking work on the event loop.** Routes are plain
  `def`; ruff `ASYNC` rules are enabled.
- [x] **ARCH-7 — DuckDB concurrency.** One connection, a cursor per
  operation, a write lock, transactions, and one uvicorn worker in the
  image.
- [x] **PERF-2 — Batching.** HF batch with `batch_size`/`truncation`; one
  multi-query RAG lookup; LLM thread pool (`ASA_LLM_CONCURRENCY`) plus an
  LRU result cache.

## Tier 3 — Analysis quality ✅

- [x] **BUG-3 + FEAT-1 — Field detection.** Samples 200 rows; scores
  string share, median length, distinctness and word-likeness; accepts
  explicit `text_fields`; returns `text_fields` in the response; 422 with
  guidance when nothing is found.
- [x] **BUG-4 — NaN handling.** At the adapter level (`None`) and in the
  normalizer (`pd.isna`).
- [x] **BUG-7 — Tokenizer and negation.** Lexicon expanded slightly.
- [x] **BUG-8 — HF truncation and label mapping.** Named, star and
  `LABEL_n` labels; probability-weighted polarity; `ASA_HF_NEUTRAL_BAND`;
  `ASA_HF_MODEL_REVISION`. Checked offline against the real SST-2 model,
  including a 3,000-word input.
- [x] **BUG-10 — Real per-field contributions** (non-paid engines),
  separate from `drivers`.
- [x] **BUG-11 — RAG only for `uses_context` engines.**
- [x] **ARCH-4 — Typed engine contract.** `SentimentResult`,
  `EngineRegistry` capabilities, `analyze_batch`, `config_snapshot`.

## Tier 4 — Persistence and product surface ✅

- [x] **ARCH-5 — Richer persistence and migrations.** `schema_version`
  with append-only migrations; run config, input hash, counts, duration,
  status; a `run_records` table; UTC timestamps. Tested against a v1.0.0
  database layout and a copy of a real local `runs.duckdb`.
- [x] **FEAT-2 — Run detail and export.** `GET /runs/{id}`,
  `/runs/{id}/export?format=csv|json`; the UI has download buttons.
- [x] **BUG-2b / FEAT-6 — Meaningful comparison.** Same `input_sha256`
  required (409 otherwise); per-record label changes and deltas;
  `POST /runs/ab`.
- [x] **FEAT-4 — `GET /api/v1/engines`.** Used by the Analyze page.
- [x] **PERF-4 — Paginated runs.** `{items, total, limit, offset}` with
  filters. *Breaking change to the `GET /runs` response shape; the
  frontend is updated.*

## Tier 5 — Cleanup and refactoring ✅

- [x] **QUAL-1 — Dead modules deleted.** `rag/store.py`, `rag/types.py`,
  `analytics/llm_explainability.py`, `analytics/per_record.py`,
  `services/workflow.py`, `harness.drift/stability`.
- [x] **QUAL-2 — Renames.** `llm_stub.py` → `llm.py`;
  `_load_all_runs` → `list_runs`; all run routes are in `run_routes.py`.
- [x] **QUAL-3 — De-duplicated.** One upload helper; `list_documents`
  delegates; scoring is done in `VectorStore.query_many`.
- [x] **QUAL-4 — Exception handlers.** `core/errors.py`; the generic 500
  hides internals and carries a `request_id`.
- [x] **ARCH-3 — Typed API contract.** Response models on every route,
  plus an OpenAPI snapshot test (`tests/test_openapi.py`; regenerate with
  `UPDATE_SNAPSHOTS=1`).
- [x] **QUAL-5 — Stricter mypy.** `check_untyped_defs` everywhere,
  `disallow_untyped_defs` for `sentiment_analyzer.*`, and `frontend/` is
  included.
- [x] **QUAL-6 — Chunking module.** `rag/chunking.py`, with
  `ASA_RAG_CHUNK_SIZE`/`_OVERLAP`.
- [x] **QUAL-7 — Tooling.** Python 3.14 everywhere (`.python-version`,
  Docker, CI), `requires-python >=3.13`, canonical version with semantic-
  release `nf` format, URLs fixed, `.pre-commit-config.yaml`, ruff
  `S`/`ASYNC`/`RUF`/`BLE`.
- [x] **PERF-5 — Unused dependencies dropped.** scikit-learn, tqdm,
  sentence-transformers, langchain-huggingface, scipy; the CUDA stack is
  gone from the Linux lock.

## Tier 6 — Operations, observability and test depth ✅ (one partial)

- [x] **OPS-3 — `/healthz` and `/readyz`.** Docker and compose
  healthchecks use `/healthz`.
- [x] **OPS-4 — Observability.** Request-ID middleware, one JSON access
  line per request (with the client id), stdlib/uvicorn routed through
  structlog, `ASA_LOG_LEVEL`, tracebacks via `logger.exception`.
  Prometheus `/metrics` covers HTTP by route template, runs, records,
  engine/RAG latency, LLM tokens/cost/cache and jobs; it's behind the API
  key and not rate limited.
- [x] **OPS-5 / SEC-6 — Docker hardening.** Non-root UID 10001, pinned
  `uv:0.10.9`, `--locked` with `--no-install-project` layering, an
  optional `.env`, `mem_limit`, and `api`/`frontend` targets.
- [x] **PERF-3 — Image size and model cache.** CPU-only torch on Linux;
  `HF_HOME` and `HOME` on the volume; revision pin setting.
- [x] **ARCH-8 — Extras and frontend config.** `ml` and `frontend` extras;
  `ASA_API_URL`/`ASA_API_KEY`; a compose `frontend` service.
- [x] **OPS-6 — Release artifacts.** After a release, the GHCR images are
  built and pushed with SBOM and provenance; semantic-release is pinned to
  v10; the debug step is removed. *Not yet exercised in CI.*
- [x] **OPS-7 / SEC-6 — Dependency automation.** Dependabot (uv, actions,
  docker), a `make audit` CI job, actions pinned by SHA.
- [x] **SEC-10 (new) — 263 vulnerable locked dependencies.** Found by the
  first CI audit run. `uv lock --upgrade` leaves 4 chromadb
  Chroma-server-only CVEs with no fix; they are accepted with a reason in
  `AUDIT_IGNORES`. `torch` (`+cpu`) is now audited under its PyPI version.
- [x] **SEC-7 — Error surfaces.** The UI shows the API `detail` and
  request ID.
- [x] **SEC-8 — Namespace semantics documented.**
- [x] **TEST-1 — Coverage.** 97.8%, gate at 90%.
- [x] **TEST-3 — Missing test classes.** Excel paths, the frontend
  `api_client`, limits, auth, the LLM failure modes, and a Docker
  boot-and-probe CI job.
- [x] **TEST-4 — Test settings isolation.** Apps are built from explicit
  `Settings(_env_file=None)`.

## Tier 7 — Later ✅

- [x] **ARCH-6b / FEAT-3 — Background jobs.** `POST /api/v1/jobs/analyze`
  (JSON/text, `/csv`, `/excel`) returns `202`; `GET /jobs[/{id}]` gives
  status and progress, with `run_id` when done; `DELETE` cancels a queued
  job. Input is validated before queueing (`AnalysisService.prepare`).
  Job limits are separate and larger. A bounded worker pool and queue
  (429 when full); a DuckDB `jobs` table (migration 3); jobs left
  unfinished are failed at startup; old jobs are pruned; shutdown waits a
  bounded time. *Inputs are held in memory, so jobs don't survive a
  restart; this is by design for the single-process deployment.* The
  frontend has a "Background job" mode with a progress bar and a jobs
  panel.
- [x] **FEAT-5 — Engine evaluation.** `POST /api/v1/evaluation/engine`
  returns accuracy, per-class P/R/F1, macro-F1 and the confusion matrix.
  There's a new Evaluate page in the UI. *Results are not persisted.*
- [x] **FEAT-7 — Time trend.** `timestamp_field` + `granularity`
  (day/week/month/year); a chart on the Analyze page.
- [x] **FEAT-8 — Versioning and rate limiting.** `/api/v1`, with `/api` as a
  hidden alias. Per-client token bucket (`ASA_RATE_LIMIT_PER_MINUTE`),
  keyed on the hashed API key or the IP, returning 429 + `Retry-After`.
- [x] **FEAT-9 — Neutral band and multilingual models.** Star-rating
  labels (e.g. multilingual `nlptown` models) map to polarity.
- [x] **FEAT-10 — LLM cost guardrails.** Per-run record caps, a result
  cache, and per-record usage (tokens, litellm cost estimate, cache hits)
  summed into `summary.llm_usage` and exported as metrics.
- [x] **Frontend gaps (found in the docs review).** RAG on/off control, a
  RAG A/B mode, the Evaluate page, and a namespace filter on the RAG page.
