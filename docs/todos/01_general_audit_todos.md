# SentimentAnalyzer — General Audit Backlog

Execution checklist for the whole-repo audit. Findings, evidence and code
references are in [`01_general_audit.md`](01_general_audit.md), and the IDs
match that file. Each tier ships as one branch/PR. Items within a tier can be
picked up independently unless a **Depends** note says otherwise.

> **Status (2026-11-25):** `v1.0.0`. Freshly filed: **nothing shipped yet.**

Effort key: **S** ≤ 0.5d · **M** ≤ 2d · **L** > 2d.
Definition of done for every item: regression test(s) added, and
`make format-check lint test` green.

---

## Tier 0 — Make the headline features work; make CI honest

- [ ] **BUG-1 — Persist the RAG vector store.** (S) Switch `VectorStore` to
  `chromadb.PersistentClient(path=…)`. Point the API, enricher and bootstrap
  script at **`settings.vector_store_path`**, not `rag_db_path`, because that
  path is the DuckDB file and a Chroma directory there would collide with
  it. Test: write with one `VectorStore`, then read with a *new* instance on
  the same path. *Depends: rename half of ARCH-2 (can land together).*
- [ ] **BUG-2a — Fix the compare crash.** (S) `RunStore.get_run` decodes
  `summary` the same way `_load_all_runs` does; extract a `_row_to_run`
  helper used by both. Test: compare two real runs and get a 200.
  *(The semantic half, BUG-2b, is in Tier 4.)*
- [ ] **OPS-1 — CI.** (S) `ci.yml`: replace `make format` with
  `make format-check`; trigger on `push` (all branches) and
  `pull_request`; add `concurrency`; use `uv sync --locked`. `release.yml`:
  run only after CI succeeds on `main` (`workflow_run` with a
  `conclusion == success` guard, or merge both into one workflow with
  `needs:`).
- [ ] **OPS-2 — Data paths onto the volume.** (S) Image `ENV`:
  `ASA_RUNS_DB_PATH=/app/data/runs.duckdb`,
  `ASA_VECTOR_STORE_PATH=/app/data/vector_store`, `HF_HOME=/app/data/hf`.
  Update `.env.example` and `docs/development.md`. Verify that run history
  survives `docker compose down && up`.
- [ ] **BUG-5 — Engine name validation.** (S) Add an `EngineName` enum (or
  `Literal`) on `AnalysisRequest` and on the CSV/Excel query params, so an
  unknown engine returns a 422. The route default comes from
  `settings.default_sentiment_engine`.
- [ ] **BUG-6 — Plain-text input.** (S) Implement `TextAdapter` (a string
  becomes one `{"text": …}` record per non-empty line) and dispatch adapters
  on `source_type`. Otherwise, remove `text` from the contract and docs.
- [ ] **TEST-2 — Fix tests that lock in bugs.** (S) The unknown-engine test
  should assert 422. The CSV test should send `sentiment_engine` as a query
  param and assert a *non-default* engine is used.

## Tier 1 — Harden before the API is reachable by anyone else

- [ ] **SEC-1 — API-key auth.** (M) `settings.api_keys: list[SecretStr]`
  and a `require_api_key` dependency (`X-API-Key`, compared with
  `secrets.compare_digest`) applied router-wide. Refuse to boot when
  `environment != "dev"` and no key is set. Optional write scope for RAG
  mutation and the `llm` engine. The frontend sends the key from
  `ASA_API_KEY`. Tests: 401 without a key or with a wrong key, 200 with the
  right key, dev mode open.
- [ ] **SEC-2 — Input limits.** (M) Add settings `max_upload_bytes`
  (default 10 MB), `max_records` (10k), `max_text_chars` (5k per record,
  truncate and flag), and `max_llm_records` (500). Enforce them before
  parsing, return 413/422 with a clear `detail`, and apply the same caps to
  RAG upload. Tests for each limit.
- [ ] **SEC-3 — Prompt-injection mitigations.** (M) Wrap record text and
  context in delimiters, and add a system-prompt clause that their content
  is data, not instructions. Use `response_format` / JSON schema where the
  provider supports it. Strict output validation (shared with BUG-9). Store
  the RAG chunk IDs used per record (ties to ARCH-5).
- [ ] **SEC-4 — CORS.** (S) Origins from `settings.cors_origins`,
  `allow_credentials=False`, methods `GET, POST, DELETE`.
- [ ] **SEC-5 — Hide docs outside dev.** (S) Turn off
  `docs_url`/`redoc_url`/`openapi_url` when `environment == "prod"`.
  *Depends: OPS-3 (the healthcheck must move off `/docs` first).*
- [ ] **BUG-9 — LLM engine robustness.** (M) Parse with a Pydantic model
  after stripping code fences, validate `label`, clamp `score`, handle
  `None` content, and set `timeout` and `num_retries`. A per-record failure
  becomes `{"error": …}` on that record (add an optional `error` field to
  `RecordSentiment`) instead of failing the whole run. Tests cover fenced
  JSON, prose, invalid label, out-of-range score and an exception from
  `completion`.
- [ ] **BUG-12 — Evaluation endpoint.** (S) Pydantic request model with a
  length-equality validator, returning 422. Rename `precision` to
  `accuracy`.
- [ ] **BUG-13 — RAG route validation.** (S) `RetrieveRequest` model
  (`query` 1–2000 chars, `top_k` 1–50, `namespace` regex
  `^[a-zA-Z0-9_-]{1,64}$`); a `UnicodeDecodeError` returns 415; namespace
  validated on upload too.

## Tier 2 — Lifespan, dependency injection and the core performance fixes

- [ ] **ARCH-2 — Config cleanup.** (S) Rename `rag_db_path` to
  `runs_db_path`, keeping the `ASA_RAG_DB_PATH` alias with a deprecation
  warning. Add a `data_dir` default. Type `environment` as a `Literal`; make
  `llm_api_key` a `SecretStr`; add `log_level` and `cors_origins`. Delete
  or wire `enable_evaluation`/`evaluation_sample_rate`. Update
  `docs/configuration.md`.
- [ ] **ARCH-1 + PERF-1 — Lifespan singletons and lazy engines.** (M)
  `create_app(settings)`; `lifespan` builds `RunStore`, `VectorStore` and an
  `EngineRegistry` once, on `app.state`. Routes take `Depends(get_service)`.
  Engines are built on first use and cached. Remove the module-level
  `store =` globals from `rag_routes`/`run_routes`. Turn the bootstrap
  script into `main()` behind a `__main__` guard, with a
  `make rag-bootstrap` target. Tests switch to `dependency_overrides`.
  Acceptance: a `rule_based` request never instantiates the HF pipeline
  (assert with a counter on the stub).
- [ ] **ARCH-6a — Stop blocking the event loop.** (S) Make
  `analyze_csv`, `analyze_excel` and `upload_document` plain `def`, or
  offload with `run_in_threadpool`. Enable ruff `ASYNC` rules so this
  doesn't regress.
- [ ] **ARCH-7 — DuckDB concurrency.** (S) One process-wide connection, a
  `cursor()` per request or operation, and the single-process constraint
  documented in `docs/architecture.md`.
- [ ] **PERF-2 — Batching.** (M) Add `SentimentEngine.analyze_batch`
  (default: loop). HF passes the list with `batch_size` and
  `truncation=True`. Enrichment uses one multi-query `collection.query`
  call. The LLM engine uses `acompletion` with a semaphore
  (`settings.llm_concurrency`, default 8). The service calls the batch API.
  *Depends: ARCH-4 for the typed result (can land together).*

## Tier 3 — Analysis quality

- [ ] **BUG-3 + FEAT-1 — Field detection.** (S) Score columns across a
  sample (by share of strings and median length), allow an explicit
  `text_fields` request param and query param, and return
  `detected_text_fields` in the response. If no field is found, return a
  422 with guidance instead of an all-neutral run.
- [ ] **BUG-4 — NaN handling.** (S) Treat `pd.isna` values as empty in
  `normalize_record`. Test with a CSV and an Excel file that have blank
  cells.
- [ ] **BUG-7 — Rule-based tokenizer and negation.** (S) Regex tokenizer
  plus a one-token negation window. Tests for `"great!"`, `"not good"` and
  `"not bad"`.
- [ ] **BUG-8 — HF truncation and label mapping.** (S) Pass
  `truncation=True`, map labels through `id2label` with an explicit
  polarity table, and add a configurable neutral band for binary models
  (FEAT-9). Tests use a stub with 3-class labels.
- [ ] **BUG-10 — Real per-field contributions.** (M) Score each detected
  text field separately and report those as `contributing_fields`, with the
  engine internals under a separate `drivers` key. Update the response
  schema, frontend chart and docs.
- [ ] **BUG-11 — Skip RAG for engines that ignore it.** (S) Add an engine
  `uses_context` flag. Enrichment and `rag.used` only apply when it's true.
- [ ] **ARCH-4 — Typed engine contract.** (M) A `SentimentResult` model
  (`label`, `polarity`, `confidence`, `drivers`), an `EngineRegistry` with
  capabilities, and each engine's score semantics documented in
  `docs/engines.md`. Anomaly detection runs on `polarity` within a single
  engine only.

## Tier 4 — Persistence and product surface

- [ ] **ARCH-5 — Richer run persistence and migrations.** (M) A
  `schema_version` table with ordered migrations. New `runs` columns:
  `source_type`, `input_sha256`, `config` JSON (engine, model and revision,
  RAG settings, text fields, app version), `records_in`/`skipped`/`failed`,
  `duration_ms`, `status`, `error`, and `created_at` in UTC. A new
  `run_records` table (`run_id`, `record_id`, `label`, `polarity`,
  `confidence`, `drivers`, `rag_chunk_ids`, `error`).
- [ ] **FEAT-2 — Run detail and export.** (M) `GET /api/runs/{id}` with
  paginated records; `GET /api/runs/{id}/export?format=csv|json`. The
  History page shows per-record results and a download button.
  *Depends: ARCH-5.*
- [ ] **BUG-2b / FEAT-6 — Meaningful RAG comparison.** (M) Compare only runs
  with the same `input_sha256` (otherwise 409). Compute per-record label
  changes, mean confidence delta and changed records. Add
  `POST /api/runs/ab` to run the same input with and without RAG in one
  call (only for `uses_context` engines). Update the History UI.
  *Depends: ARCH-5, BUG-11.*
- [ ] **FEAT-4 — `GET /api/engines`.** (S) Returns name, label,
  description, loaded, uses_context, is_paid and limits. The frontend builds
  its selector from this instead of `theme.ENGINE_LABELS`.
- [ ] **PERF-4 — Paginate `GET /api/runs`.** (S) `limit` (default 50,
  maximum 500), `offset`, and `engine`/`rag_enabled` filters; the History
  page pages through results.

## Tier 5 — Cleanup and refactoring

- [ ] **QUAL-1 — Delete dead modules.** (S) Remove `rag/store.py`,
  `rag/types.py`, `analytics/llm_explainability.py` and
  `services/workflow.py` (BUG-14), plus `harness.drift`/`stability` unless
  FEAT-5 uses them. `text_adapter.py` is handled by BUG-6.
- [ ] **QUAL-2 — Renames.** (S) `llm_stub.py` → `llm.py`;
  `_load_all_runs` → `list_runs`; move `GET /runs` into `run_routes.py`.
- [ ] **QUAL-3 — De-duplicate.** (S) One analyze helper, a shared dedupe in
  `VectorStore`, and score mapping done inside `VectorStore.query`.
- [ ] **QUAL-4 — Exception handlers.** (S) Domain exceptions plus
  `app.exception_handler`s; delete the per-route try/log/re-raise; use
  `logger.exception` in the catch-all.
- [ ] **ARCH-3 — Typed API contract.** (M) Response models for every route,
  `SentimentLabel` as a `Literal`, and a bounded `AnalysisRequest.data`.
  Snapshot the OpenAPI schema in a test.
- [ ] **QUAL-5 — Stricter mypy.** (M) `check_untyped_defs`, then
  `disallow_untyped_defs` on `sentiment_analyzer.*`; add `frontend/` to the
  ruff `src` and mypy `files`.
- [ ] **QUAL-6 — Chunking module.** (S) `rag/chunking.py` with paragraph and
  sentence-aware splitting, plus `rag_chunk_size`/`rag_chunk_overlap`
  settings.
- [ ] **QUAL-7 — Tooling nits.** (S) Align the Python version (3.13
  everywhere, or 3.14 everywhere); canonical `version`; fix the project
  URLs; add `.pre-commit-config.yaml` (ruff, ruff-format, mypy,
  end-of-file); add the ruff `S`, `ASYNC` and `RUF` rule sets.
- [ ] **PERF-5 — Drop unused dependencies.** (S) Remove `scikit-learn`
  (unless FEAT-5 uses it), `tqdm`, `sentence-transformers` and
  `langchain-huggingface`. Confirm that the RAG round-trip test (BUG-1)
  still passes with Chroma's default embedder.

## Tier 6 — Operations, observability and test depth

- [ ] **OPS-3 — Health endpoints.** (S) `/healthz` (liveness) and
  `/readyz` (DuckDB, vector store and default engine). The Dockerfile and
  compose healthchecks use `/healthz`.
- [ ] **OPS-4 — Observability.** (M) Request-ID middleware (bound into the
  structlog context and returned as `X-Request-ID`); route stdlib and
  uvicorn logs through structlog JSON; `ASA_LOG_LEVEL`; optional `/metrics`
  covering route latency, per-engine latency, records per run, LLM tokens
  and cost, and failures by type.
- [ ] **OPS-5 / SEC-6 — Dockerfile and compose hardening.** (S) Non-root
  `USER`, pinned `uv` image, `uv sync --locked --no-install-project` then
  copy the source, `env_file` with `required: false`, `mem_limit`, and a
  `frontend` service with `ASA_API_URL`.
- [ ] **PERF-3 — Image size and model cache.** (M) CPU-only `torch` index
  for the default image, `HF_HOME` on the volume (or weights baked in at
  build time), and a pinned model revision. Record image size before and
  after in the PR.
- [ ] **ARCH-8 — Dependency groups and frontend config.** (S) Optional
  extras `api`/`ml`/`llm`/`frontend`, and `ASA_API_URL` in `api_client.py`.
- [ ] **OPS-6 — Release artifacts.** (M) On a tag: build and push the
  image to GHCR with SBOM and provenance. Pin
  `python-semantic-release` and drop the debug step.
- [ ] **OPS-7 / SEC-6 — Dependency automation.** (S) Dependabot for uv,
  actions and docker; a `pip-audit`/OSV scan step in CI; actions pinned by
  SHA.
- [ ] **SEC-7 — Error surfaces.** (S) Covered mostly by QUAL-4; the
  frontend shows the API `detail` instead of the raw exception.
- [ ] **SEC-8 — Document namespace semantics.** (S) A note in
  `docs/rag.md` and the `VectorStore` docstring that a namespace is not an
  access boundary.
- [ ] **TEST-1 — RAG and compare coverage, plus a gate.** (M) API tests for
  RAG upload, list, retrieve and delete, and enrichment with a stub store;
  then `fail_under = 90`.
- [ ] **TEST-3 — Missing test classes.** (M) Excel happy path, a mocked
  frontend `api_client`, and a Docker build-and-boot smoke job that curls
  `/healthz`.
- [ ] **TEST-4 — Test settings isolation.** (S) Build the app from an
  explicit test `Settings`, with no import-time `os.environ.setdefault`.
  *Depends: ARCH-1.*

## Tier 7 — Later (pull in when there's demand)

- [ ] **ARCH-6b / FEAT-3 — Async jobs.** (L) `202` + `run_id`, a
  status/progress endpoint, a worker (`BackgroundTasks` first, a queue
  later), and frontend polling.
- [ ] **FEAT-5 — Real evaluation.** (M) Upload a labeled dataset, run the
  chosen engine, and return the confusion matrix, per-class P/R/F1 and
  macro-F1, persisted as an evaluation run.
- [ ] **FEAT-7 — Time-trend endpoint.** (S) Wire `aggregate_by_time` with
  timestamp-field detection, and add a trend chart on the Analyze page.
- [ ] **FEAT-8 — `/api/v1` and rate limiting.** (S) Version prefix (keep
  `/api` as an alias for one release); `slowapi` per-key limits.
- [ ] **FEAT-9 — Neutral band and multilingual model option.** (S) The
  neutral band ships with BUG-8; document and test a multilingual model.
- [ ] **FEAT-10 — LLM cost guardrails.** (M) Per-run record cap (SEC-2),
  `usage` tokens and estimated cost stored in the run config, and a
  content-hash result cache.
