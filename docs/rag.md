# RAG: Retrieval-Augmented Sentiment

RAG enrichment is **optional** and **read-only**: it never performs sentiment
inference itself. It retrieves domain context for each record's text and
hands that context to the sentiment engine. Only engines with
`uses_context = True` receive it; today that is the `llm` engine (see
[engines.md](engines.md)). For other engines retrieval is skipped entirely.

## Components

| File                                                                                    | Role                                                                    |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [`rag/vector_store.py`](../sentiment_analyzer/rag/vector_store.py)                      | Persistent Chroma wrapper: add, query, batched query, list, delete.     |
| [`rag/chunking.py`](../sentiment_analyzer/rag/chunking.py)                              | Splits uploaded documents into overlapping chunks.                      |
| [`rag/context_enrichment.py`](../sentiment_analyzer/rag/context_enrichment.py)          | `RAGContextEnricher`: retrieves context for a batch of texts.           |
| [`rag/models.py`](../sentiment_analyzer/rag/models.py)                                  | `RAGChunk` and `RAGResult` dataclasses.                                 |
| [`api/rag_routes.py`](../sentiment_analyzer/api/rag_routes.py)                          | HTTP endpoints: upload, list, delete, retrieve.                         |
| [`scripts/bootstrap_rag_store.py`](../sentiment_analyzer/scripts/bootstrap_rag_store.py)| Seeds the store with starter documents.                                 |

## Vector store

The store is a persistent Chroma database in the directory
`ASA_VECTOR_STORE_PATH` (default `./vector_store`; `/app/data/vector_store` in
Docker). The API and the bootstrap script use the same setting, so documents
survive restarts and seeded documents are visible to the API. The path must
be a directory, not a file such as the DuckDB runs database.

All chunks live in one collection, `rag_context`. Embeddings come from
Chroma's built-in ONNX `all-MiniLM-L6-v2` embedder, which runs locally; the
model is downloaded once on first use and cached under `~/.cache` (in Docker,
under `/app/data/home`). Tests pass an offline stub through
`create_app(..., embedding_function=...)`. Chroma telemetry is disabled.

The `VectorStore` instance is created once per process and shared via
`app.state.vector_store`.

### Document model

Each upload becomes one logical document with:

- a `document_id` (server-assigned UUID; the bootstrap script uses fixed ids),
- a `namespace` (default `"default"`): letters, digits, `-` and `_`, 1-64
  characters,
- a `source`: the original filename, returned with retrieval results,
- one or more chunks, each stored under the id `{document_id}:{chunk_index}`
  with metadata `document_id`, `namespace`, `source`, `chunk_index`.

Deleting a document deletes all its chunks.

### Namespaces are not an access boundary

A namespace is a grouping label used to filter retrieval. It does not
restrict who can read or delete documents: any client with a valid API key
can list, query and delete documents in every namespace. Do not use
namespaces to separate tenants that must not see each other's data. Uploaded
documents are treated as trusted context for the LLM engine, which is why the
RAG routes require the same API key as the rest of the API.

## Chunking

[`chunk_text`](../sentiment_analyzer/rag/chunking.py) splits a document into
chunks of at most `ASA_RAG_CHUNK_SIZE` characters (default 500):

1. Split on paragraphs (blank lines); collapse whitespace inside each.
2. A paragraph longer than the chunk size is split on sentence boundaries
   (after `.`, `!`, `?`).
3. A sentence that is still too long is split on words; a single word longer
   than the chunk size is hard-split.
4. Units are packed into chunks up to the size limit. Each new chunk starts
   with up to `ASA_RAG_CHUNK_OVERLAP` characters (default 50, rounded down to
   whole words) from the end of the previous chunk, so context that spans a
   boundary is retrievable from either side.

The overlap must be smaller than the chunk size; settings validation enforces
this. A document with no text after chunking is rejected with `422`.

## Retrieval

```python
store.query("refund delays", top_k=5, namespace="default")  # -> list[RAGChunk]
store.query_many(["text 1", "text 2"], top_k=5, namespace="default")  # -> list[list[RAGChunk]]
```

- `namespace` filters by metadata; `None` searches all namespaces.
- An empty store returns no hits without calling Chroma.
- `score` is `1 - distance`, so higher means more relevant. With the default
  embedder the collection uses cosine distance.

```python
@dataclass(frozen=True)
class RAGChunk:
    content: str
    source: str
    score: float
    document_id: str = ""
    namespace: str = ""
    chunk_id: str = ""


@dataclass
class RAGResult:
    chunks: list[RAGChunk]
    # properties:
    context: str  # "\n".join(chunk.content)
    sources: list[str]  # sorted unique non-empty sources
    chunk_ids: list[str]
```

## Enrichment in the analysis pipeline

`main.lifespan` creates a `RAGContextEnricher(vector_store, top_k=ASA_RAG_TOP_K,
namespace=ASA_RAG_NAMESPACE)` when `ASA_RAG_ENABLED=true`, and none otherwise.

In [`AnalysisService.run`](../sentiment_analyzer/services/analysis_service.py):

- **Requested** — the request's `use_rag` if given, otherwise
  `ASA_RAG_ENABLED`. Sending `use_rag: true` while RAG is disabled returns
  `422`.
- **Active** — requested, an enricher exists, and the engine has
  `uses_context`. Requesting RAG for `rule_based` or `hf_transformer` is not an
  error; it is simply inactive.
- When active, all record texts are retrieved with **one** batched
  `query_many` call. A record whose result has chunks gets
  `RAGResult.context` passed to the engine; otherwise `None`.

Per record, the response includes:

```json
"rag": {"enabled": true, "sources": ["policy.txt"], "chunks_used": 3}
```

and the stored record keeps the retrieved `rag_chunk_ids`, so you can see
which chunks influenced it.

Per run, `AnalysisResponse.rag` is a `RAGMetadata` block when
`ASA_RAG_ENABLED=true`, and `null` otherwise:

```json
{"enabled": true, "used": true, "source_count": 2}
```

- `enabled` — RAG was active for this run (requested **and** the engine uses
  context).
- `used` — at least one record received at least one chunk. Useful for
  spotting an empty store or a wrong namespace.
- `source_count` — number of distinct sources across all records.

The run's `rag_enabled` column and the `config.rag` snapshot
(`requested`, `active`, `top_k`, `namespace`) are persisted with the run.

## HTTP surface

Full schemas are in [api-reference.md](api-reference.md#rag). Paths are shown
under `/api/v1`.

| Endpoint                               | Purpose                                                       |
| -------------------------------------- | ------------------------------------------------------------- |
| `POST /api/v1/rag/documents`           | Upload a UTF-8 text file; chunk, embed and store it.          |
| `GET /api/v1/rag/documents`            | List unique documents, optionally filtered by `namespace`.    |
| `DELETE /api/v1/rag/documents/{id}`    | Delete a document and all its chunks.                         |
| `POST /api/v1/rag/retrieve`            | Preview retrieval for a query. Does **not** affect runs.      |

Use the retrieve preview to check that the store holds the right context
before running an analysis.

## Bootstrapping the store

```bash
make rag-bootstrap
# or
uv run python -m sentiment_analyzer.scripts.bootstrap_rag_store [--path DIR] [--namespace NS]
```

It writes two short starter documents into `ASA_VECTOR_STORE_PATH` (or
`--path`) under `ASA_RAG_NAMESPACE` (or `--namespace`). It is idempotent: each
starter document has a fixed id and is deleted and re-added on every run, so
running it twice does not duplicate chunks.

Stop the API first: Chroma's on-disk store should have a single writer. Treat
the starter documents as scaffolding and replace them with your own.

## Measuring RAG impact

Two ways, both in [`api/run_routes.py`](../sentiment_analyzer/api/run_routes.py):

- **`POST /api/v1/runs/ab`** — takes an `AnalysisRequest`, runs it once
  without and once with RAG, and returns both responses plus their
  comparison. The engine must use context (`llm`); otherwise it returns `422`.
- **`GET /api/v1/runs/compare?base_run_id=...&rag_run_id=...`** — compares
  two existing runs over the **same input** (matching `input_sha256`); runs of
  different inputs return `409`.

The comparison, from
[`analytics/run_comparison.py`](../sentiment_analyzer/analytics/run_comparison.py),
pairs records by `record_id` (skipping records that failed in either run) and
reports `records_compared`, `label_changes`, `label_change_rate`,
`mean_score_delta`, `mean_confidence_delta` and the list of changed records.
It describes how much the results moved; it does not judge whether RAG
helped. To judge that, compare both runs against labelled data with the
evaluation endpoints.

## Limitations

- Chunking is character-based, not token-aware or markdown-aware.
- Only UTF-8 text uploads are accepted (`415` otherwise).
- Namespaces are labels, not isolation (see above).
- One writer process per store: run a single API worker and stop it before
  bootstrapping.
