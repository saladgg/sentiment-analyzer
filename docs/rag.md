# RAG: Retrieval-Augmented Sentiment

RAG enrichment is **optional** and **read-only** — it never performs sentiment
inference itself. It retrieves domain context for a record's text and hands
that context to the sentiment engine. Today, only the LLM engine consumes the
context (see [engines.md](engines.md)).

The full pipeline gates on `ASA_RAG_ENABLED`. When disabled, the enricher is
not constructed and `AnalysisResponse.rag` is `null`.

## Components

| File                                                                          | Role                                                              |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [`rag/vector_store.py`](../sentiment_analyzer/rag/vector_store.py)            | Persistent ChromaDB wrapper. Add / query / delete / list.         |
| [`rag/context_enrichment.py`](../sentiment_analyzer/rag/context_enrichment.py)| Orchestrator — calls the store, builds a `RAGResult`.             |
| [`rag/models.py`](../sentiment_analyzer/rag/models.py)                        | `RAGChunk`, `RAGResult` dataclasses (the canonical types).        |
| [`rag/types.py`](../sentiment_analyzer/rag/types.py)                          | Legacy `RAGResult` alternative; kept for compatibility.           |
| [`rag/store.py`](../sentiment_analyzer/rag/store.py)                          | LangChain-based `Chroma` wrapper. Used by older integrations.     |
| [`api/rag_routes.py`](../sentiment_analyzer/api/rag_routes.py)                | HTTP endpoints — upload, list, delete, retrieve.                  |
| [`scripts/bootstrap_rag_store.py`](../sentiment_analyzer/scripts/bootstrap_rag_store.py) | Seeds the store with starter documents.                |

## Vector store

The default backend is local ChromaDB with the
`sentence-transformers/all-MiniLM-L6-v2` embedding model. No network access
is required to embed or retrieve.

The store persists at the path configured by `ASA_RAG_DB_PATH` (used by the
API at runtime) and `ASA_VECTOR_STORE_PATH` (used by the bootstrap script). If
you bootstrap the store, set both to the same path so the API can find the
seeded documents.

### Document model

Each upload becomes one logical document. A document has:

- A `document_id` (server-assigned UUID).
- A `namespace` (default: `"default"`) — used to isolate tenants, projects,
  or experiments.
- A `source` — the original filename or label, propagated to retrieval
  results.
- One or more chunks. The current chunker is a simple fixed-size splitter at
  500 chars (see
  [`api/rag_routes.py`](../sentiment_analyzer/api/rag_routes.py)).

### Add path

```python
store.add(
    texts=chunks,
    document_id=document_id,
    namespace="my-namespace",
    source="policy.txt",
)
```

Returns the number of chunks added. Each chunk is stored under the id
`{document_id}:{chunk_index}` so deleting a document deletes all its chunks.

### Query path

```python
store.query(query="refund delays", top_k=5, namespace="default")
```

Returns the raw Chroma response (`documents`, `metadatas`, `distances` lists).
The enricher converts that into a `RAGResult`:

```python
RAGResult(
    enabled=True,
    context="\n".join(chunks),
    sources=[unique source labels],
    chunks=[RAGChunk(content, source, score=1-distance), ...],
)
```

`score` is `1 - distance`, so higher is better.

## Enrichment in the analysis pipeline

In [`AnalysisService.run`](../sentiment_analyzer/services/analysis_service.py):

```python
if self.enricher:
    rag_result = self.enricher.enrich(text)
    if rag_result and rag_result.context:
        rag_used = True
        rag_source_count += len(rag_result.sources)

sentiment_output = engine.analyze(
    text=text,
    context=rag_result.context if rag_result else None,
)
```

`rag_used` flips to `True` the first time *any* record gets non-empty context.
That flag, plus `rag_source_count`, surfaces in the response as `RAGMetadata`:

```json
{
  "enabled": true,
  "used": true,
  "source_count": 7
}
```

`enabled` mirrors `ASA_RAG_ENABLED`. `used` is `true` only if at least one
record actually received context — useful when the store is empty or the
namespace is wrong.

## HTTP surface

Full schemas live in [api-reference.md](api-reference.md#rag). Quick map:

| Endpoint                              | Purpose                                                |
| ------------------------------------- | ------------------------------------------------------ |
| `POST /api/rag/documents`             | Upload a text file; chunks it and embeds it.           |
| `GET /api/rag/documents`              | List unique uploaded documents.                        |
| `DELETE /api/rag/documents/{id}`      | Delete a document and all its chunks.                  |
| `POST /api/rag/retrieve`              | Preview retrieval for a query (does **not** affect runs). |

The retrieve preview is for debugging — use it before running an analysis to
verify your store has the right context.

## Bootstrapping the store

Tiny one-shot seeder:

```bash
uv run python -m sentiment_analyzer.scripts.bootstrap_rag_store
```

It writes a couple of starter documents into `ASA_VECTOR_STORE_PATH`. Treat
this as scaffolding — replace it with your real domain documents.

## Comparing RAG impact

`POST /api/runs/compare` (in
[`api/run_routes.py`](../sentiment_analyzer/api/run_routes.py)) takes two run
ids — typically a baseline (no RAG) and a RAG-enriched run — and returns:

```json
{
  "comparison": {
    "confidence_gain": 0.12,
    "label_changes": 3,
    "rag_helpfulness": true
  }
}
```

The flag `rag_helpfulness` is `true` when `confidence_gain > 0.05` or any
labels changed. Logic lives in
[`analytics/run_comparison.py`](../sentiment_analyzer/analytics/run_comparison.py).
This is intentionally simple — it is meant to flag *whether* RAG mattered, not
to be a final scientific judgement.

## Limitations and gotchas

- Chunking is fixed-size at 500 chars. There is no overlap, no token-aware
  splitting, and no markdown-aware splitting. For most short policy documents
  this is fine; for long-form text you will want a smarter chunker.
- The two `RAGResult` definitions in
  [`rag/models.py`](../sentiment_analyzer/rag/models.py) and
  [`rag/types.py`](../sentiment_analyzer/rag/types.py) are not unified. The
  enricher uses `models.RAGResult` — if you write new code, prefer that one.
- `rag/store.py` is a separate LangChain-based `Chroma` wrapper. Live API
  paths route through `rag/vector_store.py`. Don't confuse the two.
- `ASA_VECTOR_STORE_PATH` and `ASA_RAG_DB_PATH` should usually match if you
  use the bootstrap script.
