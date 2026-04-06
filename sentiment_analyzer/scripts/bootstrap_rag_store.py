"""
Bootstrap script for initializing the RAG vector store.

This script populates the vector store with domain-specific
context documents used for optional RAG-based sentiment
context enrichment during analysis.

Intended usage:
- Run once (or on demand) during environment setup
- Required only when `rag_enabled = true` in application settings
- Safe to re-run if the vector store supports idempotent inserts

Design notes:
- Documents added here represent *shared domain knowledge*,
  not user-provided input.
- Context is retrieved at inference time via semantic similarity
  and injected as read-only enrichment into sentiment engines.
- This script is intentionally decoupled from the API to preserve
  determinism, security, and auditability.

Future direction:
- Introduce a secured, versioned API endpoint
  (e.g. `POST /api/rag/documents`) for managing context documents.
- Add document versioning and provenance tracking to support
  reproducible analyses and controlled updates.

This script should not be used in production request paths.
"""

from sentiment_analyzer.core.config import settings
from sentiment_analyzer.rag.vector_store import VectorStore

store = VectorStore(settings.vector_store_path)

documents = [
    {"text": "Refunds are processed within 30 days.", "source": "policy_v1"},
    {
        "text": "Delayed shipments often cause negative sentiment.",
        "source": "ops_guide",
    },
]

for doc in documents:
    store.add(
        texts=[doc["text"]],
        source=doc["source"],
    )
