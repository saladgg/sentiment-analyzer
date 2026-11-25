"""
Bootstrap script for initializing the RAG vector store.

Populates the vector store with starter domain-context documents used
for optional RAG-based sentiment enrichment.

Usage:
    make rag-bootstrap
    python -m sentiment_analyzer.scripts.bootstrap_rag_store [--path DIR] [--namespace NS]

Design notes:
- Documents added here represent *shared domain knowledge*, not user input.
- Idempotent: each starter document has a stable ID and is replaced on
  re-run, so running it twice does not duplicate chunks.
- Writes to ``ASA_VECTOR_STORE_PATH``, the same store the API reads.
- Stop the API first: Chroma's on-disk store should have one writer.
"""

import argparse

from sentiment_analyzer.core.config import get_settings
from sentiment_analyzer.rag.vector_store import VectorStore

STARTER_DOCUMENTS = [
    {
        "id": "bootstrap-policy-v1",
        "text": "Refunds are processed within 30 days.",
        "source": "policy_v1",
    },
    {
        "id": "bootstrap-ops-guide",
        "text": "Delayed shipments often cause negative sentiment.",
        "source": "ops_guide",
    },
]


def bootstrap(store: VectorStore, namespace: str) -> int:
    """(Re)load the starter documents; returns the number of chunks written."""
    added = 0
    for doc in STARTER_DOCUMENTS:
        store.delete_document(doc["id"])
        added += store.add(
            texts=[doc["text"]],
            document_id=doc["id"],
            namespace=namespace,
            source=doc["source"],
        )
    return added


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Seed the RAG vector store.")
    parser.add_argument("--path", default=settings.vector_store_path)
    parser.add_argument("--namespace", default=settings.rag_namespace)
    args = parser.parse_args(argv)

    added = bootstrap(VectorStore(args.path), args.namespace)
    print(f"Loaded {added} chunks into {args.path} (namespace={args.namespace!r})")


if __name__ == "__main__":
    main()
