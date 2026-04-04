"""
Vector store abstraction for RAG enrichment.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:
    """
    Lightweight semantic store for sentiment-related domain knowledge.
    """

    def __init__(self, persist_dir: str = ".chroma"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.store = Chroma(
            collection_name="sentiment_context",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    def search(self, query: str, k: int = 3):
        """
        Perform semantic similarity search.
        """
        return self.store.similarity_search(query, k=k)
