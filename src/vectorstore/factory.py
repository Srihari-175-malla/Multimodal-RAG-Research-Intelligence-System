"""Returns the configured vector store backend."""
from src.utils.config import settings
from src.vectorstore.base import VectorStoreBase


def get_vector_store(vector_size: int = 768) -> VectorStoreBase:
    if settings.vector_backend == "faiss":
        from src.vectorstore.faiss_store import FaissStore
        return FaissStore(vector_size=vector_size)
    from src.vectorstore.qdrant_store import QdrantStore
    return QdrantStore(vector_size=vector_size)
