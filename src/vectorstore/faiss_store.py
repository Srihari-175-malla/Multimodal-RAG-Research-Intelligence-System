"""
FAISS-backed local vector store (no external service required).
Metadata is kept in a parallel pickle file since FAISS only stores vectors.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from src.utils.config import settings
from src.utils.logging_config import get_logger
from src.vectorstore.base import VectorStoreBase

logger = get_logger(__name__)


class FaissStore(VectorStoreBase):
    def __init__(self, vector_size: int = 768, index_path: str = None, metadata_path: str = None):
        self.vector_size = vector_size
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.metadata_path = Path(metadata_path or settings.faiss_metadata_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.ids: List[str] = []
        self.payloads: List[Dict] = []

        if self.index_path.exists() and self.metadata_path.exists():
            logger.info("Loading existing FAISS index from disk")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                meta = pickle.load(f)
                self.ids = meta["ids"]
                self.payloads = meta["payloads"]
        else:
            # Inner product on normalized vectors == cosine similarity
            self.index = faiss.IndexFlatIP(vector_size)

    def upsert(self, ids: List[str], vectors: np.ndarray, payloads: List[Dict]) -> None:
        vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.ids.extend(ids)
        self.payloads.extend(payloads)
        self._persist()

    def search(self, query_vector: np.ndarray, top_k: int, filter: Optional[Dict] = None) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        query_vector = query_vector.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, min(top_k * 3 if filter else top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            payload = self.payloads[idx]
            if filter and not all(payload.get(k) == v for k, v in filter.items()):
                continue
            results.append({"id": self.ids[idx], "score": float(score), "payload": payload})
            if len(results) >= top_k:
                break
        return results

    def count(self) -> int:
        return self.index.ntotal

    def _persist(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump({"ids": self.ids, "payloads": self.payloads}, f)
