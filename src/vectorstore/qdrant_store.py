"""Qdrant-backed vector store."""
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.utils.config import settings
from src.utils.logging_config import get_logger
from src.vectorstore.base import VectorStoreBase

logger = get_logger(__name__)


class QdrantStore(VectorStoreBase):
    def __init__(self, collection_name: str = None, vector_size: int = 768, host: str = None, port: int = None):
        self.collection_name = collection_name or settings.qdrant_collection
        self.client = QdrantClient(host=host or settings.qdrant_host, port=port or settings.qdrant_port)
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={self.vector_size})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=self.vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, ids: List[str], vectors: np.ndarray, payloads: List[Dict]) -> None:
        # Qdrant point IDs must be int or UUID; we hash string chunk_ids to a stable UUID.
        import uuid

        points = []
        for i, (cid, vec, payload) in enumerate(zip(ids, vectors, payloads)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, cid))
            payload = dict(payload)
            payload["chunk_id"] = cid
            points.append(qmodels.PointStruct(id=point_id, vector=vec.tolist(), payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: np.ndarray, top_k: int, filter: Optional[Dict] = None) -> List[Dict]:
        qfilter = None
        if filter:
            must = [qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)) for k, v in filter.items()]
            qfilter = qmodels.Filter(must=must)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qfilter,
        )
        return [
            {"id": r.payload.get("chunk_id", str(r.id)), "score": r.score, "payload": r.payload}
            for r in results
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count
