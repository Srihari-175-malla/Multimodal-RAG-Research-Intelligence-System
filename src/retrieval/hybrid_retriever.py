"""
Hybrid retrieval: fuses dense (vector) and sparse (BM25) rankings using
Reciprocal Rank Fusion (RRF), which is robust to the very different score
scales the two retrievers produce.

RRF(d) = sum_over_retrievers( 1 / (k + rank_of_d) )
"""
from typing import Dict, List

from src.embeddings.embedder import Embedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.config import settings
from src.utils.logging_config import get_logger
from src.vectorstore.base import VectorStoreBase

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStoreBase,
        bm25: BM25Retriever,
        embedder: Embedder = None,
        rrf_k: int = None,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder or Embedder.get()
        self.rrf_k = rrf_k or settings.rrf_k

    def retrieve(self, query: str, dense_top_k: int = None, bm25_top_k: int = None) -> List[Dict]:
        dense_top_k = dense_top_k or settings.dense_top_k
        bm25_top_k = bm25_top_k or settings.bm25_top_k

        query_vec = self.embedder.embed([query], is_query=True)[0]
        dense_results = self.vector_store.search(query_vec, top_k=dense_top_k)
        sparse_results = self.bm25.search(query, top_k=bm25_top_k)

        fused = self._reciprocal_rank_fusion([dense_results, sparse_results])
        logger.info(
            f"Hybrid retrieval for query='{query[:50]}...': "
            f"{len(dense_results)} dense, {len(sparse_results)} sparse -> {len(fused)} fused"
        )
        return fused

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Dict]]) -> List[Dict]:
        scores: Dict[str, float] = {}
        payloads: Dict[str, Dict] = {}

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                cid = item["id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                payloads[cid] = item["payload"]

        fused = [
            {"id": cid, "score": score, "payload": payloads[cid]}
            for cid, score in scores.items()
        ]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused
