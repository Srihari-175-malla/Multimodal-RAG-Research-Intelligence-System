"""
Cross-encoder reranker: scores (query, passage) pairs jointly, which is far
more accurate than bi-encoder cosine similarity for the final top-N cut,
at the cost of being too slow to run over the whole corpus.
"""
from typing import Dict, List

from sentence_transformers import CrossEncoder

from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Reranker:
    _instance = None

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model
        logger.info(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    @classmethod
    def get(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = Reranker()
        return cls._instance

    def rerank(self, query: str, candidates: List[Dict], top_k: int = None) -> List[Dict]:
        if not candidates:
            return []
        top_k = top_k or settings.rerank_top_k
        pairs = [(query, c["payload"].get("text", "")) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]
