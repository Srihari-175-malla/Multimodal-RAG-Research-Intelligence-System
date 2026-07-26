"""
Sentence-Transformers embedding wrapper with batching and normalization.
"""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Embedder:
    _instance = None  # simple singleton so the model loads once per process

    def __init__(self, model_name: str = None, batch_size: int = None, normalize: bool = None):
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize = normalize if normalize is not None else settings.embedding_normalize
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = Embedder()
        return cls._instance

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Embed a list of texts. BGE-style models benefit from a query
        instruction prefix at retrieval time; we add one when is_query=True.
        """
        if is_query and "bge" in self.model_name.lower():
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
        )
        return embeddings

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
