"""
BM25 sparse retriever over the full chunk corpus, kept in memory.
Persisted alongside the FAISS/Qdrant index so restarts don't require
re-tokenizing every chunk.
"""
import pickle
import re
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    def __init__(self, persist_path: str = "data/index/bm25.pkl"):
        self.persist_path = Path(persist_path)
        self.chunk_ids: List[str] = []
        self.payloads: List[Dict] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None

        if self.persist_path.exists():
            self._load()

    def add(self, chunk_ids: List[str], texts: List[str], payloads: List[Dict]):
        self.chunk_ids.extend(chunk_ids)
        self.payloads.extend(payloads)
        self.corpus_tokens.extend(_tokenize(t) for t in texts)
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self._persist()

    def search(self, query: str, top_k: int = 25) -> List[Dict]:
        if self.bm25 is None or not self.chunk_ids:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"id": self.chunk_ids[i], "score": float(scores[i]), "payload": self.payloads[i]}
            for i in ranked
            if scores[i] > 0
        ]

    def _persist(self):
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "payloads": self.payloads, "corpus_tokens": self.corpus_tokens}, f
            )

    def _load(self):
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.payloads = data["payloads"]
        self.corpus_tokens = data["corpus_tokens"]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
