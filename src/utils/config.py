"""Load YAML config + environment variables into a single settings object."""
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def _deep_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class Settings:
    """Thin wrapper around config.yaml with env var overrides."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        # Environment overrides (env wins over yaml for secrets/deployment knobs)
        self.vector_backend = os.getenv("VECTOR_BACKEND", self._cfg.get("vector_backend", "qdrant"))
        self.qdrant_host = os.getenv("QDRANT_HOST", _deep_get(self._cfg, "qdrant", "host", default="localhost"))
        self.qdrant_port = int(os.getenv("QDRANT_PORT", _deep_get(self._cfg, "qdrant", "port", default=6333)))
        self.qdrant_collection = _deep_get(self._cfg, "qdrant", "collection", default="research_papers")

        self.faiss_index_path = ROOT_DIR / _deep_get(self._cfg, "faiss", "index_path", default="data/index/faiss.index")
        self.faiss_metadata_path = ROOT_DIR / _deep_get(self._cfg, "faiss", "metadata_path", default="data/index/faiss_meta.pkl")

        self.embedding_model = os.getenv("EMBEDDING_MODEL", _deep_get(self._cfg, "embedding", "model_name"))
        self.embedding_batch_size = _deep_get(self._cfg, "embedding", "batch_size", default=32)
        self.embedding_normalize = _deep_get(self._cfg, "embedding", "normalize", default=True)

        self.chunk_max_tokens = _deep_get(self._cfg, "chunking", "max_tokens", default=400)
        self.chunk_overlap_tokens = _deep_get(self._cfg, "chunking", "overlap_tokens", default=60)
        self.chunk_min_tokens = _deep_get(self._cfg, "chunking", "min_tokens", default=40)

        self.dense_top_k = _deep_get(self._cfg, "retrieval", "dense_top_k", default=25)
        self.bm25_top_k = _deep_get(self._cfg, "retrieval", "bm25_top_k", default=25)
        self.rrf_k = _deep_get(self._cfg, "retrieval", "rrf_k", default=60)
        self.final_top_k = _deep_get(self._cfg, "retrieval", "final_top_k", default=8)

        self.reranker_model = os.getenv("RERANKER_MODEL", _deep_get(self._cfg, "reranker", "model_name"))
        self.rerank_top_k = _deep_get(self._cfg, "reranker", "rerank_top_k", default=8)

        self.llm_provider = _deep_get(self._cfg, "llm", "provider", default="anthropic")
        self.llm_model = os.getenv("LLM_MODEL", _deep_get(self._cfg, "llm", "model", default="claude-sonnet-5"))
        self.llm_max_tokens = _deep_get(self._cfg, "llm", "max_tokens", default=1500)
        self.llm_temperature = _deep_get(self._cfg, "llm", "temperature", default=0.2)

        self.raw_papers_dir = ROOT_DIR / _deep_get(self._cfg, "paths", "raw_papers", default="data/raw")
        self.processed_dir = ROOT_DIR / _deep_get(self._cfg, "paths", "processed", default="data/processed")

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


settings = Settings()
