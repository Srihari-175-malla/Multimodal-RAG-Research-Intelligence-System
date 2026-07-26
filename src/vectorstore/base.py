"""Abstract interface all vector store backends implement."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class VectorStoreBase(ABC):
    @abstractmethod
    def upsert(self, ids: List[str], vectors: np.ndarray, payloads: List[Dict]) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int, filter: Optional[Dict] = None) -> List[Dict]:
        """Returns list of {"id": str, "score": float, "payload": dict}"""
        ...

    @abstractmethod
    def count(self) -> int:
        ...
