"""Unit tests for hybrid retrieval fusion logic (RRF), without loading real models."""
from unittest.mock import MagicMock

import numpy as np

from src.retrieval.hybrid_retriever import HybridRetriever


def test_reciprocal_rank_fusion_prioritizes_items_in_both_lists():
    fake_vector_store = MagicMock()
    fake_bm25 = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = np.array([[0.1, 0.2, 0.3]])

    retriever = HybridRetriever(fake_vector_store, fake_bm25, fake_embedder, rrf_k=60)

    dense_results = [
        {"id": "a", "score": 0.9, "payload": {"text": "A"}},
        {"id": "b", "score": 0.8, "payload": {"text": "B"}},
    ]
    sparse_results = [
        {"id": "b", "score": 5.0, "payload": {"text": "B"}},
        {"id": "c", "score": 4.0, "payload": {"text": "C"}},
    ]

    fused = retriever._reciprocal_rank_fusion([dense_results, sparse_results])
    fused_ids = [f["id"] for f in fused]

    # "b" appears in both lists -> should rank first
    assert fused_ids[0] == "b"
    assert set(fused_ids) == {"a", "b", "c"}


def test_hybrid_retrieve_calls_both_backends():
    fake_vector_store = MagicMock()
    fake_vector_store.search.return_value = [{"id": "a", "score": 0.9, "payload": {"text": "A"}}]
    fake_bm25 = MagicMock()
    fake_bm25.search.return_value = [{"id": "a", "score": 3.0, "payload": {"text": "A"}}]
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = np.array([[0.1, 0.2, 0.3]])

    retriever = HybridRetriever(fake_vector_store, fake_bm25, fake_embedder)
    results = retriever.retrieve("test query", dense_top_k=5, bm25_top_k=5)

    fake_vector_store.search.assert_called_once()
    fake_bm25.search.assert_called_once()
    assert len(results) == 1
    assert results[0]["id"] == "a"
