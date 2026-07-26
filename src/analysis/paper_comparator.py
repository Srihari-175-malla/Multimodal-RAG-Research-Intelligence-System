"""Compares multiple papers on a given aspect using hybrid retrieval + LLM."""
from typing import Dict, List

from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import (
    COMPARISON_SYSTEM_PROMPT,
    COMPARISON_USER_TEMPLATE,
    format_excerpts,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PaperComparator:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker, llm: LLMClient = None):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm or LLMClient()

    def compare(self, paper_ids: List[str], aspect: str, per_paper_k: int = 4) -> Dict:
        all_chunks = []
        for pid in paper_ids:
            candidates = self.retriever.retrieve(f"{aspect}", dense_top_k=15, bm25_top_k=15)
            paper_candidates = [c for c in candidates if c["payload"].get("doc_id") == pid]
            if not paper_candidates:
                logger.warning(f"No retrieved chunks for paper_id={pid} on aspect='{aspect}'")
                continue
            reranked = self.reranker.rerank(aspect, paper_candidates, top_k=per_paper_k)
            all_chunks.extend(reranked)

        excerpts = [
            {
                "text": c["payload"].get("text", ""),
                "doc_title": c["payload"].get("doc_title", c["payload"].get("doc_id")),
                "section": c["payload"].get("section"),
                "page": c["payload"].get("page_start"),
            }
            for c in all_chunks
        ]

        user_prompt = COMPARISON_USER_TEMPLATE.format(aspect=aspect, excerpts=format_excerpts(excerpts))
        answer = self.llm.generate(COMPARISON_SYSTEM_PROMPT, user_prompt)

        return {"aspect": aspect, "paper_ids": paper_ids, "answer": answer, "sources": excerpts}
