"""Finds papers/chunks that support or contradict a claim, via retrieval + LLM judgment."""
from typing import Dict

from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import (
    CLAIM_VERIFICATION_SYSTEM_PROMPT,
    CLAIM_VERIFICATION_USER_TEMPLATE,
    format_excerpts,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ClaimVerifier:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker, llm: LLMClient = None):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm or LLMClient()

    def verify(self, claim: str, top_k: int = 10) -> Dict:
        candidates = self.retriever.retrieve(claim)
        reranked = self.reranker.rerank(claim, candidates, top_k=top_k)

        excerpts = [
            {
                "text": c["payload"].get("text", ""),
                "doc_title": c["payload"].get("doc_title", c["payload"].get("doc_id")),
                "section": c["payload"].get("section"),
                "page": c["payload"].get("page_start"),
            }
            for c in reranked
        ]

        user_prompt = CLAIM_VERIFICATION_USER_TEMPLATE.format(claim=claim, excerpts=format_excerpts(excerpts))
        answer = self.llm.generate(CLAIM_VERIFICATION_SYSTEM_PROMPT, user_prompt)

        return {"claim": claim, "verdict_explanation": answer, "sources": excerpts}
