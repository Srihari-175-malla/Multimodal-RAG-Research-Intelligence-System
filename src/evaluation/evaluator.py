"""
Evaluation harness for the RAG pipeline.

Retrieval metrics (need gold chunk_ids per question):
  - Precision@k, Recall@k, MRR

Generation metrics (LLM-as-judge, no gold answer needed):
  - Faithfulness: is every claim in the answer grounded in the retrieved excerpts?
  - Answer relevance: does the answer address the question?

Expected qa_set.jsonl format (one JSON object per line):
{"question": "...", "relevant_chunk_ids": ["doc::text::3", ...]}
"relevant_chunk_ids" is optional; if omitted, only generation metrics are computed.
"""
import argparse
import json
from typing import Dict, List

from src.generation.llm_client import LLMClient
from src.pipeline import ResearchRAGPipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an evaluator for a RAG system. Given a question, the \
retrieved excerpts, and the generated answer, score the answer on two axes from 1-5:
- faithfulness: does every claim in the answer trace back to the excerpts (5) or does \
it contain unsupported/hallucinated claims (1)?
- relevance: does the answer directly address the question (5) or is it off-topic (1)?
Respond ONLY as JSON: {"faithfulness": <int>, "relevance": <int>, "reasoning": "<short>"}"""


def precision_recall_mrr(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Dict[str, float]:
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = [1 if rid in relevant_set else 0 for rid in retrieved_k]

    precision = sum(hits) / k if k else 0.0
    recall = sum(hits) / len(relevant_set) if relevant_set else 0.0

    mrr = 0.0
    for rank, rid in enumerate(retrieved_k, start=1):
        if rid in relevant_set:
            mrr = 1.0 / rank
            break

    return {"precision@k": precision, "recall@k": recall, "mrr": mrr}


class Evaluator:
    def __init__(self, pipeline: ResearchRAGPipeline = None, llm: LLMClient = None, k: int = 8):
        self.pipeline = pipeline or ResearchRAGPipeline()
        self.llm = llm or LLMClient()
        self.k = k

    def evaluate_dataset(self, qa_path: str) -> Dict:
        rows = [json.loads(line) for line in open(qa_path) if line.strip()]
        retrieval_scores, judge_scores = [], []

        for row in rows:
            question = row["question"]
            candidates = self.pipeline.retriever.retrieve(question)
            reranked = self.pipeline.reranker.rerank(question, candidates, top_k=self.k)
            retrieved_ids = [c["id"] for c in reranked]

            if "relevant_chunk_ids" in row:
                retrieval_scores.append(precision_recall_mrr(retrieved_ids, row["relevant_chunk_ids"], self.k))

            result = self.pipeline.query(question, top_k=self.k)
            judge_scores.append(self._judge(question, result["sources"], result["answer"]))

        return {
            "num_questions": len(rows),
            "retrieval": self._avg(retrieval_scores) if retrieval_scores else None,
            "generation": self._avg(judge_scores),
        }

    def _judge(self, question: str, sources: List[Dict], answer: str) -> Dict:
        from src.generation.prompt_templates import format_excerpts

        prompt = (
            f"Question: {question}\n\nExcerpts:\n{format_excerpts(sources)}\n\nGenerated answer:\n{answer}"
        )
        raw = self.llm.generate(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=300)
        try:
            cleaned = raw.strip().strip("```").replace("json", "", 1).strip()
            parsed = json.loads(cleaned)
        except Exception:
            logger.warning(f"Could not parse judge output: {raw[:200]}")
            parsed = {"faithfulness": None, "relevance": None}
        return parsed

    @staticmethod
    def _avg(dict_list: List[Dict]) -> Dict:
        keys = {k for d in dict_list for k, v in d.items() if isinstance(v, (int, float))}
        return {k: sum(d.get(k, 0) or 0 for d in dict_list) / len(dict_list) for k in keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-set", required=True, help="Path to qa_set.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    evaluator = Evaluator(k=args.top_k)
    results = evaluator.evaluate_dataset(args.qa_set)
    print(json.dumps(results, indent=2))
