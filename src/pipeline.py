"""
End-to-end orchestration: ingestion → chunking → embedding → indexing, and
query → hybrid retrieval → rerank → generation.

This is the single object the API / CLI scripts talk to.
"""
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from src.analysis.claim_verifier import ClaimVerifier
from src.analysis.paper_comparator import PaperComparator
from src.chunking.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, format_excerpts
from src.ingestion.figure_extractor import FigureExtractor
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.table_extractor import TableExtractor
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.utils.config import settings
from src.utils.logging_config import get_logger
from src.vectorstore.factory import get_vector_store

logger = get_logger(__name__)


class ResearchRAGPipeline:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.table_extractor = TableExtractor()
        self.figure_extractor = FigureExtractor()
        self.metadata_extractor = MetadataExtractor()
        self.chunker = Chunker()
        self.embedder = Embedder.get()

        self.vector_store = get_vector_store(vector_size=self.embedder.dimension)
        self.bm25 = BM25Retriever()
        self.retriever = HybridRetriever(self.vector_store, self.bm25, self.embedder)
        self.reranker = Reranker.get()
        self.llm = LLMClient()

        self.comparator = PaperComparator(self.retriever, self.reranker, self.llm)
        self.verifier = ClaimVerifier(self.retriever, self.reranker, self.llm)

        self._metadata_store: Dict[str, dict] = {}  # doc_id -> PaperMetadata dict

    # ---------------------------------------------------------------- ingest
    def ingest_pdf(self, pdf_path: str, doc_id: Optional[str] = None) -> Dict:
        doc_id = doc_id or Path(pdf_path).stem
        logger.info(f"Ingesting {pdf_path} as doc_id={doc_id}")

        parsed = self.pdf_parser.parse(pdf_path, doc_id)
        tables = self.table_extractor.extract(pdf_path)
        figures = self.figure_extractor.extract(pdf_path, doc_id)
        meta = self.metadata_extractor.extract(pdf_path, parsed)
        self._metadata_store[doc_id] = meta.__dict__

        chunks = self.chunker.chunk_document(parsed, tables=tables, figures=figures)
        if not chunks:
            logger.warning(f"No chunks produced for {pdf_path}")
            return {"doc_id": doc_id, "num_chunks": 0, "metadata": meta.__dict__}

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)

        payloads = [
            {
                "doc_id": c.doc_id,
                "doc_title": meta.title,
                "text": c.text,
                "chunk_type": c.chunk_type,
                "section": c.section,
                "page_start": c.page_start,
                "page_end": c.page_end,
            }
            for c in chunks
        ]
        chunk_ids = [c.chunk_id for c in chunks]

        self.vector_store.upsert(chunk_ids, vectors, payloads)
        self.bm25.add(chunk_ids, texts, payloads)

        logger.info(f"Ingested {doc_id}: {len(chunks)} chunks indexed")
        return {"doc_id": doc_id, "num_chunks": len(chunks), "metadata": meta.__dict__}

    def ingest_directory(self, directory: str) -> List[Dict]:
        results = []
        for pdf_path in sorted(Path(directory).glob("*.pdf")):
            try:
                results.append(self.ingest_pdf(str(pdf_path)))
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_path}: {e}")
        return results

    # ----------------------------------------------------------------- query
    def query(self, question: str, top_k: int = None) -> Dict:
        candidates = self.retriever.retrieve(question)
        reranked = self.reranker.rerank(question, candidates, top_k=top_k)

        excerpts = [
            {
                "text": c["payload"].get("text", ""),
                "doc_id": c["payload"].get("doc_id"),
                "doc_title": c["payload"].get("doc_title"),
                "section": c["payload"].get("section"),
                "page": c["payload"].get("page_start"),
            }
            for c in reranked
        ]

        user_prompt = QA_USER_TEMPLATE.format(question=question, excerpts=format_excerpts(excerpts))
        answer = self.llm.generate(QA_SYSTEM_PROMPT, user_prompt)

        return {"question": question, "answer": answer, "sources": excerpts}

    def compare_papers(self, paper_ids: List[str], aspect: str) -> Dict:
        return self.comparator.compare(paper_ids, aspect)

    def verify_claim(self, claim: str) -> Dict:
        return self.verifier.verify(claim)

    def stats(self) -> Dict:
        return {
            "num_indexed_chunks": self.vector_store.count(),
            "num_documents": len(self._metadata_store),
            "vector_backend": settings.vector_backend,
        }
