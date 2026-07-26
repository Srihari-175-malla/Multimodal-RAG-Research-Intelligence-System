"""
Lightweight metadata extraction: title, authors, abstract, and (if resolvable)
arXiv ID, using PDF metadata + heuristics over the first page's text blocks.
No network calls are made by default so ingestion works fully offline.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF

from src.ingestion.pdf_parser import ParsedDocument
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ARXIV_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
ABSTRACT_RE = re.compile(r"^abstract\b", re.IGNORECASE)


@dataclass
class PaperMetadata:
    doc_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    arxiv_id: Optional[str] = None
    year: Optional[int] = None
    num_pages: int = 0
    source_path: str = ""


class MetadataExtractor:
    def extract(self, pdf_path: str, parsed: ParsedDocument) -> PaperMetadata:
        doc = fitz.open(pdf_path)
        pdf_meta = doc.metadata or {}
        doc.close()

        title = self._extract_title(parsed) or pdf_meta.get("title") or parsed.doc_id
        authors = self._extract_authors(pdf_meta)
        abstract = self._extract_abstract(parsed)
        arxiv_id = self._extract_arxiv_id(parsed.raw_text)
        year = self._extract_year(pdf_meta, parsed.raw_text)

        return PaperMetadata(
            doc_id=parsed.doc_id,
            title=title.strip(),
            authors=authors,
            abstract=abstract.strip(),
            arxiv_id=arxiv_id,
            year=year,
            num_pages=parsed.num_pages,
            source_path=parsed.source_path,
        )

    @staticmethod
    def _extract_title(parsed: ParsedDocument) -> Optional[str]:
        # Title heuristic: largest-font heading block on page 0.
        page0_blocks = [b for b in parsed.blocks if b.page == 0]
        if not page0_blocks:
            return None
        page0_blocks.sort(key=lambda b: b.font_size, reverse=True)
        return page0_blocks[0].text

    @staticmethod
    def _extract_authors(pdf_meta: dict) -> List[str]:
        author_str = pdf_meta.get("author", "")
        if not author_str:
            return []
        parts = re.split(r",| and ", author_str)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _extract_abstract(parsed: ParsedDocument) -> str:
        collecting = False
        collected = []
        for block in parsed.blocks:
            if block.page > 1:
                break
            if ABSTRACT_RE.match(block.text.strip()):
                collecting = True
                continue
            if collecting:
                if block.is_heading:
                    break
                collected.append(block.text)
        return " ".join(collected)[:2000]

    @staticmethod
    def _extract_arxiv_id(raw_text: str) -> Optional[str]:
        m = ARXIV_RE.search(raw_text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_year(pdf_meta: dict, raw_text: str) -> Optional[int]:
        creation_date = pdf_meta.get("creationDate", "")
        m = re.search(r"D:(\d{4})", creation_date)
        if m:
            return int(m.group(1))
        m2 = re.search(r"\b(19|20)\d{2}\b", raw_text[:3000])
        return int(m2.group(0)) if m2 else None
