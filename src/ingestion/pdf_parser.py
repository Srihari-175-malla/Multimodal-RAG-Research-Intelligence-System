"""
PDF text + layout extraction using PyMuPDF.

Extracts per-page text blocks along with font-size, which is used downstream
by the chunker to detect section headings without any ML model.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TextBlock:
    page: int
    text: str
    font_size: float
    is_heading: bool = False


@dataclass
class ParsedDocument:
    doc_id: str
    source_path: str
    num_pages: int
    blocks: List[TextBlock] = field(default_factory=list)
    raw_text: str = ""


class PDFParser:
    """Extracts structured text blocks from a PDF, flagging probable headings."""

    def __init__(self, heading_font_ratio: float = 1.15):
        # A block is treated as a heading if its font size exceeds the
        # document's median body font size by this ratio.
        self.heading_font_ratio = heading_font_ratio

    def parse(self, pdf_path: str, doc_id: str) -> ParsedDocument:
        pdf_path = str(pdf_path)
        doc = fitz.open(pdf_path)
        blocks: List[TextBlock] = []
        font_sizes: List[float] = []

        raw_blocks = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # not a text block (e.g. image)
                    continue
                block_text_parts = []
                sizes = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if txt:
                            block_text_parts.append(txt)
                            sizes.append(span.get("size", 10.0))
                if not block_text_parts:
                    continue
                text = " ".join(block_text_parts).strip()
                avg_size = sum(sizes) / len(sizes) if sizes else 10.0
                font_sizes.append(avg_size)
                raw_blocks.append((page_idx, text, avg_size))

        median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0
        heading_threshold = median_size * self.heading_font_ratio

        for page_idx, text, avg_size in raw_blocks:
            is_heading = avg_size >= heading_threshold and len(text.split()) <= 15
            blocks.append(TextBlock(page=page_idx, text=text, font_size=avg_size, is_heading=is_heading))

        raw_text = "\n".join(b.text for b in blocks)
        doc.close()

        logger.info(f"Parsed {pdf_path}: {len(doc)} pages, {len(blocks)} text blocks")
        return ParsedDocument(
            doc_id=doc_id,
            source_path=pdf_path,
            num_pages=len(blocks) and max(b.page for b in blocks) + 1,
            blocks=blocks,
            raw_text=raw_text,
        )
