"""
Section-aware, token-bounded chunking.

Strategy:
1. Walk text blocks in order; a heading block starts a new "section".
2. Within a section, accumulate blocks until max_tokens is reached, then
   emit a chunk with `overlap_tokens` carried over into the next chunk so
   context isn't lost across a chunk boundary.
3. Tables and figures are chunked separately (one chunk per table/figure)
   so they can be retrieved directly and cited precisely.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken

from src.ingestion.pdf_parser import ParsedDocument
from src.ingestion.table_extractor import ExtractedTable
from src.ingestion.figure_extractor import ExtractedFigure
from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str  # "text" | "table" | "figure"
    section: Optional[str] = None
    page_start: int = 0
    page_end: int = 0
    metadata: dict = field(default_factory=dict)


class Chunker:
    def __init__(
        self,
        max_tokens: int = None,
        overlap_tokens: int = None,
        min_tokens: int = None,
    ):
        self.max_tokens = max_tokens or settings.chunk_max_tokens
        self.overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens
        self.min_tokens = min_tokens or settings.chunk_min_tokens

    def chunk_document(
        self,
        parsed: ParsedDocument,
        tables: List[ExtractedTable] = None,
        figures: List[ExtractedFigure] = None,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunks.extend(self._chunk_text_blocks(parsed))
        if tables:
            chunks.extend(self._chunk_tables(parsed.doc_id, tables))
        if figures:
            chunks.extend(self._chunk_figures(parsed.doc_id, figures))
        logger.info(f"{parsed.doc_id}: produced {len(chunks)} chunks "
                    f"({sum(1 for c in chunks if c.chunk_type=='text')} text, "
                    f"{sum(1 for c in chunks if c.chunk_type=='table')} table, "
                    f"{sum(1 for c in chunks if c.chunk_type=='figure')} figure)")
        return chunks

    def _chunk_text_blocks(self, parsed: ParsedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        current_section = "Preamble"
        buffer_texts: List[str] = []
        buffer_tokens = 0
        page_start = None
        page_end = None
        chunk_counter = 0

        def flush(carry_overlap: bool = True):
            nonlocal buffer_texts, buffer_tokens, page_start, page_end, chunk_counter
            if not buffer_texts:
                return
            text = " ".join(buffer_texts).strip()
            if _count_tokens(text) < self.min_tokens and chunks:
                # too small to stand alone — merge into previous chunk
                prev = chunks[-1]
                prev.text = (prev.text + " " + text).strip()
                prev.page_end = page_end
            else:
                chunk_counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{parsed.doc_id}::text::{chunk_counter}",
                        doc_id=parsed.doc_id,
                        text=text,
                        chunk_type="text",
                        section=current_section,
                        page_start=page_start if page_start is not None else 0,
                        page_end=page_end if page_end is not None else 0,
                    )
                )
            if carry_overlap and text:
                overlap_words = text.split()
                # approximate overlap by token budget on the tail of the text
                tail = ""
                for w in reversed(overlap_words):
                    candidate = (w + " " + tail).strip()
                    if _count_tokens(candidate) > self.overlap_tokens:
                        break
                    tail = candidate
                buffer_texts = [tail] if tail else []
                buffer_tokens = _count_tokens(tail)
            else:
                buffer_texts = []
                buffer_tokens = 0

        for block in parsed.blocks:
            if block.is_heading:
                flush(carry_overlap=False)
                current_section = block.text
                page_start = block.page
                page_end = block.page
                continue

            block_tokens = _count_tokens(block.text)
            if page_start is None:
                page_start = block.page
            page_end = block.page

            if buffer_tokens + block_tokens > self.max_tokens and buffer_texts:
                flush(carry_overlap=True)
                page_start = block.page

            buffer_texts.append(block.text)
            buffer_tokens += block_tokens

        flush(carry_overlap=False)
        return chunks

    @staticmethod
    def _chunk_tables(doc_id: str, tables: List[ExtractedTable]) -> List[Chunk]:
        chunks = []
        for i, t in enumerate(tables):
            text = f"{t.caption}\n{t.markdown}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::table::{i}",
                    doc_id=doc_id,
                    text=text,
                    chunk_type="table",
                    section=t.caption,
                    page_start=t.page,
                    page_end=t.page,
                    metadata={"caption": t.caption},
                )
            )
        return chunks

    @staticmethod
    def _chunk_figures(doc_id: str, figures: List[ExtractedFigure]) -> List[Chunk]:
        chunks = []
        for i, fg in enumerate(figures):
            text = f"{fg.caption} (image reference: {fg.image_path})"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::figure::{i}",
                    doc_id=doc_id,
                    text=text,
                    chunk_type="figure",
                    section=fg.caption,
                    page_start=fg.page,
                    page_end=fg.page,
                    metadata={"image_path": fg.image_path, "caption": fg.caption},
                )
            )
        return chunks
