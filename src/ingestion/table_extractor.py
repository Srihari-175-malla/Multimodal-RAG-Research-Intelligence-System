"""
Table extraction using pdfplumber (lightweight, no ghostscript dependency
required at import time — falls back gracefully if a page has no tables).
"""
from dataclasses import dataclass
from typing import List

import pdfplumber

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedTable:
    page: int
    table_index: int
    caption: str
    markdown: str
    raw_rows: List[List[str]]


class TableExtractor:
    def extract(self, pdf_path: str) -> List[ExtractedTable]:
        tables: List[ExtractedTable] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                page_text = page.extract_text() or ""
                for t_idx, raw in enumerate(page_tables):
                    if not raw or len(raw) < 2:
                        continue
                    caption = self._find_caption(page_text, t_idx)
                    md = self._to_markdown(raw)
                    tables.append(
                        ExtractedTable(
                            page=page_idx,
                            table_index=t_idx,
                            caption=caption,
                            markdown=md,
                            raw_rows=raw,
                        )
                    )
        logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
        return tables

    @staticmethod
    def _find_caption(page_text: str, table_index: int) -> str:
        """Heuristic: look for a line starting with 'Table N' near the table."""
        for line in page_text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("table"):
                return stripped
        return f"Table {table_index + 1}"

    @staticmethod
    def _to_markdown(rows: List[List[str]]) -> str:
        clean_rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows]
        header, *body = clean_rows
        md_lines = ["| " + " | ".join(header) + " |"]
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in body:
            row = row + [""] * (len(header) - len(row))  # pad ragged rows
            md_lines.append("| " + " | ".join(row[: len(header)]) + " |")
        return "\n".join(md_lines)
