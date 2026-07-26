"""Formats extracted paper metadata into APA and BibTeX citations."""
import re
from typing import List

from src.ingestion.metadata_extractor import PaperMetadata


class CitationGenerator:
    @staticmethod
    def to_apa(meta: PaperMetadata) -> str:
        authors = CitationGenerator._format_authors_apa(meta.authors)
        year = meta.year or "n.d."
        title = meta.title.rstrip(".")
        suffix = f" arXiv:{meta.arxiv_id}" if meta.arxiv_id else ""
        return f"{authors} ({year}). {title}.{suffix}".strip()

    @staticmethod
    def to_bibtex(meta: PaperMetadata) -> str:
        key = CitationGenerator._bibtex_key(meta)
        authors = " and ".join(meta.authors) if meta.authors else "Unknown"
        entry_type = "misc" if meta.arxiv_id else "article"
        lines = [f"@{entry_type}{{{key},"]
        lines.append(f'  title = {{{meta.title}}},')
        lines.append(f'  author = {{{authors}}},')
        if meta.year:
            lines.append(f'  year = {{{meta.year}}},')
        if meta.arxiv_id:
            lines.append(f'  eprint = {{{meta.arxiv_id}}},')
            lines.append('  archivePrefix = {arXiv},')
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _format_authors_apa(authors: List[str]) -> str:
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            return authors[0]
        if len(authors) <= 20:
            return ", ".join(authors[:-1]) + f", & {authors[-1]}"
        return ", ".join(authors[:19]) + ", ... " + authors[-1]

    @staticmethod
    def _bibtex_key(meta: PaperMetadata) -> str:
        first_author_last = "unknown"
        if meta.authors:
            first_author_last = re.sub(r"\W+", "", meta.authors[0].split()[-1].lower())
        year = meta.year or "nd"
        return f"{first_author_last}{year}"
