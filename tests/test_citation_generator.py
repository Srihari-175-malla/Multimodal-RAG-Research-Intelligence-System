"""Unit tests for citation formatting."""
from src.generation.citation_generator import CitationGenerator
from src.ingestion.metadata_extractor import PaperMetadata


def test_apa_single_author():
    meta = PaperMetadata(doc_id="d1", title="A Study of RAG", authors=["Jane Doe"], year=2023)
    apa = CitationGenerator.to_apa(meta)
    assert "Jane Doe (2023)" in apa
    assert "A Study of RAG" in apa


def test_bibtex_key_uses_last_name_and_year():
    meta = PaperMetadata(doc_id="d1", title="A Study of RAG", authors=["Jane Doe"], year=2023)
    bibtex = CitationGenerator.to_bibtex(meta)
    assert bibtex.startswith("@article{doe2023,") or bibtex.startswith("@misc{doe2023,")
