"""Unit tests for the section-aware chunker."""
from src.chunking.chunker import Chunker
from src.ingestion.pdf_parser import ParsedDocument, TextBlock


def make_parsed_doc():
    blocks = [
        TextBlock(page=0, text="1. Introduction", font_size=16, is_heading=True),
        TextBlock(page=0, text="This paper studies retrieval augmented generation. " * 20, font_size=10),
        TextBlock(page=1, text="2. Methods", font_size=16, is_heading=True),
        TextBlock(page=1, text="We use a hybrid retriever combining BM25 and dense vectors. " * 15, font_size=10),
    ]
    raw_text = "\n".join(b.text for b in blocks)
    return ParsedDocument(doc_id="test_doc", source_path="test.pdf", num_pages=2, blocks=blocks, raw_text=raw_text)


def test_chunker_creates_section_labeled_chunks():
    chunker = Chunker(max_tokens=100, overlap_tokens=10, min_tokens=5)
    parsed = make_parsed_doc()
    chunks = chunker.chunk_document(parsed)

    assert len(chunks) >= 2
    sections = {c.section for c in chunks}
    assert "1. Introduction" in sections
    assert "2. Methods" in sections


def test_chunker_respects_max_tokens_roughly():
    chunker = Chunker(max_tokens=50, overlap_tokens=5, min_tokens=5)
    parsed = make_parsed_doc()
    chunks = chunker.chunk_document(parsed)
    from src.chunking.chunker import _count_tokens

    for c in chunks:
        # allow some slack because of overlap carry-over and merge-if-too-small
        assert _count_tokens(c.text) <= 200
