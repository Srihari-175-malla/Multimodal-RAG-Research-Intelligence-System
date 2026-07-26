# Multimodal RAG Research Intelligence System

A production-grade Retrieval-Augmented Generation system for scientific literature.
It ingests research papers (PDF), extracts **text, tables, figures, and metadata**,
chunks content intelligently, embeds it, indexes it in a vector database, and answers
research questions using **hybrid retrieval (BM25 + dense) with cross-encoder reranking**.

Beyond basic Q&A it supports:
- **Multi-paper comparison** — synthesize how N papers differ on a topic
- **Claim verification** — find papers that support or contradict a claim
- **Citation generation** — APA / BibTeX generated from extracted metadata
- **Retrieval + generation evaluation** — precision/recall/MRR + LLM-judged faithfulness

## Architecture

```
Research Papers (PDF)
        │
        ▼
   PDF Parsing  ──────────────► Text │ Tables │ Figures │ Metadata
        │                          (src/ingestion/*)
        ▼
   Chunking (section-aware, token-bounded, overlapping)
        │                          (src/chunking/chunker.py)
        ▼
   Embeddings (Sentence-Transformers, batched)
        │                          (src/embeddings/embedder.py)
        ▼
   Vector DB (Qdrant, FAISS fallback)
        │                          (src/vectorstore/*)
        ▼
   Hybrid Retrieval (BM25 + dense, Reciprocal Rank Fusion)
        │                          (src/retrieval/hybrid_retriever.py)
        ▼
   Cross-Encoder Reranker
        │                          (src/retrieval/reranker.py)
        ▼
   LLM (Claude) ─── grounded answer + citations
        │                          (src/generation/*)
        ▼
   Answer + Sources
```

Additional analysis modules sit on top of the retrieval layer:
`src/analysis/paper_comparator.py`, `src/analysis/claim_verifier.py`.

## Tech stack

- **Parsing:** PyMuPDF (text/figures), pdfplumber/camelot (tables)
- **Chunking:** custom section-aware token chunker (tiktoken-based)
- **Embeddings:** `sentence-transformers` (default: `BAAI/bge-base-en-v1.5`)
- **Vector store:** Qdrant (Docker) with a FAISS local fallback
- **Sparse retrieval:** BM25 (`rank_bm25`)
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Anthropic Claude (`anthropic` SDK) — swappable via `src/generation/llm_client.py`
- **API:** FastAPI
- **UI:** Streamlit
- **Infra:** Docker + docker-compose

## Repository layout

```
config/                 YAML configuration
src/ingestion/          PDF → text/tables/figures/metadata
src/chunking/           Section-aware chunking
src/embeddings/         Embedding model wrapper
src/vectorstore/        Qdrant + FAISS backends
src/retrieval/          BM25, hybrid fusion, reranker
src/generation/         LLM client, prompts, citation formatting
src/analysis/           Paper comparison, claim verification
src/evaluation/         Retrieval + generation metrics
src/pipeline.py         End-to-end orchestration (ingest + query)
api/                    FastAPI app exposing the pipeline
frontend/app.py         Streamlit UI
scripts/                CLI entry points for bulk ingestion / index build
tests/                  Pytest unit tests
```

## Quickstart

### 1. Configure
```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env
```

### 2. Run with Docker (recommended — starts Qdrant + API + UI)
```bash
docker compose up --build
```
- API: http://localhost:8000/docs
- UI:  http://localhost:8501

### 3. Or run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# start a local Qdrant (or set VECTOR_BACKEND=faiss in config/config.yaml to skip this)
docker run -p 6333:6333 qdrant/qdrant

# bulk ingest a folder of PDFs
python scripts/ingest_papers.py --input data/raw --collection research_papers

# start the API
uvicorn api.main:app --reload --port 8000

# start the UI (separate terminal)
streamlit run frontend/app.py
```

## Example usage (API)

```bash
# Ask a research question
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What methods reduce catastrophic forgetting in continual learning?", "top_k": 8}'

# Compare papers
curl -X POST localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"paper_ids": ["arxiv:2005.11401", "arxiv:2104.08691"], "aspect": "retrieval strategy"}'

# Verify a claim
curl -X POST localhost:8000/verify_claim \
  -H "Content-Type: application/json" \
  -d '{"claim": "Larger batch sizes always improve contrastive pretraining."}'
```

## Evaluation

```bash
python -m src.evaluation.evaluator --qa-set data/eval/qa_set.jsonl
```
Reports retrieval Precision@k / Recall@k / MRR, and LLM-judged faithfulness +
answer relevance for the generation stage.

## Design notes

- **Chunking is section-aware**: headings detected from font-size heuristics are
  used as chunk boundaries first, then long sections are split by token budget
  with overlap, so a chunk never silently straddles "Methods" and "Results".
- **Tables and figures are indexed separately** with their captions and a
  back-reference to source page, so a hybrid query can retrieve a table directly
  and the generator can cite "Table 2, p.5".
- **Hybrid retrieval** blends BM25 (good for exact terminology/acronyms/model
  names) with dense retrieval (good for paraphrase/semantics) via Reciprocal
  Rank Fusion, then a cross-encoder reranks the fused candidates before they
  reach the LLM — this consistently outperforms either retriever alone on
  scientific text full of jargon and symbols.
- **Everything is swappable**: vector backend (Qdrant/FAISS), embedding model,
  reranker, and LLM are all set from `config/config.yaml` / `.env`.

## License
MIT
