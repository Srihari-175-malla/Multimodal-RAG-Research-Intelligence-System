"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import compare, evaluate, ingest, query

app = FastAPI(
    title="Multimodal RAG Research Intelligence API",
    description="Ingest research papers, ask questions, compare papers, verify claims.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(compare.router)
app.include_router(evaluate.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
