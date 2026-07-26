"""Pydantic request/response models for the FastAPI app."""
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=8, ge=1, le=30)


class SourceExcerpt(BaseModel):
    text: str
    doc_id: Optional[str] = None
    doc_title: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceExcerpt]


class CompareRequest(BaseModel):
    paper_ids: List[str]
    aspect: str


class CompareResponse(BaseModel):
    aspect: str
    paper_ids: List[str]
    answer: str
    sources: List[SourceExcerpt]


class ClaimRequest(BaseModel):
    claim: str


class ClaimResponse(BaseModel):
    claim: str
    verdict_explanation: str
    sources: List[SourceExcerpt]


class IngestResponse(BaseModel):
    doc_id: str
    num_chunks: int
    metadata: dict


class StatsResponse(BaseModel):
    num_indexed_chunks: int
    num_documents: int
    vector_backend: str
