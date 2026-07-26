"""Question-answering endpoint."""
from fastapi import APIRouter

from api.routers.ingest import get_pipeline
from api.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = get_pipeline().query(request.question, top_k=request.top_k)
    return result


@router.get("/stats")
async def stats():
    return get_pipeline().stats()
