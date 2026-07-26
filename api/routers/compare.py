"""Paper comparison and claim verification endpoints."""
from fastapi import APIRouter

from api.routers.ingest import get_pipeline
from api.schemas import ClaimRequest, ClaimResponse, CompareRequest, CompareResponse

router = APIRouter(tags=["analysis"])


@router.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    result = get_pipeline().compare_papers(request.paper_ids, request.aspect)
    return result


@router.post("/verify_claim", response_model=ClaimResponse)
async def verify_claim(request: ClaimRequest):
    result = get_pipeline().verify_claim(request.claim)
    return result
