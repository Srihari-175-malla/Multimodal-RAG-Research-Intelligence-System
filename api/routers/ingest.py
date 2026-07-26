"""PDF ingestion endpoints."""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import IngestResponse
from src.pipeline import ResearchRAGPipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])
_pipeline = None


def get_pipeline() -> ResearchRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ResearchRAGPipeline()
    return _pipeline


@router.post("", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc_id = Path(file.filename).stem
        result = get_pipeline().ingest_pdf(tmp_path, doc_id=doc_id)
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)
