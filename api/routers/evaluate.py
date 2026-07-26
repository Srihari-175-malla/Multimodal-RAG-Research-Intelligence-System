"""Evaluation endpoint — runs the evaluator against an uploaded qa_set.jsonl."""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from src.evaluation.evaluator import Evaluator

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.post("")
async def evaluate(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        evaluator = Evaluator()
        results = evaluator.evaluate_dataset(tmp_path)
        return results
    finally:
        Path(tmp_path).unlink(missing_ok=True)
