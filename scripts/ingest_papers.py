#!/usr/bin/env python
"""
Bulk-ingest a folder of PDF papers into the vector index + BM25 corpus.

Usage:
    python scripts/ingest_papers.py --input data/raw
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import ResearchRAGPipeline  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of PDF files")
    parser.add_argument("--report", default="data/processed/ingest_report.json")
    args = parser.parse_args()

    pipeline = ResearchRAGPipeline()
    results = pipeline.ingest_directory(args.input)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(results, f, indent=2, default=str)

    total_chunks = sum(r["num_chunks"] for r in results)
    logger.info(f"Ingested {len(results)} papers, {total_chunks} total chunks. Report: {args.report}")


if __name__ == "__main__":
    main()
