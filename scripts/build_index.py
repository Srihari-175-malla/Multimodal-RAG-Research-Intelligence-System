#!/usr/bin/env python
"""
Rebuild the vector + BM25 index from already-processed chunk data.
Useful after changing the embedding model or vector backend without
re-parsing every PDF from scratch (assumes chunks are cached — extend
this script to load from your own chunk cache if you persist one).
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import ResearchRAGPipeline  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=str(settings.raw_papers_dir), help="Directory of raw PDFs to re-ingest"
    )
    args = parser.parse_args()

    logger.info(f"Rebuilding index (backend={settings.vector_backend}) from {args.input}")
    pipeline = ResearchRAGPipeline()
    results = pipeline.ingest_directory(args.input)
    logger.info(f"Rebuilt index with {len(results)} documents")


if __name__ == "__main__":
    main()
