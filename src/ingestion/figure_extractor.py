"""
Figure/image extraction using PyMuPDF. Saves each embedded image to disk and
attempts to associate it with a nearby "Figure N" caption found in page text.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

CAPTION_RE = re.compile(r"^(Figure|Fig\.)\s*\d+[:.]?", re.IGNORECASE)


@dataclass
class ExtractedFigure:
    page: int
    figure_index: int
    caption: str
    image_path: str


class FigureExtractor:
    def __init__(self, output_dir: str = "data/processed/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, pdf_path: str, doc_id: str) -> List[ExtractedFigure]:
        doc = fitz.open(pdf_path)
        figures: List[ExtractedFigure] = []
        doc_out_dir = self.output_dir / doc_id
        doc_out_dir.mkdir(parents=True, exist_ok=True)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text()
            caption = self._find_caption(page_text)
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception as e:
                    logger.warning(f"Failed to extract image xref={xref} on page {page_idx}: {e}")
                    continue
                ext = base_image.get("ext", "png")
                img_path = doc_out_dir / f"page{page_idx}_img{img_idx}.{ext}"
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])
                figures.append(
                    ExtractedFigure(
                        page=page_idx,
                        figure_index=img_idx,
                        caption=caption or f"Figure on page {page_idx + 1}",
                        image_path=str(img_path),
                    )
                )
        doc.close()
        logger.info(f"Extracted {len(figures)} figures from {pdf_path}")
        return figures

    @staticmethod
    def _find_caption(page_text: str) -> Optional[str]:
        for line in page_text.splitlines():
            if CAPTION_RE.match(line.strip()):
                return line.strip()
        return None
