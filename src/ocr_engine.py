"""OCR detection and transcription engine using EasyOCR and bounding box grouping."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class MangaOCREngine:
    """
    Lightweight, CPU-optimized OCR engine for extracting story text and bounding boxes.
    """

    def __init__(
        self,
        languages: List[str] = ["en"],
        gpu: bool = False,
        min_confidence: float = 0.20
    ):
        self.languages = languages
        self.gpu = gpu
        self.min_confidence = min_confidence
        self.reader = None

    def _ensure_reader(self):
        if self.reader is None:
            if not EASYOCR_AVAILABLE:
                raise RuntimeError("EasyOCR is not yet installed in the current environment.")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)

    def extract_text(
        self,
        image: Union[np.ndarray, str, Path],
        paragraph: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run OCR detection and recognition on a manga image.
        Returns list of dicts: {'text': str, 'bbox': (x1, y1, x2, y2), 'confidence': float}
        """
        self._ensure_reader()

        # easyocr accepts image file path or numpy array
        raw_results = self.reader.readtext(
            image,
            paragraph=paragraph,
            x_ths=1.0,
            y_ths=0.5
        )

        detections = []
        for item in raw_results:
            # easyocr paragraph format: [bbox_polygon, text] or [bbox_polygon, text, confidence]
            if len(item) == 2:
                box, text = item
                conf = 0.8  # Default paragraph confidence
            elif len(item) == 3:
                box, text, conf = item
            else:
                continue

            text = text.strip()
            if not text:
                continue

            if conf < self.min_confidence:
                continue

            # Convert 4-point polygon [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] to (x1, y1, x2, y2)
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            cleaned_text = self._clean_ocr_text(text)
            if cleaned_text:
                detections.append({
                    "text": cleaned_text,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf)
                })

        return detections

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean common OCR transcription noise while preserving punctuation and stutters:
        - preserve ellipses (...), question marks (?), exclamation marks (!)
        - remove dangling edge symbols
        """
        # Collapse excessive internal whitespace
        text = re.sub(r"[ \t]+", " ", text).strip()
        # Filter out lone single-char noise that isn't valid punctuation or a/I
        if len(text) == 1 and text not in {"?", "!", ".", "I", "A", "a", "i"}:
            return ""
        return text
