"""Image loading and preprocessing routines optimized for manga speech bubbles."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk into standard BGR format."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    # Use cv2.imdecode to safely support paths with non-ascii or special characters on Windows
    with open(path, "rb") as f:
        bytes_data = bytearray(f.read())
        img_array = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode image from {path}")
    return img


def preprocess_for_ocr(
    image: np.ndarray,
    enhance_contrast: bool = True,
    scale_factor: float = 1.0
) -> np.ndarray:
    """
    Carefully preprocess manga page for OCR without destroying tiny punctuation.
    Uses CLAHE for local contrast enhancement on dark or compressed scans.
    """
    if scale_factor != 1.0:
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray


def extract_bubble_candidates(
    image: np.ndarray,
    min_area: int = 400,
    max_area_ratio: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """
    Detect candidate speech bubble bounding boxes using morphological thresholding.
    Manga speech bubbles are typically bright/white closed convex contours.
    Returns list of (x, y, w, h) bounding boxes.
    """
    h_img, w_img = image.shape[:2]
    max_area = int(h_img * w_img * max_area_ratio)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Speech bubbles are mostly white or very light gray
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Morphological closing to seal bubble outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue
        # Manga bubbles typically have aspect ratios between 0.2 and 5.0
        aspect_ratio = w / float(h)
        if 0.15 < aspect_ratio < 6.0:
            candidates.append((x, y, w, h))

    # Sort top to bottom initially
    candidates.sort(key=lambda b: (b[1], -b[0]))
    return candidates
