"""Reading order reconstruction for Japanese/English manga layouts."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Return (center_x, center_y) for a (x1, y1, x2, y2) bounding box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def manga_sort_key(bbox: Tuple[int, int, int, int], page_height: int = 1200, band_height: int = 150) -> Tuple[int, float]:
    """
    Sort key for manga reading order:
    1. Vertical band (top to bottom): bucket index = y_center // band_height
    2. Horizontal position inside band (right to left): -center_x
    """
    cx, cy = get_bbox_center(bbox)
    band_idx = int(cy // band_height)
    return (band_idx, -cx)


def sort_manga_boxes(
    boxes_with_data: List[Dict[str, Any]],
    page_width: int = 800,
    page_height: int = 1200,
    vertical_band_ratio: float = 0.12
) -> List[Dict[str, Any]]:
    """
    Reconstruct reading order for manga text bubbles on a single page.
    Default manga layout rules:
    1. Panels and bubbles are read Top-to-Bottom.
    2. Within the same panel or horizontal row, bubbles are read Right-to-Left.
    3. Connected / adjacent dialogue turns are kept in order.
    """
    if not boxes_with_data:
        return []

    band_height = max(50, int(page_height * vertical_band_ratio))

    # Pairwise comparison with fuzzy band overlap
    def is_before(item_a: Dict[str, Any], item_b: Dict[str, Any]) -> bool:
        bbox_a = item_a.get("bbox", (0, 0, 0, 0))
        bbox_b = item_b.get("bbox", (0, 0, 0, 0))

        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        # Check vertical overlap
        v_overlap = max(0, min(ay2, by2) - max(ay1, by1))
        min_h = min(ay2 - ay1, by2 - by1) if min(ay2 - ay1, by2 - by1) > 0 else 1
        overlap_ratio = v_overlap / float(min_h)

        if overlap_ratio > 0.4:
            # Overlapping vertically: rightmost comes first in Manga (RTL)
            return (ax1 + ax2) / 2.0 > (bx1 + bx2) / 2.0
        else:
            # Separated vertically: topmost comes first
            return (ay1 + ay2) / 2.0 < (by1 + by2) / 2.0

    # Stable sorting using custom comparator
    sorted_items = list(boxes_with_data)
    n = len(sorted_items)
    for i in range(n):
        for j in range(i + 1, n):
            if not is_before(sorted_items[i], sorted_items[j]):
                sorted_items[i], sorted_items[j] = sorted_items[j], sorted_items[i]

    return sorted_items
