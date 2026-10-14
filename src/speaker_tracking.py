"""Speaker identification and sequence-wide character consistency tracking."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class SpeakerTracker:
    """
    Assigns consistent character identifiers across the 3 pages of a manga sequence.
    Handles dialogue turns, speech bubble spatial proximity, and narration detection.
    """

    def __init__(
        self,
        default_speaker_prefix: str = "character",
        max_active_speakers: int = 5,
        turn_distance_threshold: float = 0.25
    ):
        self.prefix = default_speaker_prefix
        self.max_speakers = max_active_speakers
        self.turn_distance_threshold = turn_distance_threshold

    def assign_speakers(
        self,
        pages_boxes: List[List[Dict[str, Any]]],
        narration_predictor: Optional[Any] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Assign consistent speaker labels to all bubbles across the 3 consecutive pages.
        Returns pages with updated 'speaker' fields.
        """
        result_pages = []
        current_speaker_idx = 1
        last_speaker = None
        last_center = None

        for page_idx, page in enumerate(pages_boxes):
            page_results = []
            for item in page:
                text = item.get("text", "").strip()
                bbox = item.get("bbox")

                # 1. Narration check
                is_narration = False
                if narration_predictor is not None:
                    try:
                        is_narration = narration_predictor.is_narration(text, bbox)
                    except Exception:
                        is_narration = self._heuristic_is_narration(text, bbox)
                else:
                    is_narration = self._heuristic_is_narration(text, bbox)

                if is_narration:
                    speaker = "NARRATION"
                else:
                    # 2. Character dialogue assignment
                    if bbox is not None and last_center is not None and last_speaker != "NARRATION":
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        # Check spatial proximity to previous balloon (continuation vs turn-taking)
                        norm_dist = ((cx - last_center[0]) ** 2 + (cy - last_center[1]) ** 2) ** 0.5
                        if norm_dist < 200:
                            # Continuation of previous speaker
                            speaker = last_speaker
                        else:
                            # New conversational turn: alternate speaker
                            if last_speaker == f"{self.prefix}1":
                                speaker = f"{self.prefix}2"
                            elif last_speaker == f"{self.prefix}2":
                                speaker = f"{self.prefix}1"
                            else:
                                speaker = f"{self.prefix}1"
                    else:
                        speaker = f"{self.prefix}1"

                    last_speaker = speaker
                    if bbox is not None:
                        last_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

                page_results.append({
                    "speaker": speaker,
                    "text": text,
                    "bbox": bbox
                })
            result_pages.append(page_results)

        return result_pages

    def _heuristic_is_narration(self, text: str, bbox: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Heuristic rule for narration:
        - Rectangular framing / aspect ratio if bbox is wide and thin
        - Typical narration keywords: Meanwhile, Later, At that time, In year, Chapter, Next day, etc.
        """
        text_lower = text.lower().strip()
        narration_cues = [
            "meanwhile", "the next day", "later that day", "at that time",
            "and so", "it was then", "chapter", "volume", "episode",
            "suddenly,", "years ago", "several days later"
        ]
        if any(text_lower.startswith(cue) for cue in narration_cues):
            return True

        # Narration boxes are often wide captions at the top or bottom edge of panels
        if bbox is not None:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w > 400 and h < 80:  # Wide horizontal banner
                return True

        return False
