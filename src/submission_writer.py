"""Submission formatter and schema validator for Manga Task."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set


def validate_submission_format(
    predictions_path: str,
    expected_sequence_ids: Optional[Set[str]] = None,
    pages_per_sequence: int = 3
) -> Dict[str, Any]:
    """
    Strict validation of a predictions.jsonl file against the official scoring rules.
    Returns validation summary dictionary or raises ValueError on invalid schema.
    """
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found at {path}")

    seen_ids = set()
    total_lines = 0
    total_text_bubbles = 0
    empty_pages_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"Line {line_num}: Invalid JSON syntax - {e}")

            if not isinstance(row, dict) or set(row.keys()) != {"sequence_id", "pages"}:
                raise ValueError(f"Line {line_num}: Object must have exactly 'sequence_id' and 'pages' keys")

            seq_id = row["sequence_id"]
            if not isinstance(seq_id, str) or not seq_id.strip() or seq_id != seq_id.strip():
                raise ValueError(f"Line {line_num}: 'sequence_id' must be a non-empty trimmed string")

            if seq_id in seen_ids:
                raise ValueError(f"Line {line_num}: Duplicate sequence_id '{seq_id}'")
            seen_ids.add(seq_id)

            pages = row["pages"]
            if not isinstance(pages, list) or len(pages) != pages_per_sequence:
                raise ValueError(f"Line {line_num}: 'pages' must be a list of exactly {pages_per_sequence} lists")

            for p_idx, page in enumerate(pages):
                if not isinstance(page, list):
                    raise ValueError(f"Line {line_num}, page {p_idx}: Each page must be a list")
                if len(page) == 0:
                    empty_pages_count += 1
                for item in page:
                    if not isinstance(item, dict) or set(item.keys()) != {"speaker", "text"}:
                        raise ValueError(f"Line {line_num}, page {p_idx}: Line item must have exactly 'speaker' and 'text'")
                    speaker = item["speaker"]
                    text = item["text"]

                    if not isinstance(speaker, str) or not speaker.strip() or speaker != speaker.strip():
                        raise ValueError(f"Line {line_num}: speaker '{speaker}' must be non-empty trimmed string")

                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Line {line_num}: text cannot be empty; empty pages must use []")

                    if any(unicodedata.category(c) == "Cc" and not c.isspace() for c in text):
                        raise ValueError(f"Line {line_num}: text contains disallowed control characters")

                    total_text_bubbles += 1

    if expected_sequence_ids is not None:
        missing = expected_sequence_ids - seen_ids
        extra = seen_ids - expected_sequence_ids
        if missing:
            raise ValueError(f"Missing expected sequence IDs: {missing}")
        if extra:
            raise ValueError(f"Unexpected extra sequence IDs: {extra}")

    return {
        "status": "VALID",
        "total_sequences": len(seen_ids),
        "total_text_bubbles": total_text_bubbles,
        "empty_pages": empty_pages_count,
        "sequences": sorted(list(seen_ids))
    }


def write_submission_jsonl(
    predictions: List[Dict[str, Any]],
    output_path: str
) -> Path:
    """Write predictions list to JSONL format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for item in predictions:
            clean_item = {
                "sequence_id": item["sequence_id"],
                "pages": [
                    [{"speaker": row["speaker"], "text": row["text"]} for row in page]
                    for page in item["pages"]
                ]
            }
            f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
    return out
