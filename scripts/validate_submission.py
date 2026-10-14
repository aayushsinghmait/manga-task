"""Validate any predictions.jsonl submission file against the required format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.submission_writer import validate_submission_format


def main():
    parser = argparse.ArgumentParser(description="Validate submission JSONL format")
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/test_predictions.jsonl",
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="sample_submission.jsonl",
        help="Path to sample submission to check sequence IDs"
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    sample_path = Path(args.sample)

    print("=" * 60)
    print(f"Validating Submission: {pred_path}")
    print("=" * 60)

    expected_ids = None
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            expected_ids = {json.loads(line)["sequence_id"] for line in f if line.strip()}
        print(f"Found {len(expected_ids)} expected sequence IDs in sample submission.")

    try:
        report = validate_submission_format(str(pred_path), expected_sequence_ids=expected_ids)
        print("[SUCCESS] Submission is 100% compliant with task requirements!")
        print(f"  - Total Sequences:    {report['total_sequences']}")
        print(f"  - Total Text Bubbles: {report['total_text_bubbles']}")
        print(f"  - Empty Pages:        {report['empty_pages']}")
    except Exception as e:
        print(f"[FAIL] Submission validation error: {e}")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
