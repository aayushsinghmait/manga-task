"""Run baseline pipeline on development sequences and compute official baseline scores."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_dataset
from src.reading_order import sort_manga_boxes
from src.speaker_tracking import SpeakerTracker
from src.submission_writer import validate_submission_format, write_submission_jsonl


def run_baseline(limit: int = None):
    print("=" * 60)
    print("Phase 2: Running Baseline Pipeline on Development Sequences")
    print("=" * 60)

    dev_seqs, _ = load_dataset(
        sequences_path=str(project_root / "sequences.json"),
        labels_path=str(project_root / "development" / "labels.jsonl")
    )

    if limit is not None:
        dev_seqs = dev_seqs[:limit]

    print(f"Processing {len(dev_seqs)} development sequences with baseline...")

    tracker = SpeakerTracker(default_speaker_prefix="character")
    predictions = []

    for seq in dev_seqs:
        # Baseline pipeline: preserves detected text, standard reading order, baseline speaker turn assignment
        page_boxes = []
        for page in seq.pages:
            boxes = []
            for idx, line in enumerate(page.lines):
                # Simulated detection with default positioning if bbox not available
                bbox = line.bbox if line.bbox is not None else (100, 100 + idx * 80, 400, 160 + idx * 80)
                boxes.append({
                    "text": line.text,
                    "bbox": bbox
                })
            sorted_boxes = sort_manga_boxes(boxes)
            page_boxes.append(sorted_boxes)

        # Baseline speaker assignment (without learned narration classifier)
        assigned_pages = tracker.assign_speakers(page_boxes, narration_predictor=None)

        predictions.append({
            "sequence_id": seq.sequence_id,
            "pages": assigned_pages
        })

    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    baseline_out = outputs_dir / "baseline_dev_predictions.jsonl"
    write_submission_jsonl(predictions, str(baseline_out))
    print(f"[OK] Baseline predictions written to {baseline_out}")

    # Validate schema
    dev_ids = {s.sequence_id for s in dev_seqs}
    val_report = validate_submission_format(str(baseline_out), expected_sequence_ids=dev_ids)
    print(f"[OK] Schema validated: {val_report['total_sequences']} sequences, {val_report['total_text_bubbles']} bubbles.")

    # Evaluate using official score.py
    scores_out = project_root / "scores_baseline.json"
    score_cmd = [
        sys.executable,
        str(project_root / "score.py"),
        "--references", str(project_root / "development" / "labels.jsonl"),
        "--predictions", str(baseline_out),
        "--output", str(scores_out)
    ]
    print("\nRunning official score.py evaluation...")
    res = subprocess.run(score_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] score.py failed: {res.stderr}")
        sys.exit(1)

    print("Baseline Official Evaluation Scores:")
    print(res.stdout)
    print(f"Detailed scores saved to {scores_out}")
    print("=" * 60)


if __name__ == "__main__":
    run_baseline()
