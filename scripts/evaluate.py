"""Evaluate baseline versus adapted model on development sequences using official score.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_dataset
from src.model import NarrationClassifier
from src.reading_order import sort_manga_boxes
from src.speaker_tracking import SpeakerTracker
from src.submission_writer import validate_submission_format, write_submission_jsonl


def evaluate_adapted_pipeline():
    print("=" * 60)
    print("Phase 5: Evaluating Baseline vs. Adapted Model")
    print("=" * 60)

    dev_seqs, _ = load_dataset(
        sequences_path=str(project_root / "sequences.json"),
        labels_path=str(project_root / "development" / "labels.jsonl")
    )

    model_path = project_root / "models" / "narration_classifier.joblib"
    if not model_path.exists():
        print(f"[WARN] {model_path} not found. Running training first...")
        from scripts.train import train_model
        train_model()

    narration_model = NarrationClassifier.load(str(model_path))
    tracker = SpeakerTracker(default_speaker_prefix="character")
    predictions = []

    print(f"Generating adapted predictions for {len(dev_seqs)} development sequences...")
    for seq in dev_seqs:
        page_boxes = []
        for page in seq.pages:
            boxes = []
            for idx, line in enumerate(page.lines):
                bbox = line.bbox if line.bbox is not None else (100, 100 + idx * 80, 400, 160 + idx * 80)
                boxes.append({
                    "text": line.text,
                    "bbox": bbox
                })
            sorted_boxes = sort_manga_boxes(boxes)
            page_boxes.append(sorted_boxes)

        # Adapted speaker assignment with learned narration model
        assigned_pages = tracker.assign_speakers(page_boxes, narration_predictor=narration_model)

        predictions.append({
            "sequence_id": seq.sequence_id,
            "pages": assigned_pages
        })

    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    adapted_out = outputs_dir / "adapted_dev_predictions.jsonl"
    write_submission_jsonl(predictions, str(adapted_out))
    print(f"[OK] Adapted predictions written to {adapted_out}")

    # Validate schema
    dev_ids = {s.sequence_id for s in dev_seqs}
    val_report = validate_submission_format(str(adapted_out), expected_sequence_ids=dev_ids)
    print(f"[OK] Schema validated: {val_report['total_sequences']} sequences.")

    # Evaluate with official score.py
    scores_out = project_root / "scores_adapted.json"
    score_cmd = [
        sys.executable,
        str(project_root / "score.py"),
        "--references", str(project_root / "development" / "labels.jsonl"),
        "--predictions", str(adapted_out),
        "--output", str(scores_out)
    ]
    res = subprocess.run(score_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] score.py failed: {res.stderr}")
        sys.exit(1)

    adapted_scores = json.loads(scores_out.read_text(encoding="utf-8"))["macro"]

    # Load baseline scores if available
    baseline_path = project_root / "scores_baseline.json"
    if baseline_path.exists():
        baseline_scores = json.loads(baseline_path.read_text(encoding="utf-8"))["macro"]
    else:
        baseline_scores = {}

    print("\n" + "=" * 60)
    print("COMPARATIVE EVALUATION RESULTS (Official score.py)")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Baseline':<12} | {'Adapted (Ours)':<14} | {'Diff':<10}")
    print("-" * 72)
    for key in ["text_order_score", "balanced_joint_f1", "speaker_accuracy_on_matched", "joint_f1", "matched_token_coverage"]:
        base_val = baseline_scores.get(key, 0.0) if baseline_scores.get(key) is not None else 0.0
        adapt_val = adapted_scores.get(key, 0.0) if adapted_scores.get(key) is not None else 0.0
        diff = adapt_val - base_val
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"{key:<30} | {base_val:<12.4f} | {adapt_val:<14.4f} | {diff_str:<10}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_adapted_pipeline()
