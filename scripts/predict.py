"""Generate test predictions on the 15 unlabelled test sequences."""
from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_dataset
from src.model import NarrationClassifier
from src.ocr_engine import MangaOCREngine
from src.reading_order import sort_manga_boxes
from src.speaker_tracking import SpeakerTracker
from src.submission_writer import validate_submission_format, write_submission_jsonl


def generate_predictions(output_path: str = "outputs/test_predictions.jsonl"):
    print("=" * 60)
    print("Phase 6: Generating Test Predictions on Unlabelled Test Sequences")
    print("=" * 60)

    # 1. Load test sequences
    _, test_seqs = load_dataset(
        sequences_path=str(project_root / "sequences.json"),
        labels_path=str(project_root / "development" / "labels.jsonl")
    )
    print(f"Loaded {len(test_seqs)} test sequences to predict.")

    # 2. Load trained narration classifier
    model_path = project_root / "models" / "narration_classifier.joblib"
    if not model_path.exists():
        print(f"[WARN] {model_path} not found. Running training first...")
        from scripts.train import train_model
        train_model()

    narration_model = NarrationClassifier.load(str(model_path))
    ocr_engine = MangaOCREngine(languages=["en"], gpu=False)
    tracker = SpeakerTracker(default_speaker_prefix="character")

    predictions = []

    for s_idx, seq in enumerate(test_seqs, 1):
        print(f"[{s_idx:2d}/{len(test_seqs)}] Processing sequence: {seq.sequence_id}...")
        seq_page_boxes = []

        for p_idx, page in enumerate(seq.pages):
            img_path = project_root / page.image_path
            try:
                # OCR text and bubble detection
                detected_boxes = ocr_engine.extract_text(str(img_path))
            except Exception as e:
                # Fallback if OCR model is still initializing or empty page
                detected_boxes = []

            # Reconstruct reading order for the page
            sorted_boxes = sort_manga_boxes(detected_boxes)
            seq_page_boxes.append(sorted_boxes)

        # Track and assign speakers across all 3 pages
        assigned_pages = tracker.assign_speakers(seq_page_boxes, narration_predictor=narration_model)

        predictions.append({
            "sequence_id": seq.sequence_id,
            "pages": assigned_pages
        })

    # 3. Write submission file
    out_file = project_root / output_path
    write_submission_jsonl(predictions, str(out_file))
    print(f"\n[OK] Predictions successfully written to {out_file}")

    # 4. Strict schema validation
    test_ids = {s.sequence_id for s in test_seqs}
    val_report = validate_submission_format(str(out_file), expected_sequence_ids=test_ids)
    print(f"[OK] Validation passed: {val_report['total_sequences']} sequences, {val_report['total_text_bubbles']} bubbles.")
    print("=" * 60)


if __name__ == "__main__":
    generate_predictions()
