"""Validate dataset integrity, sequence ordering, image availability, and label schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from PIL import Image

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_sequences_index, load_labels_index
from src.submission_writer import validate_submission_format


def validate_dataset():
    print("=" * 60)
    print("Manga Task [Advanced] - Dataset Validation")
    print("=" * 60)

    # 1. Check sequences.json
    seq_path = project_root / "sequences.json"
    if not seq_path.exists():
        print(f"[FAIL] Missing sequences.json at {seq_path}")
        sys.exit(1)

    sequences = load_sequences_index(str(seq_path))
    print(f"[OK] sequences.json loaded: {len(sequences)} total sequences")

    dev_seqs = [s for s in sequences if s["split"] == "development"]
    test_seqs = [s for s in sequences if s["split"] == "test"]
    print(f"     - Development sequences: {len(dev_seqs)}")
    print(f"     - Test sequences: {len(test_seqs)}")

    # 2. Check image existence and integrity
    missing_images = []
    unreadable_images = []
    total_images = 0

    for s in sequences:
        imgs = s.get("images", [])
        if len(imgs) != 3:
            print(f"[FAIL] Sequence {s['sequence_id']} has {len(imgs)} images, expected exactly 3!")
            sys.exit(1)
        for img_rel in imgs:
            total_images += 1
            img_path = project_root / img_rel
            if not img_path.exists():
                missing_images.append(str(img_rel))
            else:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    unreadable_images.append((str(img_rel), str(e)))

    if missing_images:
        print(f"[FAIL] Missing {len(missing_images)} image files: {missing_images[:5]}")
        sys.exit(1)
    if unreadable_images:
        print(f"[FAIL] Corrupt or unreadable {len(unreadable_images)} images: {unreadable_images[:5]}")
        sys.exit(1)
    print(f"[OK] All {total_images} images verified intact and readable on disk.")

    # 3. Check development labels
    labels_path = project_root / "development" / "labels.jsonl"
    if not labels_path.exists():
        print(f"[FAIL] Missing development/labels.jsonl at {labels_path}")
        sys.exit(1)

    labels = load_labels_index(str(labels_path))
    print(f"[OK] development/labels.jsonl loaded: {len(labels)} sequence annotations")

    # Verify all dev sequence IDs are present in labels
    dev_ids = {s["sequence_id"] for s in dev_seqs}
    label_ids = set(labels.keys())
    diff = dev_ids - label_ids
    if diff:
        print(f"[FAIL] Development sequences missing labels: {diff}")
        sys.exit(1)
    print(f"[OK] All {len(dev_ids)} development sequences have ground truth labels.")

    # 4. Check sample submission
    sub_path = project_root / "sample_submission.jsonl"
    if not sub_path.exists():
        print(f"[FAIL] Missing sample_submission.jsonl at {sub_path}")
        sys.exit(1)

    test_ids = {s["sequence_id"] for s in test_seqs}
    sub_report = validate_submission_format(str(sub_path), expected_sequence_ids=test_ids)
    print(f"[OK] sample_submission.jsonl verified: {sub_report['total_sequences']} test sequences matching schema.")

    print("\nDataset validation PASSED with 0 errors!")
    print("=" * 60)


if __name__ == "__main__":
    validate_dataset()
