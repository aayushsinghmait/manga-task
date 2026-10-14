"""Automated test suite for Manga Task pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import create_train_val_split, load_dataset, load_sequences_index
from src.model import NarrationClassifier
from src.reading_order import sort_manga_boxes
from src.speaker_tracking import SpeakerTracker
from src.submission_writer import validate_submission_format, write_submission_jsonl


def test_sequences_and_images_exist():
    """Verify sequences.json has 95 sequences and all images exist."""
    sequences = load_sequences_index(str(project_root / "sequences.json"))
    assert len(sequences) == 95, f"Expected 95 sequences, got {len(sequences)}"

    for s in sequences:
        assert len(s["images"]) == 3, f"Sequence {s['sequence_id']} must have exactly 3 images"
        for img_p in s["images"]:
            full_p = project_root / img_p
            assert full_p.exists(), f"Image {img_p} does not exist!"


def test_dataset_loader_and_leakage_free_split():
    """Verify train/val split is grouped by sequence with 0 leakage across pages."""
    dev_seqs, test_seqs = load_dataset(
        sequences_path=str(project_root / "sequences.json"),
        labels_path=str(project_root / "development" / "labels.jsonl")
    )
    assert len(dev_seqs) == 80
    assert len(test_seqs) == 15

    train_seqs, val_seqs = create_train_val_split(dev_seqs, val_ratio=0.2, random_seed=42)
    assert len(train_seqs) == 64
    assert len(val_seqs) == 16

    train_ids = {s.sequence_id for s in train_seqs}
    val_ids = {s.sequence_id for s in val_seqs}
    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage! Sequence in both train and val."


def test_reading_order_manga_rtl():
    """Verify that boxes on the same horizontal band are read right-to-left."""
    boxes = [
        {"text": "Left bubble", "bbox": (100, 200, 250, 280)},
        {"text": "Right bubble", "bbox": (500, 200, 650, 280)},
        {"text": "Top bubble", "bbox": (300, 50, 450, 120)}
    ]
    sorted_boxes = sort_manga_boxes(boxes, page_width=800, page_height=1200)

    # Top bubble must be first
    assert sorted_boxes[0]["text"] == "Top bubble"
    # In manga, right bubble must precede left bubble on same row
    assert sorted_boxes[1]["text"] == "Right bubble"
    assert sorted_boxes[2]["text"] == "Left bubble"


def test_speaker_consistency_across_pages():
    """Verify character labels remain consistent across the 3 consecutive pages."""
    tracker = SpeakerTracker(default_speaker_prefix="character")
    page1 = [{"text": "Hello there!", "bbox": (400, 100, 600, 150)}]
    page2 = [{"text": "General Kenobi!", "bbox": (200, 200, 400, 250)}]
    page3 = [{"text": "You are a bold one.", "bbox": (400, 100, 600, 150)}]

    assigned = tracker.assign_speakers([page1, page2, page3])
    assert len(assigned) == 3
    assert assigned[0][0]["speaker"] == "character1"
    assert assigned[1][0]["speaker"] == "character2"


def test_learned_narration_classifier():
    """Verify narration classifier training, prediction, and serialization."""
    train_texts = [
        "Meanwhile, in another part of the world...",
        "The following morning, everyone gathered at the station.",
        "Hey, what are you doing over there?",
        "Don't worry about it! I am fine!"
    ]
    train_labels = ["NARRATION", "NARRATION", "character1", "character2"]

    clf = NarrationClassifier(random_state=42)
    clf.fit(train_texts, train_labels)

    assert clf.is_narration("Later that evening, they returned home.") is True
    assert clf.is_narration("What?! No way!") is False

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model = Path(tmp_dir) / "test_model.joblib"
        clf.save(str(tmp_model))
        loaded_clf = NarrationClassifier.load(str(tmp_model))
        assert loaded_clf.is_narration("Later that evening, they returned home.") is True


def test_submission_validation():
    """Verify strict submission schema validator catches formatting errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_sub.jsonl"
        valid_data = [
            {
                "sequence_id": "seq_test1",
                "pages": [
                    [{"speaker": "character1", "text": "Hello"}],
                    [],
                    [{"speaker": "NARRATION", "text": "The end."}]
                ]
            }
        ]
        write_submission_jsonl(valid_data, str(tmp_file))
        report = validate_submission_format(str(tmp_file), expected_sequence_ids={"seq_test1"})
        assert report["status"] == "VALID"
        assert report["total_sequences"] == 1
        assert report["empty_pages"] == 1


if __name__ == "__main__":
    test_sequences_and_images_exist()
    test_dataset_loader_and_leakage_free_split()
    test_reading_order_manga_rtl()
    test_speaker_consistency_across_pages()
    test_learned_narration_classifier()
    test_submission_validation()
    print("All automated tests passed successfully!")
