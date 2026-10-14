"""Data loading and sequence splitting utilities."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TextLine:
    speaker: str
    text: str
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    confidence: float = 1.0


@dataclass
class MangaPage:
    page_index: int  # 0, 1, or 2
    image_path: str
    lines: List[TextLine] = field(default_factory=list)


@dataclass
class MangaSequence:
    sequence_id: str
    split: str
    pages: List[MangaPage] = field(default_factory=list)

    @property
    def image_paths(self) -> List[str]:
        return [p.image_path for p in self.pages]


def load_sequences_index(sequences_path: str = "sequences.json") -> List[dict]:
    """Load the master sequences.json metadata."""
    path = Path(sequences_path)
    if not path.exists():
        raise FileNotFoundError(f"Sequences index not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels_index(labels_path: str = "development/labels.jsonl") -> Dict[str, List[List[dict]]]:
    """Load labels.jsonl indexed by sequence_id."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found at {path}")
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            labels[item["sequence_id"]] = item["pages"]
    return labels


def load_dataset(
    sequences_path: str = "sequences.json",
    labels_path: str = "development/labels.jsonl"
) -> Tuple[List[MangaSequence], List[MangaSequence]]:
    """
    Load development and test sequences into structured MangaSequence objects.
    Development sequences are populated with ground truth lines.
    """
    raw_sequences = load_sequences_index(sequences_path)
    labels = load_labels_index(labels_path) if Path(labels_path).exists() else {}

    dev_sequences = []
    test_sequences = []

    for seq in raw_sequences:
        seq_id = seq["sequence_id"]
        split = seq["split"]
        img_paths = seq["images"]
        pages = []

        if split == "development":
            gold_pages = labels.get(seq_id, [[], [], []])
            for idx, img_p in enumerate(img_paths):
                gold_lines = gold_pages[idx] if idx < len(gold_pages) else []
                text_lines = [
                    TextLine(speaker=g["speaker"], text=g["text"])
                    for g in gold_lines
                ]
                pages.append(MangaPage(page_index=idx, image_path=img_p, lines=text_lines))
            dev_sequences.append(MangaSequence(sequence_id=seq_id, split=split, pages=pages))
        else:
            for idx, img_p in enumerate(img_paths):
                pages.append(MangaPage(page_index=idx, image_path=img_p, lines=[]))
            test_sequences.append(MangaSequence(sequence_id=seq_id, split=split, pages=pages))

    return dev_sequences, test_sequences


def create_train_val_split(
    sequences: List[MangaSequence],
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[MangaSequence], List[MangaSequence]]:
    """
    Split development sequences into train and validation sets by COMPLETE sequence.
    Preserves all 3 consecutive pages together to avoid cross-page leakage.
    """
    shuffled = list(sequences)
    rng = random.Random(random_seed)
    rng.shuffle(shuffled)

    num_val = int(len(shuffled) * val_ratio)
    val_set = shuffled[:num_val]
    train_set = shuffled[num_val:]

    return train_set, val_set
