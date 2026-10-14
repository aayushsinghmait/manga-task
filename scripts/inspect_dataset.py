"""Inspect dataset distribution, sequence metadata, and speaker counts."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_dataset


def inspect_dataset():
    dev_seqs, test_seqs = load_dataset(
        sequences_path=str(project_root / "sequences.json"),
        labels_path=str(project_root / "development" / "labels.jsonl")
    )

    print("=" * 60)
    print("Manga Task Dataset Inspection Report")
    print("=" * 60)
    print(f"Development Sequences: {len(dev_seqs)}")
    print(f"Test Sequences:        {len(test_seqs)}")

    total_dev_pages = sum(len(s.pages) for s in dev_seqs)
    empty_pages = sum(1 for s in dev_seqs for p in s.pages if len(p.lines) == 0)
    total_lines = sum(len(p.lines) for s in dev_seqs for p in s.pages)

    speaker_counts = collections.Counter(line.speaker for s in dev_seqs for p in s.pages for line in p.lines)

    print(f"\nPage Statistics (Development):")
    print(f"  - Total Pages:        {total_dev_pages}")
    print(f"  - Empty Pages:        {empty_pages} ({empty_pages / total_dev_pages * 100:.2f}%)")
    print(f"  - Non-empty Pages:    {total_dev_pages - empty_pages}")
    print(f"  - Total Story Lines:  {total_lines}")
    print(f"  - Average Lines/Page: {total_lines / (total_dev_pages - empty_pages):.2f}")

    print(f"\nSpeaker Distribution:")
    for spk, cnt in speaker_counts.most_common():
        print(f"  - {spk:15s}: {cnt:4d} ({cnt / total_lines * 100:.2f}%)")

    chars_per_seq = []
    for s in dev_seqs:
        spks = {line.speaker for p in s.pages for line in p.lines} - {"UNKNOWN", "NARRATION"}
        chars_per_seq.append(len(spks))

    print(f"\nCharacters per Sequence (excl. UNKNOWN/NARRATION):")
    print(f"  - Min:  {min(chars_per_seq)}")
    print(f"  - Max:  {max(chars_per_seq)}")
    print(f"  - Mean: {sum(chars_per_seq) / len(chars_per_seq):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
