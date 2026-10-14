"""Train the learned narration and dialogue classification component on development data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import classification_report, f1_score
from src.data_loader import create_train_val_split, load_dataset
from src.model import NarrationClassifier


def train_model(val_ratio: float = 0.2, random_seed: int = 42):
    print("=" * 60)
    print("Phase 4: Training Learned Component (Narration Classifier)")
    print("=" * 60)

    # 1. Load dataset
    dev_seqs, _ = load_dataset(
        sequences_path=str(project_root / "sequences.json"),
        labels_path=str(project_root / "development" / "labels.jsonl")
    )

    # 2. Sequence-level train/val split (NO leakage across pages)
    train_seqs, val_seqs = create_train_val_split(dev_seqs, val_ratio=val_ratio, random_seed=random_seed)
    print(f"Sequence Split:")
    print(f"  - Train sequences: {len(train_seqs)} ({sum(len(p.lines) for s in train_seqs for p in s.pages)} lines)")
    print(f"  - Val sequences:   {len(val_seqs)} ({sum(len(p.lines) for s in val_seqs for p in s.pages)} lines)")

    # 3. Prepare training samples
    train_texts, train_labels, train_bboxes = [], [], []
    for s in train_seqs:
        for p in s.pages:
            for line in p.lines:
                train_texts.append(line.text)
                train_labels.append(line.speaker)
                train_bboxes.append(line.bbox)

    val_texts, val_labels, val_bboxes = [], [], []
    for s in val_seqs:
        for p in s.pages:
            for line in p.lines:
                val_texts.append(line.text)
                val_labels.append(line.speaker)
                val_bboxes.append(line.bbox)

    # 4. Train model
    print(f"\nTraining NarrationClassifier on {len(train_texts)} examples...")
    model = NarrationClassifier(random_state=random_seed)
    model.fit(train_texts, train_labels, train_bboxes)

    # 5. Evaluate on held-out validation set
    y_val_true = [1 if l == "NARRATION" else 0 for l in val_labels]
    y_val_probs = model.predict_proba(val_texts, val_bboxes)
    y_val_pred = [1 if p >= 0.45 else 0 for p in y_val_probs]

    f1 = f1_score(y_val_true, y_val_pred, average="macro")
    narration_f1 = f1_score(y_val_true, y_val_pred, pos_label=1)
    report = classification_report(y_val_true, y_val_pred, target_names=["Dialogue", "Narration"], output_dict=True)

    print("\nValidation Performance on Held-Out Sequences:")
    print(f"  - Macro F1:        {f1:.4f}")
    print(f"  - Narration F1:    {narration_f1:.4f}")
    print(f"  - Narration Precision: {report['Narration']['precision']:.4f}")
    print(f"  - Narration Recall:    {report['Narration']['recall']:.4f}")

    # 6. Save model artifact
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "narration_classifier.joblib"
    model.save(str(model_path))
    print(f"\n[SAVED] Model weights saved to {model_path}")

    # 7. Save training logs
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "val_ratio": val_ratio,
            "random_seed": random_seed,
            "train_sequences": len(train_seqs),
            "val_sequences": len(val_seqs),
            "macro_f1": f1,
            "narration_f1": narration_f1,
            "classification_report": report
        }, f, indent=2)
    print(f"[SAVED] Training metrics saved to {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
