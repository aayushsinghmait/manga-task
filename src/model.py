"""Machine Learning models: Narration classifier and adapted pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


class NarrationClassifier:
    """
    Learned classifier to distinguish NARRATION from character dialogue.
    Trained on development sequence annotations using linguistic and geometric signals.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            token_pattern=r"(?u)\b\w+\b|[.?!]"
        )
        self.classifier = LogisticRegression(
            class_weight="balanced",
            C=1.5,
            max_iter=1000,
            random_state=random_state
        )
        self.is_fitted = False

    def _extract_text_features(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()

    def _extract_dense_features(self, texts: List[str], bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]]) -> np.ndarray:
        feats = []
        for i, text in enumerate(texts):
            t = text.strip()
            length = len(t)
            word_count = len(t.split())
            has_exclamation = float("!" in t)
            has_question = float("?" in t)
            has_ellipsis = float("..." in t or "…" in t)
            has_quotes = float('"' in t or "'" in t)
            all_caps = float(t.isupper())

            # Bounding box features if available
            aspect_ratio = 1.0
            norm_width = 0.5
            norm_height = 0.1
            norm_y = 0.5

            if bboxes is not None and i < len(bboxes) and bboxes[i] is not None:
                bx1, by1, bx2, by2 = bboxes[i]
                bw = max(1, bx2 - bx1)
                bh = max(1, by2 - by1)
                aspect_ratio = bw / float(bh)
                norm_width = bw / 800.0
                norm_height = bh / 1200.0
                norm_y = (by1 + by2) / 2400.0

            feats.append([
                length,
                word_count,
                has_exclamation,
                has_question,
                has_ellipsis,
                has_quotes,
                all_caps,
                aspect_ratio,
                norm_width,
                norm_height,
                norm_y
            ])
        return np.array(feats, dtype=np.float32)

    def fit(
        self,
        texts: List[str],
        labels: List[str],
        bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None
    ):
        """Fit the classifier on labeled training examples."""
        # Convert labels: 1 if NARRATION, 0 otherwise
        y = np.array([1 if l == "NARRATION" else 0 for l in labels], dtype=int)

        self.vectorizer.fit(texts)
        x_tfidf = self._extract_text_features(texts)
        x_dense = self._extract_dense_features(texts, bboxes)

        x_combined = np.hstack([x_tfidf, x_dense])
        self.classifier.fit(x_combined, y)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        texts: List[str],
        bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        x_tfidf = self._extract_text_features(texts)
        x_dense = self._extract_dense_features(texts, bboxes)
        x_combined = np.hstack([x_tfidf, x_dense])
        return self.classifier.predict_proba(x_combined)[:, 1]

    def is_narration(self, text: str, bbox: Optional[Tuple[int, int, int, int]] = None, threshold: float = 0.45) -> bool:
        """Return True if text is predicted as narration."""
        if not self.is_fitted:
            return False
        prob = self.predict_proba([text], [bbox])[0]
        return bool(prob >= threshold)

    def save(self, filepath: str = "models/narration_classifier.joblib"):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, filepath: str = "models/narration_classifier.joblib") -> "NarrationClassifier":
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        return joblib.load(p)
