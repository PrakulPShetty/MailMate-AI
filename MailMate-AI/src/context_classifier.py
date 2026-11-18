"""TF-IDF + Logistic Regression context classifier wrapper.

Wraps the fitted ``vectorizer`` and ``classifier`` shipped in
``models/mailmate_model.joblib``. Text is lemmatised with the same
vocabulary-constrained normaliser used at training time before vectorising, and
label order is taken from ``classifier.classes_`` so probabilities map back to
the right categories.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .preprocessing import LemmaNormalizer

__all__ = ["ContextClassifier"]


class ContextClassifier:
    """Thin CPU inference wrapper around the serialized sklearn pipeline."""

    def __init__(self, vectorizer, classifier, normalizer: LemmaNormalizer | None = None) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.normalizer = normalizer if normalizer is not None else LemmaNormalizer.from_vectorizer(vectorizer)
        self.classes = [str(label) for label in classifier.classes_]
        self._class_index = {label: i for i, label in enumerate(self.classes)}

    @property
    def categories(self) -> list[str]:
        return list(self.classes)

    def _probabilities(self, texts: Sequence[str]) -> np.ndarray:
        lemmatised = [self.normalizer.normalize_text(text) for text in texts]
        matrix = self.vectorizer.transform(lemmatised)
        return np.asarray(self.classifier.predict_proba(matrix), dtype=float)

    def classify(self, text: str) -> dict:
        """Return the most likely category, its confidence, and all scores."""
        probabilities = self._probabilities([text or ""])[0]
        scores = {
            label: round(float(probabilities[index]), 6)
            for label, index in self._class_index.items()
        }
        best_index = int(np.argmax(probabilities))
        category = self.classes[best_index]
        return {
            "category": category,
            "confidence": round(float(probabilities[best_index]), 6),
            "scores": scores,
        }

    def category_scores(self, texts: Sequence[str], category: str) -> list[float]:
        """Probability of ``category`` for each text in one batched inference."""
        if not texts:
            return []
        index = self._class_index.get(category)
        if index is None:
            raise ValueError(f"Unknown category {category!r}; expected one of {self.classes}")
        probabilities = self._probabilities(texts)
        return [float(row[index]) for row in probabilities]
