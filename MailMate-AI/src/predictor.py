"""Prediction engine: load the Phase 2 artifact and rank next-word candidates.

This is the only module the Flask backend will import. It reads every parameter
from ``models/mailmate_model.joblib`` (``scoring_weights``, ``lambda``,
``categories``, ``tones``, ``vocab``, the count tables, ``vectorizer``,
``classifier`` and ``tone_lexicon``) and never consults the root ``config.json``.

Ranking blends the four artifact weights::

    score = w_lm * lm + w_context * context + w_tone * tone + w_freq * freq

The language-model and frequency components are normalised across the candidate
set so they share the ``[0, 1]`` scale of the tone and context components.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from time import perf_counter

import joblib

from .context_classifier import ContextClassifier
from .language_model import SPECIAL_TOKENS, NGramLanguageModel
from .preprocessing import (
    MODE_PARTIAL_WORD,
    LemmaNormalizer,
    detect_mode,
    partial_prefix,
    text_before_cursor,
    tokenize,
)
from .tone_scorer import ToneScorer

__all__ = ["DEFAULT_MODEL_PATH", "MailMatePredictor", "load_model"]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mailmate_model.joblib"

REQUIRED_KEYS = (
    "version",
    "categories",
    "tones",
    "vocab",
    "unigram_counts",
    "bigram_counts",
    "trigram_counts",
    "lambda",
    "vectorizer",
    "classifier",
    "tone_lexicon",
    "scoring_weights",
)


@lru_cache(maxsize=4)
def _load_artifact(path: str) -> dict:
    return joblib.load(path)


def load_model(path: str | Path | None = None) -> dict:
    """Load the artifact once per process and return the cached bundle.

    Repeated calls with the same path return the identical object, so the
    backend pays the deserialisation cost only on first use (~30 ms).
    """
    resolved = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {resolved}. "
            "Run notebooks/01_mailmate_nlp_training.ipynb and place "
            "mailmate_model.joblib in MailMate-AI/models/."
        )
    return _load_artifact(str(resolved.resolve()))


class MailMatePredictor:
    """CPU-only next-word suggester and context classifier."""

    def __init__(self, model: dict | None = None, model_path: str | Path | None = None) -> None:
        artifact = model if model is not None else load_model(model_path)
        missing = [key for key in REQUIRED_KEYS if key not in artifact]
        if missing:
            raise ValueError(f"Artifact is missing required keys: {missing}")

        self.artifact = artifact
        self.version = str(artifact["version"])
        self._categories = [str(label) for label in artifact["categories"]]
        self.tones = [str(tone) for tone in artifact["tones"]]
        self.weights = {key: float(value) for key, value in artifact["scoring_weights"].items()}

        config = artifact.get("config") or {}
        self.max_text_len = int(config.get("max_text_len", 2000))
        self.max_suggestions = int(config.get("max_suggestions", 3))

        self.vocab = dict(artifact["vocab"])
        self.lemmatizer = LemmaNormalizer.from_vectorizer(artifact["vectorizer"])
        self.lm = NGramLanguageModel(
            vocab=self.vocab,
            unigram_counts=artifact["unigram_counts"],
            bigram_counts=artifact["bigram_counts"],
            trigram_counts=artifact["trigram_counts"],
            lambdas=artifact["lambda"],
        )
        self.tone_scorer = ToneScorer(artifact["tone_lexicon"], self.tones)
        self.context = ContextClassifier(
            artifact["vectorizer"], artifact["classifier"], self.lemmatizer
        )

        self._candidates = [word for word in self.vocab if word not in SPECIAL_TOKENS]
        self._max_frequency = max(
            (self.lm.unigram_counts.get(word, 0) for word in self._candidates), default=1
        ) or 1

    # ------------------------------------------------------------------ basics
    @property
    def categories(self) -> list[str]:
        """Category labels in the classifier's canonical order."""
        return list(self._categories)

    def classify(self, text: str) -> dict:
        """Context classification for ``text``."""
        return self.context.classify(text)

    # --------------------------------------------------------------- prediction
    def predict(
        self,
        text: str,
        cursor_pos: int | None = None,
        category: str | None = None,
        tone: str | None = None,
        top_k: int = 3,
    ) -> dict:
        """Suggest completions for the caret position in ``text``.

        ``cursor_pos`` defaults to the end of ``text``. When the text ends
        mid-word (no trailing whitespace) candidates are filtered by that prefix
        and each suggestion is the *complete* word, suitable for replacing the
        in-progress token. Otherwise candidates are the next word to insert.

        Returns ``{"suggestions": [{"text", "score", "source"}], "category",
        "tone": {"label", "score"}, "latency_ms"}``.
        """
        started = perf_counter()
        raw = (text or "")[: self.max_text_len]
        body = text_before_cursor(raw, cursor_pos)
        mode = detect_mode(body)
        prefix = partial_prefix(body) if mode == MODE_PARTIAL_WORD else ""

        tokens = tokenize(body)
        if mode == MODE_PARTIAL_WORD and tokens:
            tokens = tokens[:-1]  # the in-progress word is not context yet

        if category is not None and category not in self._categories:
            raise ValueError(
                f"Unknown category {category!r}; expected one of {self._categories}"
            )
        if tone is not None and tone not in self.tones:
            raise ValueError(f"Unknown tone {tone!r}; expected one of {self.tones}")

        detection = self.classify(body)
        resolved_category = category or detection["category"]

        if tone is not None:
            tone_label = tone
            tone_score = self.tone_scorer.score_tokens(tokens, tone)
        else:
            tone_label, tone_score = self.tone_scorer.best_tone(tokens)

        candidates = (
            [word for word in self._candidates if word.startswith(prefix)]
            if prefix
            else list(self._candidates)
        )

        suggestions: list[dict] = []
        if candidates:
            suggestions = self._rank(
                candidates=candidates,
                body=body,
                tokens=tokens,
                category=resolved_category,
                tone=tone_label,
                top_k=max(0, min(int(top_k), self.max_suggestions)),
                join_directly=mode == MODE_PARTIAL_WORD,
            )

        latency_ms = (perf_counter() - started) * 1000.0
        return {
            "suggestions": suggestions,
            "category": resolved_category,
            "tone": {"label": tone_label, "score": round(float(tone_score), 6)},
            "latency_ms": round(latency_ms, 3),
        }

    def _rank(
        self,
        candidates: list[str],
        body: str,
        tokens: list[str],
        category: str,
        tone: str,
        top_k: int,
        join_directly: bool,
    ) -> list[dict]:
        if top_k <= 0:
            return []

        w_lm = self.weights.get("lm", 0.0)
        w_context = self.weights.get("context", 0.0)
        w_tone = self.weights.get("tone", 0.0)
        w_freq = self.weights.get("freq", 0.0)

        lm_raw = self.lm.candidate_scores(tokens, candidates)
        max_lm = max(lm_raw.values(), default=0.0) or 1.0

        features: dict[str, list[float]] = {}
        for candidate in candidates:
            features[candidate] = [
                lm_raw[candidate] / max_lm,
                self.lm.unigram_counts.get(candidate, 0) / self._max_frequency,
                self.tone_scorer.score_word(candidate, tone),
            ]

        # Cheap first pass (no classifier call) to pick a shortlist.
        shortlist = sorted(
            candidates,
            key=lambda word: w_lm * features[word][0]
            + w_tone * features[word][2]
            + w_freq * features[word][1],
            reverse=True,
        )[: self.shortlist_size]

        docs = [self._candidate_document(body, word, join_directly) for word in shortlist]
        context_scores = self.context.category_scores(docs, category)

        ranked = [
            {
                "text": word,
                "score": round(
                    w_lm * features[word][0]
                    + w_context * context_score
                    + w_tone * features[word][2]
                    + w_freq * features[word][1],
                    6,
                ),
                "source": "mixed",
            }
            for word, context_score in zip(shortlist, context_scores)
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _candidate_document(body: str, candidate: str, join_directly: bool) -> str:
        if join_directly:
            if body and not body[-1].isspace():
                body = body[: len(body) - len(partial_prefix(body))]
            return f"{body}{candidate}"
        if body and not body[-1].isspace():
            return f"{body} {candidate}"
        return f"{body}{candidate}"

    @property
    def shortlist_size(self) -> int:
        """How many candidates receive the (costlier) classifier pass."""
        return 24
