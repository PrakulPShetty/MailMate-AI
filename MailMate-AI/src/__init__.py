"""MailMate AI NLP core - runtime prediction engine (Phase 3)."""

from .context_classifier import ContextClassifier
from .evaluation import classifier_accuracy, evaluate, next_word_accuracy, perplexity
from .language_model import NGramLanguageModel
from .predictor import DEFAULT_MODEL_PATH, MailMatePredictor, load_model
from .preprocessing import LemmaNormalizer, detect_mode, tokenize
from .tone_scorer import ToneScorer

__all__ = [
    "ContextClassifier",
    "DEFAULT_MODEL_PATH",
    "LemmaNormalizer",
    "MailMatePredictor",
    "NGramLanguageModel",
    "ToneScorer",
    "classifier_accuracy",
    "detect_mode",
    "evaluate",
    "load_model",
    "next_word_accuracy",
    "perplexity",
    "tokenize",
]
