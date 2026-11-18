"""Model artifact and configuration loading for the MailMate AI backend.

The loader is the single place that knows where the artifact and root config
live. It validates the Phase 2 contract up front so a corrupt or stale export
fails loudly at startup instead of returning nonsense predictions.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
ROOT_DIR = PROJECT_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "mailmate_model.joblib"
CONFIG_PATH = ROOT_DIR / "config.json"

# ``src`` is a plain package next to ``backend``, not an installed dependency.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.predictor import MailMatePredictor, load_model  # noqa: E402

__all__ = [
    "BACKEND_DIR",
    "CONFIG_PATH",
    "MODEL_PATH",
    "PROJECT_DIR",
    "ROOT_DIR",
    "REQUIRED_ARTIFACT_KEYS",
    "ModelLoadError",
    "get_config",
    "get_predictor",
    "is_loaded",
    "load_config",
    "validate_artifact",
]

REQUIRED_ARTIFACT_KEYS: tuple[str, ...] = (
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
    "config",
)


class ModelLoadError(RuntimeError):
    """Raised when the artifact is missing, unreadable, or inconsistent."""


_lock = threading.Lock()
_predictor: MailMatePredictor | None = None
_config: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the root ``config.json`` (cached after the first successful read)."""
    resolved = Path(path) if path is not None else CONFIG_PATH
    try:
        with resolved.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ModelLoadError(f"config.json not found at {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"config.json is not valid JSON ({resolved}): {exc}") from exc
    if not isinstance(data, dict):
        raise ModelLoadError(f"config.json must contain a JSON object ({resolved})")
    return data


def get_config() -> dict[str, Any]:
    """Cached accessor for the root config."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def validate_artifact(
    artifact: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
    """Return a list of contract violations (empty means valid)."""
    problems: list[str] = []

    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing:
        problems.append(f"artifact missing required keys: {missing}")
        # Cannot trust further checks if structure is incomplete.
        return problems

    config_version = config.get("version")
    if artifact["version"] != config_version:
        problems.append(
            f"version mismatch: artifact={artifact['version']!r} config={config_version!r}"
        )

    vocab_size = len(artifact["vocab"])
    config_vocab_size = config.get("vocab_size")
    if vocab_size != config_vocab_size:
        problems.append(
            f"vocab_size mismatch: artifact={vocab_size} config={config_vocab_size!r}"
        )

    try:
        classifier_classes = list(artifact["classifier"].classes_)
    except AttributeError:
        problems.append("artifact['classifier'] has no classes_ attribute")
    else:
        if classifier_classes != list(artifact["categories"]):
            problems.append(
                "classifier.classes_ does not match artifact categories: "
                f"{classifier_classes} != {list(artifact['categories'])}"
            )

    config_categories = config.get("categories")
    if config_categories is not None and list(config_categories) != list(artifact["categories"]):
        problems.append("artifact categories do not match config.json categories")

    nested = artifact.get("config")
    if isinstance(nested, Mapping):
        if nested.get("version") not in (None, artifact["version"]):
            problems.append(
                "artifact['config'].version disagrees with artifact version: "
                f"{nested.get('version')!r} != {artifact['version']!r}"
            )
    else:
        problems.append("artifact['config'] must be a mapping")

    return problems


def get_predictor() -> MailMatePredictor:
    """Return the process-wide ``MailMatePredictor`` singleton.

    The artifact is deserialised once, validated, and wrapped once; every later
    call returns the same instance, so HTTP handlers pay no load cost.
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    with _lock:
        if _predictor is not None:
            return _predictor

        if not MODEL_PATH.is_file():
            raise ModelLoadError(
                f"model artifact not found at {MODEL_PATH}; run "
                "notebooks/01_mailmate_nlp_training.ipynb and copy "
                "mailmate_model.joblib into MailMate-AI/models/"
            )

        try:
            artifact = load_model(MODEL_PATH)
        except Exception as exc:  # corrupt/pickle-incompatible artifact
            raise ModelLoadError(f"failed to load {MODEL_PATH}: {exc}") from exc

        problems = validate_artifact(artifact, get_config())
        if problems:
            raise ModelLoadError("artifact validation failed: " + "; ".join(problems))

        _predictor = MailMatePredictor(model=artifact)
        return _predictor


def is_loaded() -> bool:
    """True once the predictor singleton has been created successfully."""
    return _predictor is not None
