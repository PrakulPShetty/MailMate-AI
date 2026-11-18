"""Standalone artifact verifier for MailMate AI.

Runs the same validation the Flask backend performs at startup, then exercises
a sample prediction. Exit code is 0 on success and 1 on any failure, so it can
gate CI or a manual pre-demo check.

Usage::

    python MailMate-AI/verify_models.py
    python verify_models.py            # from MailMate-AI/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode rich paths or dashes.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend import loader  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"


def _report(ok: bool, label: str, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    print(f"  [{PASS if ok else FAIL}] {label}{suffix}")
    return ok


def main() -> int:
    print("MailMate AI - model artifact verification")
    print(f"  artifact : {loader.MODEL_PATH}")
    print(f"  config   : {loader.CONFIG_PATH}")
    print()

    success = True

    # 1. Files exist.
    success &= _report(loader.MODEL_PATH.is_file(), "artifact file exists")
    success &= _report(loader.CONFIG_PATH.is_file(), "config.json exists")
    if not success:
        print("\nFAILED - required files are missing.")
        return 1

    # 2. Config parses, artifact loads, contract validates.
    try:
        config = loader.get_config()
        success &= _report(True, "config.json parses", f"version={config.get('version')!r}")
    except loader.ModelLoadError as exc:
        _report(False, "config.json parses", str(exc))
        print("\nFAILED - config cannot be read.")
        return 1

    from src.predictor import load_model

    try:
        artifact = load_model(loader.MODEL_PATH)
        success &= _report(True, "artifact deserialises via joblib")
    except Exception as exc:
        _report(False, "artifact deserialises via joblib", str(exc))
        print("\nFAILED - cannot load the artifact.")
        return 1

    problems = loader.validate_artifact(artifact, config)
    success &= _report(
        not problems,
        "artifact contract matches config.json",
        "; ".join(problems) if problems else f"{len(artifact)} keys, vocab={len(artifact['vocab'])}",
    )

    # 3. Predictor singleton.
    try:
        predictor = loader.get_predictor()
        success &= _report(True, "predictor initialises", f"version={predictor.version}")
    except Exception as exc:
        _report(False, "predictor initialises", str(exc))
        print("\nFAILED - predictor could not be created.")
        return 1

    # 4. Sample inference in both modes.
    samples = [
        ("next-word", "I am writing to follow up regarding my internship "),
        ("partial-word", "Please find attached my res"),
    ]
    for label, text in samples:
        try:
            result = predictor.predict(text)
            ok = bool(result["suggestions"]) and result["category"] in predictor.categories
            detail = (
                f"{[s['text'] for s in result['suggestions']]} "
                f"category={result['category']} {result['latency_ms']:.2f}ms"
            )
        except Exception as exc:
            ok, detail = False, str(exc)
        success &= _report(ok, f"{label} prediction", detail)

    try:
        classified = predictor.classify("Please find attached my resume for the position")
        success &= _report(
            classified["category"] in predictor.categories,
            "classify() returns a known category",
            f"{classified['category']} ({classified['confidence']:.2%})",
        )
    except Exception as exc:
        success &= _report(False, "classify() returns a known category", str(exc))

    print()
    print("RESULT: ALL CHECKS PASSED" if success else "RESULT: FAILURES DETECTED")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
