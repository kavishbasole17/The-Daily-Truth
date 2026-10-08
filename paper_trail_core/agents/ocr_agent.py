"""
OCR agent node — image input's entry point into the pipeline.

Only reached when `state["input_type"] == "image"` (see the conditional
entry point in pipeline.py); text/URL submissions skip this node entirely
and are unaffected by anything in this module.

Backends (config.OCR_BACKEND):
  tesseract — default. Tesseract + Hindi/English language packs, CPU-only.
              Requires the system binary — see paper_trail_core/nlp/ocr.py.
  mock      — deterministic empty result, for tests. Never select in production.
  off       — OCR disabled; any image submission abstains immediately.

Confidence gating, not guessing: if the mean OCR confidence falls below
config.OCR_CONFIDENCE_MIN, this node does not attempt to hand a low-quality
transcription to the rest of the pipeline. It returns an empty article_text
with a critical degradation instead. pipeline.py routes an empty
article_text straight to a dedicated abstain node rather than into
extract_claims — see _ocr_abstain_node's docstring there for why the
existing "no claims" adjudicator path turned out to be the wrong reuse for
this case.
"""
from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _BACKEND = settings.OCR_BACKEND
    _LANGS = settings.OCR_LANGS
    _CONF_MIN = settings.OCR_CONFIDENCE_MIN
except ImportError:  # pragma: no cover
    _BACKEND, _LANGS, _CONF_MIN = "tesseract", "hin+eng", 0.55

from paper_trail_core.agents.state import PipelineState
from paper_trail_core.degradations import degrade

log = logging.getLogger(__name__)


def _load_image(path: str):
    from PIL import Image
    return Image.open(path)


def run_ocr(image_path: str):
    """
    Run the configured OCR backend over an image file.

    Returns an `OcrResult` (paper_trail_core.nlp.ocr.OcrResult). Raises on
    genuine failure (missing file, unreadable image, backend unavailable) —
    the caller (ocr_node) is responsible for catching and degrading; this
    function stays honest about whether it actually ran.
    """
    from paper_trail_core.nlp import ocr as ocr_engine

    image = _load_image(image_path)

    if _BACKEND == "mock":
        return ocr_engine.extract_mock(image)
    if _BACKEND == "tesseract":
        return ocr_engine.extract_tesseract(image, langs=_LANGS)
    raise ValueError(f"Unknown or unimplemented OCR_BACKEND: {_BACKEND!r}")


def ocr_node(state: PipelineState) -> dict:
    """LangGraph node: extract text from the submitted image."""
    image_path = state.get("raw_image_ref")
    log.info("ocr_agent: running (backend=%s)", _BACKEND)

    if _BACKEND == "off":
        return {
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "ocr_script": "unknown",
            "article_text": "",
            "degradations": degrade(
                "ocr", "backend_off",
                "OCR is disabled (OCR_BACKEND=off); image submissions cannot "
                "be processed; the verdict is uncertain.",
                severity="critical",
            ),
        }

    if not image_path:
        return {
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "ocr_script": "unknown",
            "article_text": "",
            "degradations": degrade(
                "ocr", "no_image",
                "input_type was 'image' but no image was attached.",
                severity="critical",
            ),
        }

    try:
        result = run_ocr(image_path)
    except Exception as exc:
        # Exception detail (which can include local file paths) stays in the
        # server log only, matching every other stage's guard convention —
        # see paper_trail_core/agents/pipeline.py's with_stage_guard.
        log.error("ocr_agent: extraction failed — %s", exc, exc_info=True)
        return {
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "ocr_script": "unknown",
            "article_text": "",
            "degradations": degrade(
                "ocr", "extraction_failed",
                f"OCR extraction failed ({type(exc).__name__}); the image was not "
                "guessed at, and the verdict is uncertain.",
                severity="critical",
            ),
        }

    if result.confidence < _CONF_MIN:
        log.info("ocr_agent: confidence %.2f below floor %.2f, abstaining",
                  result.confidence, _CONF_MIN)
        return {
            "ocr_text": result.text,
            "ocr_confidence": result.confidence,
            "ocr_script": result.script,
            "article_text": "",
            "degradations": degrade(
                "ocr", "low_confidence",
                f"OCR confidence ({result.confidence:.2f}) fell below the "
                f"{_CONF_MIN:.2f} floor. Rather than verify a possibly-wrong "
                "transcription, the verdict is left uncertain.",
                severity="critical",
            ),
        }

    log.info("ocr_agent: extracted %d chars, confidence %.2f, script=%s",
              len(result.text), result.confidence, result.script)
    return {
        "ocr_text": result.text,
        "ocr_confidence": result.confidence,
        "ocr_script": result.script,
        "article_text": result.text,
        "degradations": [],
    }
