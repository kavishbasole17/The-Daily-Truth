"""
Low-level OCR engine wrapper.

Tesseract over PaddleOCR by default — see config.OCR_BACKEND and
docs/ARCHITECTURE.md's "Decisions locked in" for why (this project runs on
CPU locally; Tesseract is the lighter-weight CPU option, PaddleOCR was the
GPU-accelerated choice for a GPU that turned out not to be usable locally).

Tesseract itself is a system binary, not a pip package — `pytesseract` only
wraps a subprocess call to it. It must be installed separately, along with
the Hindi (`hin`) language pack, before this module can do real OCR:

    Windows: https://github.com/UB-Mannheim/tesseract/wiki (includes an
             installer option for additional language packs)
    Debian/Ubuntu: apt install tesseract-ocr tesseract-ocr-hin

Without it, `is_available()` returns False and the OCR agent node degrades
to the existing abstention path rather than guessing at unreadable text —
the same pattern the NLI verifier uses when torch/transformers aren't usable
(paper_trail_core/nlp/nli.py).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# Devanagari Unicode block (U+0900-U+097F) covers Hindi, Marathi, and others
# written in the same script. Good enough to answer "was this Devanagari or
# Latin script" for the OCR confidence/script-tagging requirement; it does
# not disambiguate which Devanagari-script language was detected — that's
# lang_id's job downstream (paper_trail_core/agents/lang_translation_agent.py,
# not yet built — see docs/ARCHITECTURE.md stretch tier).
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class OcrResult:
    text: str
    confidence: float           # 0-1, mean word confidence across detected blocks
    script: str                 # "devanagari" | "latin" | "mixed" | "unknown"
    n_blocks: int                # detected text blocks, for diagnostics


def detect_script(text: str) -> str:
    has_dev = bool(_DEVANAGARI_RE.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_dev and has_lat:
        return "mixed"
    if has_dev:
        return "devanagari"
    if has_lat:
        return "latin"
    return "unknown"


def preprocess(image: Image.Image) -> Image.Image:
    """
    Minimal, dependency-light preprocessing: grayscale, autocontrast, and
    upscaling small images (Tesseract's accuracy drops sharply below ~300
    DPI-equivalent). Full deskew/denoise (the brief's ask) needs an image
    library with real CV ops (OpenCV) and is deferred — see
    docs/ARCHITECTURE.md stretch tier — rather than pulling in a new heavy
    dependency for this first pass.
    """
    img = ImageOps.exif_transpose(image)  # respect phone-camera orientation
    img = img.convert("L")                # grayscale
    img = ImageOps.autocontrast(img)
    w, h = img.size
    if max(w, h) < 1000:
        scale = 1000 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception as exc:
        log.warning("ocr: tesseract backend unavailable (%s)", exc)
        return False


def is_available(backend: str) -> bool:
    if backend == "mock":
        return True
    if backend == "tesseract":
        return _tesseract_available()
    return False


def extract_mock(image: Image.Image) -> OcrResult:
    """Deterministic stand-in for tests and OCR_BACKEND=mock — no system binary needed."""
    return OcrResult(text="", confidence=0.0, script="unknown", n_blocks=0)


def extract_tesseract(image: Image.Image, langs: str = "hin+eng") -> OcrResult:
    """
    Run Tesseract over the image, returning aggregated text + mean confidence.

    Uses image_to_data rather than image_to_string so per-word confidences are
    available to average into a single score — image_to_string alone gives no
    confidence signal at all, which is exactly the number the abstention path
    needs.
    """
    import pytesseract
    from pytesseract import Output

    img = preprocess(image)
    data = pytesseract.image_to_data(img, lang=langs, output_type=Output.DICT)

    words, confs = [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        conf = float(conf)
        if word and conf >= 0:  # tesseract uses -1 for non-text blocks
            words.append(word)
            confs.append(conf)

    text = " ".join(words)
    mean_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return OcrResult(
        text=text,
        confidence=mean_conf,
        script=detect_script(text),
        n_blocks=len(words),
    )
