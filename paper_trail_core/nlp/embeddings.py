"""
Sentence-embedding singleton, used for two jobs:

  1. Evidence relevance gating — does this snippet actually address the claim,
     or does it merely share vocabulary with it? Lexical overlap cannot tell
     the difference, which is how a Wikipedia page *about* the Reserve Bank of
     India ended up counted as evidence for a specific interest-rate figure.
  2. Claim deduplication — near-identical claims waste the retrieval budget.

Uses all-MiniLM-L6-v2, which was already present in the local HuggingFace
cache (87 MB), so this adds no download. Embeddings are L2-normalised, so
cosine similarity is a plain dot product.
"""
from __future__ import annotations

import logging
import os
import sys
import threading

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _MODEL_NAME = settings.EMBED_MODEL
    _DEVICE_PREF = settings.NLI_DEVICE
except Exception:  # pragma: no cover - config is always importable in practice
    _MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    _DEVICE_PREF = "auto"

log = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def _resolve_device() -> str:
    if _DEVICE_PREF in ("cuda", "cpu"):
        return _DEVICE_PREF
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_embedder():
    """Load (once) and return the SentenceTransformer. Thread-safe."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer
        device = _resolve_device()
        log.info("embeddings: loading %s on %s", _MODEL_NAME, device)
        _model = SentenceTransformer(_MODEL_NAME, device=device)
        log.info("embeddings: ready")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """
    Encode `texts` to an (n, d) array of L2-normalised vectors.

    Returns an empty (0, 0) array for empty input so callers can branch on
    `.size` without a special case.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_embedder()
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.asarray(vecs, dtype=np.float32)


def cosine_to(query: str, candidates: list[str]) -> np.ndarray:
    """Cosine similarity of `query` against each candidate, shape (len(candidates),)."""
    if not candidates:
        return np.zeros((0,), dtype=np.float32)
    vecs = embed([query] + candidates)
    if vecs.size == 0:
        return np.zeros((len(candidates),), dtype=np.float32)
    return vecs[1:] @ vecs[0]


def pairwise_cosine(texts: list[str]) -> np.ndarray:
    """Full (n, n) cosine matrix — used for claim dedup."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = embed(texts)
    return vecs @ vecs.T


def warmup() -> None:
    """Force model load ahead of first request (called from FastAPI lifespan)."""
    try:
        embed(["warmup"])
    except Exception as exc:  # pragma: no cover
        log.warning("embeddings: warmup failed — %s", exc)
