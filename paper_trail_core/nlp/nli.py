"""
Natural Language Inference engine.

Given an evidence sentence (premise) and a claim (hypothesis), returns
probabilities over entailment / neutral / contradiction. This is the component
that decides whether retrieved evidence actually supports a claim, and it
replaces a lexical-overlap heuristic that could not detect contradiction at
all — it scored "RBI holds repo rate steady at 6.5 percent" as 0.83
*entailment* for the claim "India raised its repo rate to 6.75 percent",
because the two sentences share most of their words.

Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli.

Chosen over facebook/bart-large-mnli (which the old code named) because it is
trained on FEVER — the fact-verification benchmark, whose task is literally
"does this retrieved evidence support this claim" — plus adversarial ANLI, and
because it is roughly a quarter the size.

A safety note that is not optional: the two checkpoints use **inverted** label
orders.

    facebook/bart-large-mnli        0=contradiction 1=neutral 2=entailment
    DeBERTa-v3-base-mnli-fever-anli 0=entailment    1=neutral 2=contradiction

The old code hardcoded the first. Pointing it at this model would have
inverted every verdict — the pipeline would run without error and be exactly,
confidently wrong. Labels are therefore read from the loaded model's own
config and asserted, never assumed from position.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from config import settings
    _MODEL_NAME = settings.NLI_MODEL
    _DEVICE_PREF = settings.NLI_DEVICE
    _FP16 = settings.NLI_FP16
    _BATCH = settings.NLI_BATCH_SIZE
    _MAX_LEN = settings.NLI_MAX_LENGTH
except Exception:  # pragma: no cover
    _MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    _DEVICE_PREF, _FP16, _BATCH, _MAX_LEN = "auto", True, 32, 256

log = logging.getLogger(__name__)

REQUIRED_LABELS = {"entailment", "neutral", "contradiction"}

# Premises longer than this are cut before tokenizing. Belt and braces
# alongside truncation="only_first" — keeps tokenizer work bounded on the
# occasional multi-kilobyte scraped snippet.
MAX_PREMISE_CHARS = 1500

_bundle = None
_lock = threading.Lock()


@dataclass(frozen=True)
class NLIResult:
    entail: float
    neutral: float
    contradict: float
    backend: str = "transformer"

    @property
    def label(self) -> str:
        return max(
            (("entailment", self.entail), ("neutral", self.neutral), ("contradiction", self.contradict)),
            key=lambda kv: kv[1],
        )[0]


def build_label_index(id2label: dict) -> dict[str, int]:
    """
    Map label name -> logit index, from the model's own config.

    Raises rather than guessing: an unrecognised label set means we cannot
    know which column is which, and silently picking an order is how you ship
    a system that inverts entailment and contradiction without anyone noticing.
    """
    idx = {str(v).lower().strip(): int(k) for k, v in id2label.items()}
    missing = REQUIRED_LABELS - idx.keys()
    if missing:
        raise ValueError(
            f"NLI model config is missing label(s) {sorted(missing)}; "
            f"got id2label={id2label}. Refusing to guess the logit order."
        )
    return idx


def _resolve_device() -> str:
    if _DEVICE_PREF in ("cuda", "cpu"):
        return _DEVICE_PREF
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_nli():
    """Load (once) and return (tokenizer, model, label_index, device)."""
    global _bundle
    if _bundle is not None:
        return _bundle
    with _lock:
        if _bundle is not None:
            return _bundle

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(0)
        device = _resolve_device()
        log.info("nli: loading %s on %s", _MODEL_NAME, device)

        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        kwargs = {}
        if device == "cuda" and _FP16:
            kwargs["dtype"] = torch.float16
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME, **kwargs)
        model.to(device)
        model.eval()

        label_index = build_label_index(model.config.id2label)
        log.info("nli: ready — label_index=%s", label_index)
        _bundle = (tokenizer, model, label_index, device)
    return _bundle


def infer_batch(premises: list[str], hypotheses: list[str]) -> list[NLIResult]:
    """
    Run NLI over aligned (premise, hypothesis) pairs.

    Truncation is `only_first`, so a long premise gets cut but the hypothesis —
    the claim under test — always survives intact. Combined with the token cap
    that claim extraction applies, the claim is never silently truncated away.

    Softmax runs in fp32 even when the model is fp16: half precision loses
    resolution in the distribution tails, and the tails are exactly where a
    confident contradiction lives.
    """
    if not premises:
        return []
    if len(premises) != len(hypotheses):
        raise ValueError(f"premise/hypothesis length mismatch: {len(premises)} vs {len(hypotheses)}")

    import torch

    tokenizer, model, idx, device = get_nli()
    i_ent, i_neu, i_con = idx["entailment"], idx["neutral"], idx["contradiction"]

    premises = [p[:MAX_PREMISE_CHARS] for p in premises]
    results: list[NLIResult] = []

    with torch.inference_mode():
        for start in range(0, len(premises), _BATCH):
            batch_p = premises[start:start + _BATCH]
            batch_h = hypotheses[start:start + _BATCH]
            enc = tokenizer(
                batch_p, batch_h,
                padding=True,
                truncation="only_first",
                max_length=_MAX_LEN,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                results.append(
                    NLIResult(
                        entail=float(row[i_ent]),
                        neutral=float(row[i_neu]),
                        contradict=float(row[i_con]),
                    )
                )
    return results


def warmup() -> None:
    """Force model load ahead of first request (called from FastAPI lifespan)."""
    try:
        infer_batch(["The sky is blue."], ["The sky is blue."])
        log.info("nli: warmup complete")
    except Exception as exc:  # pragma: no cover
        log.warning("nli: warmup failed — %s", exc)


def is_available() -> bool:
    """Whether the transformer backend can actually run, without raising."""
    try:
        get_nli()
        return True
    except Exception as exc:
        log.warning("nli: transformer backend unavailable — %s", exc)
        return False
