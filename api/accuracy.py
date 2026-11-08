"""
Accuracy self-check: is every pipeline stage running at full strength?

Config alone can't answer that — a stage can be *set* to its best backend and
still be running a fallback (model failed to download, binary not installed,
key missing, quota exhausted). This checks what will actually run.

    GET /health/accuracy            fast: config + keys + cheap probes
    GET /health/accuracy?deep=true  also loads the NLI model, the sentence
                                    encoder and the 22-model ensemble to prove
                                    they work (slow the first time)

The same report is logged at startup, so a degraded deployment says so in its
logs. Individual verdicts additionally carry `degradations` describing anything
that ran below full strength for that specific check.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Any

log = logging.getLogger(__name__)

try:
    from config import settings
except ImportError:  # pragma: no cover
    settings = None


def _stage(name: str, ok: bool, running: str, detail: str = "", fix: str = "") -> dict[str, Any]:
    return {"stage": name, "full_accuracy": ok, "running": running,
            "detail": detail, "fix": "" if ok else fix}


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def accuracy_report(deep: bool = False) -> dict[str, Any]:
    s = settings
    from paper_trail_core.nlp.llm import llm_configured
    llm_ok = llm_configured()
    stages: list[dict[str, Any]] = []

    # 1. Claim extraction
    if s.CLAIM_BACKEND == "llm" and llm_ok:
        stages.append(_stage("claim_extraction", True, f"LLM ({s.LLM_MODEL})",
                             "Claude extracts self-contained claims; grounded against the text."))
    else:
        stages.append(_stage(
            "claim_extraction", False,
            "local heuristic" if s.CLAIM_BACKEND != "naive" else "naive splitter",
            "Sentence ranking only: misses claims spread over sentences and can't resolve "
            "pronouns ('he said…').",
            "Set ANTHROPIC_API_KEY in .env (and CLAIM_BACKEND=llm)."))

    # 2. Language ID + translation
    lang_ok = _has("langdetect")
    trans_ok = s.TRANSLATION_BACKEND == "llm" and llm_ok
    stages.append(_stage(
        "translation", lang_ok and trans_ok,
        f"{'langdetect' if lang_ok else 'NO language detection'} + "
        f"{'LLM translation' if trans_ok else 'no translation'}",
        "Non-English (e.g. Hindi) articles are translated before NLI." if trans_ok else
        "Non-English articles can't be translated, so their claims are checked as-is "
        "against English evidence; expect 'uncertain'.",
        "pip install langdetect; set ANTHROPIC_API_KEY." if not lang_ok else "Set ANTHROPIC_API_KEY."))

    # 3. Evidence retrieval
    providers = []
    if s.NEWS_API_KEY:
        providers.append("NewsAPI")
    if s.TAVILY_API_KEY:
        providers.append("Tavily")
    if s.SERPAPI_KEY:
        providers.append("SerpAPI")
    providers.append("Wikipedia")
    if s.REGION_AWARE and s.REGIONAL_NEWS_PROVIDER != "none":
        providers.append(f"regional news ({s.REGIONAL_NEWS_PROVIDER})")
    news_detail = ""
    try:
        from paper_trail_core.agents.evidence_retriever import newsapi_status
        st = newsapi_status()
        if st.get("in_cooldown") or st.get("cooldown"):
            news_detail = " NewsAPI is currently rate-limited (daily quota hit)."
    except Exception:
        pass
    live = s.EVIDENCE_BACKEND == "live"
    broad = bool(s.TAVILY_API_KEY or s.SERPAPI_KEY)
    stages.append(_stage(
        "evidence_retrieval", live and bool(s.NEWS_API_KEY) and not news_detail,
        f"{s.EVIDENCE_BACKEND}: " + ", ".join(providers),
        ("Searches trusted news + reference sources." +
         ("" if broad else " No general web search key, so coverage is news + Wikipedia only.")
         + news_detail),
        "Set EVIDENCE_BACKEND=live and NEWS_API_KEY." if not live or not s.NEWS_API_KEY
        else "Wait for the NewsAPI quota to reset, or add TAVILY_API_KEY."))

    # 4. NLI (the core judgement)
    if s.NLI_BACKEND != "transformer":
        stages.append(_stage("nli", False, s.NLI_BACKEND,
                             "Word overlap can't detect contradiction.",
                             "Set NLI_BACKEND=transformer."))
    elif not (_has("torch") and _has("transformers")):
        stages.append(_stage("nli", False, "unavailable (torch/transformers not installed)",
                             "Verdicts will be 'uncertain'.",
                             "pip install torch transformers sentencepiece protobuf"))
    else:
        ok, device = True, "unknown"
        if deep:
            from paper_trail_core.nlp import nli
            ok = nli.is_available()
            try:
                device = nli._resolve_device()
            except Exception:
                pass
        stages.append(_stage(
            "nli", ok, f"transformer {s.NLI_MODEL}" + (f" on {device}" if deep else ""),
            ("Loaded and working." if deep and ok else "Installed (run deep=true to load it).")
            + (" CPU is slower but exactly as accurate as GPU." if device == "cpu" else ""),
            "The model failed to load (see server log): check disk space / network for the "
            "first download."))

    # 5. Relevance gating (sentence embeddings)
    if not _has("sentence_transformers"):
        stages.append(_stage("relevance_gating", False, "unavailable",
                             "Evidence can't be filtered for relevance.",
                             "pip install sentence-transformers"))
    else:
        ok = True
        if deep:
            try:
                from paper_trail_core.nlp.embeddings import get_embedder
                get_embedder()
            except Exception as exc:
                ok = False
                log.warning("accuracy: embedder failed: %s", exc)
        stages.append(_stage("relevance_gating", ok, f"{s.EMBED_MODEL}, cosine >= {s.RELEVANCE_MIN}",
                             "", "The sentence encoder failed to load (see server log)."))

    # 6. Style ensemble (22 models + stacking)
    if s.ENSEMBLE_BACKEND != "real":
        stages.append(_stage("style_ensemble", False, s.ENSEMBLE_BACKEND, "",
                             "Set ENSEMBLE_BACKEND=real."))
    else:
        loaded, hybrid, missing = None, None, []
        if deep:
            from paper_trail_core.agents.stylistic_ensemble_agent import _get_registry
            reg = _get_registry()
            if reg is not None:
                loaded = reg.total_loaded
                hybrid = reg.hybrid_meta_learner is not None
                have = set(reg.classical) | set(getattr(reg, "dl_models", {}) or {})
                missing = [b for b in reg.hybrid_base_names if b not in have]
        ok = (not deep) or (bool(hybrid) and not missing)
        detail = ("Not loaded (run deep=true)." if not deep else
                  f"{loaded} models loaded; stacking model {'present' if hybrid else 'MISSING'}"
                  + (f"; base models missing: {', '.join(missing)}" if missing else "") + ".")
        stages.append(_stage("style_ensemble", ok, "real", detail,
                             "Install the packages for the missing models (xgboost, lightgbm, "
                             "catboost, tensorflow) — see requirements.txt."))

    # 7. Fusion adjudicator
    try:
        from paper_trail_core.agents import fusion
        d = fusion.describe()
        stages.append(_stage("fusion", bool(d.get("trained")),
                             d.get("method", "unknown"),
                             f"Calibrated to {s.TARGET_PRECISION:.0%} precision on held-out data."
                             if d.get("trained") else "Hand-set weights instead of the trained model.",
                             f"Missing {s.FUSION_MODEL_PATH}."))
    except Exception as exc:
        stages.append(_stage("fusion", False, "error", str(exc), "See server log."))

    # 8. Location awareness
    stages.append(_stage(
        "location", s.REGION_AWARE and s.GEOCODER_BACKEND == "nominatim",
        f"{s.GEOCODER_BACKEND}" if s.REGION_AWARE else "off",
        "Claims about a place also get regional news.",
        "Set REGION_AWARE=true and GEOCODER_BACKEND=nominatim."))

    # 9. OCR (image input)
    ocr_ok, ocr_detail = False, ""
    if s.OCR_BACKEND == "tesseract":
        try:
            from paper_trail_core.nlp import ocr
            ocr_ok = ocr.is_available("tesseract")
        except Exception as exc:
            ocr_detail = str(exc)
    stages.append(_stage(
        "ocr", ocr_ok, s.OCR_BACKEND if ocr_ok else f"{s.OCR_BACKEND} (not installed)",
        f"Languages: {s.OCR_LANGS}." if ocr_ok else
        ("Image checks will abstain. " + ocr_detail).strip(),
        "Install the Tesseract binary + Hindi pack (UB-Mannheim build on Windows) and "
        "pip install pytesseract."))

    # 10. Chat replies
    stages.append(_stage(
        "chat_replies", llm_ok and getattr(s, "QA_ENABLED", True),
        f"LLM ({getattr(s, 'QA_MODEL', '')}) with verify_claim tool" if llm_ok else "rule-based",
        "", "Set ANTHROPIC_API_KEY."))

    full = all(st["full_accuracy"] for st in stages)
    return {
        "full_accuracy": full,
        "deep": deep,
        "summary": "All stages at full accuracy." if full else
        "Running below full accuracy: " + ", ".join(st["stage"] for st in stages if not st["full_accuracy"]),
        "stages": stages,
    }


def log_report(deep: bool) -> None:
    try:
        report = accuracy_report(deep=deep)
    except Exception as exc:  # never block startup on the self-check
        log.warning("accuracy self-check failed: %s", exc)
        return
    if report["full_accuracy"]:
        log.info("accuracy: %s", report["summary"])
        return
    log.warning("accuracy: %s", report["summary"])
    for st in report["stages"]:
        if not st["full_accuracy"]:
            log.warning("accuracy:   %-18s running=%s | fix: %s", st["stage"], st["running"], st["fix"])
