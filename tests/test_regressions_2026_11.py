"""
Regression tests for the 2026-11-16 bug-fix pass (see AGENTS.md, Part 6).

Each test pins one defect that silently lowered accuracy:

  * a hung evidence provider discarded every claim's evidence, not just its own
  * a year sitting next to a percentage was read as a percentage
  * LLM claims that drew on two adjacent sentences were dropped as hallucinated
  * the four Keras base models failed to load under the installed Keras
  * the API and the pipeline each loaded the whole model registry
"""
from __future__ import annotations

import os
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from paper_trail_core.agents import evidence_retriever as ER
from paper_trail_core.nlp.numeric import extract_numbers, numeric_conflict

# ── Evidence retrieval survives one slow claim ────────────────────────────────

def test_slow_claim_does_not_discard_other_claims_evidence(monkeypatch):
    """
    Before: `pool.map` waited for the slowest claim, the stage guard timed the
    whole node out, and the fallback reported *every* claim as unverifiable.
    Now the slow claim alone is reported, with status "timeout".
    """
    release = threading.Event()

    def fake_retrieve(claim, idx, **_kw):
        if idx == 1:
            release.wait(4.0)          # hangs well past the 1 s deadline below
        return {"claim": claim, "snippets": [{"text": "x", "relevance": 0.7, "trust": 0.9}],
                "status": "ok", "cache_hits": 0, "queries": 1}

    monkeypatch.setattr(ER, "retrieve_for_claim", fake_retrieve)
    monkeypatch.setattr(ER, "_MIN_DEADLINE_S", 0.5)
    monkeypatch.setattr(ER, "_DEADLINE_MARGIN_S", 0.0)
    monkeypatch.setattr(ER, "_STAGE_BUDGET_S", 1.0)

    started = time.time()
    try:
        out = ER.retrieve_evidence_node({"claims": ["fast one", "slow one", "fast two"],
                                         "article_text": "x" * 500, "claim_locations": []})
    finally:
        release.set()

    assert time.time() - started < 3.5
    statuses = [e["status"] for e in out["evidence"]]
    assert statuses == ["ok", "timeout", "ok"]
    assert out["retrieval_stats"]["n_with_evidence"] == 2
    assert out["retrieval_stats"]["n_timeout"] == 1
    assert any(d["kind"] == "retrieval_timeout" for d in out["degradations"])


def test_claim_whose_retrieval_raises_does_not_fail_the_node(monkeypatch):
    def fake_retrieve(claim, idx, **_kw):
        if idx == 0:
            raise RuntimeError("provider exploded")
        return {"claim": claim, "snippets": [{"text": "x", "relevance": 0.7, "trust": 0.9}],
                "status": "ok", "cache_hits": 0, "queries": 1}

    monkeypatch.setattr(ER, "retrieve_for_claim", fake_retrieve)
    out = ER.retrieve_evidence_node({"claims": ["a", "b"], "article_text": "x" * 500,
                                     "claim_locations": []})
    assert [e["status"] for e in out["evidence"]] == ["error", "ok"]


# ── Numeric units stop at the neighbouring figure ─────────────────────────────

def test_year_beside_a_percentage_is_a_year():
    nums = dict(extract_numbers("Inflation was 5 percent in 2019, up from 4.1 percent in 2018."))
    assert nums[2019.0] == "year"
    assert nums[2018.0] == "year"
    assert nums[5.0] == "percent"


def test_preceding_currency_symbol_still_binds():
    assert ("currency" in {u for _, u in extract_numbers("worth $5 billion")})
    assert ("currency" in {u for _, u in extract_numbers("worth Rs. 1,500 crore")})


def test_different_year_same_percentage_is_a_conflict():
    assert numeric_conflict("Tesla revenue grew 5 percent in 2023",
                            "Tesla revenue grew 5 percent in 2022") == 1.0


# ── LLM claim grounding compares against sentence pairs ───────────────────────

def test_hallucination_check_embeds_adjacent_sentence_pairs():
    from paper_trail_core.agents import claim_extractor as CE

    text = ("The minister spoke in Pune on Monday. He said the bridge would open in March. "
            "Traffic has been diverted since January.")
    seen: list[list[str]] = []

    def fake_embed(texts):
        seen.append(list(texts))
        n = len(texts)
        # Make the claim match only the two-sentence window (index 1 + pair 0).
        vecs = np.zeros((n, 4), dtype=np.float32)
        vecs[0] = [1, 0, 0, 0]              # the claim
        vecs[1 + 3] = [1, 0, 0, 0]          # first sentence pair, after the 3 sentences
        return vecs

    with patch("paper_trail_core.nlp.embeddings.embed", fake_embed):
        kept = CE._reject_hallucinations(
            text, ["The minister said in Pune on Monday that the bridge would open in March."])
    assert kept, "a claim grounded in two adjacent sentences must survive"
    # 1 claim + 3 sentences + 2 adjacent pairs were embedded
    assert len(seen[0]) == 1 + 3 + 2


# ── Keras models saved by a newer Keras still load ────────────────────────────

def test_forward_keys_are_stripped_recursively():
    from api.model_loader import ModelRegistry

    cfg = {"class_name": "Sequential", "config": {"layers": [
        {"class_name": "Embedding", "config": {"input_dim": 3, "quantization_config": None}},
        {"class_name": "Dense", "config": {"units": 1, "quantization_config": None}},
    ]}}
    cleaned = ModelRegistry._strip_forward_keys(cfg)
    assert "quantization_config" not in cleaned["config"]["layers"][0]["config"]
    assert "quantization_config" not in cleaned["config"]["layers"][1]["config"]
    assert cleaned["config"]["layers"][0]["config"]["input_dim"] == 3


@pytest.mark.skipif(not os.path.exists(os.path.join("models", "deep_learning", "lstm.keras")),
                    reason="shipped Keras model not present")
def test_shipped_keras_model_loads():
    tf = pytest.importorskip("tensorflow")
    from api.model_loader import ModelRegistry

    model = ModelRegistry()._load_keras_model(tf, os.path.join("models", "deep_learning", "lstm.keras"))
    assert model is not None
    assert len(model.layers) > 1


# ── One registry per process ──────────────────────────────────────────────────

def test_shared_registry_is_a_singleton():
    from api import model_loader as ML

    with patch.object(ML, "_shared", None), patch.object(ML, "_shared_load_error", None):
        a = ML.get_shared_registry(load=False)
        b = ML.get_shared_registry(load=False)
    assert a is b
