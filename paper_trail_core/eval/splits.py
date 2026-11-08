"""
Leak-free evaluation splits.

Why this module exists
----------------------
The base models in models/ are frozen, and *every* file under dataset/ was fed
into `preprocess.load_all_datasets()` when they were trained. So the only rows
those models have never seen are the 20% test partition of the original split.

Anything else is a memorisation measurement. Concretely: scoring the hybrid
ensemble on dataset/Adversarial/adversarial_style.csv returns 300/300, because
export_models.py oversamples that source 15x into the *training* partition —
roughly 80% of that file is training data and the rest is template-identical
to it.

This module reproduces the original partition exactly, then hands back the
**raw** `text_content` for the held-out rows. That last part matters:
`load_all_datasets()` returns `text_clean`, which is lowercased with all
digits and punctuation stripped — useless for claim extraction and NLI, which
need sentence boundaries, numbers and capitalisation.

Reproducing the split is safe because `train_test_split` is deterministic
given the same array length, `random_state`, `test_size` and `stratify`
vector. We verify that rather than trust it (see `_assert_split_matches`).
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocess import get_train_test_split, load_all_datasets

log = logging.getLogger(__name__)

# Must match export_models.py:76 exactly, or the "held-out" rows are not held out.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Sources excluded from the headline metric, with the reason.
#
#   Adversarial — oversampled 15x into training (export_models.py:76). Its rows
#                 are template-generated, so even the held-out fifth is
#                 near-identical to trained rows. Reported separately as an
#                 explicit memorisation baseline, never folded into headline.
#   BuzzFeed    — `text_content` is `Category + " " + Page` (preprocess.py:120),
#                 i.e. metadata, not article text. There is nothing to fact-check.
EXCLUDED_FROM_HEADLINE = ("Adversarial", "BuzzFeed")

_CACHE: pd.DataFrame | None = None
_DISK_CACHE = os.path.join(_ROOT, "results", "holdout_cache.parquet")


def _assert_split_matches(df: pd.DataFrame, test_idx: np.ndarray) -> None:
    """
    Verify our index-based split reproduces preprocess's value-based split.

    If sklearn's internals ever change, or preprocess.py's ordering shifts, we
    must fail loudly here — silently evaluating on training data is exactly the
    failure mode this module exists to prevent.
    """
    _, x_test_ref, _, _ = get_train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    ours = df.loc[test_idx, "text_clean"].to_numpy()

    if len(ours) != len(x_test_ref) or not np.array_equal(ours, x_test_ref):
        raise RuntimeError(
            "Held-out split does not reproduce preprocess.get_train_test_split. "
            "Refusing to continue — the evaluation set may contain training rows. "
            f"(ours={len(ours)} rows, reference={len(x_test_ref)} rows)"
        )
    log.info("splits: verified held-out partition matches training-time split (%d rows)", len(ours))


def get_holdout_frame(force_reload: bool = False) -> pd.DataFrame:
    """
    Return the 20% partition the frozen models were never trained on, with raw text.

    Columns: text_content (raw), text_clean, label (0=fake, 1=real), source.

    The result is cached to disk: `load_all_datasets()` reads ~115 MB of CSVs
    and then cleans and de-duplicates ~100k rows, which takes several minutes
    every time. Evaluation runs shouldn't pay that repeatedly. Delete
    `results/holdout_cache.parquet` (or pass `force_reload`) to rebuild.
    """
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE

    if not force_reload and os.path.exists(_DISK_CACHE):
        try:
            _CACHE = pd.read_parquet(_DISK_CACHE)
            log.info("splits: loaded cached holdout (%d rows) from %s", len(_CACHE), _DISK_CACHE)
            return _CACHE
        except Exception as exc:
            log.warning("splits: could not read %s (%s) — rebuilding", _DISK_CACHE, exc)

    df = load_all_datasets()
    df = df.reset_index(drop=True)

    # Split the *index*, stratified the same way, so we can recover raw text.
    idx = df.index.to_numpy()
    y = df["label"].to_numpy()
    _, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    _assert_split_matches(df, test_idx)

    holdout = df.loc[test_idx, ["text_content", "text_clean", "label", "source"]].copy()
    holdout = holdout.reset_index(drop=True)

    log.info(
        "splits: holdout = %d rows (fake=%d real=%d) across sources %s",
        len(holdout),
        int((holdout.label == 0).sum()),
        int((holdout.label == 1).sum()),
        sorted(holdout["source"].unique().tolist()),
    )

    try:
        os.makedirs(os.path.dirname(_DISK_CACHE), exist_ok=True)
        holdout.to_parquet(_DISK_CACHE, index=False)
        log.info("splits: cached holdout to %s", _DISK_CACHE)
    except Exception as exc:
        log.warning("splits: could not cache holdout (%s)", exc)

    _CACHE = holdout
    return holdout


def headline_frame(holdout: pd.DataFrame | None = None) -> pd.DataFrame:
    """Held-out rows eligible for the headline metric (see EXCLUDED_FROM_HEADLINE)."""
    holdout = get_holdout_frame() if holdout is None else holdout
    return holdout[~holdout["source"].isin(EXCLUDED_FROM_HEADLINE)].reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    n_per_class: int,
    seed: int = RANDOM_STATE,
    min_chars: int = 120,
) -> pd.DataFrame:
    """
    Balanced sample for evaluation runs.

    The pipeline costs seconds per article (retrieval + NLI), so we evaluate on a
    sample rather than all ~19k held-out rows. Fixed seed, sorted output — the
    same command always evaluates the same articles.

    `min_chars` drops rows too short to extract a checkable claim from (LIAR
    statements and FakeNewsNet titles are often a dozen words). Those are kept
    in the frame for source-level breakdowns but excluded from sampling, since
    an evidence-grounded pipeline has nothing to work with on a bare headline.
    """
    eligible = df[df["text_content"].str.len() >= min_chars]

    parts = []
    for label in (0, 1):
        rows = eligible[eligible["label"] == label]
        take = min(n_per_class, len(rows))
        if take < n_per_class:
            log.warning(
                "splits: only %d rows available for label=%d (asked for %d)",
                take, label, n_per_class,
            )
        parts.append(rows.sample(n=take, random_state=seed))

    out = pd.concat(parts).sort_index().reset_index(drop=True)
    log.info("splits: sampled %d rows (%d per class requested)", len(out), n_per_class)
    return out


def dev_eval_split(df: pd.DataFrame, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Halve a sample into dev (for fitting fusion + choosing thresholds) and eval
    (for the reported numbers). Fitting and reporting on the same rows would
    reproduce the very problem this rebuild is fixing.
    """
    dev_idx, eval_idx = train_test_split(
        df.index.to_numpy(), test_size=0.5, random_state=seed, stratify=df["label"].to_numpy()
    )
    dev = df.loc[sorted(dev_idx)].reset_index(drop=True)
    ev = df.loc[sorted(eval_idx)].reset_index(drop=True)
    return dev, ev
