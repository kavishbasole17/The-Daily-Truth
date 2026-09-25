# Cleanup plan (Phase 1b)

Per the project brief: nothing below is deleted yet. This is the plan for
sign-off. Every item was checked against `git ls-files`, `grep` for
references, and (for notebooks) actually executed as a script to see what
path it reads from — see `docs/CURRENT_STATE.md` §7 for how these were found.

Cleanup will happen on a separate branch (`cleanup/repo-tidy`), in its own
commit(s), after this plan is approved, so it can be reverted independently
of any other work.

---

## Files to delete

| Path | Size | Reason |
|---|---|---|
| `tmp_full.diff` | 164 KB | Tracked scratch diff file at repo root. Not referenced by any code, config, test, or doc. Looks like an accidental `git add` of a working diff. |
| `mergeDataset.ipynb` | 14 KB | Reads from a hardcoded personal path (`c:\Users\kbkav\OneDrive\Desktop\capstone`) that doesn't exist on this machine or in this repo, and writes to `dataset/final_dataset/`, which does not exist and is not tracked. Nothing imports or references this notebook. Superseded by `preprocess.py`'s `load_all_datasets()`/`get_train_test_split()`, which is what `export_models.py` and every notebook in `notebooks/` actually use. |
| `models/TF-IDF_LogReg.ipynb` | — | Reads from `../dataset/final_dataset/{train,test}_split.csv`, which does not exist (see above — only `mergeDataset.ipynb` ever produced it, and that notebook is itself dead). Duplicate of `notebooks/01_logistic_regression.ipynb`, which uses the live `preprocess.py` pipeline instead. Not referenced anywhere. |

## Config fix (not a deletion)

| Change | Reason |
|---|---|
| Remove `.env.example` from `.gitignore`, then `git add .env.example` | Currently gitignored alongside `.env` (line grouped under "Backend runtime / generated artifacts"), so it is **not tracked in git at all** — a fresh clone has no template to copy, despite the README's step 2 (`cp .env.example .env`) depending on it. `.env.example` has no secrets (it's placeholder values like `sk-ant-...`); it's the opposite of something that should be ignored. |

## `.gitignore` addition

Add a rule to stop stray root-level diff/patch files from being committed
again: `*.diff`, `*.patch`.

---

## Flagged, not included in this pass — needs a separate explicit decision

**`dataset/` (231 MB) and `models/` (1.3 GB) are committed directly to git**,
not produced by a fresh-clone script. This is real bloat, but fixing it
properly is a bigger, higher-blast-radius action than the items above:

- `git rm --cached` on these paths + `.gitignore` them stops *future* growth,
  but the 1.5 GB already sits in git history regardless — the repo doesn't
  get smaller unless history is rewritten (`git filter-repo`/BFG), which
  rewrites every commit hash and requires a force-push. That breaks any
  existing clone (yours, if you have more than one, or a teammate's, if they
  ever pull this repo).
- Even the non-destructive half (stop tracking new changes) means a fresh
  `git clone` would no longer have `dataset/` or `models/` populated —
  `export_models.py` already regenerates `models/` from `dataset/`, so that
  half is safe once documented, but `dataset/` itself (the raw ISOT/LIAR/
  FakeNewsNet/MultiFC/etc. source data) has no download script yet, only a
  loader (`preprocess.py`) that assumes the files are already present. Per
  the brief's own rule — "never delete datasets or checkpoints without a
  script that can regenerate them" — a download script would need to exist
  and be verified first.

**Recommendation:** leave both tracked for now. If you want this fixed,
say so explicitly as its own step, separate from this cleanup pass — it's a
one-way operation on shared history, not a tidy-up.

---

## Explicitly NOT touched (per the brief's rules, confirmed by search)

- `notebooks/` (25 notebooks + `notebooks/figures/`, 158 tracked files,
  22 MB) — referenced by the README as intentional per-algorithm EDA
  documentation; kept.
- `dataset/`, `models/` — see above.
- `.env` — correctly gitignored (has real secrets), left as-is.
- `catboost_info/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`,
  `paper_trail.db`, `cache/` — already correctly gitignored and **not
  tracked in git** (verified via `git ls-files`), so there is nothing to
  clean up in git; they're just local generated artifacts sitting on disk.
  No action needed.
- Any teammate code — none exists in this repo (confirmed with you
  directly), so nothing to avoid here yet.

---

## After approval

1. `git checkout -b cleanup/repo-tidy`
2. Delete the three files above; fix `.gitignore` (`.env.example` +
   `*.diff`/`*.patch` addition); `git add .env.example`.
3. Commit.
4. Run `pytest tests/ -v` and confirm the API (`uvicorn api.main:app`) still
   starts, to confirm nothing broke.
5. Leave the branch for you to review/merge — not merged automatically.
