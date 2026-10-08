# Current state (as of 2026-09-25)

Snapshot of what exists in this repo before the multilingual/OCR/RAG upgrade
described in the project brief. Written from a full read of the codebase, not
from the README's description of itself — where the two disagree, this
document follows the code.

---

## 1. Architecture, as built

**Two systems living in one FastAPI app:**

1. **Legacy classifier bank** (`GET /models`, `POST /predict`, `/predict/all`,
   `/predict/batch`, `/results*`) — direct model inference, no pipeline, no
   persistence. Exists for the benchmark/demo comparison across 22 models.
2. **Agentic verification pipeline** (`POST /articles/submit` → async job →
   `GET /articles/{id}`) — the actual capstone deliverable. A 5-node
   **LangGraph** `StateGraph` (`paper_trail_core/agents/pipeline.py`):

   ```
   extract_claims → retrieve_evidence → verify_claims → run_ensemble → adjudicate
   ```

   Every node is wrapped in `with_stage_guard`: a per-stage timeout
   (`STAGE_TIMEOUT_S`, default 90s) and a typed fallback. A stage that throws
   or times out never crashes the request or silently degrades — it returns a
   `Degradation` record (`paper_trail_core/degradations.py`) that rides along
   with the verdict to the DB and the API response. This degradation-tracking
   design is worth preserving and extending to new stages (OCR, translation,
   region classification) rather than rebuilding.

   Each stage names its own backend independently via `config.py` /
   `Settings` (`CLAIM_BACKEND`, `NLI_BACKEND`, `EVIDENCE_BACKEND`,
   `ENSEMBLE_BACKEND`) — a deliberate fix for a previous single `MOCK_MODE`
   flag that degraded two unrelated stages at once. New stages should follow
   the same per-stage-backend convention.

   Submission is **async**: `POST /articles/submit` returns 202 immediately,
   a `BackgroundTasks` job runs the pipeline and writes the result, client
   polls `GET /articles/{id}`.

**Persistence:** SQLAlchemy + SQLite by default (`paper_trail.db`,
Postgres-ready via `DATABASE_URL`). Three tables: `articles`, `verdicts`
(verdict + confidence + JSON-encoded claim verdicts + JSON-encoded
degradations + fused_score). The `review_queue` table was removed along with
human review. `api/orm.py` states `articles.id` is the intended shared FK
for teammates' tables; no teammate tables exist in this repo (see §5).

**Auth:** JWT (`api/auth.py`), two hardcoded demo users (`admin`/`admin123`,
`analyst`/`analyst123`). Fine for a capstone demo; would need real user
storage before any wider deployment — out of scope for this brief unless
asked.

## 2. Models — what's actually trained vs. off-the-shelf

**Trained by this team** (`export_models.py`, English-only, TF-IDF /
Keras-tokenizer features):
- 18 classical ML models (logistic regression, NB variants, SVM, decision
  tree, random forest, extra trees, AdaBoost, gradient boosting, XGBoost,
  LightGBM, CatBoost, ridge, SGD, passive-aggressive, perceptron, KNN, MLP).
- 4 deep-learning models (LSTM, GRU, CNN, CNN+LSTM) via TensorFlow/Keras.
- 1 hybrid stacking meta-learner (`models/hybrid/stacking_lr.joblib`) over
  the above. **Has a known label-leakage bug**, already self-documented in
  `docs/VERIFICATION_PIPELINE.md`: the meta-learner is trained on
  `np.random.choice(X_train)` rows — the *same* rows the base models were
  fitted on — rather than out-of-fold predictions, so its 0.99 accuracy is
  not a real number. Fix is a `cross_val_predict` refit; flagged in the
  existing docs as the single highest-value fix available, not yet done.
- 1 calibrated fusion adjudicator (`models/fusion/adjudicator_v1.joblib`,
  `scripts/train_fusion.py`) — a logistic regression over a frozen
  13-feature vector combining the ensemble score with evidence-retrieval
  features (coverage, entail/contradict mass, numeric conflict, etc.). This
  is the closest thing in the repo to a "learned aggregator" and is what the
  brief's §9 (aggregator + abstention) already does, minus multilingual/OCR
  inputs.

**Off-the-shelf, not fine-tuned by this team:**
- NLI verifier: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (English-only,
  pretrained on FEVER+ANLI, used as-is). This is the brief's "Model 3"
  territory — the brief's ask is to fine-tune XLM-R/mDeBERTa for
  SUPPORTED/REFUTED/NEI; the current verifier is the closest analogue and the
  natural baseline to compare a new fine-tuned model against, not something
  to discard.
- Embeddings for relevance gating: `sentence-transformers/all-MiniLM-L6-v2`
  (English-only in practice; not evaluated on Indic text).

**Does not exist at all:**
- A fine-tuned multilingual stylistic classifier (brief's "Model 1" —
  MuRIL/IndicBERT). Today's stylistic signal is the English-only classical/DL
  ensemble above.
- Any region classifier (brief's "Model 2"). No "region" concept exists
  anywhere in the codebase, schema, or config.
- Any fine-tuned claim-evidence NLI model trained by this team (brief's
  "Model 3" as a training exercise, not just off-the-shelf inference).

## 3. Data

`dataset/` — **231 MB, all committed directly to git**, all English:

| Source | Size | Used by |
|---|---|---|
| ISOT (Fake.csv/True.csv) | 111 MB | export_models.py — has a known dateline artifact (real rows start "WASHINGTON (Reuters) -"), already documented |
| MultiFC | 74 MB | export_models.py |
| FakeNewsNet (politifact/gossipcop) | 43 MB | export_models.py |
| LIAR Dataset | 2.9 MB | export_models.py |
| BuzzFeed | 360 KB | export_models.py |
| BSDetector | 64 KB | export_models.py |
| Adversarial (synthetic) | 52 KB | export_models.py — has a known grammatical tell, oversampled 15×, memorisation-only, already documented |
| golden (hand-written) | 20 KB | `scripts/evaluate_pipeline.py --slice golden` — 30 novel English articles, the only leak-free eval set |

No Hindi/Telugu/Tamil data anywhere. No labeled region metadata anywhere.
Trained artifacts (`models/classical/*.joblib`, `models/deep_learning/*.keras`,
`models/hybrid/*.joblib`, ~tens of MB) are also committed directly to git
rather than produced by a reproducible download+train step a fresh clone can
run unattended (`export_models.py` *is* that reproducible step — the issue is
the outputs are committed on top of it, doubling the repo's weight for no
reason). Both the raw datasets and the trained artifacts are cleanup-plan
candidates for a later phase (replace with a download/train script + `.gitignore`,
not delete the capability).

## 4. Retrieval — no vector store exists

There is **no persistent index, vector DB, or curated source corpus** today.
`retrieve_evidence` queries **NewsAPI** (keyed, free tier 100 req/day,
~1-month recency window, English-only search) and **Wikipedia** (keyless,
unmetered) live, at request time, then gates results by MiniLM cosine
relevance. Results are cached to flat JSON files in `cache/evidence/`
(gitignored, disk cache only — not a queryable index).

This means the brief's "region-partitioned vector index, LOCAL → GLOBAL →
MANUAL cascade, curated trusted sources per region" is a **net-new subsystem**,
not a partitioning of something that already exists. There is nothing to
migrate; a vector store (FAISS/Chroma/similar) and `data/sources/<region>.yaml`
curation would be built from scratch.

## 5. Entry points

- `uvicorn api.main:app --reload` → FastAPI on :8000, Swagger at `/docs`.
- `cd frontend && npm run dev` → Next.js on :3000 (teammate Phani's module;
  do not modify — `frontend/src/app/verify/page.jsx` reads `claim_verdicts`
  fields directly with no null-checks, per `tests/test_pipeline_e2e.py`'s
  comment, so any API schema change must stay backward compatible or that
  page breaks silently).
- `python export_models.py [--skip-dl]` → trains/saves all 22+ models.
- `py scripts/evaluate_pipeline.py --slice {golden|holdout|adversarial}` →
  the pipeline eval harness (accuracy, calibration/ECE, coverage-vs-precision,
  ensemble-only/evidence-only/fused ablation). **This already does a
  meaningful chunk of the brief's §4 "agent evaluation" requirement** for the
  existing English pipeline — extend it for OCR/multilingual/region metrics
  rather than writing a parallel `scripts/eval_all.py` from zero.
- `py scripts/train_fusion.py` → refits the adjudicator.
- `python -m paper_trail_core.registry.cli` → registry browser CLI.
- `pytest tests/ -v` → 107 tests (`tests/test_adjudicator.py`,
  `test_api.py`, `test_claim_extractor.py`, `test_ensemble.py`,
  `test_evidence_retriever.py`, `test_pipeline_e2e.py`, `test_verifier.py`),
  all offline/mocked — no GPU, trained models, or API keys required to run
  them, which is a pattern worth keeping as new nodes are added.

**Teammates' modules are not present in this repo at all** — no
social-graph/propagation code, no moderation tooling, no separate
source-credibility/analytics module beyond the `review_queue`/`articles`
tables' FK comment addressed "for Person D". Either they live in separate
repos, or they haven't been started/merged yet. Worth confirming with the
user before assuming "don't touch teammate code" has anything concrete to
avoid touching yet.

## 6. Config / secrets

`.env` exists locally with a real `ANTHROPIC_API_KEY` and
`LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-haiku-4-5-20251001` — usable
today for LLM-backed steps (translation fallback, OCR fallback, rationale
generation, `CLAIM_BACKEND=llm`). `OPENAI_API_KEY` slot is present but not
confirmed populated with a real key — treat as unavailable unless the user
says otherwise. No search-provider keys are required for the current
pipeline (Wikipedia is keyless); `NEWS_API_KEY`/`TAVILY_API_KEY`/`SERPAPI_KEY`
are optional upgrades already wired in `SEARCH_PROVIDER`.

`requirements.txt` already lists `langchain-anthropic`, `langchain-openai`,
`anthropic`, `openai` as optional deps (for `CLAIM_BACKEND=llm`) — there is
**no unified LLM gateway module** yet (no central routing/retry/fallback
layer across stages); each stage that wants an LLM would currently call a
provider SDK directly. The brief's §4 "LLM gateway" is net-new.

## 7. Bugs / inconsistencies found (independent of the upgrade brief)

1. **`.env.example` is gitignored** (`.gitignore` line 18) despite the
   README's setup instructions saying `cp .env.example .env` as step 2. It is
   *not* tracked in git (confirmed via `git ls-files`) — a fresh clone has no
   `.env` template at all. This looks unintentional (it's grouped under
   "Backend runtime / generated artifacts" next to `.env`, which correctly
   *should* be ignored, but the `.example` file is the opposite of a secret —
   it's the thing meant to be shared). Worth fixing regardless of the rest of
   this project.
2. **`tmp_full.diff`** (164 KB) is tracked in git at the repo root — looks
   like an accidentally committed scratch diff file. Cleanup candidate.
3. Two already-self-documented, unfixed issues carried over from
   `docs/VERIFICATION_PIPELINE.md` (not rediscovered here, just noted so the
   final report doesn't imply they were missed): the stacking meta-learner
   in-sample leakage (§2 above), and the adversarial dataset's grammatical
   tell in `scripts/generate_adversarial_dataset.py`.
4. Raw datasets (231 MB) and trained model artifacts committed directly to
   git rather than fetched/produced by script on a fresh clone (see §3).

## 8. Net-new work implied by the brief vs. this codebase

| Brief component | Status |
|---|---|
| Image input + OCR | Does not exist |
| Language ID + translation (Hindi/Telugu/Tamil) | Does not exist; pipeline assumes English throughout |
| Region classifier ("Model 2") | Does not exist |
| Region-partitioned retrieval / vector store / `data/sources/<region>.yaml` | Does not exist (see §4) |
| Fine-tuned multilingual stylistic classifier ("Model 1") | Does not exist (current ensemble is English-only, untuned for Indic) |
| Fine-tuned claim-evidence NLI ("Model 3") | Does not exist (current verifier is off-the-shelf English NLI) |
| Claim extraction, retrieval cascade, NLI verification, stylistic scoring, fusion+abstention, stage-guard/degradation tracking | **Already built and working for English text**; extend, don't rebuild |
| Multi-level rationale (phrase/claim/article) with citation-faithfulness check | `reasoning_trace` exists at article level only; no phrase-level attribution, no per-claim cited-URL rationale, no faithfulness check |
| LLM gateway (routing/fallback/retry, prompt versioning, guardrails) | Does not exist; ad hoc optional LLM calls only |
| Golden eval set (100+ cases, multi-language, text+image) | 30-case English-only golden set exists; needs multilingual/image expansion |
| Human-review → retraining loop | Removed: human review is no longer part of the system |
| `scripts/eval_all.py` | Does not exist as one script; `scripts/evaluate_pipeline.py` already covers a meaningful subset for the English pipeline |
