# AGENTS.md

Guide for people and AI coding agents working on **The Daily Truth / Paper Trail**, plus a full
log of the changes an AI agent (Claude Code) made on **2026-11-08**, with why each was made and
how to back it out. If something broke after that date, start with the **Backtracking** section.

---

## Working in this repo

| Task | Command (from the repo root unless noted) |
|---|---|
| Backend tests | `pytest` (427 tests, all offline: network and models are stubbed) |
| Python lint | `ruff check .` (config in `pyproject.toml`; must say *All checks passed!*) |
| Frontend lint | `cd frontend && npx next lint` (config in `frontend/.eslintrc.json`) |
| Frontend build | `cd frontend && npx next build` |
| Run the API | `uvicorn api.main:app --reload` |
| Run the UI | `cd frontend && npm run dev` |
| Is every stage at full accuracy? | `GET http://localhost:8000/health/accuracy?deep=true` |

Rules of thumb:
- **Never retrain or re-export models** (`export_models.py`, `scripts/train_fusion.py`) as a side effect of other work. The files in `models/` are the shipped artifacts.
- **Keep `scikit-learn==1.7.2`.** `models/` was pickled with it (see change 5.1).
- Every pipeline stage must turn failures into a **degradation** on the verdict, never a crash and never a silent fallback. If you add a fallback, add a `degrade(...)` call next to it.
- `.env` is git-ignored and holds real keys. Never commit it; document new settings in `.env.example` and `config.py`.

---

## Commit map

| Commit | Contains |
|---|---|
| `439030d` *added auth and chat history* | Part 1 below |
| `ffc2b02` *feature: links can now be sent…* | Parts 2, 3 and 4 below |
| `dc944cc` *lint issues and minor bugs fixed* | Part 5 below (lint + bug fixes) |
| *uncommitted at time of writing* | Parts 6 and 7 below (accuracy bug fixes + LLM judge, 2026-11-16) |
| *not in git* (`.env` is ignored) | `.env` edits listed in Part 4 |

To see exactly what a part changed: `git show <commit> -- <file>`.

---

## Part 1 — Accounts, one free check, chat history (`439030d`)

**Why:** the app had no real login. The frontend silently fetched an `admin/admin123` token for
every request, so anyone had admin rights and nothing was saved per user. The request was:
an auth page, one free query without signing in, sign-in for follow-ups, saved previous chats,
a test user, and a Google sign-in placeholder.

| Change | Files | Why |
|---|---|---|
| Users stored in the DB instead of a hard-coded dict | `api/orm.py` (`User`), `api/auth.py` | Needed for registration and history. `admin/admin123` and `analyst/analyst123` are still seeded so old scripts keep working. |
| Seeded **test account** `testuser` / `test@dailytruth.app` / `Test@1234` | `api/auth.py` (`TEST_USER`, `seed_users`) | Requested. Turn it off with `SEED_TEST_USER=false`. |
| `POST /auth/register`, `GET /auth/me`, `GET /auth/guest`; `POST /auth/token` accepts username **or** email | `api/routers/auth.py` | Sign-up, session restore, and the free-check counter for the UI. |
| `POST /auth/google` (returns **501** until `GOOGLE_CLIENT_ID` is set) | `api/routers/auth.py`, `frontend/src/components/GoogleSignInButton.jsx` | Placeholder the team will finish. Token verification and account linking are already written; it only needs the client ID in `.env` and `frontend/.env.local`. |
| **One free check per client IP**, then 401 `detail.code = "guest_limit_reached"` | `api/orm.py` (`GuestUsage`), `api/auth.py` (`consume_guest_query`), `api/routers/articles.py` (`_assign_owner`) | Counted server-side so clearing the browser doesn't reset it. `GUEST_QUERY_LIMIT` sets the number. |
| Conversations: every signed-in check belongs to a chat | `api/orm.py` (`Conversation`, `Article.user_id/conversation_id/guest_key/kind`), `api/routers/conversations.py` | "View previous chats". `POST /conversations/claim` moves the guest's free check into their new chat after sign-in. |
| Saved checks are private to their owner | `api/routers/articles.py` (`_can_read`) | Before, any article was readable by id. Guest checks stay readable by id. |
| New DB columns added automatically on startup | `api/database.py` (`_ADDED_COLUMNS`) | `create_all` never adds columns to existing tables. This keeps existing `paper_trail.db` files working without a migration tool. |
| Login page, navbar account menu, auth context | `frontend/src/app/login/page.jsx`, `frontend/src/context/AuthContext.jsx`, `frontend/src/components/Navbar.jsx`, `frontend/src/styles/auth.css` | UI for the above. The token lives in `localStorage` (`dt_token`). |
| "Previous chats" sidebar + sign-in gate on `/verify` | `frontend/src/components/ChatHistorySidebar.jsx`, `frontend/src/app/verify/page.jsx`, `frontend/src/styles/chat.css` | UI for history and the one-free-check rule. |
| API client sends the real token; the hard-coded admin login is removed | `frontend/src/services/api.js` | Security fix. |
| `JWT_EXPIRE_MINUTES` default 60 → 1440 | `config.py` | Users were logged out mid-session after an hour. |

**Known limits:** IP-based guest counting means users behind one NAT share the free check. There's
no password reset yet. The token in `localStorage` is readable by any script on the page, so keep XSS out.

---

## Part 2 — Link checks, linked sources, highlights, sharing, feedback (`ffc2b02`)

**Why:** the request was user-facing features to make the project stand out. These five were chosen.

| Change | Files | Why |
|---|---|---|
| **Check a link**: paste a URL (or `{"url": ...}`) and the server fetches and extracts the article | `api/url_fetch.py`, `api/routers/articles.py` (`_run_pipeline_job`, `ArticleSubmitRequest`) | URL submissions used to be rejected. The fetcher refuses private/loopback/metadata IPs on every redirect (SSRF guard), and caps size (3 MB), redirects (4) and time (15 s). Failures are stored in `Article.error_message`. |
| Evidence now carries `urls`, `published_at`, `trust` per snippet | `paper_trail_core/agents/state.py`, `paper_trail_core/agents/verifier.py`, `ClaimVerdictOut` in `api/routers/articles.py` | So the UI can link each snippet to its article. Fields are additive with defaults, so old stored verdicts still load. |
| Verdict UI pulled into one shared component; sources are clickable links with date + trust dot | `frontend/src/components/VerdictReport.jsx` | Used by both the chat and the public share page so they can't drift apart. |
| Each claim highlighted in the checked text, coloured by verdict | `frontend/src/lib/highlight.js` | Exact match first, then best word overlap (≥ 60%), because the extractor rewrites claims slightly. |
| `full_text`, `source_title`, `error`, `kind=url` on `ArticleOut` | `api/routers/articles.py` | Needed for highlighting link/image checks and for showing why a check failed. |
| **Shareable verdicts**: `POST/DELETE /articles/{id}/share`, public `GET /share/{slug}`, page `/v/<slug>` with a generated preview image | `api/routers/share.py`, `frontend/src/app/v/[slug]/*`, `frontend/src/lib/shared.js`, `frontend/src/components/CheckActions.jsx` | Shared pages expose no owner info. The preview image runs on the **edge runtime** because Next 14's Node runtime can't find `@vercel/og`'s font on Windows. Needs `NEXT_PUBLIC_SITE_URL`, and `API_INTERNAL_URL` in Docker (set in `docker-compose.yml`). |
| **Feedback** 👍/👎 with a reason, `PUT/DELETE /articles/{id}/feedback` | `api/orm.py` (`Feedback`), `api/routers/articles.py`, `frontend/src/components/CheckActions.jsx` | Stores the verdict at rating time as labelled data for retraining the fusion model later. |
| `trafilatura` dependency | `requirements.txt` | Better article extraction. Without it a stdlib parser is used. |

---

## Part 3 — Chat continuity (`ffc2b02`)

**Why:** users wanted to discuss a specific finding, push back on it, and get a verdict, in one
continuous chat.

| Change | Files | Why |
|---|---|---|
| `POST /conversations/{id}/messages` (202 + background job) and `GET …/messages/{mid}` to poll | `api/routers/conversations.py` (`_run_chat_job`, `send_message`) | A reply can run a whole new check, which takes seconds, so it's async like verification. |
| Chat engine with a `verify_claim` tool (Claude, manual tool loop, max 2 checks per reply) | `api/chat.py` | Claude decides when to check, and new checks become evidence for later messages. Model `QA_MODEL` (default `claude-opus-5-5`), with server-side refusal fallback enabled. |
| Rule-based fallback when no LLM key is set | `api/chat.py` (`route_message`, `resolve_statement`, `_heuristic_reply`) | The feature still works without a key. `resolve_statement` turns "I read it was 1887" into a checkable claim ("The Eiffel Tower was completed in 1887."). |
| `ChatQuestion` gained `status, stage, focus, article_ids, error_message` | `api/orm.py`, `api/database.py` | Async state, the "Discuss" focus, and the checks a reply ran. |
| Assistant-run checks are `Article.kind = "chat"` and hidden from `turns` | `api/routers/conversations.py` (`_detail`) | They belong to the reply that ran them. |
| One reply in progress per chat; a reply stuck for 5+ minutes stops blocking | `send_message` | Prevents overlapping jobs; recovers after a server restart. |
| UI: "Discuss" buttons, focus chip, auto-routing hint with override, follow-up suggestions, inline check cards | `frontend/src/app/verify/page.jsx`, `frontend/src/components/VerdictReport.jsx`, `frontend/src/styles/features.css` | Short messages go to chat; articles, links and images start a new check. |
| Legacy `POST /conversations/{id}/ask` kept | `api/qa.py`, `api/routers/conversations.py` | Synchronous, evidence-only answers. Still tested, but no longer used by the UI. |

---

## Part 4 — Maximum accuracy (`ffc2b02` + local `.env`)

**Why:** the request was maximum accuracy at all times, proof that it's happening, and a list of
the hard limits. A real (not mocked) pipeline run found the problems below.

| Change | Files | Why |
|---|---|---|
| **Pinned `scikit-learn==1.7.2`** | `requirements.txt` | `models/` was pickled with 1.7.2. Newer versions failed to load `gradient_boosting` and warned of *invalid results* for the other models. Fresh installs, including Docker, were affected. |
| New LLM helper that's safe for current models | `paper_trail_core/nlp/llm.py` | The old code read `content[0].text` with `max_tokens=512`. Current Claude models put a thinking block first, so extraction would have silently fallen back to the heuristic on every call. The helper reads text blocks only, streams, rejects refusals and truncation, and parses JSON tolerantly. |
| Claim extraction and translation use the helper; input caps 4,000 / 8,000 → 50,000 chars | `paper_trail_core/agents/claim_extractor.py`, `translation_agent.py` | Claims after roughly the first 600 words of an article were never seen. |
| Extraction prompt rewritten (resolve pronouns, keep numbers/dates exact, treat the article as untrusted) | `claim_extractor.py` (`_PROMPT`) | The NLI step needs self-contained claims, and the article text could otherwise carry prompt injection. |
| `CLAIM_BACKEND=llm` and `LLM_MODEL=claude-opus-5-5` by default; no key → heuristic **plus** an `llm_not_configured` degradation | `config.py`, `.env.example`, `claim_extractor.py` | Use the strongest extractor when possible, and never fall back silently. |
| `GET /health/accuracy[?deep=true]` + startup log | `api/accuracy.py`, `api/main.py` | Reports what each stage is **actually** running, whether that's full strength, and the fix. Never returns secret values. |
| `docs/ACCURACY.md` | new | Every hard limit (10-char minimum, 50k maximum, 5 claims, NewsAPI 100/day, 256-token NLI input, …) plus measured results. |
| `requirements.txt` notes: TensorFlow and `anthropic` marked **required** for full accuracy | `requirements.txt` | The stacking ensemble needs the 4 TensorFlow models as inputs. |
| **Local `.env` (not in git):** `CLAIM_BACKEND=llm`, `LLM_MODEL=claude-opus-5-5`, empty `ANTHROPIC_API_KEY=` placeholder | `.env` | The original was backed up outside the repo. **Paste a real key there.** Until then extraction, translation and chat run in reduced mode (shown in `/health/accuracy`). |

**Deliberately not changed:** decision thresholds (`CLAIM_CONTRADICT_MIN`, `NLI_MAX_LENGTH`, …).
Fusion was calibrated with the current values, so changing them needs a `scripts/evaluate_pipeline.py`
run first.

---

## Part 5 — Lint cleanup and bug fixes (uncommitted at time of writing)

### Lint setup
| Change | Files | Why |
|---|---|---|
| `pyproject.toml` with the ruff config | new | Reproducible lint. It ignores only `B008` (FastAPI `Depends()` defaults are the framework idiom) and `BLE001` (the pipeline deliberately catches everything and records a degradation). |
| 459 → 0 ruff findings | many `.py` files | Mostly mechanical: `Optional[X]` → `X \| None`, import order, `datetime.UTC`, parenthesised multi-line strings, `log.exception(...)` instead of `log.error(..., exc_info=True)`, NaN checks via `math.isfinite`. None of the 47 "implicit string concatenation" warnings was a missing-comma bug. |
| ESLint (`eslint@8.57.1`, `eslint-config-next@14.2.35`, `.eslintrc.json`) | `frontend/package.json`, lockfile | The project had no ESLint setup, so `next lint` couldn't run. |
| News thumbnails `<img>` → `next/image` (`fill` + `unoptimized`) | `frontend/src/components/NewsFeed.jsx`, `frontend/src/styles/feed.css` | Fixes the 3 lint warnings. `unoptimized` because publisher hosts can't be allow-listed. |
| `.ruff_cache/` git-ignored | `.gitignore` | |

### Bugs fixed
| # | Bug | Fix | Files |
|---|---|---|---|
| 1 | **Stage timeouts never fired.** Leaving `with ThreadPoolExecutor()` waits for the timed-out stage, so a 0.5 s budget returned after 3.0 s and a hung source could stall a check indefinitely. | Executor created without `with`; `shutdown(wait=False, cancel_futures=True)` in `finally`. Regression test `test_stage_timeout_returns_within_budget`. | `paper_trail_core/agents/pipeline.py` |
| 1b | Consequence of #1: once timeouts work, the LLM stages would be cut off at 90 s on long articles. | New `LLM_STAGE_TIMEOUT_S=300` for translation and claim extraction; the frontend's poll timeout went from 3 to 10 minutes. | `config.py`, `pipeline.py`, `.env.example`, `frontend/src/services/api.js` |
| 2 | Stacking base models that failed to load (e.g. no TensorFlow) were silently replaced with 0.5, and the verdict claimed full strength. | `ModelRegistry.missing_hybrid_bases()` plus a `base_models_missing` degradation. Test added. | `api/model_loader.py`, `paper_trail_core/agents/stylistic_ensemble_agent.py` |
| 3 | Single/Consensus/Batch/Models pages crashed on a 422, because FastAPI's `detail` is a list of objects. | Use `errorMessage()` from `services/api.js`; `ErrorState` ignores non-string messages. | `frontend/src/app/{single,consensus,batch,models}/page.jsx`, `frontend/src/components/ErrorState.jsx` |
| 4 | If the model list failed to load, "Verify" silently did nothing. | Show an error on load failure and on submit. | `single/page.jsx`, `batch/page.jsx` |
| 5 | One batch line under 5 characters made the API reject the whole batch. | Client-side check names the offending line. | `batch/page.jsx` |
| 6 | One malformed `pubDate` threw inside `.map()` and emptied the whole RSS feed. | `toIso()` parses safely per item. | `frontend/src/app/api/news/route.js` |
| 7 | A `javascript:` link in an RSS feed would end up in an `href` (XSS). | `safeUrl()` allows only http(s), for links and images. | `frontend/src/app/api/news/route.js` |
| 8 | Sidebar message count included assistant-run checks and left out the user's messages. | Count user checks plus questions. | `api/routers/conversations.py` (`_summary`) |
| 9 | No warning when `JWT_SECRET_KEY` is the built-in default (tokens forgeable). | Startup warning. | `api/main.py` |

---

## Part 6 — Accuracy bug fixes (2026-11-16, uncommitted at time of writing)

**Why:** a full read of the codebase with the goal "maximum accuracy for the Paper Trail
workflow". Two of the defects below meant the style ensemble and (on GPU machines) the NLI
model were silently running below full strength. Regression tests for every fix are in
`tests/test_regressions_2026_11.py`.

| # | Bug | Fix | Files |
|---|---|---|---|
| 1 | **All four Keras base models (lstm, gru, cnn, cnn_lstm) failed to load.** They were saved with Keras 3.14, whose layer configs carry `quantization_config: null`; Keras 3.11 (bundled with TensorFlow 2.20) rejects that key. The stacking ensemble ran with 0.5 in their place on every article (and said so in `base_models_missing`). `custom_objects` cannot fix it because Keras resolves built-in layers by module name first. | On that exact error the `.keras` archive is rewritten to a temp copy with the unknown keys removed from `config.json` and loaded from there. Weights are untouched, so predictions are the ones the models were trained to give. | `api/model_loader.py` (`_load_keras_model`, `_strip_forward_keys`) |
| 2 | **NLI model could not load on any CUDA machine.** `from_pretrained(..., dtype=float16)` is only accepted by transformers ≥ 4.56; on 4.55 the kwarg reaches the model constructor and raises. The verifier then fell back to word overlap and every verdict was forced `uncertain`. | Try `dtype`, fall back to `torch_dtype`. | `paper_trail_core/nlp/nli.py` |
| 3 | A single slow evidence provider took the whole retrieval stage past its 90 s budget, and the stage guard then discarded the evidence of every claim that *had* finished. | Per-claim futures with a shared deadline 5 s inside the stage budget. The slow claim alone is reported (status `timeout`, degradation `retrieval_timeout`); a claim whose retrieval raises is likewise recorded alone instead of failing the node. | `paper_trail_core/agents/evidence_retriever.py` |
| 4 | A year next to a percentage was read as a percentage ("5 percent in 2019" → 2019 = percent), because the unit window around each figure extended 12 chars back and 22 forward across neighbouring numbers. | The window stops at the neighbouring figure; a preceding unit must sit within 5 chars. | `paper_trail_core/nlp/numeric.py` |
| 5 | LLM-extracted claims were dropped as "hallucinated" when they drew on two adjacent sentences (the prompt asks the model to resolve pronouns and dates from context, which is exactly when that happens). | The grounding check also compares against adjacent-sentence pairs; the 0.5 cosine bar is unchanged. | `paper_trail_core/agents/claim_extractor.py` |
| 6 | LLM calls ran at the model's default `medium` effort. | `complete_json` asks for `effort: high` (claim extraction, translation); falls back cleanly on an SDK that predates `output_config`. | `paper_trail_core/nlp/llm.py` |
| 7 | The API and the pipeline's ensemble node each built their own `ModelRegistry`: every model loaded twice and sat in memory twice. | One process-wide registry (`get_shared_registry`). | `api/model_loader.py`, `api/main.py`, `paper_trail_core/agents/stylistic_ensemble_agent.py` |
| 8 | A failed link check showed "Verification failed. Please try again." in the live chat even though the server had stored the real reason (paywall, no article text, private address). | The poller surfaces `error`. | `frontend/src/services/api.js` |
| 9 | For a link check the pipeline received the URL as typed, not the final URL after redirects, so the locator's country-from-domain context could be wrong. | Pass `article.url`. | `api/routers/articles.py` |
| 10 | Chat and Q&A each had their own idea of "is an LLM key set", different from the pipeline's. | All three use `llm_configured()`. | `api/chat.py`, `api/qa.py` |
| 11 | Installed `scikit-learn` was 1.7.1, not the pinned 1.7.2 (`InconsistentVersionWarning` on every classical model). | `pip install scikit-learn==1.7.2` (environment only, no code). | — |

**Deliberately not changed:** decision thresholds and all scoring constants (`CLAIM_*_MIN`,
`RELEVANCE_MIN`, `CONTRADICTION_FLOOR`, …). The fusion model was calibrated with them. The one
scoring-adjacent change (#4) only affects the numeric-conflict *feature*, which the fusion model
weighs; re-run `scripts/evaluate_pipeline.py --slice golden` before retraining fusion.

---

## Part 7 — Independent LLM fact-check blended into the verdict (2026-11-16, uncommitted)

**Why:** the request was for the final verdict to come from *both* the agentic workflow and a
direct LLM judgement, run simultaneously, weighted 0.6 LLM / 0.4 agentic, "or whichever
guarantees an almost perfect output". Nothing guarantees that; the closest thing is two
independent checkers that must agree, so the weights are the default and an agreement guard
sits on top. Details in `docs/ACCURACY.md` ("How the final verdict is reached").

| Change | Files | Why |
|---|---|---|
| `llm_judge.py`: Claude Opus 5.5 with `web_search_20260209` + `web_fetch_20260209`, effort high, handles `pause_turn`, refusal and truncation; returns `LLMJudgement` (verdict, P(real), confidence, summary, claims, sources). | `paper_trail_core/agents/llm_judge.py` | The second, independent checker. Grounded in live web search so it is not limited to the model's training data. |
| `blend.py`: pure function. 0.6/0.4 weights with discounts (no sources, low LLM confidence, degraded NLI, zero coverage), **agreement** commits, **disagreement → uncertain**, LLM-alone needs confidence ≥ `LLM_JUDGE_MIN_CONFIDENCE` and a source. | `paper_trail_core/agents/blend.py` | Averaging a disagreement produces a confident-looking wrong number; abstaining is the honest output. |
| Pipeline starts the judge in a background thread *before* the graph runs and blends after it; images judge after OCR. Every missing-judge path is a degradation (`llm_judge_not_configured`, `_failed`, `_timeout`, `judges_disagree`). The reasoning trace gains `INDEPENDENT LLM CHECK` and `FINAL BLEND` blocks. | `paper_trail_core/agents/pipeline.py` (`_start_judge`, `_apply_judge`), `VerdictResult.llm_judge` | Simultaneous execution; the agentic verdict is never lost (kept in `llm_judge.blend.agentic_verdict`). |
| Settings `LLM_JUDGE*` (on by default, needs `ANTHROPIC_API_KEY`). | `config.py`, `.env.example` | |
| Persisted as `verdicts.llm_judge` (JSON, added by `ensure_columns`), served as `VerdictOut.llm_judge`; `ai_model_version` → `pipeline-v3-evidence-fusion-llm-judge`. | `api/orm.py`, `api/database.py`, `api/routers/articles.py` | |
| "Independent AI fact-check" section in the verdict report, with the blend line. | `frontend/src/components/VerdictReport.jsx`, `frontend/src/styles/features.css` | |
| `/health/accuracy` stage `llm_judge`; eval harness records `llm_p_real` / `llm_confidence` / `blend_rule` per article and includes the judge settings in its cache key. | `api/accuracy.py`, `scripts/evaluate_pipeline.py` | So the weight can later be learned (`scripts/train_fusion.py`) rather than asserted. |
| Tests off by default (`LLM_JUDGE=False` in `tests/conftest.py`); 23 tests in `tests/test_llm_judge.py` stub the call. | `tests/` | |

**Cost/latency:** one Opus call with up to 8 web searches per check, run in parallel with the
pipeline, so wall time is roughly `max(pipeline, LLM)`. `LLM_JUDGE_MAX_SEARCHES` is the lever.

**Not done (next step if wanted):** learn the weights. Run
`scripts/evaluate_pipeline.py --slice golden` with the judge on, then fit the blend weight (or add
`llm_p_real` as a fusion feature and retrain) instead of trusting 0.6/0.4.

**Still below full accuracy on this machine (config, not code):** the local `.env` has
`CLAIM_BACKEND=heuristic`, no `ANTHROPIC_API_KEY`, and `LLM_MODEL=claude-haiku-4-5-20251001`.
For full accuracy set `CLAIM_BACKEND=llm`, `LLM_MODEL=claude-opus-5-5` and a real key, as
`.env.example` shows. `GET /health/accuracy?deep=true` reports this.

---

## Backtracking

**Symptom → most likely cause**

| Symptom | Look at |
|---|---|
| Every verdict `uncertain` / "stage timed out" | Part 5 #1: timeouts now actually fire. Check `STAGE_TIMEOUT_S` / `LLM_STAGE_TIMEOUT_S` and slow sources in the log. |
| Claims look worse / `llm_not_configured` in limitations | Part 4: `ANTHROPIC_API_KEY` empty in `.env`. |
| `llm_failed` degradation | `paper_trail_core/nlp/llm.py`: refusal, truncation, or a bad key (see server log). |
| Models fail to load / `InconsistentVersionWarning` | Part 4: scikit-learn isn't 1.7.2. |
| `base_models_missing` on verdicts | TensorFlow not installed (Part 5 #2 only *reports* it), or a Keras config mismatch Part 6 #1 doesn't cover (look for "Unrecognized keyword arguments" in the log and add the key to `_KERAS_FORWARD_KEYS`). |
| NLI "unavailable" on a GPU box | Part 6 #2: `dtype` vs `torch_dtype`; or set `NLI_FP16=false`. |
| `retrieval_timeout` in limitations | Part 6 #3: one claim's providers were slow; the others were still checked. Raise `STAGE_TIMEOUT_S` or look at the slow provider in the log. |
| `llm_judge_not_configured` on every verdict | Part 7: `LLM_JUDGE=true` but no `ANTHROPIC_API_KEY`. Set the key, or `LLM_JUDGE=false` to go back to pipeline-only verdicts. |
| Many verdicts `uncertain` with `judges_disagree` | Part 7: the pipeline and the LLM check commit to opposite sides. Read `verdict.llm_judge` to see which is wrong; this is the guard working, not a crash. |
| Checks slower than before | Part 7: wall time is `max(pipeline, LLM check)`; lower `LLM_JUDGE_MAX_SEARCHES` or `LLM_JUDGE_TIMEOUT_S`. |
| 401 `guest_limit_reached` in dev | Part 1: clear the `guest_usage` table or raise `GUEST_QUERY_LIMIT`. |
| Can't read an article by id | Part 1: saved checks are owner-only; send the owner's token. |
| "no such column" errors | Part 1/3: `ensure_columns()` in `api/database.py` (SQLite only; other databases need a migration). |
| Share preview image 500s | Part 2: edge runtime in `frontend/src/app/v/[slug]/opengraph-image.jsx`; `API_INTERNAL_URL` must be reachable from the Next server. |
| Chat reply stuck "Thinking…" | Part 3: `_run_chat_job` failed; see the server log. Stale replies unblock after 5 minutes. |
| Lint failing after a pull | `ruff check .` / `npx next lint`; config in `pyproject.toml` and `frontend/.eslintrc.json`. |

**Reverting a part**

```bash
# Part 1 (auth/history) or Parts 2–4 (features/accuracy): revert the commit
git revert 439030d     # or ffc2b02
# Part 6 while still uncommitted: discard per file, e.g.
git restore api/model_loader.py
```

Reverting is order-sensitive: Parts 2–4 build on Part 1's `Conversation` / `Article.user_id`, so
revert `ffc2b02` before `439030d`. DB columns added by `ensure_columns()` stay in existing
databases after a revert. They're nullable and harmless.
