"""
Integration tests for the FastAPI endpoints.

Uses FastAPI's TestClient with mocked model registry and pipeline so no
trained models or API keys are required.

Run with: pytest tests/test_api.py -v
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ── App fixture ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Return a TestClient backed by the FastAPI app."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///./test_paper_trail.db"}):
        # Skip transformer warmup — these tests exercise HTTP contracts and mock
        # the pipeline out entirely, so loading a 370 MB NLI model would add
        # tens of seconds for nothing.
        from config import settings
        settings.WARM_MODELS = False

        from api.database import create_tables, engine
        from api.main import app
        create_tables()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    # SQLite keeps the file handle open until the engine's connection pool is
    # disposed; on Windows that makes the unlink fail outright rather than
    # deferring it as POSIX would.
    engine.dispose()
    if os.path.exists("test_paper_trail.db"):
        try:
            os.remove("test_paper_trail.db")
        except OSError:
            pass  # a leftover test DB is not worth failing the suite over


# ── Auth fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def admin_token(client):
    resp = client.post("/auth/token", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture(scope="module")
def analyst_token(client):
    resp = client.post("/auth/token", json={"username": "analyst", "password": "analyst123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


# ── Info endpoints ─────────────────────────────────────────────────────────────

def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "version" in data


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "models_loaded" in data


# ── Auth endpoints ─────────────────────────────────────────────────────────────

def test_token_valid_credentials(client):
    resp = client.post("/auth/token", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_token_invalid_credentials(client):
    resp = client.post("/auth/token", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401


def test_token_unknown_user(client):
    resp = client.post("/auth/token", json={"username": "nobody", "password": "pass"})
    assert resp.status_code == 401


# ── Article endpoints ──────────────────────────────────────────────────────────

SAMPLE_ARTICLE = (
    "Scientists at the University of Cambridge have discovered a new form of "
    "renewable energy that could power entire cities. The breakthrough was "
    "published in the journal Nature on Tuesday after five years of research."
)

_MOCK_VERDICT = {
    "verdict": "likely_real",
    "confidence": 0.75,
    "claim_verdicts": [
        {
            "claim": "Scientists discovered a new form of renewable energy.",
            "verdict": "entailment",
            "confidence": 0.82,
            "evidence_snippets": ["Cambridge researchers announced a breakthrough."],
        }
    ],
    "ensemble_score": 0.72,
    "reasoning_trace": "Ensemble score 0.72. NLI: 1 supporting. Combined: 0.75.",
}


def _reset_guest_usage():
    from api.database import SessionLocal
    from api.orm import GuestUsage
    db = SessionLocal()
    try:
        db.query(GuestUsage).delete()
        db.commit()
    finally:
        db.close()


def _uncertain_vr():
    from paper_trail_core.agents.pipeline import VerdictResult
    return VerdictResult(verdict="uncertain", confidence=0.0, claim_verdicts=[],
                         ensemble_score=0.5, reasoning_trace="")


def test_one_free_query_then_auth_required(client):
    from config import settings
    settings.GUEST_QUERY_LIMIT = 1
    _reset_guest_usage()

    assert client.get("/auth/guest").json()["remaining"] == 1
    with patch("api.routers.articles.run_verification_pipeline", return_value=_uncertain_vr()):
        first = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE})
        assert first.status_code == 202
        assert first.json()["conversation_id"] is None
        # the guest's own check is readable without a token
        assert client.get(f"/articles/{first.json()['id']}").status_code == 200

        second = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE})
    assert second.status_code == 401
    assert second.json()["detail"]["code"] == "guest_limit_reached"
    assert client.get("/auth/guest").json()["remaining"] == 0


# ── Accounts + chat history ────────────────────────────────────────────────────

def test_test_user_can_sign_in_by_username_or_email(client):
    for ident in ("testuser", "test@dailytruth.app", "TestUser"):
        resp = client.post("/auth/token", json={"username": ident, "password": "Test@1234"})
        assert resp.status_code == 200, ident
        assert resp.json()["user"]["username"] == "testuser"


def test_register_then_me(client):
    import uuid
    name = f"newbie{uuid.uuid4().hex[:8]}"
    resp = client.post("/auth/register", json={
        "username": name, "email": f"{name}@example.com",
        "password": "longenough1", "full_name": "New Bie",
    })
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["email"] == f"{name}@example.com"

    dup = client.post("/auth/register", json={
        "username": name, "email": "other@example.com", "password": "longenough1",
    })
    assert dup.status_code == 409
    short = client.post("/auth/register", json={
        "username": "shorty", "email": "s@example.com", "password": "short",
    })
    assert short.status_code == 422


def test_google_sign_in_reports_not_configured(client):
    from config import settings
    settings.GOOGLE_CLIENT_ID = None
    resp = client.post("/auth/google", json={"credential": "x"})
    assert resp.status_code == 501


def _signup(client, name):
    import uuid
    name = f"{name}{uuid.uuid4().hex[:8]}"   # unique even if the test DB survives a run
    resp = client.post("/auth/register", json={
        "username": name, "email": f"{name}@example.com", "password": "password123",
    })
    assert resp.status_code == 201
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_signed_in_checks_build_a_chat_history(client):
    alice = _signup(client, "alice")
    bob = _signup(client, "bob")

    with patch("api.routers.articles.run_verification_pipeline", return_value=_uncertain_vr()):
        first = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE}, headers=alice)
        conv_id = first.json()["conversation_id"]
        assert conv_id is not None
        follow = client.post("/articles/submit",
                             json={"text": "A follow-up claim about the same story.",
                                   "conversation_id": conv_id},
                             headers=alice)
        assert follow.json()["conversation_id"] == conv_id
        # bob can't post into alice's chat
        assert client.post("/articles/submit",
                           json={"text": SAMPLE_ARTICLE, "conversation_id": conv_id},
                           headers=bob).status_code == 404

    chats = client.get("/conversations", headers=alice).json()
    assert [c["id"] for c in chats] == [conv_id]
    assert chats[0]["message_count"] == 2

    detail = client.get(f"/conversations/{conv_id}", headers=alice).json()
    assert [t["text"] for t in detail["turns"]] == [
        SAMPLE_ARTICLE, "A follow-up claim about the same story."]
    assert detail["turns"][0]["verdict"]["verdict"] == "uncertain"

    # privacy: bob and anonymous callers can't read alice's chat or checks
    assert client.get(f"/conversations/{conv_id}", headers=bob).status_code == 404
    assert client.get("/conversations").status_code == 401
    assert client.get(f"/articles/{first.json()['id']}").status_code == 404
    assert client.get(f"/articles/{first.json()['id']}", headers=alice).status_code == 200

    renamed = client.patch(f"/conversations/{conv_id}", json={"title": "Energy story"}, headers=alice)
    assert renamed.json()["title"] == "Energy story"
    assert client.delete(f"/conversations/{conv_id}", headers=alice).status_code == 204
    assert client.get("/conversations", headers=alice).json() == []


def test_guest_check_is_claimed_into_a_chat_after_sign_in(client):
    from config import settings
    settings.GUEST_QUERY_LIMIT = 1
    _reset_guest_usage()

    with patch("api.routers.articles.run_verification_pipeline", return_value=_uncertain_vr()):
        guest = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE})
    guest_id = guest.json()["id"]

    carol = _signup(client, "carol")
    claimed = client.post("/conversations/claim", json={"article_ids": [guest_id]}, headers=carol)
    assert claimed.status_code == 201
    assert [t["id"] for t in claimed.json()["turns"]] == [guest_id]
    # can't be claimed twice
    dave = _signup(client, "dave")
    assert client.post("/conversations/claim", json={"article_ids": [guest_id]},
                       headers=dave).status_code == 404


def test_submit_article_no_text_or_url(client, admin_token):
    resp = client.post(
        "/articles/submit",
        json={},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 422


def test_submit_and_retrieve_article(client, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    mock_vr = VerdictResult(
        verdict="likely_real",
        confidence=0.75,
        claim_verdicts=_MOCK_VERDICT["claim_verdicts"],
        ensemble_score=0.72,
        reasoning_trace="Test trace.",
    )

    with patch("api.routers.articles.run_verification_pipeline", return_value=mock_vr):
        resp = client.post(
            "/articles/submit",
            json={"text": SAMPLE_ARTICLE},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    # Submission is asynchronous: 202 with no verdict attached yet.
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "processing"
    assert body["verdict"] is None

    # TestClient runs BackgroundTasks before returning, so the pipeline has
    # already finished by the time we poll.
    article_id = body["id"]
    get_resp = client.get(f"/articles/{article_id}",
                          headers={"Authorization": f"Bearer {admin_token}"})
    assert get_resp.status_code == 200
    done = get_resp.json()
    assert done["id"] == article_id
    assert done["status"] == "completed"
    assert done["verdict"]["verdict"] == "likely_real"
    assert done["verdict"]["ensemble_score"] == pytest.approx(0.72, abs=1e-4)
    assert "needs_human_review" not in done["verdict"]


def test_get_article_not_found(client):
    resp = client.get("/articles/999999")
    assert resp.status_code == 404


def test_submit_uncertain_verdict_is_returned_without_review_queue(client, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    mock_vr = VerdictResult(
        verdict="uncertain",
        confidence=0.1,
        claim_verdicts=[],
        ensemble_score=0.5,
        reasoning_trace="Unable to determine.",
    )

    with patch("api.routers.articles.run_verification_pipeline", return_value=mock_vr):
        resp = client.post(
            "/articles/submit",
            json={"text": SAMPLE_ARTICLE},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    assert resp.status_code == 202
    article_id = resp.json()["id"]

    done = client.get(f"/articles/{article_id}",
                      headers={"Authorization": f"Bearer {admin_token}"}).json()
    assert done["status"] == "completed"
    assert done["verdict"]["verdict"] == "uncertain"
    assert "needs_human_review" not in done["verdict"]


def test_review_queue_endpoints_are_gone(client, admin_token):
    assert client.get("/review-queue").status_code == 404
    resp = client.post(
        "/review-queue/1/resolve",
        json={"human_verdict": "likely_fake"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 404


# ── Region-aware verification ──────────────────────────────────────────────────

def test_submit_passes_the_region_to_the_pipeline(client, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    vr = VerdictResult(verdict="uncertain", confidence=0.0, claim_verdicts=[],
                       ensemble_score=0.5, reasoning_trace="")
    with patch("api.routers.articles.run_verification_pipeline", return_value=vr) as run:
        resp = client.post(
            "/articles/submit",
            json={"text": SAMPLE_ARTICLE, "region": "Pune"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert resp.status_code == 202
    assert run.call_args.args[2] == "Pune"


def test_region_is_optional(client, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    vr = VerdictResult(verdict="uncertain", confidence=0.0, claim_verdicts=[],
                       ensemble_score=0.5, reasoning_trace="")
    with patch("api.routers.articles.run_verification_pipeline", return_value=vr) as run:
        resp = client.post(
            "/articles/submit",
            json={"text": SAMPLE_ARTICLE},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert resp.status_code == 202
    assert run.call_args.args[2] is None


def test_claim_location_round_trips_through_storage(client, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    location = {
        "name": "Pune", "label": "Pune, Pune District, Maharashtra, India", "level": "city",
        "lat": 18.52, "lon": 73.86, "country": "India", "country_code": "in",
        "state": "Maharashtra", "district": "Pune District", "city": "Pune", "locality": None,
        "confidence": 0.95, "method": "nominatim", "source": "claim",
        "mentions": ["Pune"], "ambiguous": False, "also": [],
    }
    claim = {
        "claim": "The dam released water into the river near Pune.", "verdict": "entailment",
        "confidence": 0.8, "evidence_snippets": ["Dam discharge floods Pune."],
        "location": location, "search_scope": "Pune Maharashtra (city)",
        "regional_evidence": 1, "in_region": [True],
        "evidence_meta": [{"url": "https://punemirror.com/a", "domain": "punemirror.com",
                           "published_at": "2026-10-28T09:00:00Z", "provider": "newsapi",
                           "trust": 0.8, "stance": "repeats", "stance_score": 0.91}],
        "origin": {"basis": "fact_check", "claimant": "Facebook post", "claim_date": "2026-10-27",
                   "claimant_source": "boomlive.in",
                   "timeline": [{"date": "2026-10-28", "domain": "punemirror.com",
                                 "url": "https://punemirror.com/a", "kind": "repeats",
                                 "label": "", "trust": 0.8}],
                   "fact_checks": [], "fact_check_status": "ok"},
    }
    vr = VerdictResult(verdict="likely_real", confidence=0.8, claim_verdicts=[claim],
                       ensemble_score=0.7, reasoning_trace="trace")
    with patch("api.routers.articles.run_verification_pipeline", return_value=vr):
        resp = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE},
                           headers={"Authorization": f"Bearer {admin_token}"})
    body = client.get(f"/articles/{resp.json()['id']}",
                      headers={"Authorization": f"Bearer {admin_token}"}).json()

    cv = body["verdict"]["claim_verdicts"][0]
    assert cv["location"]["name"] == "Pune" and cv["location"]["level"] == "city"
    assert cv["location"]["country_code"] == "in" and cv["location"]["confidence"] == 0.95
    assert cv["search_scope"] == "Pune Maharashtra (city)"
    assert cv["regional_evidence"] == 1 and cv["in_region"] == [True]
    assert cv["evidence_meta"][0]["stance"] == "repeats"
    assert cv["evidence_meta"][0]["published_at"] == "2026-10-28T09:00:00Z"
    assert cv["origin"]["claimant"] == "Facebook post"
    assert cv["origin"]["timeline"][0]["kind"] == "repeats"


def test_claims_stored_before_regions_existed_still_load(client, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    old_claim = {"claim": "An old claim.", "verdict": "neutral", "confidence": 0.5,
                 "evidence_snippets": []}
    vr = VerdictResult(verdict="uncertain", confidence=0.1, claim_verdicts=[old_claim],
                       ensemble_score=0.5, reasoning_trace="")
    with patch("api.routers.articles.run_verification_pipeline", return_value=vr):
        resp = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE},
                           headers={"Authorization": f"Bearer {admin_token}"})
    cv = client.get(f"/articles/{resp.json()['id']}",
                    headers={"Authorization": f"Bearer {admin_token}"}).json()["verdict"]["claim_verdicts"][0]
    assert cv["location"] is None and cv["regional_evidence"] == 0 and cv["in_region"] == []
    assert cv["evidence_meta"] == []
    assert cv["origin"] is None


# ── Model registry endpoint ────────────────────────────────────────────────────

def test_model_registry_no_auth(client):
    resp = client.get("/models/registry")
    assert resp.status_code == 200
    body = resp.json()
    assert "total" in body
    assert "models" in body
    assert isinstance(body["models"], list)


def test_model_registry_invalid_sort(client):
    resp = client.get("/models/registry?sort_by=invalid_metric")
    assert resp.status_code == 422


def test_model_registry_top_n(client):
    resp = client.get("/models/registry?top_n=3")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) <= 3


# ── Check a link ───────────────────────────────────────────────────────────────

def _vr_with_evidence():
    from paper_trail_core.agents.pipeline import VerdictResult
    claim = {
        "claim": "The Eiffel Tower is in Paris.", "verdict": "entailment", "confidence": 0.9,
        "evidence_snippets": ["The Eiffel Tower is a wrought-iron tower in Paris, France."],
        "sources": ["wikipedia.org"], "urls": ["https://en.wikipedia.org/wiki/Eiffel_Tower"],
        "published_at": [""], "trust": [0.8],
    }
    return VerdictResult(verdict="likely_real", confidence=0.9, claim_verdicts=[claim],
                         ensemble_score=0.8, reasoning_trace="trace")


def test_link_check_fetches_then_verifies(client):
    from api.url_fetch import FetchedArticle
    alice = _signup(client, "linker")
    page = FetchedArticle(url="https://news.example.com/story",
                          title="Tower story", text="The Eiffel Tower is in Paris. " * 20)
    with patch("api.routers.articles.fetch_article", return_value=page) as fetch, \
         patch("api.routers.articles.run_verification_pipeline", return_value=_vr_with_evidence()) as run:
        # a bare link in the text box counts as a link check
        resp = client.post("/articles/submit", json={"text": "https://news.example.com/story"}, headers=alice)
    assert resp.status_code == 202 and resp.json()["kind"] == "url"
    fetch.assert_called_once_with("https://news.example.com/story")
    assert run.call_args.args[0].startswith("The Eiffel Tower")

    done = client.get(f"/articles/{resp.json()['id']}", headers=alice).json()
    assert done["status"] == "completed"
    assert done["source_title"] == "Tower story"
    assert done["full_text"].startswith("The Eiffel Tower")
    cv = done["verdict"]["claim_verdicts"][0]
    assert cv["urls"] == ["https://en.wikipedia.org/wiki/Eiffel_Tower"] and cv["trust"] == [0.8]


def test_link_check_reports_why_fetch_failed(client):
    from api.url_fetch import FetchError
    alice = _signup(client, "linkfail")
    with patch("api.routers.articles.fetch_article", side_effect=FetchError("The page needs a login.")), \
         patch("api.routers.articles.run_verification_pipeline") as run:
        resp = client.post("/articles/submit", json={"url": "https://paywalled.example.com/x"}, headers=alice)
    run.assert_not_called()
    done = client.get(f"/articles/{resp.json()['id']}", headers=alice).json()
    assert done["status"] == "failed" and done["error"] == "The page needs a login."


def test_non_http_links_are_rejected(client):
    alice = _signup(client, "badlink")
    resp = client.post("/articles/submit", json={"url": "file:///etc/passwd"}, headers=alice)
    assert resp.status_code == 422


def test_url_guard_blocks_private_addresses():
    from api.url_fetch import FetchError, validate_url
    for bad in ("http://127.0.0.1/", "http://localhost:8000/", "http://169.254.169.254/latest",
                "http://10.0.0.5/", "ftp://example.com/", "http://user:pw@example.com/"):
        with pytest.raises(FetchError):
            validate_url(bad)


def test_fallback_extractor_reads_paragraphs():
    from api.url_fetch import _FallbackParser
    html = ("<html><head><title>T</title><meta property='og:title' content='Real title'></head>"
            "<body><nav><p>Menu menu menu menu menu menu menu menu menu menu</p></nav>"
            "<p>" + "Body sentence about the event. " * 3 + "</p>"
            "<script>var x = 'not text';</script></body></html>")
    p = _FallbackParser()
    p.feed(html)
    assert p.og_title == "Real title"
    assert len(p.paragraphs) == 1 and p.paragraphs[0].startswith("Body sentence")


# ── Sharing ────────────────────────────────────────────────────────────────────

def test_share_link_is_public_and_revocable(client):
    alice = _signup(client, "sharer")
    bob = _signup(client, "notowner")
    with patch("api.routers.articles.run_verification_pipeline", return_value=_vr_with_evidence()):
        aid = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE}, headers=alice).json()["id"]

    assert client.post(f"/articles/{aid}/share", headers=bob).status_code == 404
    share = client.post(f"/articles/{aid}/share", headers=alice).json()
    assert share["path"] == f"/v/{share['slug']}"
    # idempotent
    assert client.post(f"/articles/{aid}/share", headers=alice).json()["slug"] == share["slug"]

    public = client.get(f"/share/{share['slug']}")       # no token
    assert public.status_code == 200
    body = public.json()
    assert body["text"] == SAMPLE_ARTICLE and body["verdict"]["verdict"] == "likely_real"
    assert "user_id" not in body and "conversation_id" not in body

    assert client.delete(f"/articles/{aid}/share", headers=alice).status_code == 204
    assert client.get(f"/share/{share['slug']}").status_code == 404


# ── Feedback ───────────────────────────────────────────────────────────────────

def test_feedback_is_saved_updated_and_returned(client):
    alice = _signup(client, "rater")
    bob = _signup(client, "snoop")
    with patch("api.routers.articles.run_verification_pipeline", return_value=_vr_with_evidence()):
        aid = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE}, headers=alice).json()["id"]

    assert client.put(f"/articles/{aid}/feedback", json={"rating": 1}).status_code == 401
    assert client.put(f"/articles/{aid}/feedback", json={"rating": 1}, headers=bob).status_code == 404
    assert client.put(f"/articles/{aid}/feedback", json={"rating": 5}, headers=alice).status_code == 422

    ok = client.put(f"/articles/{aid}/feedback", json={"rating": 1}, headers=alice)
    assert ok.status_code == 200 and ok.json()["rating"] == 1
    down = client.put(f"/articles/{aid}/feedback",
                      json={"rating": -1, "reason": "bad_sources", "comment": "old article"},
                      headers=alice).json()
    assert down == {"rating": -1, "reason": "bad_sources", "comment": "old article"}
    assert client.get(f"/articles/{aid}", headers=alice).json()["feedback"]["rating"] == -1

    from api.database import SessionLocal
    from api.orm import Feedback
    db = SessionLocal()
    try:
        rows = db.query(Feedback).filter(Feedback.article_id == aid).all()
        assert len(rows) == 1 and rows[0].verdict_at_rating == "likely_real"
    finally:
        db.close()


# ── Follow-up questions ────────────────────────────────────────────────────────

def test_ask_answers_from_the_chat_with_citations(client):
    alice = _signup(client, "asker")
    with patch("api.routers.articles.run_verification_pipeline", return_value=_vr_with_evidence()):
        conv_id = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE},
                              headers=alice).json()["conversation_id"]

    def fake_llm(question, ctx, history):
        assert "[1] (wikipedia.org)" in ctx.text and "Eiffel" in ctx.text
        return "Wikipedia places the tower in Paris [1]."

    from config import settings
    with patch.object(settings, "ANTHROPIC_API_KEY", "sk-ant-test"), \
         patch.object(settings, "LLM_PROVIDER", "anthropic"), \
         patch("api.qa._ask_anthropic", side_effect=fake_llm):
        resp = client.post(f"/conversations/{conv_id}/ask",
                           json={"question": "Why is claim 1 supported?"}, headers=alice)
    assert resp.status_code == 201
    body = resp.json()
    assert body["mode"] == "llm" and "[1]" in body["answer"]
    assert body["citations"][0]["url"] == "https://en.wikipedia.org/wiki/Eiffel_Tower"

    detail = client.get(f"/conversations/{conv_id}", headers=alice).json()
    assert [q["question"] for q in detail["questions"]] == ["Why is claim 1 supported?"]


def test_ask_without_llm_falls_back_to_evidence(client):
    alice = _signup(client, "nollm")
    with patch("api.routers.articles.run_verification_pipeline", return_value=_vr_with_evidence()):
        conv_id = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE},
                              headers=alice).json()["conversation_id"]
    from config import settings
    with patch.object(settings, "ANTHROPIC_API_KEY", None), \
         patch.object(settings, "LLM_PROVIDER", "anthropic"):
        body = client.post(f"/conversations/{conv_id}/ask",
                           json={"question": "Where is the Eiffel Tower?"}, headers=alice).json()
    assert body["mode"] == "extractive"
    assert body["citations"] and "wrought-iron" in body["citations"][0]["snippet"]


def test_ask_is_private(client):
    alice = _signup(client, "private")
    bob = _signup(client, "intruder")
    with patch("api.routers.articles.run_verification_pipeline", return_value=_vr_with_evidence()):
        conv_id = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE},
                              headers=alice).json()["conversation_id"]
    assert client.post(f"/conversations/{conv_id}/ask", json={"question": "hi there"},
                       headers=bob).status_code == 404


# ── Chat continuity ────────────────────────────────────────────────────────────

def test_route_message_rules():
    from api.chat import route_message
    assert route_message("Is it true that the tower was finished in 1887?") == \
        ("verify", "The tower was finished in 1887.")
    assert route_message("Actually the BBC reported it opened in 1887 instead")[0] == "verify"
    assert route_message("Why is claim 2 contradicted?")[0] == "answer"
    assert route_message("Which source is most reliable?")[0] == "answer"
    assert route_message("So what's your verdict?")[0] == "verdict"


def _chat_setup(client, name):
    user = _signup(client, name)
    with patch("api.routers.articles.run_verification_pipeline", return_value=_vr_with_evidence()):
        first = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE}, headers=user).json()
    return user, first["conversation_id"], first["id"]


def _pushback_vr():
    from paper_trail_core.agents.pipeline import VerdictResult
    claim = {
        "claim": "The Eiffel Tower was completed in 1887.", "verdict": "contradiction",
        "confidence": 0.85, "evidence_snippets": ["The tower was completed in March 1889."],
        "sources": ["britannica.com"], "urls": ["https://www.britannica.com/topic/Eiffel-Tower"],
        "published_at": [""], "trust": [0.9],
    }
    return VerdictResult(verdict="likely_fake", confidence=0.85, claim_verdicts=[claim],
                         ensemble_score=0.3, reasoning_trace="t")


def _no_llm():
    from config import settings
    return patch.object(settings, "ANTHROPIC_API_KEY", None)


def test_pushback_runs_a_new_check_and_continues_the_chat(client):
    user, conv_id, first_id = _chat_setup(client, "pusher")

    with _no_llm(), patch("api.routers.articles.run_verification_pipeline",
                          return_value=_pushback_vr()) as run:
        sent = client.post(f"/conversations/{conv_id}/messages",
                           json={"message": "But I read it was completed in 1887, not 1889",
                                 "focus": {"article_id": first_id, "claim_index": 0}},
                           headers=user)
    assert sent.status_code == 202 and sent.json()["status"] == "pending"
    run.assert_called_once()
    # "But I read it was…" is turned into a self-contained claim about the focused finding
    assert run.call_args.args[0] == "The Eiffel Tower was completed in 1887, not 1889."

    reply = client.get(f"/conversations/{conv_id}/messages/{sent.json()['id']}", headers=user).json()
    assert reply["status"] == "done" and reply["mode"] == "heuristic"
    assert "likely misinformation" in reply["answer"].lower()
    assert reply["focus"] == {"article_id": first_id, "claim_index": 0}
    assert len(reply["checks"]) == 1 and reply["checks"][0]["kind"] == "chat"
    assert reply["citations"][0]["url"] == "https://www.britannica.com/topic/Eiffel-Tower"

    detail = client.get(f"/conversations/{conv_id}", headers=user).json()
    # the assistant's check lives in the message, not in the user's list of checks
    assert [t["id"] for t in detail["turns"]] == [first_id]
    assert detail["questions"][0]["checks"][0]["verdict"]["verdict"] == "likely_fake"

    # continuity: the next reply can draw on evidence from the check run above
    with _no_llm():
        nxt = client.post(f"/conversations/{conv_id}/messages",
                          json={"message": "Which source says when it was completed?"},
                          headers=user).json()
    nxt = client.get(f"/conversations/{conv_id}/messages/{nxt['id']}", headers=user).json()
    assert any("britannica" in c["source"] for c in nxt["citations"])


def test_focused_verdict_request(client):
    user, conv_id, first_id = _chat_setup(client, "verdicter")
    with _no_llm():
        sent = client.post(f"/conversations/{conv_id}/messages",
                           json={"message": "What's your verdict on this one?",
                                 "focus": {"article_id": first_id, "claim_index": 0}},
                           headers=user).json()
    reply = client.get(f"/conversations/{conv_id}/messages/{sent['id']}", headers=user).json()
    assert reply["answer"].startswith("My verdict on this finding: **Supported**")
    assert "Eiffel Tower is in Paris" in reply["answer"]


def test_chat_focus_must_be_in_this_chat_and_chat_is_private(client):
    user, conv_id, _ = _chat_setup(client, "owner1")
    other, _, other_article = _chat_setup(client, "owner2")
    assert client.post(f"/conversations/{conv_id}/messages",
                       json={"message": "hi", "focus": {"article_id": other_article}},
                       headers=user).status_code == 404
    assert client.post(f"/conversations/{conv_id}/messages", json={"message": "hi"},
                       headers=other).status_code == 404
    assert client.post(f"/conversations/{conv_id}/messages",
                       json={"message": "hi"}).status_code == 401


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason, self.content, self.stop_details = stop_reason, content, None


def test_llm_tool_loop_runs_verify_and_cites_new_evidence(client, monkeypatch):
    import sys
    import types
    user, conv_id, first_id = _chat_setup(client, "llmuser")

    calls = []
    script = [
        _Resp("tool_use", [_Block(type="text", text="Let me check that."),
                           _Block(type="tool_use", id="tu1", name="verify_claim",
                                  input={"statement": "The Eiffel Tower was completed in 1887."})]),
        _Resp("end_turn", [_Block(type="text",
                                  text="That's contradicted: it was completed in 1889 [2]. "
                                       "My verdict stands: Supported for the original claim [1].")]),
    ]

    class FakeMessages:
        def create(self, **kwargs):
            # snapshot: the loop keeps appending to the same messages list
            calls.append({**kwargs, "messages": list(kwargs["messages"])})
            return script[len(calls) - 1]

    class FakeClient:
        def __init__(self, **kw):
            self.beta = types.SimpleNamespace(messages=FakeMessages())
            self.messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))
    from config import settings
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")

    with patch("api.routers.articles.run_verification_pipeline", return_value=_pushback_vr()):
        sent = client.post(f"/conversations/{conv_id}/messages",
                           json={"message": "I heard it was finished in 1887?",
                                 "focus": {"article_id": first_id, "claim_index": 0}},
                           headers=user).json()
    reply = client.get(f"/conversations/{conv_id}/messages/{sent['id']}", headers=user).json()

    assert reply["mode"] == "llm" and reply["status"] == "done"
    assert [c["n"] for c in reply["citations"]] == [1, 2]
    assert reply["citations"][1]["source"] == "britannica.com"
    assert len(reply["checks"]) == 1
    # the model saw the focus, the tool, and got the tool result back
    first_prompt = calls[0]["messages"][-1]["content"]
    assert "<focus>" in first_prompt and "Claim 1.1" in first_prompt
    assert calls[0]["tools"][0]["name"] == "verify_claim"
    tool_result = calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu1"
    assert '"overall_verdict": "likely misinformation"' in tool_result["content"]
