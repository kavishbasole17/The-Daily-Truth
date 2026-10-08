"""
Integration tests for the FastAPI endpoints.

Uses FastAPI's TestClient with mocked model registry and pipeline so no
trained models or API keys are required.

Run with: pytest tests/test_api.py -v
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# ── App fixture ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Return a TestClient backed by the FastAPI app."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///./test_paper_trail.db"}):
        # Skip transformer warmup — these tests exercise HTTP contracts and mock
        # the pipeline out entirely, so loading a 370 MB NLI model would add
        # tens of seconds for nothing.
        from config import settings
        settings.WARM_MODELS = False

        from api.main import app
        from api.database import create_tables, engine
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


def test_submit_article_requires_auth(client):
    resp = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE})
    assert resp.status_code == 401


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
    get_resp = client.get(f"/articles/{article_id}")
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

    done = client.get(f"/articles/{article_id}").json()
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
    }
    vr = VerdictResult(verdict="likely_real", confidence=0.8, claim_verdicts=[claim],
                       ensemble_score=0.7, reasoning_trace="trace")
    with patch("api.routers.articles.run_verification_pipeline", return_value=vr):
        resp = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE},
                           headers={"Authorization": f"Bearer {admin_token}"})
    body = client.get(f"/articles/{resp.json()['id']}").json()

    cv = body["verdict"]["claim_verdicts"][0]
    assert cv["location"]["name"] == "Pune" and cv["location"]["level"] == "city"
    assert cv["location"]["country_code"] == "in" and cv["location"]["confidence"] == 0.95
    assert cv["search_scope"] == "Pune Maharashtra (city)"
    assert cv["regional_evidence"] == 1 and cv["in_region"] == [True]


def test_claims_stored_before_regions_existed_still_load(client, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    old_claim = {"claim": "An old claim.", "verdict": "neutral", "confidence": 0.5,
                 "evidence_snippets": []}
    vr = VerdictResult(verdict="uncertain", confidence=0.1, claim_verdicts=[old_claim],
                       ensemble_score=0.5, reasoning_trace="")
    with patch("api.routers.articles.run_verification_pipeline", return_value=vr):
        resp = client.post("/articles/submit", json={"text": SAMPLE_ARTICLE},
                           headers={"Authorization": f"Bearer {admin_token}"})
    cv = client.get(f"/articles/{resp.json()['id']}").json()["verdict"]["claim_verdicts"][0]
    assert cv["location"] is None and cv["regional_evidence"] == 0 and cv["in_region"] == []


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
