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
    """Return a TestClient backed by the FastAPI app with an in-memory SQLite DB."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Use in-memory SQLite for tests
    with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///./test_paper_trail.db"}):
        from api.main import app
        from api.database import create_tables
        create_tables()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

    # Clean up test DB
    if os.path.exists("test_paper_trail.db"):
        os.remove("test_paper_trail.db")


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
    "needs_human_review": False,
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
        needs_human_review=False,
    )

    with patch("api.routers.articles.run_verification_pipeline", return_value=mock_vr):
        resp = client.post(
            "/articles/submit",
            json={"text": SAMPLE_ARTICLE},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "completed"
    assert body["verdict"]["verdict"] == "likely_real"
    assert body["verdict"]["ensemble_score"] == pytest.approx(0.72, abs=1e-4)
    assert body["verdict"]["needs_human_review"] is False

    # Retrieve by ID
    article_id = body["id"]
    get_resp = client.get(f"/articles/{article_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == article_id


def test_get_article_not_found(client):
    resp = client.get("/articles/999999")
    assert resp.status_code == 404


def test_submit_uncertain_creates_review_queue_item(client, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    mock_vr = VerdictResult(
        verdict="uncertain",
        confidence=0.1,
        claim_verdicts=[],
        ensemble_score=0.5,
        reasoning_trace="Unable to determine.",
        needs_human_review=True,
    )

    with patch("api.routers.articles.run_verification_pipeline", return_value=mock_vr):
        resp = client.post(
            "/articles/submit",
            json={"text": SAMPLE_ARTICLE},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    assert resp.status_code == 201
    assert resp.json()["verdict"]["needs_human_review"] is True

    # Should appear in review queue
    rq_resp = client.get("/review-queue")
    assert rq_resp.status_code == 200
    items = rq_resp.json()["items"]
    assert len(items) >= 1


# ── Review-queue endpoints ─────────────────────────────────────────────────────

def test_review_queue_no_auth_allowed(client):
    resp = client.get("/review-queue")
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "total" in body
    assert "pages" in body


def test_resolve_requires_auth(client):
    resp = client.post("/review-queue/1/resolve",
                       json={"human_verdict": "likely_fake"})
    assert resp.status_code == 401


def test_resolve_queue_item(client, analyst_token, admin_token):
    """Submit an uncertain article, then resolve it as an analyst."""
    from paper_trail_core.agents.pipeline import VerdictResult

    mock_vr = VerdictResult(
        verdict="uncertain",
        confidence=0.05,
        claim_verdicts=[],
        ensemble_score=0.5,
        reasoning_trace="Uncertain.",
        needs_human_review=True,
    )

    with patch("api.routers.articles.run_verification_pipeline", return_value=mock_vr):
        submit_resp = client.post(
            "/articles/submit",
            json={"text": "Controversial claim that needs review."},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert submit_resp.status_code == 201

    # Find the queued item
    rq_resp = client.get("/review-queue?status=pending")
    assert rq_resp.status_code == 200
    items = rq_resp.json()["items"]
    assert items, "Expected at least one pending item"
    queue_id = items[0]["id"]

    # Resolve it
    resolve_resp = client.post(
        f"/review-queue/{queue_id}/resolve",
        json={"human_verdict": "likely_fake", "human_notes": "Checked manually."},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert resolve_resp.status_code == 200
    body = resolve_resp.json()
    assert body["human_verdict"] == "likely_fake"
    assert body["status"] == "resolved"


def test_resolve_already_resolved_returns_409(client, analyst_token, admin_token):
    from paper_trail_core.agents.pipeline import VerdictResult

    mock_vr = VerdictResult(
        verdict="uncertain", confidence=0.0, claim_verdicts=[],
        ensemble_score=0.5, reasoning_trace="", needs_human_review=True,
    )

    with patch("api.routers.articles.run_verification_pipeline", return_value=mock_vr):
        submit_resp = client.post(
            "/articles/submit",
            json={"text": "Another uncertain article for double-resolve test."},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
    assert submit_resp.status_code == 201

    rq_resp = client.get("/review-queue?status=pending")
    items = rq_resp.json()["items"]
    queue_id = items[0]["id"]

    # First resolve — should succeed
    client.post(
        f"/review-queue/{queue_id}/resolve",
        json={"human_verdict": "likely_real"},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )

    # Second resolve — should 409
    resp = client.post(
        f"/review-queue/{queue_id}/resolve",
        json={"human_verdict": "likely_fake"},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert resp.status_code == 409


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
