# SPDX-License-Identifier: Apache-2.0
"""HTTP-surface tests: the /score API-key gate and the open /health probe.

The screener is stood in by `FakeScreener` and every request uses
`sonnet_judgement="off"`, so /score returns the arithmetic verdict without any
network. Auth posture is driven by overriding `get_settings` (what the gate
dependency reads) per test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.fakes import FakeScreener

from scorer import __version__, api
from scorer.config import Settings, get_settings
from scorer.domain.logic import SCORER_MODEL_SENTINEL

# A minimal valid request that never escalates (judgement="off") — so the call
# stays arithmetic-only and never reaches the (faked) screener's network path.
_REQUEST = {
    "posting": {
        "kind": "job",
        "source_id": "s",
        "external_id": "1",
        "canonical_key": "s::1",
        "url": "https://x/y",
        "title": "Engineer",
        "company": "Acme",
        "description": "..",
        "posted_at": "2026-01-01T00:00:00Z",
        "fetched_at": "2026-01-01T00:00:00Z",
        "location_text": "Remote",
        "remote": "yes",
        "seniority_hint": "senior",
    },
    "profile": {"body": "x"},
    "sonnet_judgement": "off",
}


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the screener faked out (no network on /score)."""
    monkeypatch.setattr(api, "_screener", FakeScreener())
    return TestClient(api.app)


def _with_keys(keys: str):
    """Override the gate's settings so it sees `keys` as the accepted list."""
    return lambda: Settings().model_copy(update={"scorer_api_keys": keys})


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    api.app.dependency_overrides.clear()


def test_health_is_open(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_score_open_when_no_keys_configured(client):
    """Empty key list ⇒ auth disabled ⇒ /score accepts an unauthenticated call."""
    api.app.dependency_overrides[get_settings] = _with_keys("")
    assert client.post("/score", json=_REQUEST).status_code == 200


def test_score_rejects_missing_key(client):
    api.app.dependency_overrides[get_settings] = _with_keys("secret-1,secret-2")
    resp = client.post("/score", json=_REQUEST)
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_score_rejects_wrong_key(client):
    api.app.dependency_overrides[get_settings] = _with_keys("secret-1")
    resp = client.post("/score", json=_REQUEST, headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_score_accepts_bearer_key(client):
    api.app.dependency_overrides[get_settings] = _with_keys("secret-1,secret-2")
    resp = client.post("/score", json=_REQUEST, headers={"Authorization": "Bearer secret-2"})
    assert resp.status_code == 200


def test_score_accepts_x_api_key_header(client):
    api.app.dependency_overrides[get_settings] = _with_keys("secret-1")
    resp = client.post("/score", json=_REQUEST, headers={"X-API-Key": "secret-1"})
    assert resp.status_code == 200


# ── response contracts ───────────────────────────────────────────────────────


def test_score_returns_a_provenance_stamped_verdict(client):
    """A 200 carries a real, fully-formed Verdict — not just a status code."""
    api.app.dependency_overrides[get_settings] = _with_keys("")  # auth off
    body = client.post("/score", json=_REQUEST).json()
    assert body["decision"] in {"apply", "maybe", "skip"}
    assert 0 <= body["match_score"] <= 100
    assert isinstance(body["reasoning"], str) and body["reasoning"]
    # Provenance is stamped on the way out (single-sourced in score()).
    assert body["version"] == __version__
    assert body["scorer"] == SCORER_MODEL_SENTINEL


def test_unconfigured_vocabulary_returns_422_with_violations(client):
    """An unconfigured vocabulary value is mapped to a 422 whose body lists each
    violation as {field, value, allowed} — the structured error contract."""
    api.app.dependency_overrides[get_settings] = _with_keys("")  # auth off; fail on vocab
    bad = {
        **_REQUEST,
        "profile": {"body": "x", "preferences": {"working_country": "ZZ"}},
    }
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422
    payload = resp.json()
    assert "detail" in payload
    violation = next(
        v for v in payload["violations"] if v["field"] == "profile.preferences.working_country"
    )
    assert violation["value"] == "ZZ"
    assert "ZZ" not in violation["allowed"]


def test_dimension_mismatch_returns_422_with_detail_list(client):
    """Mismatched embedding dimensions hit the model_validator → ValidationError,
    which the hand-rolled handler maps to FastAPI's default {"detail": [...]} shape.
    Pins that contract now that /score parses the body itself."""
    api.app.dependency_overrides[get_settings] = _with_keys("")  # auth off
    bad = {
        **_REQUEST,
        "profile": {
            "body": "x",
            "preferences": {
                "candidate_domains": [
                    {"tag": "a", "gloss": "a", "vector": [1.0, 0.0]},
                    {"tag": "b", "gloss": "b", "vector": [1.0, 0.0, 0.0]},
                ]
            },
        },
    }
    resp = client.post("/score", json=bad)
    assert resp.status_code == 422
    payload = resp.json()
    assert isinstance(payload["detail"], list)
    assert any("dimension" in err["msg"] for err in payload["detail"])


def test_malformed_json_body_returns_422(client):
    """A body that isn't valid JSON raises a pydantic ValidationError from
    model_validate_json — same 422 path, not a 500."""
    api.app.dependency_overrides[get_settings] = _with_keys("")  # auth off
    resp = client.post("/score", content=b"{not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert isinstance(resp.json()["detail"], list)
