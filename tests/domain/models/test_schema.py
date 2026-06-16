# SPDX-License-Identifier: Apache-2.0
"""Behavioral contract tests for `scorer.domain.models`.

Only the bits that aren't plain pydantic field declarations: union
discrimination, the gate-default, forward-compat (extra fields tolerated),
the contract bounds, and a real-posting round-trip.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from scorer.domain.models import (
    Contract,
    GatePreferences,
    Job,
    LedgerEntry,
    Posting,
    Preferences,
    Profile,
    ScoreRequest,
    SemanticTag,
    Verdict,
)

_POSTING = TypeAdapter(Posting)


def _job(**over) -> dict:
    return {
        "kind": "job",
        "source_id": "s",
        "external_id": "1",
        "canonical_key": "s::1",
        "url": "https://x/y",
        "title": "Engineer",
        "company": "Acme",
        "description": "..",
        "posted_at": "2026-06-12T00:00:00Z",
        "fetched_at": "2026-06-12T00:00:00Z",
        "location_text": "Remote",
        "remote": "yes",
        "seniority_hint": "senior",
        **over,
    }


def _contract(**over) -> dict:
    return {
        "kind": "contract",
        "source_id": "s",
        "external_id": "2",
        "canonical_key": "s::2",
        "url": "https://x/y",
        "title": "Contractor",
        "company": "Acme",
        "description": "..",
        "posted_at": "2026-06-12T00:00:00Z",
        "fetched_at": "2026-06-12T00:00:00Z",
        "location_text": "Remote",
        "remote": "yes",
        "engagement_type": "hourly",
        "duration_hint": "long",
        **over,
    }


# A synthetic, anonymized posting↔profile, scoped to the round-trip test below.
# Carries the resolved scoring inputs on each side plus extra columns
# (id/status) the contract must tolerate. No real posting or profile data.
_GOLDEN: dict = {
    "posting": _job(
        source_id="examplejobs",
        external_id="42007",
        canonical_key="examplejobs::42007",
        url="https://example.com/jobs/42007",
        title="Backend Engineer",
        company="Globex Corp",
        description="Remote backend engineer (Go, Kubernetes, PostgreSQL).",
        seniority_hint="unspecified",
        role_region="americas",
        eligibility_gate="none",
        salary_min_annual_usd=130000,
        salary_max_annual_usd=170000,
        salary_currency="USD",
        salary_period="year",
        abilities=[
            {"tag": "Go", "ordinal": 0, "gloss": "Go language", "vector": [1.0, 0.0]},
            {
                "tag": "Kubernetes",
                "ordinal": 1,
                "gloss": "Kubernetes orchestration",
                "vector": [0.0, 1.0],
            },
            {
                "tag": "PostgreSQL",
                "ordinal": 2,
                "gloss": "PostgreSQL database",
                "vector": [1.0, 1.0],
            },
        ],
        role_families=[
            {"tag": "software-engineering", "gloss": "software engineering", "vector": [1.0, 0.0]}
        ],
        domains=[{"tag": "logistics", "gloss": "logistics supply chain", "vector": [0.0, 1.0]}],
        id="11111111-1111-4111-8111-111111111111",
        status="scored",
    ),
    "profile": {
        "body": "# Backend Engineer\n\nRemote. Distributed systems and backend services.",
        "ledger": [
            {"tag": "Go", "tier": "core", "gloss": "Go language", "vector": [1.0, 0.0]},
            {
                "tag": "Kubernetes",
                "tier": "proficient",
                "gloss": "Kubernetes orchestration",
                "vector": [0.0, 1.0],
            },
            {
                "tag": "PostgreSQL",
                "tier": "core",
                "gloss": "PostgreSQL database",
                "vector": [1.0, 1.0],
            },
        ],
        "preferences": {
            "salary_min_annual_usd": 120000,
            "preferred_engagement": "job",
            "excluded_industries": ["gambling"],
            "candidate_domains": [
                {"tag": "fintech", "gloss": "fintech payments", "vector": [0.0, 1.0]},
                {"tag": "saas", "gloss": "saas cloud software", "vector": [0.5, 0.5]},
            ],
            "gate": {
                "allowed_work_arrangements": ["remote"],
                "allowed_seniorities": ["mid", "senior", "staff"],
                "allowed_regions": ["americas", "global", "unknown"],
                "work_countries": ["CA"],
                "allowed_role_families": [
                    {
                        "tag": "software-engineering",
                        "gloss": "software engineering",
                        "vector": [1.0, 0.0],
                    }
                ],
            },
        },
    },
}


# ── 1. discriminated union on `kind` ────────────────────────────────────────


def test_kind_discriminates_to_job():
    assert isinstance(_POSTING.validate_python(_job()), Job)


def test_kind_discriminates_to_contract():
    assert isinstance(_POSTING.validate_python(_contract()), Contract)


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        _POSTING.validate_python(_job(kind="internship"))


def test_job_requires_seniority_hint():
    job = _job()
    del job["seniority_hint"]
    with pytest.raises(ValidationError):
        _POSTING.validate_python(job)


@pytest.mark.parametrize("field", ["engagement_type", "duration_hint"])
def test_contract_requires_engagement_and_duration(field):
    contract = _contract()
    del contract[field]
    with pytest.raises(ValidationError):
        _POSTING.validate_python(contract)


# ── 2. the gate-default gotcha ──────────────────────────────────────────────


def test_preferences_gate_defaults_to_none():
    # None ⇒ the baseline skips gating (GATE=1). This default is load-bearing.
    assert Preferences().gate is None
    assert Profile(body="x").preferences.gate is None


def test_explicit_empty_gate_is_a_real_gate_not_none():
    # The flagged gotcha: an explicit GatePreferences{} is a REAL (empty) gate,
    # distinct from None — empty allowlists would zero real signals. Kairos must
    # send None to skip gating, never {}.
    prefs = Preferences(gate=GatePreferences())
    assert prefs.gate is not None
    assert prefs.gate.allowed_work_arrangements == []


# ── 3. forward-compat: extra fields tolerated ───────────────────────────────


def test_extra_posting_columns_are_ignored():
    job = _POSTING.validate_python(_job(id="uuid", status="scored", surprise=1))
    assert not hasattr(job, "surprise")
    assert "surprise" not in job.model_dump()


def test_scorerequest_tolerates_extra_kairos_fields():
    req = ScoreRequest.model_validate(
        {"posting": _job(legacy_domain="ecommerce"), "profile": {"body": "x"}}
    )
    assert isinstance(req.posting, Job)


def test_sonnet_judgement_defaults_to_wide():
    # Default routing when Kairos omits the field; "wide" is load-bearing.
    assert ScoreRequest(posting=_job(), profile={"body": "x"}).sonnet_judgement == "wide"


def test_sonnet_judgement_rejects_unknown_value():
    with pytest.raises(ValidationError):
        ScoreRequest(posting=_job(), profile={"body": "x"}, sonnet_judgement="loud")


# ── 4. contract bounds ──────────────────────────────────────────────────────


@pytest.mark.parametrize("score", [-1, 101, 1000])
def test_match_score_out_of_range_rejected(score):
    with pytest.raises(ValidationError):
        Verdict(decision="apply", match_score=score, reasoning="r")


def test_match_score_bounds_are_inclusive():
    assert Verdict(decision="skip", match_score=0, reasoning="r").match_score == 0
    assert Verdict(decision="apply", match_score=100, reasoning="r").match_score == 100


def test_empty_profile_body_rejected():
    with pytest.raises(ValidationError):
        Profile(body="")


@pytest.mark.parametrize("iso2", ["us", "USA", "U1"])
def test_country_iso2_pattern_rejected(iso2):
    with pytest.raises(ValidationError):
        _POSTING.validate_python(_job(country_iso2=iso2))


def test_country_iso2_valid():
    job = _POSTING.validate_python(_job(country_iso2="US"))
    assert job.country_iso2 == "US"


@pytest.mark.parametrize("currency", ["US", "USDX"])
def test_salary_currency_length_rejected(currency):
    with pytest.raises(ValidationError):
        _POSTING.validate_python(_job(salary_currency=currency))


@pytest.mark.parametrize("hours", [0, -1])
def test_contract_hours_per_week_must_be_positive(hours):
    with pytest.raises(ValidationError):
        _POSTING.validate_python(_contract(hours_per_week_hint=hours))


# ── 4b. SemanticTag.vector positive-dimension guard ─────────────────────────
# The real guard against a dim-0 embedding: ScoreRequest's validator only checks
# dimension *consistency* (all-empty vectors share dim 0 and would pass), so this
# field constraint is what rejects [] outright.


def test_semantic_tag_rejects_empty_vector():
    with pytest.raises(ValidationError):
        SemanticTag(tag="t", gloss="g", vector=[])


def test_ledger_entry_rejects_empty_vector():
    # LedgerEntry subclasses SemanticTag, so it inherits the min_length=1 guard.
    with pytest.raises(ValidationError):
        LedgerEntry(tag="t", tier="core", gloss="g", vector=[])


# ── 5. real-posting round-trip ──────────────────────────────────────────────


def test_golden_request_parses_and_round_trips():
    req = ScoreRequest(posting=_GOLDEN["posting"], profile=_GOLDEN["profile"])
    # Each side carries its resolved scoring inputs.
    assert isinstance(req.posting, Job)
    assert req.posting.abilities and req.posting.domains
    assert req.profile.ledger and req.profile.preferences.gate is not None
    # Extra Kairos columns were dropped, so a dump→reparse is a fixed point.
    assert ScoreRequest(**req.model_dump()) == req
    assert "status" not in req.posting.model_dump()
