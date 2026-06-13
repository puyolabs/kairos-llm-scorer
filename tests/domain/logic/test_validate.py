# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the request-vocabulary guard.

Every request value that keys into an operator-configured map must be a configured
word (or a recall-safe sentinel); otherwise the request is rejected with the
allowed words. The tuning is the committed example config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scorer.config as cfg
from scorer.config import load_scorer_tuning
from scorer.domain.logic.validate import (
    RequestVocabularyError,
    validate_request_vocabulary,
)
from scorer.domain.models import ScoreRequest

TUNING = load_scorer_tuning(
    Path(cfg.__file__).resolve().parents[2] / "config" / "scorer.example.toml"
)

_BASE = {
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
}


def _request(*, posting_over: dict | None = None, prefs: dict | None = None) -> ScoreRequest:
    posting = {**_BASE["posting"], **(posting_over or {})}
    profile = {**_BASE["profile"], "preferences": prefs or {}}
    return ScoreRequest.model_validate({"posting": posting, "profile": profile})


def _validate(**kw) -> None:
    validate_request_vocabulary(_request(**kw), TUNING)


# ── passes ───────────────────────────────────────────────────────────────────


def test_clean_request_passes():
    _validate(
        posting_over={"role_region": "europe", "eligibility_gate": "eu-only"},
        prefs={"working_country": "DE", "gate": {"work_countries": ["DE", "US"]}},
    )


@pytest.mark.parametrize(
    "posting_over",
    [
        {"remote": "unknown"},  # sentinel
        {"role_region": "global"},  # sentinel
        {"role_region": "unknown"},  # sentinel
        {"role_region": None},  # unresolved
        {"eligibility_gate": "none"},  # sentinel
        {"eligibility_gate": "unknown"},  # sentinel
        {"eligibility_gate": None},  # unresolved
    ],
)
def test_recall_safe_sentinels_pass(posting_over: dict):
    _validate(posting_over=posting_over)


def test_absent_working_country_and_gate_pass():
    _validate()  # no working_country, no gate ⇒ nothing to check


# ── rejections ─────────────────────────────────────────────────────────────


def test_unconfigured_working_country_rejected():
    with pytest.raises(RequestVocabularyError) as ei:
        _validate(prefs={"working_country": "ZZ"})
    (v,) = ei.value.violations
    assert v.field == "profile.preferences.working_country"
    assert v.value == "ZZ"
    assert "US" in v.allowed and "ZZ" not in v.allowed


def test_unconfigured_work_country_rejected():
    with pytest.raises(RequestVocabularyError) as ei:
        _validate(prefs={"gate": {"work_countries": ["US", "ZZ"]}})
    (v,) = ei.value.violations
    assert v.field == "profile.preferences.gate.work_countries"
    assert v.value == "ZZ"


def test_unconfigured_allowed_region_rejected():
    # An owner-declared gate region absent from region_utc_offset is rejected;
    # sentinels (global/unknown) in the same list still pass.
    with pytest.raises(RequestVocabularyError) as ei:
        _validate(prefs={"gate": {"allowed_regions": ["europe", "global", "ZZ"]}})
    (v,) = ei.value.violations
    assert v.field == "profile.preferences.gate.allowed_regions"
    assert v.value == "ZZ"


def test_unconfigured_eligibility_gate_rejected():
    # A free-string gate the config never declared is a vocabulary violation
    # (distinct from eligibility_allows, which treats an unknown gate as a pass).
    with pytest.raises(RequestVocabularyError) as ei:
        _validate(posting_over={"eligibility_gate": "mars-only-real"})
    (v,) = ei.value.violations
    assert v.field == "posting.eligibility_gate"
    assert v.value == "mars-only-real"


def test_role_region_unconfigured_in_map_rejected():
    # region_utc_offset is the sole source of truth for valid regions; a posting
    # region absent from it (here, 'europe') is rejected by the vocabulary guard.
    trimmed = TUNING.model_copy(update={"region_utc_offset": {"americas": -5.0}})
    with pytest.raises(RequestVocabularyError) as ei:
        validate_request_vocabulary(_request(posting_over={"role_region": "europe"}), trimmed)
    (v,) = ei.value.violations
    assert v.field == "posting.role_region"
    assert v.allowed == ["americas"]


def test_multiple_violations_reported_together():
    trimmed = TUNING.model_copy(update={"region_utc_offset": {"americas": -5.0}})
    request = _request(posting_over={"role_region": "europe"}, prefs={"working_country": "ZZ"})
    with pytest.raises(RequestVocabularyError) as ei:
        validate_request_vocabulary(request, trimmed)
    fields = {v.field for v in ei.value.violations}
    assert fields == {"posting.role_region", "profile.preferences.working_country"}
