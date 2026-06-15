# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for `scorer.application.score`.

Exercises the real two-stage flow end to end: the arithmetic `deterministic_score`
runs for real (no monkeypatching the math), and the LLM stage is stood in by
`FakeScreener`. The decision band is forced by setting the per-call thresholds
*around the request's actual baseline score*, so these tests assert both the
arithmetic and the escalation routing that sits on top of it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from tests.fakes import FakeScreener

import scorer.config as cfg
from scorer import __version__
from scorer.application.score import _should_escalate, score
from scorer.config import Settings, load_scorer_tuning
from scorer.domain.logic import SCORER_MODEL_SENTINEL, deterministic_score
from scorer.domain.logic.validate import RequestVocabularyError
from scorer.domain.models import Decision, ScoreRequest, SonnetJudgement, Verdict

# Tuning from the committed example (the real scorer.toml may be absent in CI).
TUNING = load_scorer_tuning(
    Path(cfg.__file__).resolve().parents[2] / "config" / "scorer.example.toml"
)

# The `application` package re-exports the `score` function, shadowing the
# `score` submodule for attribute access — so `scorer.application.score` resolves
# to the function. Reach the real module via sys.modules to patch its globals.
_score_module = importlib.import_module("scorer.application.score")

# A neutral remote/senior job against a bare profile — no abilities, no prefs, so
# every axis lands on its neutral prior. Its arithmetic score is computed live in
# `baseline_score` below; we never hard-code the number.
_NEUTRAL = {
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

_BUILD_SHA = "deadbeef"
# A verdict the fake returns when escalated — deliberately *overturns* the band
# and carries poisoned provenance, so the assertions prove (a) the screener's
# verdict is what surfaces and (b) `score()` overwrites model-set provenance.
_SCREENER_VERDICT = Verdict(
    decision="skip",
    match_score=3,
    reasoning="overturned by screener",
    version="POISON",
    build_sha="POISON",
    scorer="POISON",
)


def _request(judgement: SonnetJudgement) -> ScoreRequest:
    return ScoreRequest.model_validate({**_NEUTRAL, "sonnet_judgement": judgement})


@pytest.fixture
def baseline_score() -> int:
    """The request's real arithmetic score (band-independent: thresholds = 0)."""
    return deterministic_score(
        _request("off"), tuning=TUNING, apply_threshold=0, maybe_threshold=0
    ).match_score


def _force_band(
    monkeypatch, baseline_score: int, band: Decision, escalate_floor: int = 101
) -> Settings:
    """Pin Settings whose thresholds put `baseline_score` in the wanted band.

    `escalate_floor` defaults to 101 (unreachable) so a forced `skip` never
    escalates on score — these band-routing tests isolate the decision logic.
    The floor's own behavior is covered by the dedicated near-miss tests below.
    """
    if band == "apply":
        apply_t, maybe_t = baseline_score, baseline_score
    elif band == "maybe":
        apply_t, maybe_t = baseline_score + 1, baseline_score
    else:  # skip
        apply_t, maybe_t = baseline_score + 1, baseline_score + 1
    # Settings fields populate by env-var alias, not field name, so we can't pass
    # them to the constructor — copy a default instance, overriding by field name.
    settings = Settings().model_copy(
        update={
            "apply_threshold": apply_t,
            "maybe_threshold": maybe_t,
            "escalate_floor": escalate_floor,
            "build_sha": _BUILD_SHA,
        }
    )
    monkeypatch.setattr(_score_module, "get_settings", lambda: settings)
    return settings


# ── vocabulary guard runs before scoring ────────────────────────────────────


@pytest.mark.anyio
async def test_score_rejects_unconfigured_vocabulary_before_screening(monkeypatch, baseline_score):
    """score() runs the vocabulary guard first: a bad value raises and the screener
    is never reached (the guard fails closed, before any baseline/escalation)."""
    _force_band(monkeypatch, baseline_score, "apply")
    fake = FakeScreener(_SCREENER_VERDICT)
    bad = ScoreRequest.model_validate(
        {
            **_NEUTRAL,
            "sonnet_judgement": "wide",
            "profile": {"body": "x", "preferences": {"working_country": "ZZ"}},
        }
    )
    with pytest.raises(RequestVocabularyError):
        await score(bad, screener=fake)
    assert fake.call_count == 0  # rejected before the LLM stage


# ── _should_escalate: the routing truth table ───────────────────────────────


@pytest.mark.parametrize(
    "decision,score,judgement,expected",
    [
        # off — never, regardless of band/score
        ("apply", 80, "off", False),
        ("maybe", 50, "off", False),
        ("skip", 30, "off", False),
        # narrow — apply only (the tailoring pass)
        ("apply", 80, "narrow", True),
        ("maybe", 50, "narrow", False),
        ("skip", 30, "narrow", False),
        # wide — maybe/apply always; skip only when its score reaches the floor (25)
        ("apply", 80, "wide", True),
        ("maybe", 50, "wide", True),
        ("skip", 30, "wide", True),  # near-miss skip ≥ floor → escalates
        ("skip", 24, "wide", False),  # just under the floor → does not
        ("skip", 0, "wide", False),  # hard-gate skip → never
    ],
)
def test_should_escalate_truth_table(
    decision: Decision, score: int, judgement: SonnetJudgement, expected: bool
):
    baseline = Verdict(decision=decision, match_score=score, reasoning="x")
    assert _should_escalate(baseline, judgement, escalate_floor=25) is expected


# ── score(): real baseline, then the fake screener iff the mode escalates ────


@pytest.mark.parametrize(
    "band,judgement,escalates",
    [
        ("apply", "off", False),
        ("maybe", "off", False),
        ("skip", "off", False),
        ("apply", "narrow", True),
        ("maybe", "narrow", False),
        ("skip", "narrow", False),
        ("apply", "wide", True),
        ("maybe", "wide", True),
        ("skip", "wide", False),
    ],
)
@pytest.mark.anyio
async def test_score_escalation_routing(
    monkeypatch, baseline_score: int, band: Decision, judgement: SonnetJudgement, escalates: bool
):
    _force_band(monkeypatch, baseline_score, band)
    fake = FakeScreener(_SCREENER_VERDICT)
    request = _request(judgement)

    verdict = await score(request, screener=fake)

    if escalates:
        assert fake.call_count == 1
        # The fake's verdict surfaces (overturned to skip), not the baseline.
        assert verdict.decision == "skip"
        assert verdict.reasoning == "overturned by screener"
        # Substance comes from the screener (3), not the baseline (60) — proves the
        # provenance stamp doesn't clobber the verdict's own score.
        assert verdict.match_score == _SCREENER_VERDICT.match_score == 3
        # The screener was handed the arithmetic baseline it must refine.
        _, passed_baseline = fake.calls[0]
        assert passed_baseline.decision == band
        assert passed_baseline.match_score == baseline_score
    else:
        assert fake.call_count == 0
        # Untouched arithmetic verdict surfaces with its real band.
        assert verdict.decision == band
        assert verdict.match_score == baseline_score


@pytest.mark.anyio
async def test_wide_escalates_near_miss_skip_at_floor(monkeypatch, baseline_score):
    """A `skip`-band verdict still escalates under `wide` when its score reaches the
    floor — the near-miss band the LLM most often overturns."""
    _force_band(monkeypatch, baseline_score, "skip", escalate_floor=baseline_score)
    fake = FakeScreener(_SCREENER_VERDICT)

    verdict = await score(_request("wide"), screener=fake)

    assert fake.call_count == 1  # score ≥ floor → escalated despite skip band
    assert verdict.decision == "skip"  # the screener's overturned verdict surfaces
    assert verdict.reasoning == "overturned by screener"


@pytest.mark.anyio
async def test_wide_does_not_escalate_skip_below_floor(monkeypatch, baseline_score):
    """A `skip` whose score is under the floor (a hard-gate-style skip) never escalates."""
    _force_band(monkeypatch, baseline_score, "skip", escalate_floor=baseline_score + 1)
    fake = FakeScreener(_SCREENER_VERDICT)

    verdict = await score(_request("wide"), screener=fake)

    assert fake.call_count == 0  # below floor → no call
    assert verdict.decision == "skip"
    assert verdict.match_score == baseline_score  # untouched arithmetic verdict


@pytest.mark.anyio
async def test_provenance_is_stamped_on_the_arithmetic_path(monkeypatch, baseline_score):
    """The deterministic verdict gets version/build_sha/scorer stamped on exit."""
    settings = _force_band(monkeypatch, baseline_score, "maybe")
    verdict = await score(_request("off"), screener=FakeScreener())

    assert verdict.version == __version__
    assert verdict.build_sha == settings.build_sha == _BUILD_SHA
    assert verdict.scorer == SCORER_MODEL_SENTINEL


@pytest.mark.anyio
async def test_provenance_overwrites_screener_supplied_values(monkeypatch, baseline_score):
    """Provenance is single-sourced in score(): model-set fields are overwritten."""
    _force_band(monkeypatch, baseline_score, "apply")
    fake = FakeScreener(_SCREENER_VERDICT)  # carries POISON provenance

    verdict = await score(_request("wide"), screener=fake)

    assert verdict.version == __version__
    assert verdict.build_sha == _BUILD_SHA
    assert verdict.scorer == SCORER_MODEL_SENTINEL
    # model_copy is non-mutating: the screener's verdict object is untouched, so
    # the shared module-level constant keeps its POISON provenance for other tests.
    assert verdict is not _SCREENER_VERDICT
    assert _SCREENER_VERDICT.version == "POISON"


@pytest.mark.anyio
async def test_screener_substance_passes_through_only_provenance_changes(
    monkeypatch, baseline_score
):
    """On escalation the screener's verdict surfaces byte-for-byte — only the three
    provenance fields are stamped; every substantive field is preserved."""
    _force_band(monkeypatch, baseline_score, "apply")
    rich = Verdict(
        decision="apply",
        match_score=88,
        reasoning="rich screener reasoning",
        risks_and_gaps=["gap A", "gap B"],
        tailoring_hints=["hint 1", "hint 2"],
        version="POISON",
        build_sha="POISON",
        scorer="POISON",
    )

    verdict = await score(_request("wide"), screener=FakeScreener(rich))

    # Substantive fields pass through unchanged.
    assert verdict.decision == "apply"
    assert verdict.match_score == 88
    assert verdict.reasoning == "rich screener reasoning"
    assert verdict.risks_and_gaps == ["gap A", "gap B"]
    assert verdict.tailoring_hints == ["hint 1", "hint 2"]
    # Only provenance is rewritten.
    assert verdict.version == __version__
    assert verdict.build_sha == _BUILD_SHA
    assert verdict.scorer == SCORER_MODEL_SENTINEL
