# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for `scorer.domain.logic.baseline`.

Two layers: the pure axis helpers (seniority ladder, timezone overlap, domain
match, the rescale/rounding primitives) and the `deterministic_score`
integration (gate zeroing, injected decision bands, risk surfacing, and that the
generalized seniority/timezone inputs move the final score).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scorer.config as cfg
from scorer.config import load_scorer_tuning
from scorer.domain.logic.baseline import (
    _closeness_weight,
    _cosine,
    _domain_score,
    _round_half_up,
    _tier_fit,
    _timezone_score,
    clamp01,
    deterministic_score,
)
from scorer.domain.models import ScoreRequest, SemanticTag, SeniorityHint

# Load tuning from the committed example so the suite runs without the gitignored
# real config/scorer.toml (mirrors the prompt tests' use of screener.example.xml).
TUNING = load_scorer_tuning(
    Path(cfg.__file__).resolve().parents[2] / "config" / "scorer.example.toml"
)


def _request(
    *,
    posting_over: dict | None = None,
    profile_over: dict | None = None,
    prefs: dict | None = None,
) -> ScoreRequest:
    """An all-default remote senior job faced against a bare profile.

    Every tag carries a vector, so the closeness path is always live; the
    technical axis still credits an exact `tag` match at full tier first.
    """
    posting = {
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
        **(posting_over or {}),
    }
    profile: dict = {"body": "x", **(profile_over or {})}
    if prefs is not None:
        profile["preferences"] = prefs
    return ScoreRequest.model_validate({"posting": posting, "profile": profile})


def _score(req: ScoreRequest, *, apply: int = 65, maybe: int = 35):
    return deterministic_score(req, tuning=TUNING, apply_threshold=apply, maybe_threshold=maybe)


# ── 1. rescale + rounding primitives ────────────────────────────────────────


@pytest.mark.parametrize(
    "v,lo,hi,expected",
    [
        (0.5, 0.0, 1.0, 0.5),
        (1.5, 0.0, 1.0, 1.0),  # clamped high
        (-1.0, 0.0, 1.0, 0.0),  # clamped low
        (0.6, 0.6, 1.0, 0.0),  # at floor
        (0.8, 0.6, 1.0, 0.5),  # midpoint of [0.6, 1.0]
        (0.5, 1.0, 1.0, 0.0),  # degenerate hi <= lo
    ],
)
def test_clamp01(v, lo, hi, expected):
    assert clamp01(v, lo, hi) == pytest.approx(expected)


@pytest.mark.parametrize("x,expected", [(0.5, 1), (1.5, 2), (2.5, 3), (2.4, 2), (64.5, 65)])
def test_round_half_up_not_bankers(x, expected):
    # 2.5 → 3 (half-up), where Python's round() would give 2 (banker's).
    assert _round_half_up(x) == expected


# ── 2. seniority axis — distance to the candidate's own level ────────────────


@pytest.mark.parametrize(
    "posting,expected",
    [("senior", 1.0), ("staff", 0.8), ("mid", 0.8), ("junior", 0.6)],
)
def test_tier_fit_relative_to_senior_candidate(posting: SeniorityHint, expected: float):
    assert _tier_fit(posting, "senior", TUNING) == pytest.approx(expected)


def test_tier_fit_pivots_with_candidate():
    # The same posting scores differently depending on who the candidate is.
    assert _tier_fit("junior", "junior", TUNING) == 1.0
    assert _tier_fit("junior", "staff", TUNING) == pytest.approx(0.4)  # 3 steps × step 0.2


@pytest.mark.parametrize(
    "posting,candidate",
    [("unspecified", "senior"), (None, "senior"), ("senior", None), (None, None)],
)
def test_tier_fit_neutral_when_either_side_unstated(
    posting: SeniorityHint | None, candidate: SeniorityHint | None
):
    assert _tier_fit(posting, candidate, TUNING) == 0.6


# ── 3. remote axis — working-hours overlap ──────────────────────────────────


@pytest.mark.parametrize(
    "region,country,expected",
    [
        ("americas", "US", 1.0),  # US (-5) same tz as americas
        ("europe", "US", 1.0),  # 6h apart → 8h overlap → full shift
        ("mea", "US", 0.75),  # 8h apart → 6h overlap
        ("apac", "US", 0.125),  # 13h apart → 1h overlap
        ("apac", "SG", 1.0),  # symmetric: SG (+8) candidate, apac role
        ("americas", "SG", 0.125),  # symmetric mirror of the americas case
        ("mea", "IN", 1.0),  # IN (+5.5) fractional offset
    ],
)
def test_timezone_overlap(region: str, country: str, expected: float):
    assert _timezone_score(region, country, TUNING) == pytest.approx(expected)


def test_timezone_global_is_full():
    # `global` fits any timezone ⇒ full credit, with or without a candidate country.
    assert _timezone_score("global", "US", TUNING) == 1.0
    assert _timezone_score("global", None, TUNING) == 1.0


@pytest.mark.parametrize(
    "region,country",
    [
        ("unknown", "US"),  # role unplaceable
        (None, "US"),  # role unplaceable
        ("apac", None),  # candidate has no working country
        ("apac", "ZZ"),  # candidate country not in the offset table
    ],
)
def test_timezone_unplaceable_floors(region: str | None, country: str | None):
    # An unplaceable side sits at the axis floor (normalizes to 0), not the max —
    # no free remote credit for region-unknown postings (ADR 0056 §2).
    assert _timezone_score(region, country, TUNING) == 0.5


def test_timezone_region_absent_from_offset_table_floors():
    # A non-sentinel region the config never mapped (here 'antarctica') is
    # unplaceable on the job side ⇒ floor, even with a placeable candidate.
    assert _timezone_score("antarctica", "US", TUNING) == TUNING.remote_unknown


# ── 4. domain axis — closeness bands over the gloss-vectors ──────────────────


def _dt(vec: list[float]) -> SemanticTag:
    """A domain SemanticTag carrying a given vector (tag/gloss are unread here)."""
    return SemanticTag(tag="d", gloss="d", vector=vec)


@pytest.mark.parametrize(
    "posting_vec,cand_vec,expected",
    [
        ([1.0, 0.0], [1.0, 0.0], (TUNING.domain_direct, "direct")),  # cosine 1.0 ≥ 0.90
        ([1.0, 0.0], [0.0, 1.0], (TUNING.domain_mismatch, "mismatch")),  # cosine 0.0 ≤ 0.75
        ([1.0, 0.0], [0.8, 0.6], (TUNING.domain_transferable, "transferable~")),  # cosine 0.8
    ],
)
def test_domain_score_vector_bands(posting_vec, cand_vec, expected):
    assert _domain_score([_dt(posting_vec)], [_dt(cand_vec)], tuning=TUNING) == expected


@pytest.mark.parametrize(
    "domains,cands",
    [
        ([], [_dt([1.0, 0.0])]),  # no posting domains
        ([_dt([1.0, 0.0])], []),  # owner declares no candidate domains
    ],
)
def test_domain_score_empty_side_is_neutral(domains, cands):
    # Either side empty ⇒ no signal ⇒ transferable (neutral), not a mismatch.
    assert _domain_score(domains, cands, tuning=TUNING) == (
        TUNING.domain_transferable,
        "transferable",
    )


# ── 5. deterministic_score — gate, bands, risks, structure ──────────────────


def test_gate_zeroes_score_to_skip():
    # Posting is remote, owner gate allows onsite only ⇒ modality factor 0.
    req = _request(
        prefs={"gate": {"allowed_work_arrangements": ["onsite"], "allowed_seniorities": ["senior"]}}
    )
    v = _score(req)
    assert v.match_score == 0
    assert v.decision == "skip"
    assert "gated" in v.reasoning and "modality" in v.reasoning


def test_injected_bands_drive_the_decision():
    req = _request()
    s = _score(req).match_score
    assert _score(req, apply=0, maybe=0).decision == "apply"  # everything applies
    assert _score(req, apply=101, maybe=s + 1).decision == "skip"  # nothing reaches a band
    assert _score(req, apply=s + 1, maybe=0).decision == "maybe"  # between the bands


def test_score_is_bounded_and_baseline_sets_no_tailoring_hints():
    v = _score(_request())
    assert 0 <= v.match_score <= 100
    assert v.tailoring_hints == []  # tailoring is the Sonnet stage's job
    assert "Deterministic score" in v.reasoning


def test_matched_ability_raises_technical_axis():
    # An exact `tag` match ("Go" == "Go") earns full tier credit; an empty ledger
    # earns only the baseline prior — so the matched run outscores the unmatched.
    ability = {"tag": "Go", "ordinal": 0, "gloss": "Go", "vector": [1.0, 0.0]}
    unmatched = _score(_request(posting_over={"abilities": [ability]}))
    matched = _score(
        _request(
            posting_over={"abilities": [ability]},
            profile_over={
                "ledger": [{"tag": "Go", "tier": "core", "gloss": "Go", "vector": [1.0, 0.0]}]
            },
        )
    )
    assert matched.match_score > unmatched.match_score


def test_higher_ranked_ability_dominates_technical_axis():
    # Ability credit is weighted by ordinal (weight = 26 - ordinal), so matching
    # the top-ranked ability must beat matching a lower-ranked one — same two
    # abilities, same single match, only *which* one is matched differs.
    go = {"tag": "Go", "ordinal": 0, "gloss": "Go", "vector": [1.0, 0.0]}
    rust = {"tag": "Rust", "ordinal": 20, "gloss": "Rust", "vector": [0.0, 1.0]}
    abilities = {"abilities": [go, rust]}
    top_matched = _score(
        _request(
            posting_over=abilities,
            profile_over={
                "ledger": [{"tag": "Go", "tier": "core", "gloss": "Go", "vector": [1.0, 0.0]}]
            },
        )
    )
    low_matched = _score(
        _request(
            posting_over=abilities,
            profile_over={
                "ledger": [{"tag": "Rust", "tier": "core", "gloss": "Rust", "vector": [0.0, 1.0]}]
            },
        )
    )
    # Without ordinal weighting both would tie; strict-greater proves the weight.
    assert top_matched.match_score > low_matched.match_score


def test_engagement_mismatch_lowers_the_score():
    # Same posting; only the engagement preference differs. The ×factor penalty
    # must actually drop the final score, not just annotate the reasoning.
    matched = _score(_request(prefs={"preferred_engagement": "either"}))
    mismatched = _score(_request(prefs={"preferred_engagement": "contract"}))  # posting is a job
    assert mismatched.match_score < matched.match_score


def test_top_unmatched_surfaces_most_important_ability():
    # With several unmatched abilities, the flagged one is the lowest-ordinal
    # (most important), not just the first or last seen.
    abilities = [
        {"tag": "Rust", "ordinal": 5, "gloss": "Rust", "vector": [1.0, 0.0]},
        {"tag": "Cobol", "ordinal": 0, "gloss": "Cobol", "vector": [0.0, 1.0]},
    ]
    v = _score(_request(posting_over={"abilities": abilities}))
    note = next(r for r in v.risks_and_gaps if "Most-important posting ability" in r)
    assert "Cobol" in note and "Rust" not in note


def test_aligned_timezone_outscores_opposite():
    apac_role = {"role_region": "apac"}
    far = _score(_request(posting_over=apac_role, prefs={"working_country": "US"}))
    near = _score(_request(posting_over=apac_role, prefs={"working_country": "SG"}))
    assert near.match_score > far.match_score


def test_matching_seniority_outscores_distant():
    posting = {"seniority_hint": "junior"}
    exact = _score(_request(posting_over=posting, prefs={"candidate_seniority": "junior"}))
    distant = _score(_request(posting_over=posting, prefs={"candidate_seniority": "staff"}))
    assert exact.match_score > distant.match_score


# ── 6. risk surfacing ───────────────────────────────────────────────────────


def test_empty_ledger_is_flagged():
    v = _score(_request())
    assert any("ledger is empty" in r for r in v.risks_and_gaps)


def test_top_unmatched_ability_is_flagged():
    rust = {"tag": "Rust", "ordinal": 0, "gloss": "Rust", "vector": [1.0]}
    v = _score(_request(posting_over={"abilities": [rust]}))
    assert any("Most-important posting ability" in r and "Rust" in r for r in v.risks_and_gaps)


def test_engagement_mismatch_is_flagged_and_penalizes():
    req = _request(
        posting_over={"kind": "contract", "engagement_type": "hourly", "duration_hint": "long"},
        prefs={"preferred_engagement": "job"},
    )
    v = _score(req)
    assert "engagement mismatch" in v.reasoning
    assert any("Engagement mismatch" in r for r in v.risks_and_gaps)


def test_salary_below_floor_is_flagged():
    v = _score(
        _request(
            posting_over={"salary_max_annual_usd": 50_000},
            prefs={"salary_min_annual_usd": 120_000},
        )
    )
    assert any("below floor" in r for r in v.risks_and_gaps)


# ── 7. cosine similarity primitive ──────────────────────────────────────────


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),  # identical direction
        ([1.0, 0.0], [0.0, 1.0], 0.0),  # orthogonal
        ([1.0, 0.0], [-1.0, 0.0], -1.0),  # opposite
        ([1.0, 2.0, 2.0], [2.0, 4.0, 4.0], 1.0),  # scale-invariant
    ],
)
def test_cosine(a, b, expected):
    assert _cosine(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a,b",
    [
        ([1.0, 0.0], [1.0]),  # length mismatch ⇒ no match
        ([0.0, 0.0], [1.0, 0.0]),  # zero vector ⇒ no match (no NaN)
    ],
)
def test_cosine_degenerate_returns_zero(a, b):
    assert _cosine(a, b) == 0.0


# ── 8. closeness weight — the relatedness band ──────────────────────────────


def test_closeness_weight_saturates_and_floors():
    # exact_sim=0.95, related_sim_floor=0.75 in the example tuning.
    assert _closeness_weight(0.99, TUNING) == 1.0  # ≥ exact ⇒ full credit
    assert _closeness_weight(0.95, TUNING) == 1.0  # at exact boundary
    assert _closeness_weight(0.75, TUNING) == 0.0  # at floor boundary
    assert _closeness_weight(0.5, TUNING) == 0.0  # below floor ⇒ no relatedness


def test_closeness_weight_is_linear_between():
    # Midpoint of [0.75, 0.95] ⇒ 0.5.
    mid = (TUNING.related_sim_floor + TUNING.exact_sim) / 2
    assert _closeness_weight(mid, TUNING) == pytest.approx(0.5)


# ── 9. vector relatedness in the technical axis ─────────────────────────────

_VEC_ABILITY = {"tag": "Golang", "ordinal": 0, "gloss": "Golang", "vector": [1.0, 0.0]}
_VEC_LEDGER = [{"tag": "Go", "tier": "core", "gloss": "Go", "vector": [1.0, 0.0]}]


def test_related_vector_match_beats_unmatched_and_is_noted():
    # "Golang" doesn't exactly match the ledger's "Go", but an identical gloss
    # vector (cosine 1.0 ≥ exact_sim) earns full tier credit via the always-on
    # closeness path.
    related = _score(
        _request(
            posting_over={"abilities": [_VEC_ABILITY]},
            profile_over={"ledger": _VEC_LEDGER},
        )
    )
    unmatched = _score(_request(posting_over={"abilities": [_VEC_ABILITY]}))
    assert related.match_score > unmatched.match_score
    assert "related 1 (Golang)" in related.reasoning


def test_exact_match_preferred_over_weaker_related():
    # An exact string match is credited as "matched", not "related", even with a
    # vector present — the exact path takes precedence.
    v = _score(
        _request(
            posting_over={
                "abilities": [{"tag": "Go", "ordinal": 0, "gloss": "Go", "vector": [1.0, 0.0]}]
            },
            profile_over={"ledger": _VEC_LEDGER},
        )
    )
    assert "matched 1/1" in v.reasoning
    assert "related 0" in v.reasoning


def test_matched_ability_takes_the_stronger_related_credit():
    # "Go" matches a low-tier (ramping=0.4) ledger row exactly, but a second row
    # ("Golang", core=1.0, identical vector) offers higher related credit. The
    # ability stays "matched" yet earns the stronger credit (max of the two).
    ability = {"abilities": [{"tag": "Go", "ordinal": 0, "gloss": "Go", "vector": [1.0, 0.0]}]}
    exact_only = _score(
        _request(
            posting_over=ability,
            profile_over={
                "ledger": [{"tag": "Go", "tier": "ramping", "gloss": "Go", "vector": [1.0, 0.0]}]
            },
        )
    )
    with_stronger_related = _score(
        _request(
            posting_over=ability,
            profile_over={
                "ledger": [
                    {"tag": "Go", "tier": "ramping", "gloss": "Go", "vector": [1.0, 0.0]},
                    {"tag": "Golang", "tier": "core", "gloss": "Golang", "vector": [1.0, 0.0]},
                ]
            },
        )
    )
    assert with_stronger_related.match_score > exact_only.match_score
    assert "matched 1/1" in with_stronger_related.reasoning  # still classified as an exact match


def test_weak_related_below_baseline_is_unmatched():
    # cosine 0.78 ∈ (floor 0.75, exact 0.95) ⇒ weight 0.15 ⇒ credit 0.15×0.4=0.06,
    # which does not clear baseline_credit (0.1) ⇒ the ability falls to unmatched,
    # not "related".
    ability = {"tag": "Go", "ordinal": 0, "gloss": "Go", "vector": [1.0, 0.0]}
    weak = {"tag": "Ada", "tier": "ramping", "gloss": "Ada", "vector": [0.78, 0.6258]}  # cos≈0.78
    v = _score(_request(posting_over={"abilities": [ability]}, profile_over={"ledger": [weak]}))
    assert "matched 0/1" in v.reasoning
    assert "related 0" in v.reasoning
    assert "unmatched: Go" in v.reasoning
