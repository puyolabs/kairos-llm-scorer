# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for `scorer.domain.logic.gate`.

The six binary factors, the load-bearing recall-safe invariant (absence never
rejects), the location-derived `eligibility` factor (the candidate's
`work_countries` vs the posting's gate), and the `GateResult` contract (product,
read-only `factors`, canonically-ordered `.zeroed`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scorer import config as cfg
from scorer.config import load_scorer_tuning
from scorer.domain.logic.gate import GateResult, gate_factors
from scorer.domain.models import GatePreferences, SemanticTag, SeniorityHint

# Tuning from the committed example (mirrors test_baseline.py); the role-family
# gate reads `role_gate_sim` (0.82) off it for the closeness threshold.
TUNING = load_scorer_tuning(
    Path(cfg.__file__).resolve().parents[2] / "config" / "scorer.example.toml"
)

# Role-family vectors: identical ⇒ cosine 1.0 (>= role_gate_sim ⇒ pass), orthogonal
# ⇒ cosine 0.0 (< threshold ⇒ reject). One axis per distinct family keeps it clean.
_ROLE_VEC = {
    "software-engineering": [1.0, 0.0],
    "design": [0.0, 1.0],
}


def _st(tag: str) -> SemanticTag:
    return SemanticTag(tag=tag, gloss=tag, vector=_ROLE_VEC[tag])


# A preference set that accepts the all-passing input below; tests override one
# axis at a time to drive a single factor to 0.
_ALLOW = dict(
    allowed_work_arrangements=["remote"],
    allowed_seniorities=["senior"],
    allowed_regions=["americas"],
    work_countries=["CA"],
    allowed_role_families=[_st("software-engineering")],
    salary_min_annual_usd=120_000,
)


def _prefs(**over) -> GatePreferences:
    return GatePreferences(**{**_ALLOW, **over})


def _call(*, prefs: GatePreferences | None = None, **over) -> GateResult:
    """Evaluate the gate over an all-passing baseline, overriding per test."""
    args = dict(
        remote="yes",
        seniority_hint="senior",
        role_region="americas",
        eligibility_gate="none",
        role_families=[_st("software-engineering")],
        salary_max_annual_usd=170_000,
        tuning=TUNING,
    )
    return gate_factors(prefs=_prefs() if prefs is None else prefs, **{**args, **over})


# ── 0. baseline + the load-bearing invariant ────────────────────────────────


def test_all_factors_pass_gate_is_one():
    res = _call()
    assert res.gate == 1
    assert set(res.factors.values()) == {1}
    assert res.zeroed == []


@pytest.mark.parametrize("absent_seniority", ["unspecified", None])
def test_absence_never_rejects(absent_seniority: SeniorityHint | None):
    # The whole rule: null / unknown / unspecified inputs pass even against the
    # most restrictive (empty-allowlist) prefs. Absence of signal never gates.
    res = gate_factors(
        remote="unknown",
        seniority_hint=absent_seniority,
        role_region=None,
        eligibility_gate=None,
        role_families=[],
        salary_max_annual_usd=None,
        tuning=TUNING,
        prefs=GatePreferences(),
    )
    assert res.gate == 1
    assert res.zeroed == []


# ── 1. modality ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("remote", ["no", "hybrid"])
def test_modality_rejects_arrangement_outside_allowlist(remote):
    res = _call(remote=remote)  # prefs allow "remote" only
    assert res.factors["modality"] == 0
    assert res.gate == 0


def test_modality_unknown_remote_passes():
    res = _call(remote="unknown", prefs=_prefs(allowed_work_arrangements=[]))
    assert res.factors["modality"] == 1


# ── 2. seniority ────────────────────────────────────────────────────────────


def test_seniority_rejects_outside_allowlist():
    res = _call(seniority_hint="junior")
    assert res.factors["seniority"] == 0


@pytest.mark.parametrize("hint", ["unspecified", None])
def test_seniority_unspecified_or_none_passes(hint):
    # Passes even though "unspecified"/None is not in the allowlist.
    res = _call(seniority_hint=hint)
    assert res.factors["seniority"] == 1


# ── 3. region ───────────────────────────────────────────────────────────────


def test_region_rejects_outside_allowlist():
    res = _call(role_region="europe")
    assert res.factors["region"] == 0


def test_region_none_passes():
    res = _call(role_region=None, prefs=_prefs(allowed_regions=[]))
    assert res.factors["region"] == 1


# ── 4. eligibility (DERIVED from the candidate's work_countries) ─────────────


def test_eligibility_rejects_when_location_outside_gate():
    # Canada-based candidate, US-only posting ⇒ not authorized ⇒ reject.
    res = _call(eligibility_gate="us-only", prefs=_prefs(work_countries=["CA"]))
    assert res.factors["eligibility"] == 0


def test_eligibility_passes_when_location_satisfies_gate():
    # 'DE' is an EU member ⇒ satisfies 'eu-only'.
    res = _call(eligibility_gate="eu-only", prefs=_prefs(work_countries=["DE"]))
    assert res.factors["eligibility"] == 1


def test_eligibility_none_gate_passes():
    res = _call(eligibility_gate=None, prefs=_prefs(work_countries=["CA"]))
    assert res.factors["eligibility"] == 1


def test_eligibility_empty_work_countries_passes():
    # No location signal ⇒ recall-safe pass even against a real gate.
    res = _call(eligibility_gate="us-only", prefs=_prefs(work_countries=[]))
    assert res.factors["eligibility"] == 1


# ── 5. role_family (recall-safe, vector closeness) ──────────────────────────


def test_role_family_rejects_when_resolved_and_none_close():
    # 'design' is orthogonal to the allowed 'software-engineering' ⇒ cosine 0 ⇒ reject.
    res = _call(role_families=[_st("design")])
    assert res.factors["role_family"] == 0


def test_role_family_empty_passes():
    # Unresolved (empty) passes even with a non-matching allowlist — recall-safe.
    res = _call(role_families=[])
    assert res.factors["role_family"] == 1


def test_role_family_empty_allowlist_rejects_resolved_role():
    # The owner allows nothing ⇒ any resolved posting role zeroes.
    res = _call(prefs=_prefs(allowed_role_families=[]), role_families=[_st("software-engineering")])
    assert res.factors["role_family"] == 0


def test_role_family_any_close_entry_passes():
    # Closeness to ANY allowed vector is enough (the SE entry clears the threshold).
    res = _call(role_families=[_st("design"), _st("software-engineering")])
    assert res.factors["role_family"] == 1


# ── 6. salary ───────────────────────────────────────────────────────────────


def test_salary_rejects_when_top_of_range_below_floor():
    res = _call(salary_max_annual_usd=100_000)  # floor is 120_000
    assert res.factors["salary"] == 0


def test_salary_at_floor_passes():
    # Gates only when strictly below — equal to the floor passes.
    res = _call(salary_max_annual_usd=120_000)
    assert res.factors["salary"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"salary_max_annual_usd": None},  # comp not parsed
        {"prefs": GatePreferences(salary_min_annual_usd=None)},  # no floor set
    ],
)
def test_salary_passes_when_either_side_absent(kwargs):
    res = _call(**kwargs)
    assert res.factors["salary"] == 1


# ── 7. GateResult contract ──────────────────────────────────────────────────


def test_gate_is_zero_iff_any_factor_zero():
    res = _call(remote="no", role_region="europe")  # two factors zeroed
    assert res.gate == 0


def test_zeroed_lists_rejecting_factors_in_canonical_order():
    res = _call(remote="no", role_region="europe", salary_max_annual_usd=1)
    assert res.zeroed == ["modality", "region", "salary"]


def test_factors_mapping_is_read_only():
    res = _call()
    with pytest.raises(TypeError):
        res.factors["modality"] = 0  # type: ignore[index]
