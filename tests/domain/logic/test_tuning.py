# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for `scorer.domain.logic.tuning.ScorerTuning`.

The invariants the loader can't catch (weights sum/keys, tier-credit keys), the
derived `axis_range`, the prompt-rubric rendering, and that the model is frozen.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import scorer.config as cfg
from scorer.config import load_scorer_tuning
from scorer.domain.logic.tuning import ScorerTuning

TUNING = load_scorer_tuning(
    Path(cfg.__file__).resolve().parents[2] / "config" / "scorer.example.toml"
)


def _kwargs(**over) -> dict:
    base = dict(
        weights={"technical": 0.45, "seniority": 0.20, "domain": 0.15, "remote": 0.20},
        tier_credit={"core": 1.0, "proficient": 0.8, "ramping": 0.5},
        baseline_credit=0.05,
        no_abilities_t=0.5,
        exact_sim=0.97,
        related_sim_floor=0.78,
        domain_direct=1.0,
        domain_transferable=0.5,
        domain_mismatch=0.1,
        domain_direct_sim=0.90,
        domain_mismatch_sim=0.75,
        role_gate_sim=0.82,
        seniority_ladder={"junior": 0, "mid": 1, "senior": 2, "staff": 3},
        seniority_step=0.25,
        seniority_neutral=0.7,
        work_window_hours=14.0,
        full_shift_hours=8.0,
        remote_unknown=0.6,
        region_utc_offset={"americas": -5.0, "europe": 1.0, "apac": 8.0, "mea": 3.0},
        country_utc_offset={"US": -5.0, "DE": 1.0, "SG": 8.0},
        engagement_mismatch_factor=0.9,
        remote_to_arrangement={"yes": "remote", "hybrid": "hybrid", "no": "onsite"},
        eligibility_countries={"us-only": frozenset({"US"}), "eu-only": frozenset({"DE"})},
    )
    base.update(over)
    return base


# ── 1. validators ───────────────────────────────────────────────────────────


def test_valid_tuning_builds():
    assert ScorerTuning(**_kwargs()).weights["technical"] == 0.45


def test_weights_are_normalized_to_sum_to_one():
    raw = {"technical": 2.0, "seniority": 1.0, "domain": 1.0, "remote": 1.0}  # sums to 5.0
    tuning = ScorerTuning(**_kwargs(weights=raw))
    assert abs(sum(tuning.weights.values()) - 1.0) < 1e-9
    assert tuning.weights["technical"] == 0.4  # 2/5
    assert tuning.weights["seniority"] == 0.2  # 1/5


def test_weights_with_nonpositive_sum_rejected():
    bad = {"technical": 0.0, "seniority": 0.0, "domain": 0.0, "remote": 0.0}
    with pytest.raises(ValidationError, match="positive sum"):
        ScorerTuning(**_kwargs(weights=bad))


def test_weights_keys_must_be_the_four_axes():
    with pytest.raises(ValidationError, match="technical/seniority/domain/remote"):
        ScorerTuning(**_kwargs(weights={"technical": 0.5, "seniority": 0.3, "domain": 0.2}))


def test_tier_credit_must_define_all_tiers():
    with pytest.raises(ValidationError, match="core/proficient/ramping"):
        ScorerTuning(**_kwargs(tier_credit={"core": 1.0, "proficient": 0.8}))


def test_tuning_is_frozen():
    with pytest.raises(ValidationError):
        TUNING.seniority_step = 0.5


# ── 2. derived axis_range ────────────────────────────────────────────────────


def test_axis_range_derives_from_tuning():
    r = TUNING.axis_range
    assert r["technical"] == (TUNING.baseline_credit, 1.0)
    assert r["domain"] == (TUNING.domain_mismatch, 1.0)
    assert r["remote"] == (TUNING.remote_unknown, 1.0)
    # Seniority floor = max ladder distance (span 3) × step.
    assert r["seniority"] == pytest.approx((1.0 - TUNING.seniority_step * 3, 1.0))


# ── 3. prompt rubric rendering ───────────────────────────────────────────────


def test_axis_weight_lines_renders_canonical_rubric():
    lines = TUNING.axis_weight_lines()
    assert lines == (
        "  - Technical fit: 40%\n"
        "  - Seniority fit: 20%\n"
        "  - Domain fit: 20%\n"
        "  - Remote/timezone fit: 20%"
    )
