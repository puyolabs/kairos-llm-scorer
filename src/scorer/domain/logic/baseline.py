# SPDX-License-Identifier: Apache-2.0

"""The deterministic baseline scorer.

``deterministic_score`` is the arithmetic core of the system: it turns a posting
and a candidate profile into a 0–100 ``Verdict`` with no LLM call. It blends
four normalized axes (technical, seniority, domain, remote) by the configured
weights, multiplies in a soft engagement penalty and the hard gate from
``gate``, then maps the result to an apply/maybe/skip decision and a
human-readable reasoning string. The screener LLM runs downstream only for
borderline cases this stage can't settle on its own.
"""

from __future__ import annotations

import math

from ..models import (
    Decision,
    Remote,
    ScoreRequest,
    SemanticTag,
    SeniorityHint,
    Verdict,
)
from .gate import gate_factors
from .tuning import ScorerTuning
from .vec import _closeness_weight, _cosine

# Identifies scores produced by this arithmetic path (vs. an LLM model id).
SCORER_MODEL_SENTINEL = "kairos-arithmetic-v3"


def clamp01(v: float, lo: float, hi: float) -> float:
    """Rescale ``v`` from the ``[lo, hi]`` range into ``[0.0, 1.0]``, clamped.

    Returns ``0.0`` for a degenerate range (``hi <= lo``).
    """
    if hi <= lo:
        return 0.0
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


def _round_half_up(x: float) -> int:
    """Round to the nearest int with ties going up (banker's-rounding-free)."""
    return math.floor(x + 0.5)


def _normalize(s: str) -> str:
    """Casefold and trim a tag for exact-name comparison."""
    return s.strip().lower()


def _tier_fit(
    posting_hint: SeniorityHint | None, candidate: SeniorityHint | None, tuning: ScorerTuning
) -> float:
    """Score seniority fit from the distance between the two ladder rungs.

    Full credit at the same level, falling ``seniority_step`` per rung apart
    (floored at 0). Returns the neutral score if either side is unmapped.
    """
    pl: int | None = tuning.seniority_ladder.get(posting_hint or "")
    cl: int | None = tuning.seniority_ladder.get(candidate or "")
    if pl is None or cl is None:
        return tuning.seniority_neutral
    return max(0.0, 1.0 - tuning.seniority_step * abs(pl - cl))


def _timezone_score(
    region: str | None, candidate_country: str | None, tuning: ScorerTuning
) -> float:
    """Score remote/timezone fit from the candidate↔region working-hour overlap.

    ``global`` regions are a perfect fit; an unplaceable region or country
    (missing, ``unknown``, or unmapped) yields the ``remote_unknown`` floor.
    Otherwise the score is the fraction of a full shift the two UTC offsets
    share within the configured working window.
    """
    if region == "global":
        return 1.0
    if region is None or region == "unknown":
        return tuning.remote_unknown
    candidate_offset_utc: float | None = (
        tuning.country_utc_offset.get(candidate_country) if candidate_country else None
    )
    if candidate_offset_utc is None:
        return tuning.remote_unknown
    job_offset: float | None = tuning.region_utc_offset.get(region)
    if job_offset is None:
        return tuning.remote_unknown
    overlap = max(0.0, tuning.work_window_hours - abs(candidate_offset_utc - job_offset))
    return min(1.0, overlap / tuning.full_shift_hours)


def _domain_score(
    domains: list[SemanticTag],
    candidate_domains: list[SemanticTag],
    *,
    tuning: ScorerTuning,
) -> tuple[float, str]:
    """Score domain fit by the best cosine match across the two tag sets.

    Banded by similarity: ``direct`` at/above ``domain_direct_sim``, ``mismatch``
    at/below ``domain_mismatch_sim``, ``transferable`` in between. With no tags
    on either side, defaults to ``transferable`` (neutral, recall-safe).

    Returns:
        A ``(credit, band_label)`` pair; the label feeds the reasoning string.
    """
    posting_vecs = [e.vector for e in domains]
    candidate_vecs = [e.vector for e in candidate_domains]
    if not posting_vecs or not candidate_vecs:
        return tuning.domain_transferable, "transferable"
    sim = max(_cosine(p, c) for p in posting_vecs for c in candidate_vecs)
    if sim >= tuning.domain_direct_sim:
        return tuning.domain_direct, "direct"
    if sim <= tuning.domain_mismatch_sim:
        return tuning.domain_mismatch, "mismatch"
    return tuning.domain_transferable, "transferable~"


def _num(x: float) -> str:
    """Format a score compactly: drop the decimal when the value is integral."""
    return str(int(x)) if x == int(x) else str(x)


def deterministic_score(
    request: ScoreRequest,
    *,
    tuning: ScorerTuning,
    apply_threshold: int,
    maybe_threshold: int,
) -> Verdict:
    """Score a posting against a profile and return a graded ``Verdict``.

    Computes the four axes, normalizes each into ``[0, 1]`` via ``axis_range``,
    blends them by ``tuning.weights``, then applies the engagement penalty and
    the hard gate before mapping to a 0–100 score and an apply/maybe/skip
    decision. Also assembles the reasoning string and any risk/gap notes.

    Args:
        request: The posting + profile to score.
        tuning: Active scorer tuning (weights, thresholds, lookup tables).
        apply_threshold: Minimum score for an ``apply`` decision.
        maybe_threshold: Minimum score for a ``maybe`` decision (below → ``skip``).

    Returns:
        A ``Verdict`` with the decision, 0–100 score, reasoning, and risk notes.
    """
    posting = request.posting
    profile = request.profile
    prefs = profile.preferences

    seniority_hint: SeniorityHint | None = getattr(posting, "seniority_hint", None)
    remote: Remote = posting.remote
    role_region: str | None = posting.role_region
    eligibility_gate = posting.eligibility_gate
    kind = posting.kind
    salary_max = posting.salary_max_annual_usd
    salary_min_floor = prefs.salary_min_annual_usd
    preferred_engagement = prefs.preferred_engagement

    abilities = posting.abilities
    ledger = profile.ledger

    matched_names: list[str] = []
    related_names: list[str] = []
    unmatched_names: list[str] = []
    top_unmatched: tuple[int, str] | None = None

    technical = tuning.no_abilities_t
    if abilities:
        weighted_credit = 0.0
        total_weight = 0.0
        for ability in abilities:
            weight = max(1, 26 - ability.ordinal)
            name = ability.tag

            key = _normalize(name)
            exact_credit: float | None = None
            for row in ledger:
                if _normalize(row.tag) == key:
                    c = tuning.tier_credit[row.tier]
                    exact_credit = c if exact_credit is None else max(exact_credit, c)

            related_credit: float | None = None
            for row in ledger:
                w = _closeness_weight(_cosine(ability.vector, row.vector), tuning)
                if w > 0.0:
                    c = w * tuning.tier_credit[row.tier]
                    related_credit = c if related_credit is None else max(related_credit, c)

            if exact_credit is not None:
                matched_names.append(name)
                credit = exact_credit
                if related_credit is not None:
                    credit = max(exact_credit, related_credit)
            elif related_credit is not None and related_credit > tuning.baseline_credit:
                related_names.append(name)
                credit = related_credit
            else:
                unmatched_names.append(name)
                if top_unmatched is None or ability.ordinal < top_unmatched[0]:
                    top_unmatched = (ability.ordinal, name)
                credit = tuning.baseline_credit

            weighted_credit += weight * credit
            total_weight += weight
        technical = weighted_credit / total_weight if total_weight > 0 else tuning.no_abilities_t

    seniority = _tier_fit(seniority_hint, prefs.candidate_seniority, tuning)
    domain_value, domain_kind = _domain_score(
        posting.domains,
        prefs.candidate_domains,
        tuning=tuning,
    )
    remote_tz = _timezone_score(role_region, prefs.working_country, tuning)

    gate_prefs = prefs.gate
    gate_result = (
        gate_factors(
            remote=remote,
            seniority_hint=seniority_hint,
            role_region=role_region,
            eligibility_gate=eligibility_gate,
            role_families=posting.role_families,
            salary_max_annual_usd=salary_max,
            prefs=gate_prefs,
            tuning=tuning,
        )
        if gate_prefs is not None
        else None
    )
    gate = gate_result.gate if gate_result is not None else 1

    engagement_mismatch = preferred_engagement != "either" and preferred_engagement != kind
    engagement_factor = tuning.engagement_mismatch_factor if engagement_mismatch else 1.0

    axis_range = tuning.axis_range
    n_t = clamp01(technical, *axis_range["technical"])
    n_s = clamp01(seniority, *axis_range["seniority"])
    n_d = clamp01(domain_value, *axis_range["domain"])
    n_r = clamp01(remote_tz, *axis_range["remote"])

    raw = (
        tuning.weights["technical"] * n_t
        + tuning.weights["seniority"] * n_s
        + tuning.weights["domain"] * n_d
        + tuning.weights["remote"] * n_r
    )
    raw_score = _round_half_up(100 * raw * engagement_factor * gate)
    score = min(100, max(0, raw_score))

    decision: Decision = (
        "apply" if score >= apply_threshold else "maybe" if score >= maybe_threshold else "skip"
    )

    matched_note = f": {', '.join(matched_names)}" if matched_names else ""
    related_note = f", related {len(related_names)}" + (
        f" ({', '.join(related_names)})" if related_names else ""
    )
    unmatched_note = ", ".join(unmatched_names) if unmatched_names else "none"
    engagement_note = (
        f", engagement mismatch (prefers {preferred_engagement}, posting is {kind}) "
        f"×{tuning.engagement_mismatch_factor}"
        if engagement_mismatch
        else ""
    )
    zeroed = gate_result.zeroed if gate_result is not None and gate_result.gate == 0 else []
    gated_note = f"gated: {', '.join(zeroed)}; " if zeroed else ""
    domain_str = "|".join(e.tag for e in posting.domains) if posting.domains else "none"
    reasoning = (
        f"Deterministic score {score} (v3, full-range): {gated_note}"
        f"technical {technical:.2f}→n{n_t:.2f} "
        f"(matched {len(matched_names)}/{len(abilities)} posting abilities{matched_note}"
        f"{related_note}; unmatched: {unmatched_note}), "
        f"seniority {seniority_hint or 'unspecified'}→{_num(seniority)}→n{n_s:.2f}, "
        f"domain {domain_str}→{domain_kind}→{_num(domain_value)}→n{n_d:.2f}, "
        f"remote {remote}/{role_region or 'unknown'}→{_num(remote_tz)}→n{n_r:.2f}"
        f"{engagement_note}."
    )

    risks_and_gaps: list[str] = []
    if top_unmatched is not None:
        risks_and_gaps.append(f'Most-important posting ability not in ledger: "{top_unmatched[1]}"')
    if engagement_mismatch:
        risks_and_gaps.append(
            f"Engagement mismatch: owner prefers {preferred_engagement}, posting is a {kind}."
        )
    if salary_max is not None and salary_min_floor is not None and salary_max < salary_min_floor:
        risks_and_gaps.append(
            f"Top of salary (USD {salary_max}) below floor (USD {salary_min_floor})."
        )
    if not ledger:
        risks_and_gaps.append("Candidate skill ledger is empty — technical fit not assessed.")

    return Verdict(
        decision=decision,
        match_score=score,
        reasoning=reasoning,
        risks_and_gaps=risks_and_gaps,
    )
