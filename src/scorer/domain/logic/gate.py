# SPDX-License-Identifier: Apache-2.0

"""Hard eligibility gates for the deterministic scorer.

Where the four axes produce a graded fit, the gate is binary: it answers "is
this posting even worth surfacing?" ``gate_factors`` evaluates six independent
hard constraints (modality, seniority, region, eligibility, role-family,
salary); any singular rejection zeroes the overall gate, which ``baseline`` multiplies
into the final score to force it to 0. Gates fire only on the candidate's
explicit preferences (``GatePreferences``) — absent a preference, a factor
passes, keeping the gate recall-safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from ..models import (
    GatePreferences,
    Remote,
    SemanticTag,
    SeniorityHint,
    WorkArrangement,
)
from .eligibility import eligibility_allows
from .tuning import ScorerTuning
from .vec import cosine_matrix


def work_arrangement_from_remote(
    remote: Remote, mapping: Mapping[str, str]
) -> WorkArrangement | None:
    """Translate a posting ``remote`` value to a profile ``WorkArrangement``.

    Args:
        remote: The posting's remote flag (``yes``/``hybrid``/``no``/``unknown``).
        mapping: ``tuning.remote_to_arrangement``; deliberately omits ``unknown``
            so an unplaceable posting maps to ``None`` and passes the gate.

    Returns:
        The corresponding ``WorkArrangement``, or ``None`` when unmapped. The
        config only ever holds valid arrangement strings, so the lookup result
        is cast back to the ``WorkArrangement`` literal.
    """
    return cast("WorkArrangement | None", mapping.get(remote))


@dataclass(frozen=True)
class GateResult:
    """Outcome of the hard-gate evaluation.

    Attributes:
        factors: Per-constraint pass/fail map (1 = pass, 0 = fail), keyed by
            factor name (modality/seniority/region/eligibility/role_family/salary).
        gate: The combined gate, 1 only if every factor passed, else 0.
    """

    factors: Mapping[str, int]
    gate: int

    @property
    def zeroed(self) -> list[str]:
        """Names of the factors that failed, for the score's reasoning string."""
        return [name for name, value in self.factors.items() if value == 0]


def gate_factors(
    *,
    remote: Remote,
    seniority_hint: SeniorityHint | None,
    role_region: str | None,
    eligibility_gate: str | None,
    role_families: list[SemanticTag],
    salary_max_annual_usd: int | None,
    prefs: GatePreferences,
    tuning: ScorerTuning,
) -> GateResult:
    """Evaluate the six hard constraints into a combined pass/fail gate.

    Each factor passes by default and fails only on an explicit conflict with
    the candidate's preferences, so missing data never hard-skips a posting.

    Args:
        remote: Marking a non-local work status, tied to a work configuration for a type.
        seniority_hint: Posting seniority; ``None``/``unspecified`` always passes.
        role_region: Posting region checked against ``prefs.allowed_regions``.
        eligibility_gate: Posting eligibility key, resolved via ``eligibility_allows``.
        role_families: Posting role-family tags; gated by cosine vs. allowed
            families against ``tuning.role_gate_sim``.
        salary_max_annual_usd: Posting salary ceiling vs. ``prefs.salary_min_*``.
        prefs: The candidate's gate preferences driving every factor.
        tuning: Supplies thresholds and the remote→arrangement / eligibility maps.

    Returns:
        A ``GateResult`` with the per-factor breakdown and the combined gate.
    """
    arrangement = work_arrangement_from_remote(remote, tuning.remote_to_arrangement)
    f_modality = (
        0 if arrangement is not None and arrangement not in prefs.allowed_work_arrangements else 1
    )

    f_seniority = (
        0
        if seniority_hint is not None
        and seniority_hint != "unspecified"
        and seniority_hint not in prefs.allowed_seniorities
        else 1
    )

    f_region = 0 if role_region is not None and role_region not in prefs.allowed_regions else 1

    f_eligibility = (
        1
        if eligibility_allows(eligibility_gate, prefs.work_countries, tuning.eligibility_countries)
        else 0
    )

    posting_role_vecs = [e.vector for e in role_families]
    allowed_role_vecs = [e.vector for e in prefs.allowed_role_families]
    if not role_families:
        f_role_family = 1
    elif not allowed_role_vecs:
        f_role_family = 0
    else:
        sim = float(cosine_matrix(posting_role_vecs, allowed_role_vecs).max())
        f_role_family = 1 if sim >= tuning.role_gate_sim else 0

    f_salary = (
        0
        if prefs.salary_min_annual_usd is not None
        and salary_max_annual_usd is not None
        and salary_max_annual_usd < prefs.salary_min_annual_usd
        else 1
    )

    factors = MappingProxyType(
        {
            "modality": f_modality,
            "seniority": f_seniority,
            "region": f_region,
            "eligibility": f_eligibility,
            "role_family": f_role_family,
            "salary": f_salary,
        }
    )
    gate = 1 if all(v == 1 for v in factors.values()) else 0
    return GateResult(factors=factors, gate=gate)
