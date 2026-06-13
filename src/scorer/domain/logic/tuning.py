# SPDX-License-Identifier: Apache-2.0

"""Operator-supplied tuning for the deterministic scorer.

``ScorerTuning`` is the single frozen knob set that parameterizes every axis of
the deterministic score. It is loaded once from ``scorer.toml`` (see
``config.py`` and ``config/scorer.example.toml``) and threaded read-only through
the scoring logic — ``baseline`` (per-axis credit), ``vec`` (relatedness curve),
``gate`` (hard-gate thresholds), and ``validate`` (vocabulary universes). None
of it comes from the posting or the user; it is per-deployment policy, tuned by
sweeping against the eval harness.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

# Display labels for the four scoring axes, used by ``axis_weight_lines``.
_AXIS_LABELS = {
    "technical": "Technical fit",
    "seniority": "Seniority fit",
    "domain": "Domain fit",
    "remote": "Remote/timezone fit",
}


class ScorerTuning(BaseModel):
    """Frozen, validated tuning parameters for one scorer deployment.

    Field groups map to the four scoring axes plus the hard gates. Values are
    illustrative placeholders in the example config; real ones are chosen by
    sweeping against an eval harness. The loader fails loud on any missing key.
    """

    model_config = ConfigDict(frozen=True)

    # ── Axis blend ───────────────────────────────────────────────────────────
    # The four axis weights; keys must be exactly technical/seniority/domain/
    # remote (enforced by ``_check``). Raw values may be any non-negative ratio
    # — they are normalized to sum to 1.0 by ``_normalize_weights``. Applied in
    # baseline.
    weights: dict[str, float]

    # ── Technical axis ───────────────────────────────────────────────────────
    # Credit a matched ledger skill added, keyed by tier; must define
    # core/proficient/ramping (enforced by ``_check``).
    tier_credit: dict[str, float]
    # Credit for a posting ability with no ledger match; also the floor used to
    # rescale the technical axis.
    baseline_credit: float
    # Neutral technical score when the posting has zero extracted abilities.
    no_abilities_t: float

    # ── Relatedness curve (vec) ──────────────────────────────────────────────
    # Cosine >= this counts as an exact match (full tier credit).
    exact_sim: float
    # Cosine <= this is unrelated; between the floor and exact_sim, credit ramps
    # linearly. Falls back to baseline_credit below the floor.
    related_sim_floor: float

    # ── Domain axis ──────────────────────────────────────────────────────────
    # Credit bands for the domain axis: direct / transferable / mismatch.
    domain_direct: float
    domain_transferable: float
    domain_mismatch: float
    # Cosine thresholds selecting the band: >= direct_sim → direct,
    # <= mismatch_sim → mismatch, between → transferable.
    domain_direct_sim: float
    domain_mismatch_sim: float

    # ── Role-family hard gate (gate) ─────────────────────────────────────────
    # Min cosine for a posting role to pass the role-family gate; conservative,
    # since failing it hard-skips the posting.
    role_gate_sim: float

    # ── Seniority axis ───────────────────────────────────────────────────────
    # Ordinal rungs (e.g., junior=0..staff=3); ladder distance drives the axis.
    seniority_ladder: dict[str, int]
    # Credits lost per ladder step away from the candidate's level.
    seniority_step: float
    # Neutral score when either side's seniority is unstated.
    seniority_neutral: float

    # ── Remote / timezone axis ───────────────────────────────────────────────
    # Assumed working window per side, in hours (e.g., 14 for 6am–8pm local).
    work_window_hours: float
    # Timezone overlap, in hours, that counts as a full match.
    full_shift_hours: float
    # Axis floor for an unplaceable role (unknown region/country/remote).
    remote_unknown: float
    # Representative UTC offset per posting RoleRegion.
    region_utc_offset: dict[str, float]
    # Representative UTC offset per candidate working country (ISO 3166-1
    # alpha-2); unmapped countries are unplaceable, never wrongly rejected.
    country_utc_offset: dict[str, float]

    # ── Engagement & modality ────────────────────────────────────────────────
    # Soft multiplier when the posting kind misses the owner's engagement pref.
    engagement_mismatch_factor: float
    # Posting ``remote`` value → profile WorkArrangement (the modality gate).
    remote_to_arrangement: dict[str, str]

    # ── Eligibility hard gate ────────────────────────────────────────────────
    # Each posting EligibilityGate key → the set of admitted ISO-2 countries
    # (regions joined in config); consumed by ``eligibility_allows``.
    eligibility_countries: dict[str, frozenset[str]]

    @model_validator(mode="before")
    @classmethod
    def _normalize_weights(cls, data: object) -> object:
        """Rescale ``weights`` to sum to 1.0, preserving their relative ratios.

        Lets operators express weights in any units (e.g., 2/1/1/1) without
        hand-balancing to 1.0. Requires a positive total; an all-zero or
        negative sum has no meaningful normalization and raises ``ValueError``.
        Key validation is deferred to ``_check``.
        """
        if isinstance(data, dict) and isinstance(data.get("weights"), dict):
            weights: dict[str, float] = data["weights"]
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("weights must have a positive sum")
            data = {**data, "weights": {k: v / total for k, v in weights.items()}}
        return data

    @model_validator(mode="after")
    def _check(self) -> ScorerTuning:
        """Enforce the invariants the scoring math relies on.

        ``weights`` must cover exactly the four axes (their sum is normalized
        to 1.0 in ``_normalize_weights``); tier_credit must define the three
        skill tiers. Raises ``ValueError`` otherwise.
        """
        if set(self.weights) != {"technical", "seniority", "domain", "remote"}:
            raise ValueError("weights must have keys technical/seniority/domain/remote")
        if not {"core", "proficient", "ramping"} <= set(self.tier_credit):
            raise ValueError("tier_credit must define core/proficient/ramping")
        return self

    @property
    def axis_range(self) -> dict[str, tuple[float, float]]:
        """Per-axis ``(min, max)`` raw-score span, used to normalize each axis.

        The minimum is each axis's worst-case credit (baseline/floor/mismatch,
        or the widest seniority-ladder gap); the maximum is always 1.0.
        """
        span = max(self.seniority_ladder.values()) - min(self.seniority_ladder.values())
        return {
            "technical": (self.baseline_credit, 1.0),
            "seniority": (1.0 - self.seniority_step * span, 1.0),
            "remote": (self.remote_unknown, 1.0),
            "domain": (self.domain_mismatch, 1.0),
        }

    def axis_weight_lines(self) -> str:
        """Render the axis weights as human-readable percentage bullet lines."""
        return "\n".join(
            f"  - {_AXIS_LABELS[k]}: {round(v * 100)}%" for k, v in self.weights.items()
        )
