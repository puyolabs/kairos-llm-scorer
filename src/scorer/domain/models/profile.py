# SPDX-License-Identifier: Apache-2.0
"""The candidate profile a posting is scored against.

A `Profile` is the free-text `body`, a `ledger` of skills with proficiency
tiers, and `Preferences`. Preferences split into soft signals (used to
nudge the score) and a hard `gate` (used to disqualify outright).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .shared import CountryISO2, SemanticTag, SeniorityHint

SkillTier = Literal["core", "proficient", "ramping"]
PreferredEngagement = Literal["job", "contract", "either"]
WorkArrangement = Literal["remote", "hybrid", "onsite"]


class LedgerEntry(SemanticTag):
    """A candidate skill, embedded for matching, with its proficiency tier."""

    tier: SkillTier


class GatePreferences(BaseModel):
    """Hard disqualifiers. An empty allowlist means "allow nothing".

    Distinct from `Preferences.gate is None`, which skips gating entirely:
    a present-but-empty gate would zero out every signal it covers.
    """

    model_config = ConfigDict(extra="ignore")

    allowed_work_arrangements: list[WorkArrangement] = Field(default_factory=list)
    allowed_seniorities: list[SeniorityHint] = Field(default_factory=list)
    allowed_regions: list[str] = Field(default_factory=list)
    work_countries: list[CountryISO2] = Field(default_factory=list)
    allowed_role_families: list[SemanticTag] = Field(default_factory=list)
    salary_min_annual_usd: int | None = None


class Preferences(BaseModel):
    """Soft scoring signals plus the optional hard `gate`."""

    model_config = ConfigDict(extra="ignore")

    salary_min_annual_usd: int | None = None
    rate_min_hourly_usd: int | None = None
    preferred_engagement: PreferredEngagement = "either"
    excluded_industries: list[str] = Field(default_factory=list)
    candidate_domains: list[SemanticTag] = Field(default_factory=list)
    candidate_seniority: SeniorityHint | None = None
    working_country: CountryISO2 | None = None
    gate: GatePreferences | None = Field(
        default=None, description="Hard disqualifiers; None skips gating entirely"
    )


class Profile(BaseModel):
    """The candidate: narrative body, skill ledger, and preferences."""

    body: str = Field(min_length=1, max_length=200_000)
    ledger: list[LedgerEntry] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)
