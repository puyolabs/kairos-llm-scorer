# SPDX-License-Identifier: Apache-2.0
"""The job/contract posting being scored.

`Posting` is a discriminated union on `kind`: a `Job` or a `Contract`.
Both share `PostingBase`; each adds the fields specific to its engagement.
`extra="ignore"` lets Kairos send richer rows (ids, status, …) without
breaking the contract — unknown columns are silently dropped.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .shared import CountryISO2, SemanticTag, SeniorityHint

Remote = Literal["yes", "hybrid", "no", "unknown"]
EngagementType = Literal["hourly", "fixed"]
DurationHint = Literal["short", "medium", "long", "open", "unknown"]
PostingKind = Literal["job", "contract"]


class PostingAbility(SemanticTag):
    """A required skill on a posting, ranked by its significance."""

    ordinal: int = Field(description="Listing order; 0 is most prominent")


class PostingBase(BaseModel):
    """Fields common to every posting, regardless of engagement kind."""

    model_config = ConfigDict(extra="ignore")

    source_id: str = Field(description="Board/source the posting came from")
    external_id: str = Field(description="Posting id within that source")
    canonical_key: str = Field(description="Stable cross-source key, `source_id::external_id`")
    url: str
    title: str
    company: str
    description: str
    posted_at: str = Field(description="ISO-8601 timestamp the source published it")
    fetched_at: str = Field(description="ISO-8601 timestamp Kairos ingested it")
    location_text: str
    remote: Remote
    role_region: str | None = Field(default=None, description="Region gate is matched against")
    eligibility_gate: str | None = Field(
        default=None, description="Work-authorization requirement, if stated"
    )
    country_iso2: CountryISO2 | None = None
    salary_min_annual_usd: int | None = None
    salary_max_annual_usd: int | None = None
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    salary_period: str | None = None
    abilities: list[PostingAbility] = Field(
        default_factory=list, description="Required skills, embedded for matching"
    )
    domains: list[SemanticTag] = Field(
        default_factory=list, description="Industry/domain tags, embedded for matching"
    )
    role_families: list[SemanticTag] = Field(
        default_factory=list, description="Role-family tags, embedded for matching"
    )


class Job(PostingBase):
    """A permanent/salaried posting."""

    kind: Literal["job"] = "job"
    seniority_hint: SeniorityHint
    salary_range_text: str | None = None


class Contract(PostingBase):
    """A fixed-term/freelance posting with rate and duration."""

    kind: Literal["contract"] = "contract"
    engagement_type: EngagementType
    rate_text: str | None = None
    duration_hint: DurationHint
    hours_per_week_hint: int | None = Field(default=None, gt=0)


#: A posting, discriminated on `kind` into `Job` or `Contract`.
Posting = Annotated[Job | Contract, Field(discriminator="kind")]
