# SPDX-License-Identifier: Apache-2.0
"""The scoring result returned to Kairos.

A `Verdict` carries the recommendation, the numeric fit, and the rationale,
plus provenance fields stamped at response time for forensics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["apply", "maybe", "skip"]


class Verdict(BaseModel):
    """The recommendation, score, rationale, and build provenance."""

    decision: Decision = Field(description="Recommended action for the candidate")
    match_score: int = Field(ge=0, le=100, description="0–100 across the four fit axes")
    reasoning: str = Field(description="One paragraph citing concrete profile evidence")
    risks_and_gaps: list[str] = Field(
        default_factory=list, description="Concerns that lowered the score"
    )
    tailoring_hints: list[str] = Field(
        default_factory=list, description="How to strengthen an application"
    )

    version: str = Field(default="", description="kairos-llm-scorer package version")
    build_sha: str = Field(default="", description="Deploy build SHA (forensics)")
    scorer: str = Field(default="", description="Arithmetic scorer behavior version")
