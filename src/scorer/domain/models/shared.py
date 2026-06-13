# SPDX-License-Identifier: Apache-2.0
"""Primitives shared across postings, profiles, and gate preferences.

`SemanticTag` is the common shape for every embedded concept (skills,
domains, role families): both sides of a match carry pre-resolved glosses
and vectors, so scoring is pure arithmetic with no embedding calls.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

SeniorityHint = Literal["junior", "mid", "senior", "staff", "unspecified"]

#: Uppercase ISO 3166-1 alpha-2 country code (e.g. "US", "CA").
CountryISO2 = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]


class SemanticTag(BaseModel):
    """An embedded concept: human-readable label plus its vector."""

    tag: str = Field(description="Canonical identifier for the concept")
    gloss: str = Field(description="Text that was embedded to produce `vector`")
    vector: list[float] = Field(description="Pre-computed embedding of `gloss`")
