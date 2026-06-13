# SPDX-License-Identifier: Apache-2.0
"""The scoring request: one posting paired with one profile."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .posting import Posting
from .profile import Profile

#: How freely the LLM pass may override the arithmetic baseline.
#: "wide" lets it move the score broadly, "narrow" only nudges, "off" disables it.
SonnetJudgement = Literal["wide", "narrow", "off"]


class ScoreRequest(BaseModel):
    """A posting↔profile pair plus the LLM-override policy to apply."""

    posting: Posting
    profile: Profile
    sonnet_judgement: SonnetJudgement = Field(
        default="wide", description="LLM-override breadth; defaults to wide when omitted"
    )
