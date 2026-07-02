# SPDX-License-Identifier: Apache-2.0
"""The scoring result returned to Kairos.

A `Verdict` carries the recommendation, the numeric fit, and the rationale,
plus provenance fields stamped at response time for forensics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["apply", "maybe", "skip"]

# Which stage produced the surfaced verdict: the deterministic arithmetic scorer,
# or the LLM screener that overturned it. Stamped in `application.score`.
Method = Literal["deterministic", "llm"]


class Verdict(BaseModel):
    """The recommendation, score, rationale, and build provenance.

    This is the *core* verdict both stages produce — the arithmetic baseline and
    the LLM screener (it is the screener's structured-output schema, so it must
    stay free of wire-only fields). The wire response is `ScoreResult`, which adds
    `method`/`baseline` provenance on top.
    """

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


class Baseline(BaseModel):
    """The deterministic pre-LLM score, retained when the screener overturned it.

    A flat snapshot of the arithmetic verdict (no provenance) so Kairos can show
    both grades side by side whenever the LLM took the lead. Present only when
    `ScoreResult.method == "llm"`.
    """

    decision: Decision = Field(description="The arithmetic recommendation")
    match_score: int = Field(ge=0, le=100, description="The arithmetic 0–100 fit score")
    reasoning: str = Field(default="", description="The arithmetic rationale")
    risks_and_gaps: list[str] = Field(default_factory=list)
    tailoring_hints: list[str] = Field(default_factory=list)


class ScoreResult(Verdict):
    """The `POST /score` wire response: the surfaced verdict plus its provenance.

    `method` says which stage produced the surfaced fields; `baseline` carries the
    deterministic pre-LLM score, present only when the LLM screener took the lead
    (`method == "llm"`) — so a caller can label deterministic vs. LLM grades and,
    when the LLM led, see both. Kept distinct from `Verdict` so the screener's
    structured-output schema (`output_format=Verdict`) never sees these fields and
    the model is never asked to invent them.
    """

    method: Method = Field(description="Which stage produced the surfaced verdict")
    baseline: Baseline | None = Field(
        default=None,
        description="Deterministic pre-LLM score; present only when method == 'llm'",
    )
