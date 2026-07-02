# SPDX-License-Identifier: Apache-2.0
"""Domain models for the scoring contract.

The request side (`ScoreRequest` = `Posting` + `Profile`) and the response
side (`Verdict`), built from shared embedded primitives (`SemanticTag`).
"""

from __future__ import annotations

from .posting import (
    Contract,
    Job,
    Posting,
    Remote,
)
from .profile import (
    GatePreferences,
    LedgerEntry,
    Preferences,
    Profile,
    WorkArrangement,
)
from .request import ScoreRequest, SonnetJudgement
from .shared import SemanticTag, SeniorityHint
from .verdict import Baseline, Decision, Method, ScoreResult, Verdict

__all__ = [
    "SemanticTag",
    "SeniorityHint",
    "Posting",
    "Job",
    "Contract",
    "Remote",
    "Profile",
    "LedgerEntry",
    "Preferences",
    "GatePreferences",
    "WorkArrangement",
    "ScoreRequest",
    "SonnetJudgement",
    "Verdict",
    "ScoreResult",
    "Baseline",
    "Decision",
    "Method",
]
