# SPDX-License-Identifier: Apache-2.0

"""Pre-scoring vocabulary validation.

Guards the scorer against requests whose enum-like string fields (remote, role
region, eligibility gate, working country) are not part of the active tuning
config — values the scoring math could otherwise silently treat as unplaceable.
``validate_request_vocabulary`` runs first in the scoring flow; on any mismatch
it raises ``RequestVocabularyError``, which ``api.py`` maps to an HTTP 422 with
the per-field violation list.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import ScoreRequest
from .tuning import ScorerTuning

# Sentinel values accepted for a field regardless of the configured vocabulary;
# they encode "unspecified / no constraint" and pass scoring on their own path.
_REMOTE_PASS = {"unknown"}
_REGION_PASS = {"global", "unknown"}
_GATE_PASS = {"none", "unknown"}


@dataclass(frozen=True)
class VocabViolation:
    """A single request field whose value is not in the configured vocabulary.

    Attributes:
        field: Dotted path of the offending field (e.g. ``posting.remote``).
        value: The unrecognized value supplied.
        allowed: Sorted list of values the config accepts for this field.
    """

    field: str
    value: str
    allowed: list[str]


class RequestVocabularyError(ValueError):
    """Raised when a request carries one or more unconfigured vocabulary values.

    Aggregates every ``VocabViolation`` from a single validation pass so the
    caller sees all problems at once. ``api.py`` surfaces ``violations`` as the
    structured body of a 422 response.
    """

    def __init__(self, violations: list[VocabViolation]) -> None:
        self.violations = violations
        summary = "; ".join(
            f"{v.field}={v.value!r} is not a configured value — allowed: {', '.join(v.allowed)}"
            for v in violations
        )
        super().__init__(summary)


def validate_request_vocabulary(request: ScoreRequest, tuning: ScorerTuning) -> None:
    """Validate a request's vocabulary fields against the tuning config.

    Checks each enum-like field on the posting and the profile preferences
    against the universe of values the ``tuning`` config knows about, allowing
    ``None`` and the per-field passthrough sentinels. Collects all violations
    before failing so the caller gets a complete picture.

    Args:
        request: The scoring request whose posting and preferences are checked.
        tuning: Active config supplying each field's accepted-value universe.

    Raises:
        RequestVocabularyError: If any checked field holds an unconfigured value.
    """
    posting = request.posting
    prefs = request.profile.preferences
    violations: list[VocabViolation] = []

    def check(field: str, value: str | None, allowed: set[str], passthrough: set[str]) -> None:
        """Record a violation unless ``value`` is None, a sentinel, or allowed."""
        if value is None or value in passthrough or value in allowed:
            return
        violations.append(VocabViolation(field, value, sorted(allowed)))

    check("posting.remote", posting.remote, set(tuning.remote_to_arrangement), _REMOTE_PASS)
    check("posting.role_region", posting.role_region, set(tuning.region_utc_offset), _REGION_PASS)
    check(
        "posting.eligibility_gate",
        posting.eligibility_gate,
        set(tuning.eligibility_countries),
        _GATE_PASS,
    )

    check(
        "profile.preferences.working_country",
        prefs.working_country,
        set(tuning.country_utc_offset),
        set(),
    )
    if prefs.gate is not None:
        region_universe = set(tuning.region_utc_offset)
        for region in prefs.gate.allowed_regions:
            check("profile.preferences.gate.allowed_regions", region, region_universe, _REGION_PASS)
        universe: set[str] = set().union(*tuning.eligibility_countries.values())
        for country in prefs.gate.work_countries:
            check("profile.preferences.gate.work_countries", country, universe, set())

    if violations:
        raise RequestVocabularyError(violations)
