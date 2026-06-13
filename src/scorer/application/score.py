# SPDX-License-Identifier: Apache-2.0

"""The two-stage scoring use case — arithmetic first, LLM screener on escalation.

``score`` is the application entry point behind ``POST /score``: it validates the
request vocabulary, runs the deterministic ``deterministic_score``, and escalates
to the injected ``ScreenerPort`` only when the request's ``sonnet_judgement`` mode
and the baseline band call for it (``_should_escalate``). Provenance
(version / build_sha / scorer) is stamped here and single-sourced, so neither the
arithmetic nor the LLM stage can set it.
"""

from __future__ import annotations

from .. import __version__
from ..config import get_scorer_tuning, get_settings
from ..domain.logic import (
    SCORER_MODEL_SENTINEL,
    deterministic_score,
    validate_request_vocabulary,
)
from ..domain.models import ScoreRequest, SonnetJudgement, Verdict
from .screener_port import ScreenerPort


def _should_escalate(baseline: Verdict, judgement: SonnetJudgement, *, escalate_floor: int) -> bool:
    """Decide whether the baseline verdict should be sent to the LLM screener.

    The request's ``sonnet_judgement`` mode sets how eager escalation is:

    - ``off``: never escalate — the arithmetic verdict is final.
    - ``narrow``: only an ``apply`` baseline (the tailoring pass).
    - ``wide``: any ``maybe``/``apply``, plus a ``skip`` whose score still reaches
      ``escalate_floor`` — the near-miss band the model most often overturns. A
      hard-gated ``skip`` (score below the floor) never escalates.
    """
    if judgement == "off":
        return False
    if judgement == "narrow":
        return baseline.decision == "apply"
    return baseline.decision in ("maybe", "apply") or baseline.match_score >= escalate_floor


def score(request: ScoreRequest, *, screener: ScreenerPort) -> Verdict:
    """Score a request end to end and return the provenance-stamped verdict.

    Validates the request vocabulary (raising on unconfigured values), computes
    the deterministic baseline, then escalates to ``screener`` iff
    ``_should_escalate`` says so. The surfaced verdict — baseline or screener — is
    copied with the canonical provenance fields stamped, so those are
    single-sourced here rather than trusted from either stage.

    Args:
        request: The posting + profile to score, carrying its ``sonnet_judgement``.
        screener: The LLM port invoked when the baseline escalates.

    Returns:
        The final ``Verdict`` with version / build_sha / scorer stamped.

    Raises:
        RequestVocabularyError: If the request carries unconfigured vocabulary.
    """
    settings = get_settings()
    tuning = get_scorer_tuning()
    validate_request_vocabulary(request, tuning)
    baseline = deterministic_score(
        request,
        tuning=tuning,
        apply_threshold=settings.apply_threshold,
        maybe_threshold=settings.maybe_threshold,
    )
    verdict = (
        baseline
        if not _should_escalate(
            baseline, request.sonnet_judgement, escalate_floor=settings.escalate_floor
        )
        else screener.screen(request, baseline=baseline)
    )
    return verdict.model_copy(
        update={
            "version": __version__,
            "build_sha": settings.build_sha,
            "scorer": SCORER_MODEL_SENTINEL,
        }
    )
