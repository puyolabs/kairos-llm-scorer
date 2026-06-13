# SPDX-License-Identifier: Apache-2.0

"""Vector math primitives for embedding-based matching.

Two pure helpers shared by the scoring logic: ``_cosine`` measures embedding
similarity (used by ``baseline`` for ability/domain matches and by ``gate`` for
the role-family gate), and ``_closeness_weight`` maps a raw similarity onto the
tuning-defined relatedness curve (used by ``baseline`` to scale technical
credit).
"""

from __future__ import annotations

import math

from .tuning import ScorerTuning


def _cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors.

    Defensive rather than strict: returns ``0.0`` (treat as unrelated) when the
    vectors differ in length or either has zero magnitude, instead of raising.

    Args:
        a: First vector.
        b: Second vector, expected to share ``a``'s dimensionality.

    Returns:
        Cosine similarity in ``[-1.0, 1.0]``, or ``0.0`` on a length mismatch or
        a zero-norm input.
    """
    if len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _closeness_weight(sim: float, tuning: ScorerTuning) -> float:
    """Map a cosine similarity onto the configured relatedness curve.

    A piecewise-linear ramp: full credit at/above ``exact_sim``, no credit
    at/below ``related_sim_floor``, and a linear interpolation between the two.

    Args:
        sim: Cosine similarity to weight, typically from ``_cosine``.
        tuning: Supplies the ``exact_sim`` ceiling and ``related_sim_floor``.

    Returns:
        A weight in ``[0.0, 1.0]`` scaling how much credit the match earns.
    """
    if sim >= tuning.exact_sim:
        return 1.0
    if sim <= tuning.related_sim_floor:
        return 0.0
    return (sim - tuning.related_sim_floor) / (tuning.exact_sim - tuning.related_sim_floor)
