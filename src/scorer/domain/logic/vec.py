# SPDX-License-Identifier: Apache-2.0

"""Vectorized embedding-match primitives.

Two numpy helpers shared by the scoring logic: ``cosine_matrix`` computes the
all-pairs cosine similarity between two sets of embeddings in a single matmul
(used by ``baseline`` for ability/domain matches and by ``gate`` for the
role-family gate), and ``closeness_weights`` maps raw similarities onto the
tuning-defined relatedness curve (used by ``baseline`` to scale technical credit).

Callers pass uniform-dimension embeddings — ``ScoreRequest`` validates that
invariant at the request boundary (→ HTTP 422), so these helpers assume
rectangular input. A zero-magnitude row is treated as unrelated (similarity
``0.0``) rather than producing a NaN.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .tuning import ScorerTuning


def cosine_matrix(rows: list[list[float]], cols: list[list[float]]) -> NDArray[np.float64]:
    """All-pairs cosine similarity between two sets of equal-dimension vectors.

    Args:
        rows: ``M`` vectors, each of dimension ``D``.
        cols: ``N`` vectors, each of dimension ``D``.

    Returns:
        An ``M×N`` array where entry ``(i, j)`` is the cosine similarity of
        ``rows[i]`` and ``cols[j]``. A zero-magnitude vector yields ``0.0`` over
        its row/column (treated as unrelated, never NaN). Returns an ``M×N`` zero
        array when either side is empty.
    """
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    if r.size == 0 or c.size == 0:
        return np.zeros((len(rows), len(cols)), dtype=np.float64)
    denom = np.linalg.norm(r, axis=1, keepdims=True) @ np.linalg.norm(c, axis=1, keepdims=True).T
    # where=denom!=0 leaves zero-norm pairs at 0.0 instead of dividing by zero.
    return np.divide(r @ c.T, denom, out=np.zeros_like(denom), where=denom != 0.0)


def closeness_weights(sims: NDArray[np.float64], tuning: ScorerTuning) -> NDArray[np.float64]:
    """Map cosine similarities onto the configured relatedness curve, elementwise.

    A piecewise-linear ramp: full credit (``1.0``) at/above ``exact_sim``, none
    (``0.0``) at/below ``related_sim_floor``, linear interpolation between.

    Args:
        sims: Array of cosine similarities (any shape).
        tuning: Supplies the ``exact_sim`` ceiling and ``related_sim_floor``.

    Returns:
        An array of weights in ``[0.0, 1.0]``, same shape as ``sims``.
    """
    span = tuning.exact_sim - tuning.related_sim_floor
    return np.clip((sims - tuning.related_sim_floor) / span, 0.0, 1.0)
