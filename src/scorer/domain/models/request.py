# SPDX-License-Identifier: Apache-2.0
"""The scoring request: one posting paired with one profile."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .posting import Posting
from .profile import Profile
from .shared import SemanticTag

#: How freely the LLM pass may override the arithmetic baseline.
#: "wide" lets it move the score broadly, "narrow" only nudges, "off" disables it.
SonnetJudgement = Literal["wide", "narrow", "off"]


def _iter_vectors(obj: object) -> Iterator[list[float]]:
    """Yield every embedding vector reachable from a model tree.

    Recurses through nested models and lists, surfacing each ``SemanticTag``'s
    vector (``LedgerEntry`` included, since it subclasses ``SemanticTag``).
    """
    if isinstance(obj, SemanticTag):
        yield obj.vector
    elif isinstance(obj, BaseModel):
        for field_name in type(obj).model_fields:
            yield from _iter_vectors(getattr(obj, field_name))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_vectors(item)


class ScoreRequest(BaseModel):
    """A posting↔profile pair plus the LLM-override policy to apply."""

    posting: Posting
    profile: Profile
    sonnet_judgement: SonnetJudgement = Field(
        default="wide", description="LLM-override breadth; defaults to wide when omitted"
    )

    @model_validator(mode="after")
    def _embeddings_share_one_dimension(self) -> ScoreRequest:
        """Require every embedding in the request to share one positive dimension.

        All vectors come from a single embedding model, so a dimension mismatch
        signals a malformed request, not a legitimate input. Failing here (→ HTTP
        422) keeps the vectorized scoring math rectangular and surfaces the
        producer bug rather than silently scoring the mismatch as unrelated.
        """
        dims = {len(v) for v in _iter_vectors(self)}
        if len(dims) > 1:
            raise ValueError(
                f"all embedding vectors must share one dimension; found {sorted(dims)}"
            )
        return self
