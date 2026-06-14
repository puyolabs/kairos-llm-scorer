# SPDX-License-Identifier: Apache-2.0

"""The HTTP surface: the FastAPI app, its routes, and error mapping.

Two routes — an open ``GET /health`` liveness probe and the authenticated
``POST /score`` that runs the two-stage scorer. Cross-cutting concerns live here
as ASGI wiring: ``require_api_key`` guards ``/score``, and the domain's
``RequestVocabularyError`` is mapped to a 422 carrying a structured per-field
violation body. The screener is a module-level singleton (built once at import);
tests swap it by patching ``_screener``.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .application.score import score
from .application.screener_port import ScreenerPort
from .auth import require_api_key
from .domain.logic import RequestVocabularyError
from .domain.models import ScoreRequest, Verdict
from .infrastructure import AnthropicScreener

app = FastAPI(title="kairos-llm-scorer", version=__version__)


@app.exception_handler(RequestVocabularyError)
def _vocabulary_error_handler(_request: Request, exc: RequestVocabularyError) -> JSONResponse:
    """Map a domain ``RequestVocabularyError`` to a 422 with its violation list.

    Surfaces every ``VocabViolation`` as ``{field, value, allowed}`` so the caller
    sees all unconfigured values at once, not just the first one rejected.
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": str(exc),
            "violations": [
                {"field": v.field, "value": v.value, "allowed": v.allowed} for v in exc.violations
            ],
        },
    )


_screener: ScreenerPort = AnthropicScreener()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — open (no API key), returns a static ok status."""
    return {"status": "ok"}


@app.post("/score", response_model=Verdict, dependencies=[Depends(require_api_key)])
def score_posting(request: ScoreRequest) -> Verdict:
    """Score one posting against the profile; guarded by ``require_api_key``.

    Delegates to the application ``score`` use case with the configured screener.
    A ``RequestVocabularyError`` it raises becomes a 422 via the handler above;
    the success body is the provenance-stamped ``Verdict``.
    """
    return score(request, screener=_screener)
