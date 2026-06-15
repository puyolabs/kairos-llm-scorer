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
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import ValidationError

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


@app.exception_handler(ValidationError)
def _validation_error_handler(_request: Request, exc: ValidationError) -> JSONResponse:
    """Map a pydantic ``ValidationError`` to FastAPI's default 422 shape.

    ``/score`` parses its body by hand with ``model_validate_json`` (see below) to
    skip FastAPI's slower ``json.loads`` + ``model_validate`` two-step, which means
    its built-in ``RequestValidationError`` handler no longer fires for this route.
    Reproducing the ``{"detail": [...]}`` body here keeps the wire contract identical
    for malformed JSON, missing fields, and the dimension-uniformity check.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors(include_url=False))},
    )


_screener: ScreenerPort = AnthropicScreener()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — open (no API key), returns a static ok status."""
    return {"status": "ok"}


@app.post(
    "/score",
    response_model=Verdict,
    dependencies=[Depends(require_api_key)],
    openapi_extra={
        "requestBody": {
            "content": {"application/json": {"schema": ScoreRequest.model_json_schema()}},
            "required": True,
        }
    },
)
async def score_posting(http_request: Request) -> Verdict:
    """Score one posting against the profile; guarded by ``require_api_key``.

    The body is parsed by hand with ``ScoreRequest.model_validate_json`` — pydantic
    -core's fused jiter (Rust) parse+validate — instead of FastAPI's default
    ``json.loads`` + ``model_validate``, which built throwaway Python float graphs for
    the ~22k embedding floats before validating. ``require_api_key`` stays a
    dependency so it short-circuits *before* this handler reads the ~360 KB body, and
    ``openapi_extra`` preserves the ``ScoreRequest`` schema in ``/docs``.

    Delegates to the application ``score`` use case with the configured screener.
    Async so the event loop stays free to serve other requests while an escalated
    call awaits the LLM. A ``ValidationError`` (malformed JSON, bad fields, or a
    dimension mismatch) becomes a 422 via the handler above, as does a
    ``RequestVocabularyError`` raised in ``score``; the success body is the
    provenance-stamped ``Verdict``.
    """
    request = ScoreRequest.model_validate_json(await http_request.body())
    return await score(request, screener=_screener)
