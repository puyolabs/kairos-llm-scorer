# SPDX-License-Identifier: Apache-2.0
"""Test doubles for the scorer's ports — no network, no provider SDKs.

`FakeScreener` implements `ScreenerPort` so use-case and API tests can exercise
the escalation path without hitting Anthropic. It records every call and returns
a verdict you control (a fixed one, or one computed per request).
"""

from __future__ import annotations

from collections.abc import Callable

from scorer.domain.models import ScoreRequest, Verdict


class FakeScreener:
    """A `ScreenerPort` that returns a canned verdict and records its calls.

    Pass `verdict` for a fixed result, or `responder` for a per-call function
    `(request, baseline) -> Verdict`. With neither, it echoes the baseline (the
    "model agreed with the arithmetic" case).
    """

    def __init__(
        self,
        verdict: Verdict | None = None,
        *,
        responder: Callable[[ScoreRequest, Verdict], Verdict] | None = None,
    ) -> None:
        if verdict is not None and responder is not None:
            raise ValueError("pass verdict or responder, not both")
        self._verdict = verdict
        self._responder = responder
        self.calls: list[tuple[ScoreRequest, Verdict]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def screen(self, request: ScoreRequest, *, baseline: Verdict) -> Verdict:
        self.calls.append((request, baseline))
        if self._responder is not None:
            return self._responder(request, baseline)
        if self._verdict is not None:
            return self._verdict
        return baseline
