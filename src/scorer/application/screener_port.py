# SPDX-License-Identifier: Apache-2.0

"""The port the use case depends on for the LLM screening stage.

``ScreenerPort`` is a structural ``Protocol`` so ``application`` carries no
provider SDK dependency: the real ``AnthropicScreener`` (infrastructure) and the
test ``FakeScreener`` both satisfy it by shape alone, keeping the use case
testable without a network.
"""

from __future__ import annotations

from typing import Protocol

from ..domain.models import ScoreRequest, Verdict


class ScreenerPort(Protocol):
    """Refines a deterministic baseline verdict into a final one via an LLM."""

    async def screen(self, request: ScoreRequest, *, baseline: Verdict) -> Verdict:
        """Return a refined verdict for ``request``, given the arithmetic ``baseline``.

        Async so the LLM round-trip is awaited on the event loop rather than
        holding a worker thread — letting one process keep many escalations in
        flight (see ``application.score``).
        """
        ...
