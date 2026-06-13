# SPDX-License-Identifier: Apache-2.0

"""The Anthropic-backed screener — the LLM half of the two-stage scorer.

``AnthropicScreener`` implements ``ScreenerPort``: given a request and the
deterministic baseline, it calls the configured model with the screener prompt
and parses a refined ``Verdict`` via structured output. It runs only for the
borderline cases the arithmetic stage escalates (see ``application.score``); the
default ``api`` wiring constructs one, and tests substitute ``FakeScreener``.

``_supports_effort`` probes whether the model advertises the requested reasoning
``effort`` so the optional ``output_config`` is sent only when honored; the probe
is best-effort and never allowed to break the screen call.
"""

from __future__ import annotations

from anthropic import Anthropic, AnthropicError
from anthropic.types import MessageParam, OutputConfigParam

from ....config import get_scorer_tuning, get_settings
from ....domain.models import ScoreRequest, Verdict
from ..prompt import ScreenerPromptAdapter

# Per-model memo of effort-capability probes, so the models.retrieve round-trip
# happens at most once per model per process.
_EFFORT_SUPPORT: dict[str, bool] = {}


def _supports_effort(client: Anthropic, model: str, effort: str) -> bool:
    """Best-effort probe: does ``model`` advertise the requested ``effort`` level?

    Caches the answer per model. Only the network lookup can fail (caught and
    treated as "unsupported" so the screen call proceeds without
    ``output_config``); the capability shape is parsed defensively with explicit
    ``isinstance`` guards rather than a broad ``try``. A transient lookup failure
    returns ``False`` *without* caching, so a later call can re-probe.

    Returns:
        ``True`` only if the model reports effort support and, when it breaks
        support down per level, the specific ``effort`` is supported.
    """
    cached = _EFFORT_SUPPORT.get(model)
    if cached is not None:
        return cached
    try:
        capabilities = client.models.retrieve(model).capabilities
    except AnthropicError:
        return False
    caps = capabilities.model_dump() if hasattr(capabilities, "model_dump") else capabilities
    caps = caps if isinstance(caps, dict) else {}
    effort_caps = caps.get("effort")
    effort_caps = effort_caps if isinstance(effort_caps, dict) else {}
    supported = bool(effort_caps.get("supported", False))
    if supported:
        level = effort_caps.get(effort)
        if isinstance(level, dict) and "supported" in level:
            supported = bool(level["supported"])
    _EFFORT_SUPPORT[model] = supported
    return supported


class AnthropicScreener:
    """``ScreenerPort`` adapter that refines a baseline verdict via an LLM call.

    Holds a lazily-constructed Anthropic client and the prompt adapter. Settings
    and tuning are read fresh per ``screen`` call (both are ``lru_cache``-backed),
    so config is picked up without rebuilding the screener.
    """

    def __init__(self, prompt: ScreenerPromptAdapter | None = None) -> None:
        self._prompt = prompt or ScreenerPromptAdapter()
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic:
        """Return the cached Anthropic client, constructing it on first use.

        Deferred so importing the module (and constructing the screener at API
        wire-up) never requires an API key — only an actual ``screen`` call does.
        """
        client = self._client
        if client is None:
            client = self._client = Anthropic(api_key=get_settings().anthropic_api_key or None)
        return client

    def screen(self, request: ScoreRequest, *, baseline: Verdict) -> Verdict:
        """Call the model to produce a refined ``Verdict`` from the baseline.

        Builds the cached system blocks and the user turn, sends the request with
        structured ``Verdict`` output, and attaches ``output_config`` only when
        the model supports the configured effort level.

        Args:
            request: The posting + profile being scored.
            baseline: The deterministic verdict the model is asked to refine.

        Returns:
            The model's parsed ``Verdict``.

        Raises:
            RuntimeError: If the response carries no parsable verdict (e.g. the
                model hit a stop reason before completing structured output).
        """
        settings = get_settings()
        tuning = get_scorer_tuning()
        system = self._prompt.build_system_blocks(
            request.profile,
            axis_weights=tuning.axis_weight_lines(),
            apply_threshold=settings.apply_threshold,
            maybe_threshold=settings.maybe_threshold,
        )
        user = self._prompt.build_user_message(
            request.posting, request.profile.preferences, baseline=baseline
        )
        messages: list[MessageParam] = [{"role": "user", "content": user}]
        client = self._get_client()
        params: dict = {
            "model": settings.scorer_model,
            "max_tokens": settings.scorer_max_tokens,
            "system": system,
            "messages": messages,
            "output_format": Verdict,
        }
        if _supports_effort(client, settings.scorer_model, settings.scorer_effort):
            params["output_config"] = OutputConfigParam(effort=settings.scorer_effort)
        resp = client.messages.parse(**params)
        verdict = resp.parsed_output
        if verdict is None:
            raise RuntimeError(
                f"screener returned no parsable verdict (stop_reason={resp.stop_reason})"
            )
        return verdict
