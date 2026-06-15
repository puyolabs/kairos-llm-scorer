# SPDX-License-Identifier: Apache-2.0
"""Tests for the async `AnthropicScreener` adapter (the sync→async migration).

`FakeScreener` proves the port seam, but not the real `await client.messages.parse`
/ `await client.models.retrieve` rewrites — exactly where an async-migration bug
hides. These inject a fake `AsyncAnthropic` client (via `_get_client`) and exercise
the happy path, the no-verdict guard, and the best-effort `_supports_effort` probe.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from anthropic import AnthropicError

import scorer.infrastructure.screener.anthropic_adapter.screener as screener_mod
from scorer.domain.models import ScoreRequest, Verdict
from scorer.infrastructure.screener import prompt as prompt_mod
from scorer.infrastructure.screener.anthropic_adapter.screener import (
    AnthropicScreener,
    _supports_effort,
)
from scorer.infrastructure.screener.prompt import ScreenerPromptAdapter

# The real screener.xml is gitignored IP (absent in CI), so back the adapter with
# the committed example — same trick test_prompt.py uses.
_EXAMPLE = prompt_mod._PROMPT_FILE.with_name("screener.example.xml")


def _screener() -> AnthropicScreener:
    return AnthropicScreener(prompt=ScreenerPromptAdapter(prompt_file=_EXAMPLE))


class _FakeMessages:
    def __init__(self, parsed_output, stop_reason):
        self._parsed_output = parsed_output
        self._stop_reason = stop_reason
        self.calls: list[dict] = []

    async def parse(self, **params):
        self.calls.append(params)
        return SimpleNamespace(parsed_output=self._parsed_output, stop_reason=self._stop_reason)


class _FakeModels:
    def __init__(self, capabilities, error):
        self._capabilities = capabilities
        self._error = error
        self.calls: list[str] = []

    async def retrieve(self, model):
        self.calls.append(model)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(capabilities=self._capabilities)


class _FakeClient:
    """Stands in for `AsyncAnthropic`: awaitable `messages.parse` / `models.retrieve`."""

    def __init__(
        self, *, parsed_output=None, stop_reason="end_turn", capabilities=None, error=None
    ):
        self.messages = _FakeMessages(parsed_output, stop_reason)
        self.models = _FakeModels(capabilities, error)


@pytest.fixture(autouse=True)
def _clear_effort_cache():
    """The per-model effort memo persists across calls — reset it per test."""
    screener_mod._EFFORT_SUPPORT.clear()
    yield
    screener_mod._EFFORT_SUPPORT.clear()


def _request() -> ScoreRequest:
    return ScoreRequest.model_validate(
        {
            "posting": {
                "kind": "job",
                "source_id": "s",
                "external_id": "1",
                "canonical_key": "s::1",
                "url": "https://x/y",
                "title": "Engineer",
                "company": "Acme",
                "description": "..",
                "posted_at": "2026-01-01T00:00:00Z",
                "fetched_at": "2026-01-01T00:00:00Z",
                "location_text": "Remote",
                "remote": "yes",
                "seniority_hint": "senior",
            },
            "profile": {"body": "x"},
        }
    )


def _baseline() -> Verdict:
    return Verdict(decision="maybe", match_score=42, reasoning="baseline")


# ── screen(): happy path + no-verdict guard ──────────────────────────────────


@pytest.mark.anyio
async def test_screen_returns_parsed_verdict(monkeypatch):
    screener = _screener()
    parsed = Verdict(decision="apply", match_score=77, reasoning="parsed by model")
    # capabilities=None ⇒ effort unsupported ⇒ no output_config on the parse call.
    fake = _FakeClient(parsed_output=parsed, capabilities=None)
    monkeypatch.setattr(screener, "_get_client", lambda: fake)

    verdict = await screener.screen(_request(), baseline=_baseline())

    assert verdict is parsed
    assert len(fake.messages.calls) == 1
    assert "output_config" not in fake.messages.calls[0]


@pytest.mark.anyio
async def test_screen_sends_output_config_when_effort_supported(monkeypatch):
    screener = _screener()
    parsed = Verdict(decision="apply", match_score=77, reasoning="parsed")
    fake = _FakeClient(parsed_output=parsed, capabilities={"effort": {"supported": True}})
    monkeypatch.setattr(screener, "_get_client", lambda: fake)

    await screener.screen(_request(), baseline=_baseline())

    assert "output_config" in fake.messages.calls[0]


@pytest.mark.anyio
async def test_screen_raises_when_no_parsable_verdict(monkeypatch):
    screener = _screener()
    fake = _FakeClient(parsed_output=None, stop_reason="max_tokens", capabilities=None)
    monkeypatch.setattr(screener, "_get_client", lambda: fake)

    with pytest.raises(RuntimeError, match="max_tokens"):
        await screener.screen(_request(), baseline=_baseline())


# ── _supports_effort(): best-effort probe ────────────────────────────────────


@pytest.mark.anyio
async def test_supports_effort_swallows_anthropic_error():
    client = _FakeClient(error=AnthropicError("lookup failed"))
    assert await _supports_effort(client, "some-model", "high") is False
    # A transient failure is NOT cached, so a later call can re-probe.
    assert "some-model" not in screener_mod._EFFORT_SUPPORT


@pytest.mark.anyio
async def test_supports_effort_true_and_cached_when_advertised():
    client = _FakeClient(capabilities={"effort": {"supported": True}})
    assert await _supports_effort(client, "m", "high") is True
    assert screener_mod._EFFORT_SUPPORT["m"] is True


@pytest.mark.anyio
async def test_supports_effort_false_when_level_unsupported():
    caps = {"effort": {"supported": True, "high": {"supported": False}}}
    client = _FakeClient(capabilities=caps)
    assert await _supports_effort(client, "m", "high") is False


@pytest.mark.anyio
async def test_supports_effort_reads_capabilities_via_model_dump():
    caps = SimpleNamespace(model_dump=lambda: {"effort": {"supported": True}})
    client = _FakeClient(capabilities=caps)
    assert await _supports_effort(client, "m", "low") is True


@pytest.mark.anyio
async def test_supports_effort_uses_cache_without_network():
    screener_mod._EFFORT_SUPPORT["m"] = True
    client = _FakeClient(error=AnthropicError("must not be called"))
    assert await _supports_effort(client, "m", "high") is True
    assert client.models.calls == []  # cache hit short-circuits the round-trip
