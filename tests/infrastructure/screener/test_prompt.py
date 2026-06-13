# SPDX-License-Identifier: Apache-2.0
"""Tests for the screener prompt builder.

Chiefly the anti-desync property: the rendered prompt's rubric is interpolated
from the scorer's canonical source (axis weights from the domain, decision bands
from Settings), so the prompt cannot drift from the code. Plus the load-time
guarantees (raise on a missing file / section) and the cache breakpoint.

Runs against the committed `screener.example.xml` (the gitignored real one may be
absent in CI), so it also pins that the shipped template uses the placeholders.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from anthropic.types import TextBlockParam

import scorer.config as cfg
from scorer.config import load_scorer_tuning
from scorer.domain.models import LedgerEntry, Profile, ScoreRequest, Verdict
from scorer.infrastructure.screener import prompt as prompt_mod
from scorer.infrastructure.screener.prompt import ScreenerPromptAdapter

_EXAMPLE = prompt_mod._PROMPT_FILE.with_name("screener.example.xml")

# Tuning from the committed example (the gitignored real scorer.toml may be absent).
TUNING = load_scorer_tuning(
    Path(cfg.__file__).resolve().parents[2] / "config" / "scorer.example.toml"
)


def _system(*, apply: int = 65, maybe: int = 35) -> list[TextBlockParam]:
    adapter = ScreenerPromptAdapter(prompt_file=_EXAMPLE)
    return adapter.build_system_blocks(
        Profile(body="PROFILE_MARKER"),
        axis_weights=TUNING.axis_weight_lines(),
        apply_threshold=apply,
        maybe_threshold=maybe,
    )


# ── anti-desync: the rubric comes from code, not the template ────────────────


def test_axis_weights_are_injected_from_the_domain():
    instructions = _system()[0]["text"]
    # The exact rendered lines from the domain source appear verbatim, and every
    # live weight is present — so changing the tuning's weights flows into the prompt.
    assert TUNING.axis_weight_lines() in instructions
    for v in TUNING.weights.values():
        assert f"{round(v * 100)}%" in instructions


def test_decision_bands_are_injected_from_settings():
    instructions = _system(apply=70, maybe=40)[0]["text"]
    assert ">= 70" in instructions  # apply band
    assert "< 40" in instructions  # skip band
    # the defaults are NOT what got rendered — the bands really are dynamic
    assert ">= 65" not in instructions


def test_template_holds_placeholders_not_hardcoded_numbers():
    raw = _EXAMPLE.read_text(encoding="utf-8")
    assert "{axis_weights}" in raw
    assert "{apply_threshold}" in raw and "{maybe_threshold}" in raw


# ── structure: cache breakpoint + substitution ──────────────────────────────


def test_cache_breakpoint_is_on_the_profile_block_only():
    blocks = _system()
    assert len(blocks) == 2
    assert "cache_control" not in blocks[0]  # instructions are not the breakpoint
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert "PROFILE_MARKER" in blocks[1]["text"]


# ── profile block: ledger rendering ─────────────────────────────────────────


def test_resolved_ledger_renders_as_bullets():
    # A non-empty ledger renders one "- tag [tier]: gloss" line per skill into
    # the cached profile block.
    adapter = ScreenerPromptAdapter(prompt_file=_EXAMPLE)
    profile = Profile(
        body="PROFILE_MARKER",
        ledger=[
            LedgerEntry(tag="Go", tier="core", gloss="systems language", vector=[1.0, 0.0]),
            LedgerEntry(tag="Rust", tier="ramping", gloss="memory-safe", vector=[0.0, 1.0]),
        ],
    )
    profile_block = adapter.build_system_blocks(
        profile, axis_weights=TUNING.axis_weight_lines(), apply_threshold=65, maybe_threshold=35
    )[1]["text"]
    assert "- Go [core]: systems language" in profile_block
    assert "- Rust [ramping]: memory-safe" in profile_block
    assert "(none resolved)" not in profile_block


def test_empty_ledger_renders_placeholder():
    block = _system()[1]["text"]  # default Profile has no ledger
    assert "(none resolved)" in block


# ── user message: posting / preferences / baseline are interpolated ──────────


def test_user_message_interpolates_posting_prefs_and_baseline():
    adapter = ScreenerPromptAdapter(prompt_file=_EXAMPLE)
    # Posting is a discriminated union, so build it through ScoreRequest.
    request = ScoreRequest.model_validate(
        {
            "posting": {
                "kind": "job",
                "source_id": "s",
                "external_id": "1",
                "canonical_key": "s::1",
                "url": "https://x/y",
                "title": "UNIQUE_TITLE_MARKER",
                "company": "Acme",
                "description": "..",
                "posted_at": "2026-01-01T00:00:00Z",
                "fetched_at": "2026-01-01T00:00:00Z",
                "location_text": "Remote",
                "remote": "yes",
                "seniority_hint": "senior",
            },
            "profile": {"body": "x", "preferences": {"preferred_engagement": "contract"}},
        }
    )
    baseline = Verdict(decision="maybe", match_score=42, reasoning="BASELINE_MARKER")

    msg = adapter.build_user_message(
        request.posting, request.profile.preferences, baseline=baseline
    )

    # Each of the three is serialized as JSON into its template slot.
    assert "UNIQUE_TITLE_MARKER" in msg  # posting
    assert '"preferred_engagement": "contract"' in msg  # preferences
    assert "BASELINE_MARKER" in msg and '"match_score": 42' in msg  # baseline verdict


def test_user_message_strips_vectors_but_keeps_glosses():
    # Vectors are scorer-only: they must never reach the LLM prompt (≈3k wasted
    # tokens/tag), but the tag's gloss — what the model actually reads — must stay.
    adapter = ScreenerPromptAdapter(prompt_file=_EXAMPLE)
    request = ScoreRequest.model_validate(
        {
            "posting": {
                "kind": "job",
                "source_id": "s",
                "external_id": "1",
                "canonical_key": "s::1",
                "url": "https://x/y",
                "title": "t",
                "company": "Acme",
                "description": "..",
                "posted_at": "2026-01-01T00:00:00Z",
                "fetched_at": "2026-01-01T00:00:00Z",
                "location_text": "Remote",
                "remote": "yes",
                "seniority_hint": "senior",
                "abilities": [
                    {
                        "tag": "Go",
                        "ordinal": 0,
                        "gloss": "GLOSS_MARKER",
                        "vector": [0.123456, 0.654321],
                    }
                ],
            },
            "profile": {"body": "x", "preferences": {"preferred_engagement": "contract"}},
        }
    )
    baseline = Verdict(decision="maybe", match_score=42, reasoning="r")

    msg = adapter.build_user_message(
        request.posting, request.profile.preferences, baseline=baseline
    )

    assert '"vector"' not in msg  # no vector key reaches the prompt
    assert "0.123456" not in msg  # nor any vector component value
    assert '"gloss": "GLOSS_MARKER"' in msg  # the gloss the model reads survives


# ── load-time guarantees ────────────────────────────────────────────────────


def test_missing_screener_file_raises(tmp_path):
    adapter = ScreenerPromptAdapter(prompt_file=tmp_path / "nope.xml")
    with pytest.raises(FileNotFoundError):
        adapter.build_system_blocks(
            Profile(body="x"), axis_weights="x", apply_threshold=65, maybe_threshold=35
        )


def test_missing_section_raises(tmp_path):
    partial = tmp_path / "partial.xml"
    partial.write_text("<prompt><instructions>hi</instructions></prompt>", encoding="utf-8")
    adapter = ScreenerPromptAdapter(prompt_file=partial)
    with pytest.raises(ValueError):
        adapter.build_system_blocks(
            Profile(body="x"), axis_weights="x", apply_threshold=65, maybe_threshold=35
        )
