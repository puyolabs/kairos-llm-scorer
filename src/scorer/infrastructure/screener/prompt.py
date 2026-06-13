# SPDX-License-Identifier: Apache-2.0

"""Builds the screener LLM prompt from the operator-authored XML template.

``ScreenerPromptAdapter`` is the seam between the prompt text (``screener.xml`` —
IP and gitignored; ``screener.example.xml`` ships as the public template) and the
Anthropic call in ``anthropic_adapter.screener``. It splits the template into
three sections and interpolates the *canonical* rubric at render time — axis
weights from ``ScorerTuning``, decision bands from ``Settings`` — so the prompt
can never silently drift from the arithmetic scorer it backstops. The profile
block carries the prompt-cache breakpoint; the posting, preferences, and
deterministic baseline go in the per-call user message.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from xml.etree.ElementTree import fromstring

from anthropic.types import TextBlockParam

from ...config import config_dir
from ...domain.models import LedgerEntry, Posting, Preferences, Profile, Verdict

# Default template path; the loader falls back to screener.example.xml in tests
# and unprovisioned deploys (the real screener.xml is gitignored).
_PROMPT_FILE = config_dir() / "screener.xml"
# The three sections every screener template must define, in render order.
_SECTIONS = ("instructions", "profile_wrapper", "user_template")


def _strip_vectors(node: object) -> object:
    """Recursively drop every ``vector`` key from a dumped model tree.

    The LLM reads each tag's ``gloss``; its 768-float ``vector`` is consumed only
    by the arithmetic scorer. Sending vectors would bloat the posting/preferences
    payload by ~3k tokens per tag for zero signal, so they are stripped here — at
    the LLM boundary only, leaving every other ``model_dump`` (request round-trips,
    scoring) carrying the vectors it needs.
    """
    if isinstance(node, dict):
        return {k: _strip_vectors(v) for k, v in node.items() if k != "vector"}
    if isinstance(node, list):
        return [_strip_vectors(x) for x in node]
    return node


def _dump_without_vectors(model: object) -> str:
    """JSON-serialize a model for the prompt with all vectors stripped."""
    return json.dumps(_strip_vectors(model.model_dump(mode="json")), indent=2)


def _render_ledger(ledger: list[LedgerEntry]) -> str:
    """Render the candidate's resolved skill ledger as markdown bullets.

    Returns a ``(none resolved)`` placeholder for an empty ledger so the prompt
    still reads cleanly when no skills were extracted.
    """
    if not ledger:
        return "(none resolved)"
    return "\n".join(f"- {e.tag} [{e.tier}]: {e.gloss}" for e in ledger)


@lru_cache
def _load_template(path: Path) -> dict[str, str]:
    """Parse and cache the three prompt sections from the template at ``path``.

    Memoized per path (``lru_cache``) so the file is read and parsed once per
    process. Fails loud rather than rendering a half-formed prompt.

    Raises:
        FileNotFoundError: If the template file is absent (likely an
            unprovisioned ``screener.xml``).
        ValueError: If any of the three required sections is missing or empty.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"screener prompt not found at {path}. The real screener.xml is "
            "IP and gitignored; provision it using screener.example.xml."
        )
    root = fromstring(path.read_text(encoding="utf-8"))
    sections = {}
    for tag in _SECTIONS:
        node = root.find(tag)
        if node is None or node.text is None:
            raise ValueError(f"screener.xml is missing a non-empty <{tag}> section")
        sections[tag] = node.text.strip()
    return sections


class ScreenerPromptAdapter:
    """Renders the screener's system blocks and user message from the template.

    Stateless aside from the template path; the parsed template is process-cached
    by ``_load_template``. Inject a different ``prompt_file`` (e.g. the example)
    in tests.
    """

    def __init__(self, prompt_file: Path = _PROMPT_FILE) -> None:
        self._prompt_file = prompt_file

    def build_system_blocks(
        self, profile: Profile, *, axis_weights: str, apply_threshold: int, maybe_threshold: int
    ) -> list[TextBlockParam]:
        """Build the two cached system blocks: instructions, then the profile.

        The instructions block interpolates the canonical rubric (axis weights,
        decision bands) so it tracks the code. The profile block carries the
        ``ephemeral`` cache-control breakpoint — it is the largest stable prefix,
        so caching it there maximizes prompt-cache reuse across postings for the
        same candidate.

        Args:
            profile: The candidate profile; its ``body`` and ``ledger`` fill the
                profile block.
            axis_weights: Pre-rendered axis-weight rubric lines from
                ``ScorerTuning.axis_weight_lines``.
            apply_threshold: Minimum score for an ``apply`` decision (from Settings).
            maybe_threshold: Minimum score for a ``maybe`` decision (from Settings).

        Returns:
            A two-element list of Anthropic ``TextBlockParam`` system blocks.
        """
        tpl = _load_template(self._prompt_file)
        instructions = tpl["instructions"].format(
            axis_weights=axis_weights,
            apply_threshold=apply_threshold,
            maybe_threshold=maybe_threshold,
        )
        return [
            {"type": "text", "text": instructions},
            {
                "type": "text",
                "text": tpl["profile_wrapper"].format(
                    profile=profile.body, ledger=_render_ledger(profile.ledger)
                ),
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def build_user_message(
        self, posting: Posting, preferences: Preferences, *, baseline: Verdict
    ) -> str:
        """Render the per-call user turn: the posting, prefs, and baseline verdict.

        All three are serialized as indented JSON into the ``user_template`` so
        the model sees the exact arithmetic baseline it is asked to refine.
        """
        return _load_template(self._prompt_file)["user_template"].format(
            posting=_dump_without_vectors(posting),
            preferences=_dump_without_vectors(preferences),
            baseline=baseline.model_dump_json(indent=2),
        )
