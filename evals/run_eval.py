# SPDX-License-Identifier: Apache-2.0
"""Eval harness — the load-bearing part of this repo.

Runs each golden example through both the LLM screener (the two-stage `score()`)
and the deterministic baseline, then scores both against the owner's hand-labeled
decision. Emits decision-agreement %, score MAE, a confusion matrix, and (for the
LLM) cost to report.md.

The golden set is real: postings pulled from the Kairos DB, faced against the
owner's actual profile/preferences, labeled by the owner's real apply/maybe/skip
application decision. The candidate side is shared, so it lives once in
`profile.json`; each golden line carries one posting plus its expected decision.

The LLM stage runs only when an `ANTHROPIC_API_KEY` and the (gitignored, IP)
`screener.xml` are both present; otherwise the harness reports the baseline alone
and notes the LLM column as not run. Pass `--no-llm` to force baseline-only, or
`--limit N` to score only the first N examples.

LLM verdicts are cached to `evals/.eval_cache.json` (gitignored), keyed by
model + judgement + the full posting×profile request. Re-running replays cached
verdicts instead of re-billing the same call; pass `--no-cache` to force fresh calls.

Usage: uv run python -m evals.run_eval [--no-llm] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Make `scorer` importable when run as a plain script (not just `-m`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scorer.application.score import _should_escalate, score  # noqa: E402
from scorer.config import get_scorer_tuning, get_settings  # noqa: E402
from scorer.domain.logic import deterministic_score  # noqa: E402
from scorer.domain.models import Decision, Profile, ScoreRequest, Verdict  # noqa: E402
from scorer.infrastructure.screener.prompt import _PROMPT_FILE  # noqa: E402

HERE = Path(__file__).parent
GOLDEN = HERE / "golden.jsonl"
PROFILE = HERE / "profile.json"
REPORT = HERE / "report.md"
CACHE = HERE / ".eval_cache.json"

DECISIONS: tuple[Decision, ...] = ("apply", "maybe", "skip")

# Per-MTok USD prices for the screener model. Cost is an estimate — verify against
# current Anthropic pricing before quoting it. Keyed by a model-id substring.
PRICING = {
    "sonnet": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    "haiku": {"input": 0.80, "output": 4.0, "cache_write": 1.0, "cache_read": 0.08},
    "opus": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
}


# ── Golden I/O ──────────────────────────────────────────────────────────────


@dataclass
class GoldenCase:
    """One labeled example: a posting faced against the shared profile."""

    id: str
    expected_decision: Decision
    expected_score: int | None
    notes: str
    request: ScoreRequest


def load_golden() -> list[GoldenCase]:
    """Load the shared profile + each posting line into validated requests."""
    if not PROFILE.exists() or not GOLDEN.exists() or not GOLDEN.read_text().strip():
        return []
    profile_raw = json.loads(PROFILE.read_text())
    profile = Profile.model_validate(profile_raw)
    cases: list[GoldenCase] = []
    for line in GOLDEN.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        request = ScoreRequest.model_validate(
            {
                "posting": obj["posting"],
                "profile": profile.model_dump(),
                "sonnet_judgement": obj.get("sonnet_judgement", "wide"),
            }
        )
        cases.append(
            GoldenCase(
                id=obj["id"],
                expected_decision=obj["expected_decision"],
                expected_score=obj.get("expected_score"),
                notes=obj.get("notes", ""),
                request=request,
            )
        )
    return cases


# ── Cost / cache metering ───────────────────────────────────────────────────


@dataclass
class Usage:
    """Accumulated token usage across the LLM calls actually made."""

    calls: int = 0
    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    def add(self, u: object) -> None:
        self.calls += 1
        self.input += getattr(u, "input_tokens", 0) or 0
        self.output += getattr(u, "output_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0

    def cost_usd(self, model: str) -> float | None:
        price = next((p for k, p in PRICING.items() if k in model), None)
        if price is None:
            return None
        return (
            self.input * price["input"]
            + self.output * price["output"]
            + self.cache_write * price["cache_write"]
            + self.cache_read * price["cache_read"]
        ) / 1_000_000


def _metered_screener(usage: Usage):
    """An AnthropicScreener that records token usage without touching prod code.

    Wraps the Anthropic client so every `messages.parse` response feeds `usage`;
    the screener's own prompt/parse logic is reused untouched (no duplication).
    """
    from scorer.infrastructure import AnthropicScreener

    class _MessagesProxy:
        def __init__(self, inner) -> None:
            self._inner = inner

        async def parse(self, **kw):
            resp = await self._inner.parse(**kw)
            usage.add(resp.usage)
            return resp

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _ClientProxy:
        def __init__(self, inner) -> None:
            self._inner = inner
            self.messages = _MessagesProxy(inner.messages)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class MeteredScreener(AnthropicScreener):
        def _get_client(self):
            return _ClientProxy(super()._get_client())

    return MeteredScreener()


# ── Metrics ─────────────────────────────────────────────────────────────────


@dataclass
class Column:
    """One scorer's results across the golden set."""

    name: str
    verdicts: list[Verdict | None] = field(default_factory=list)
    ran: bool = True
    note: str = ""

    def agreement(self, cases: list[GoldenCase]) -> float | None:
        pairs = [(c, v) for c, v in zip(cases, self.verdicts, strict=True) if v is not None]
        if not pairs:
            return None
        hits = sum(1 for c, v in pairs if v.decision == c.expected_decision)
        return hits / len(pairs)

    def mae(self, cases: list[GoldenCase]) -> float | None:
        errs = [
            abs(v.match_score - c.expected_score)
            for c, v in zip(cases, self.verdicts, strict=True)
            if v is not None and c.expected_score is not None
        ]
        return sum(errs) / len(errs) if errs else None

    def confusion(self, cases: list[GoldenCase]) -> dict[tuple[Decision, Decision], int]:
        """Counts keyed (expected, predicted)."""
        m = Counter()
        for c, v in zip(cases, self.verdicts, strict=True):
            if v is not None:
                m[(c.expected_decision, v.decision)] += 1
        return m


# ── Report rendering ────────────────────────────────────────────────────────


def _pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "—"


def _confusion_table(col: Column, cases: list[GoldenCase]) -> str:
    m = col.confusion(cases)
    header = "| expected ↓ / predicted → | " + " | ".join(DECISIONS) + " |"
    sep = "|" + "---|" * (len(DECISIONS) + 1)
    rows = [
        "| **" + exp + "** | " + " | ".join(str(m.get((exp, pred), 0)) for pred in DECISIONS) + " |"
        for exp in DECISIONS
    ]
    return "\n".join([header, sep, *rows])


def render_report(
    cases: list[GoldenCase], llm: Column, base: Column, usage: Usage, model: str, escalated: int
) -> str:
    n = len(cases)
    dist = Counter(c.expected_decision for c in cases)
    dist_str = ", ".join(f"{d}={dist.get(d, 0)}" for d in DECISIONS)

    # Cost reflects tokens actually billed this run; a run with no fresh billing has
    # nothing to measure, so it reads "—" rather than a misleading $0.00.
    cost = usage.cost_usd(model) if usage.calls else None
    cost_per_100 = (cost / escalated * 100) if cost is not None and escalated else None

    def cell(col: Column, value: str) -> str:
        return value if col.ran else f"_not run — {col.note}_"

    lines = [
        "# Eval report",
        "",
        f"_Generated by `evals/run_eval.py` against `golden.jsonl` "
        f"({n} synthetic postings — mechanic probes + a realistic spectrum — "
        f"faced against a synthetic profile; "
        f"labels = the intended apply/maybe/skip ground truth; {dist_str})._",
        "",
        f"- **LLM screener:** {'`' + model + '`' if llm.ran else 'not run — ' + llm.note}",
        f"- **LLM calls made:** {escalated} (the baseline escalated these of {n})"
        if llm.ran
        else "- **LLM calls made:** 0",
        "- **Score MAE** is measured against band-midpoint targets implied by the "
        "decision thresholds (apply→82, maybe→50, skip→17).",
        "",
        "| Metric                       | LLM | Deterministic baseline |",
        "|------------------------------|-----|------------------------|",
        f"| Decision agreement vs golden | {cell(llm, _pct(llm.agreement(cases)))} "
        f"| {_pct(base.agreement(cases))} |",
        f"| Score MAE vs golden          | {cell(llm, _fmt(llm.mae(cases)))} "
        f"| {_fmt(base.mae(cases))} |",
        f"| Cost / 100 postings          | {cell(llm, _usd(cost_per_100))} | n/a |",
        "",
        "## Confusion — deterministic baseline",
        "",
        _confusion_table(base, cases),
    ]
    if llm.ran:
        lines += ["", "## Confusion — LLM screener", "", _confusion_table(llm, cases)]

    lines += [
        "",
        "## Per-example",
        "",
        "| id | expected | baseline | LLM | notes |",
        "|----|----------|----------|-----|-------|",
    ]
    for c, bv, lv in zip(cases, base.verdicts, llm.verdicts, strict=True):
        b = f"{bv.decision}/{bv.match_score}" if bv else "—"
        ll = f"{lv.decision}/{lv.match_score}" if lv else "—"
        notes = c.notes.replace("|", "/")[:70]
        lines.append(f"| `{c.id[:28]}` | {c.expected_decision} | {b} | {ll} | {notes} |")
    return "\n".join(lines) + "\n"


def _fmt(x: float | None) -> str:
    return f"{x:.1f}" if x is not None else "—"


def _usd(x: float | None) -> str:
    return f"${x:.2f}" if x is not None else "—"


# ── Verdict cache ───────────────────────────────────────────────────────────
# Persistent across runs: a posting×profile faced under the same model+judgement
# yields the same verdict, so we replay it from disk instead of re-billing Haiku.
# Keyed by a hash of (model, judgement, full request) so any change busts the key.


def _cache_key(req: ScoreRequest, model: str) -> str:
    payload = json.dumps(
        {"model": model, "judgement": req.sonnet_judgement, "request": req.model_dump(mode="json")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text())
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE.write_text(json.dumps(cache, indent=0))


# ── Entry point ─────────────────────────────────────────────────────────────


def _llm_available() -> tuple[bool, str]:
    if not get_settings().anthropic_api_key:
        return False, "ANTHROPIC_API_KEY not set"
    if not _PROMPT_FILE.exists():
        return False, "screener.xml not provisioned (IP, gitignored)"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judgement",
        choices=("off", "narrow", "wide"),
        default="off",
        help="LLM escalation mode (default off = no network). wide/narrow make real billed calls.",
    )
    parser.add_argument("--limit", type=int, default=None, help="score only the first N")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the on-disk verdict cache and re-bill every LLM call",
    )
    args = parser.parse_args()

    cases = load_golden()
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no golden examples found — is evals/golden.jsonl populated?")

    settings = get_settings()
    usage = Usage()

    tuning = get_scorer_tuning()
    base = Column("baseline")
    for c in cases:
        base.verdicts.append(
            deterministic_score(
                c.request,
                tuning=tuning,
                apply_threshold=settings.apply_threshold,
                maybe_threshold=settings.maybe_threshold,
            )
        )

    llm = Column("llm")
    # How many postings the baseline escalates to the screener — a deterministic
    # property of the baseline band + judgement mode, independent of whether each
    # call was freshly billed or replayed from cache. The report counts these.
    escalated = sum(
        _should_escalate(bv, args.judgement, escalate_floor=settings.escalate_floor)
        for bv in base.verdicts
        if bv is not None
    )
    if args.judgement == "off":
        # The score.py off-switch: no escalation, no screener call, no network.
        llm.ran = False
        llm.note = "escalation off (pass --judgement wide to enable real Sonnet calls)"
        llm.verdicts = [None] * len(cases)
    else:
        available, why = _llm_available()
        if not available:
            raise SystemExit(f"--judgement {args.judgement} needs the LLM but {why}.")
        # Force the chosen mode on every request so the run's cost is one explicit knob.
        screener = _metered_screener(usage)
        cache = {} if args.no_cache else _load_cache()

        async def _run_llm() -> int:
            """Score every escalating case under one event loop.

            ``score()`` is async; running the whole sequential pass in a single
            ``asyncio.run`` keeps the screener's cached ``AsyncAnthropic`` client
            bound to one loop (per-call ``asyncio.run`` would reuse it across
            closed loops). Returns the cache-hit count for the run summary.
            """
            hits = 0
            for c in cases:
                req = c.request.model_copy(update={"sonnet_judgement": args.judgement})
                key = _cache_key(req, settings.scorer_model)
                if not args.no_cache and key in cache:
                    llm.verdicts.append(Verdict.model_validate(cache[key]))
                    hits += 1
                    continue
                try:
                    verdict = await score(req, screener=screener)
                    llm.verdicts.append(verdict)
                    cache[key] = verdict.model_dump(mode="json")  # only real calls get cached
                except Exception as exc:  # one bad call shouldn't sink the whole run
                    print(f"  ! LLM failed on {c.id}: {exc}", file=sys.stderr)
                    llm.verdicts.append(None)
            return hits

        cache_hits = asyncio.run(_run_llm())
        if not args.no_cache:
            _save_cache(cache)
        if cache_hits:
            print(f"  verdict cache: {cache_hits}/{len(cases)} replayed (no LLM call)")

    report = render_report(cases, llm, base, usage, settings.scorer_model, escalated)
    REPORT.write_text(report)

    print(f"Scored {len(cases)} examples → {REPORT}")
    print(f"  baseline agreement: {_pct(base.agreement(cases))}  MAE: {_fmt(base.mae(cases))}")
    if llm.ran:
        print(f"  LLM agreement:      {_pct(llm.agreement(cases))}  MAE: {_fmt(llm.mae(cases))}")
        print(f"  LLM escalations: {escalated}  (billed this run: {usage.calls})")
    else:
        print(f"  LLM column: not run — {llm.note}")


if __name__ == "__main__":
    main()
