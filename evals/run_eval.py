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

Artifacts are written per model family: the report to `evals/report.<slug>.md`
(committed) and the verdict cache to `evals/.eval_cache.<slug>.json` (gitignored),
where `<slug>` is the model's pricing family (sonnet/haiku/opus). The cache is keyed
by model + judgement + the vector-stripped request + baseline — only what the
screener actually sends Anthropic — so re-running replays cached verdicts instead of
re-billing; pass `--no-cache` to force fresh calls, or `--rekey` to migrate a cache
built under the legacy whole-request key (no LLM calls).

Usage: uv run python -m evals.run_eval [--judgement off|narrow|wide] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
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

DECISIONS: tuple[Decision, ...] = ("apply", "maybe", "skip")

# Per-MTok USD prices for the screener model. Cost is an estimate — verify against
# current Anthropic pricing before quoting it. Keyed by a model-id substring.
PRICING = {
    "sonnet": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    "haiku": {"input": 0.80, "output": 4.0, "cache_write": 1.0, "cache_read": 0.08},
    "opus": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
}


def _model_slug(model: str) -> str:
    """Short family tag for naming per-model artifacts (cache + report).

    Matches the PRICING family keys (``sonnet``/``haiku``/``opus``) so the files
    written line up with the committed ``report.<slug>.md`` / ``.eval_cache.<slug>.json``
    products; falls back to a sanitized model id for an unrecognized family.
    """
    for family in PRICING:
        if family in model:
            return family
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _cache_path(model: str) -> Path:
    """On-disk verdict cache for ``model`` (gitignored, one file per family)."""
    return HERE / f".eval_cache.{_model_slug(model)}.json"


def _report_path(model: str) -> Path:
    """Markdown report for ``model`` (committed, one file per family)."""
    return HERE / f"report.{_model_slug(model)}.md"


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
# yields the same verdict, so we replay it from disk instead of re-billing the LLM.
#
# Keyed by a hash of *only what the screener actually sends to Anthropic*: model,
# judgement, the request with embedding vectors stripped, and the deterministic
# baseline. The prompt builder drops every 768-float vector before the LLM sees it
# (`prompt._strip_vectors`); the vectors influence the verdict solely through the
# baseline, which is in the key. So re-embedding the profile or moving to a binary
# vector wire format leaves the key untouched (no re-bill), while a genuine change
# to the posting/profile text or the baseline the model reads still busts it.
#
# Migrating from the legacy whole-request key is lossless via `--rekey` (no calls).


def _strip_vectors(node: object) -> object:
    """Drop every ``vector`` key from a dumped tree — they never reach the LLM."""
    if isinstance(node, dict):
        return {k: _strip_vectors(v) for k, v in node.items() if k != "vector"}
    if isinstance(node, list):
        return [_strip_vectors(x) for x in node]
    return node


def _cache_key(req: ScoreRequest, model: str, baseline: Verdict) -> str:
    """Hash the verdict-determining inputs: model, judgement, stripped request, baseline."""
    payload = json.dumps(
        {
            "model": model,
            "judgement": req.sonnet_judgement,
            "request": _strip_vectors(req.model_dump(mode="json")),
            "baseline": baseline.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _legacy_cache_key(req: ScoreRequest, model: str) -> str:
    """The pre-vector-strip key (whole request hashed). Used only to migrate caches."""
    payload = json.dumps(
        {"model": model, "judgement": req.sonnet_judgement, "request": req.model_dump(mode="json")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict[str, dict], cache_path: Path) -> None:
    cache_path.write_text(json.dumps(cache, indent=0))


def _rekey_cache(
    cases: list[GoldenCase],
    baselines: list[Verdict | None],
    model: str,
    judgement: str,
    cache_path: Path,
) -> None:
    """Re-key the existing cache from the legacy whole-request key to the new key.

    Lossless and billing-free: for each golden case it recomputes the legacy key
    (raw request) and the new key (stripped request + baseline) and copies the
    cached verdict across. ``judgement`` must match the mode the cache was built
    under (e.g. ``wide``) — both keys embed it, exactly as ``_run_llm`` forces it.
    Old entries are kept (idempotent, harmless). Run this *before* regenerating
    ``profile.json`` — once the vectors change, the legacy keys no longer match
    what's on disk and the mapping is lost.
    """
    cache = _load_cache(cache_path)
    migrated = already = missing = 0
    for c, bv in zip(cases, baselines, strict=True):
        if bv is None:
            continue
        req = c.request.model_copy(update={"sonnet_judgement": judgement})
        old = _legacy_cache_key(req, model)
        new = _cache_key(req, model, bv)
        if new in cache:
            already += 1
        elif old in cache:
            cache[new] = cache[old]
            migrated += 1
        else:
            missing += 1
    _save_cache(cache, cache_path)
    print(
        f"rekey ({model} → {cache_path.name}): {migrated} migrated, "
        f"{already} already new-keyed, {missing} not in legacy cache "
        f"(those re-bill if escalated)."
    )


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
    parser.add_argument(
        "--rekey",
        action="store_true",
        help="migrate the cache to the new key (no LLM calls), then exit; "
        "use the judgement the cache was built under, e.g. --judgement wide --rekey",
    )
    args = parser.parse_args()

    cases = load_golden()
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no golden examples found — is evals/golden.jsonl populated?")

    settings = get_settings()
    usage = Usage()
    cache_path = _cache_path(settings.scorer_model)
    report_path = _report_path(settings.scorer_model)

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

    if args.rekey:
        if args.judgement == "off":
            raise SystemExit(
                "--rekey needs the judgement the cache was built under, e.g. --judgement wide"
            )
        _rekey_cache(cases, base.verdicts, settings.scorer_model, args.judgement, cache_path)
        return

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
        cache = {} if args.no_cache else _load_cache(cache_path)

        async def _run_llm() -> int:
            """Score every escalating case under one event loop.

            ``score()`` is async; running the whole sequential pass in a single
            ``asyncio.run`` keeps the screener's cached ``AsyncAnthropic`` client
            bound to one loop (per-call ``asyncio.run`` would reuse it across
            closed loops). Returns the cache-hit count for the run summary.
            """
            hits = 0
            for c, bv in zip(cases, base.verdicts, strict=True):
                req = c.request.model_copy(update={"sonnet_judgement": args.judgement})
                key = _cache_key(req, settings.scorer_model, bv)
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
            _save_cache(cache, cache_path)
        if cache_hits:
            print(f"  verdict cache: {cache_hits}/{len(cases)} replayed (no LLM call)")

    report = render_report(cases, llm, base, usage, settings.scorer_model, escalated)
    report_path.write_text(report)

    print(f"Scored {len(cases)} examples → {report_path}")
    print(f"  baseline agreement: {_pct(base.agreement(cases))}  MAE: {_fmt(base.mae(cases))}")
    if llm.ran:
        print(f"  LLM agreement:      {_pct(llm.agreement(cases))}  MAE: {_fmt(llm.mae(cases))}")
        print(f"  LLM escalations: {escalated}  (billed this run: {usage.calls})")
    else:
        print(f"  LLM column: not run — {llm.note}")


if __name__ == "__main__":
    main()
