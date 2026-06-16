# SPDX-License-Identifier: Apache-2.0
"""Tests for the eval verdict cache: keying and the legacy→new migration.

`_rekey_cache`'s whole point is a lossless, billing-free migration — a silent key
mismatch forces re-billing real LLM calls. These pin: `_strip_vectors` drops only
`vector` keys at any depth; `_cache_key` is invariant to vector changes but busts
on a baseline/text change; and `_rekey_cache` copies legacy entries to the new key
with correct migrated/already/missing accounting.
"""

from __future__ import annotations

import json

from evals.run_eval import (
    GoldenCase,
    _cache_key,
    _legacy_cache_key,
    _rekey_cache,
    _strip_vectors,
)

from scorer.domain.models import ScoreRequest, Verdict


def _request(*, title: str = "Engineer", vector: list[float] | None = None) -> ScoreRequest:
    vector = vector if vector is not None else [1.0, 0.0]
    return ScoreRequest.model_validate(
        {
            "posting": {
                "kind": "job",
                "source_id": "s",
                "external_id": "1",
                "canonical_key": "s::1",
                "url": "https://x/y",
                "title": title,
                "company": "Acme",
                "description": "..",
                "posted_at": "2026-01-01T00:00:00Z",
                "fetched_at": "2026-01-01T00:00:00Z",
                "location_text": "Remote",
                "remote": "yes",
                "seniority_hint": "senior",
                "abilities": [{"tag": "Go", "ordinal": 0, "gloss": "Go", "vector": vector}],
            },
            "profile": {"body": "x"},
            "sonnet_judgement": "wide",
        }
    )


def _baseline(score: int = 70) -> Verdict:
    return Verdict(decision="apply", match_score=score, reasoning="b")


# ── _strip_vectors ───────────────────────────────────────────────────────────


def test_strip_vectors_removes_only_vector_keys_at_all_depths():
    tree = {
        "vector": [1.0],
        "tag": "t",
        "nested": {"vector": [2.0], "gloss": "g"},
        "items": [{"vector": [3.0], "k": 1}, {"k": 2}],
    }
    assert _strip_vectors(tree) == {
        "tag": "t",
        "nested": {"gloss": "g"},
        "items": [{"k": 1}, {"k": 2}],
    }


# ── _cache_key invariants ─────────────────────────────────────────────────────


def test_cache_key_invariant_to_vector_changes():
    # Re-embedding the profile changes vectors but not what the LLM sees, so the
    # key (and thus the cached verdict) must survive — no re-bill.
    base = _baseline()
    k1 = _cache_key(_request(vector=[1.0, 0.0]), "sonnet", base)
    k2 = _cache_key(_request(vector=[0.5, 0.5]), "sonnet", base)
    assert k1 == k2


def test_cache_key_busts_on_baseline_change():
    req = _request()
    assert _cache_key(req, "sonnet", _baseline(70)) != _cache_key(req, "sonnet", _baseline(71))


def test_cache_key_busts_on_posting_text_change():
    base = _baseline()
    assert _cache_key(_request(title="A"), "sonnet", base) != _cache_key(
        _request(title="B"), "sonnet", base
    )


# ── _rekey_cache migration ─────────────────────────────────────────────────────


def test_rekey_cache_migrates_legacy_entries(tmp_path, capsys):
    model, judgement = "claude-sonnet-4-6", "wide"

    r_migrate, bv_migrate = _request(title="Migrate"), _baseline(70)
    r_already, bv_already = _request(title="Already"), _baseline(50)
    r_missing, bv_missing = _request(title="Missing"), _baseline(80)
    r_none = _request(title="NoBaseline")  # baseline None ⇒ skipped entirely

    cases = [
        GoldenCase("m", "apply", None, "", r_migrate),
        GoldenCase("a", "maybe", None, "", r_already),
        GoldenCase("x", "apply", None, "", r_missing),
        GoldenCase("n", "apply", None, "", r_none),
    ]
    baselines = [bv_migrate, bv_already, bv_missing, None]

    # Seed the cache: one entry under the LEGACY key (to migrate) and one already
    # under the NEW key (counts as "already new-keyed"). The "missing" case has
    # neither key present.
    legacy_key = _legacy_cache_key(r_migrate, model)
    already_key = _cache_key(r_already, model, bv_already)
    verdict_payload = {"decision": "apply", "match_score": 72, "reasoning": "cached"}
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({legacy_key: verdict_payload, already_key: {"old": True}}))

    _rekey_cache(cases, baselines, model, judgement, cache_path)

    result = json.loads(cache_path.read_text())
    new_key = _cache_key(r_migrate, model, bv_migrate)
    assert result[new_key] == verdict_payload  # verdict copied to the new key
    assert legacy_key in result  # legacy entry kept (idempotent)

    out = capsys.readouterr().out
    assert "1 migrated" in out
    assert "1 already new-keyed" in out
    assert "1 not in legacy cache" in out
