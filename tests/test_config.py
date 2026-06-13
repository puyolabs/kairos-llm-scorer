# SPDX-License-Identifier: Apache-2.0
"""Tests for the config boundary: the scorer-tuning loader, config_dir
resolution, and the Settings decision-band invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import scorer.config as cfg
from scorer.config import (
    Settings,
    config_dir,
    get_scorer_tuning,
    get_settings,
    load_scorer_tuning,
)

_EXAMPLE = Path(cfg.__file__).resolve().parents[2] / "config" / "scorer.example.toml"


# ── 1. load_scorer_tuning ────────────────────────────────────────────────────


def test_load_scorer_tuning_from_example():
    t = load_scorer_tuning(_EXAMPLE)
    assert abs(sum(t.weights.values()) - 1.0) < 1e-9
    assert t.tier_credit["core"] == 1.0
    assert t.region_utc_offset["apac"] == 8.0
    assert t.country_utc_offset["US"] == -5.0


def test_load_scorer_tuning_missing_file_fails_loud(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="scorer.example.toml"):
        load_scorer_tuning(tmp_path / "absent.toml")


def test_load_scorer_tuning_composes_eligibility_gates():
    # Each gate's country set is the UNION of its named regions' lists.
    t = load_scorer_tuning(_EXAMPLE)
    assert t.eligibility_countries["us-only"] == frozenset({"US"})  # single region
    assert t.eligibility_countries["north-america-only"] == frozenset({"US", "CA"})  # us ∪ ca
    assert t.eligibility_countries["emea-only"] == frozenset(  # eu ∪ uk ∪ mea
        {"DE", "FR", "ES", "GB", "AE", "ZA"}
    )
    # `none` / `unknown` carry no restriction and are deliberately absent.
    assert "none" not in t.eligibility_countries
    assert "unknown" not in t.eligibility_countries


def test_get_scorer_tuning_is_cached():
    # conftest points KAIROS_CONFIG_DIR at the example and clears the cache, so
    # this loads the example tuning; repeated calls return the one cached object.
    first = get_scorer_tuning()
    assert get_scorer_tuning() is first
    assert abs(sum(first.weights.values()) - 1.0) < 1e-9


# ── 2. config_dir resolution ─────────────────────────────────────────────────


def test_config_dir_honors_env(monkeypatch):
    monkeypatch.setenv("KAIROS_CONFIG_DIR", "/tmp/somewhere")
    assert config_dir() == Path("/tmp/somewhere")


def test_config_dir_defaults_to_repo_config(monkeypatch):
    monkeypatch.delenv("KAIROS_CONFIG_DIR", raising=False)
    assert config_dir() == Path(cfg.__file__).resolve().parents[2] / "config"


# ── 3. Settings decision-band invariants ─────────────────────────────────────


def test_default_settings_are_valid():
    s = Settings()
    assert s.escalate_floor <= s.maybe_threshold <= s.apply_threshold


def test_get_settings_is_cached():
    get_settings.cache_clear()
    first = get_settings()
    assert get_settings() is first  # lru_cache hands back the same instance


# ── 4. API-key parsing (the /score auth gate's source of truth) ──────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("secret-1,secret-2", frozenset({"secret-1", "secret-2"})),
        (" a , b ,, c ", frozenset({"a", "b", "c"})),  # stripped; blank entries dropped
        ("", frozenset()),  # unset ⇒ empty ⇒ auth disabled
        ("   ", frozenset()),  # all-whitespace ⇒ empty
        ("dup,dup", frozenset({"dup"})),  # de-duplicated by the set
    ],
)
def test_accepted_api_keys_parses_csv(raw: str, expected: frozenset[str]):
    settings = Settings().model_copy(update={"scorer_api_keys": raw})
    assert settings.accepted_api_keys() == expected


def test_apply_below_maybe_rejected(monkeypatch):
    monkeypatch.setenv("KAIROS_APPLY_THRESHOLD", "30")
    monkeypatch.setenv("KAIROS_MAYBE_THRESHOLD", "40")
    with pytest.raises(ValidationError, match="apply_threshold must be"):
        Settings()


def test_escalate_floor_above_maybe_rejected(monkeypatch):
    # The backstop invariant: escalate_floor must sit at or below the maybe band.
    monkeypatch.setenv("KAIROS_MAYBE_THRESHOLD", "35")
    monkeypatch.setenv("KAIROS_ESCALATE_FLOOR", "50")
    with pytest.raises(ValidationError, match="escalate_floor must be"):
        Settings()


def test_escalate_floor_equal_to_maybe_is_allowed(monkeypatch):
    monkeypatch.setenv("KAIROS_MAYBE_THRESHOLD", "35")
    monkeypatch.setenv("KAIROS_ESCALATE_FLOOR", "35")
    assert Settings().escalate_floor == 35
