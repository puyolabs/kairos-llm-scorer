# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for `scorer.domain.logic.eligibility`.

The location-derived eligibility check: gate membership (incl. the composed
regional gates), the recall-safe passes (`none`/`unknown`/unrecognized gate,
empty `work_countries`), and 'any one country satisfies' semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scorer.config as cfg
from scorer.config import load_scorer_tuning
from scorer.domain.logic.eligibility import eligibility_allows

_COUNTRIES = load_scorer_tuning(
    Path(cfg.__file__).resolve().parents[2] / "config" / "scorer.example.toml"
).eligibility_countries


@pytest.mark.parametrize(
    ("gate", "country", "allowed"),
    [
        ("us-only", "US", True),
        ("us-only", "CA", False),
        ("canada-only", "CA", True),
        ("uk-only", "GB", True),
        ("uk-only", "DE", False),
        ("north-america-only", "CA", True),
        ("north-america-only", "MX", False),  # tech 'NA' = US+CA; MX → latam-only
        ("eu-only", "DE", True),
        ("eu-only", "GB", False),  # UK left the EU
        ("emea-only", "DE", True),  # Europe ⊂ EMEA
        ("emea-only", "GB", True),
        ("emea-only", "AE", True),  # Middle East
        ("emea-only", "ZA", True),  # Africa
        ("emea-only", "US", False),
        ("latam-only", "BR", True),
        ("latam-only", "MX", True),
        ("latam-only", "US", False),
        ("apac-only", "JP", True),
        ("apac-only", "AU", True),  # ANZ ⊂ APAC
        ("apac-only", "US", False),
        ("anz-only", "AU", True),
        ("anz-only", "JP", False),  # in APAC but not ANZ
    ],
)
def test_gate_membership(gate: str, country: str, allowed: bool):
    assert eligibility_allows(gate, [country], _COUNTRIES) is allowed


@pytest.mark.parametrize("gate", ["none", "unknown", None])
def test_no_restriction_always_passes(gate):
    # No real restriction ⇒ passes even for an unlisted country.
    assert eligibility_allows(gate, ["XX"], _COUNTRIES) is True


def test_unrecognized_gate_passes():
    # A gate value not in the table is recall-safe (never silently rejects).
    assert eligibility_allows("mars-only", ["US"], _COUNTRIES) is True


def test_empty_work_countries_passes():
    # No location signal ⇒ recall-safe pass against a real gate.
    assert eligibility_allows("us-only", [], _COUNTRIES) is True


def test_any_one_country_satisfies():
    # Multiple authorizations: one match is enough.
    assert eligibility_allows("us-only", ["DE", "US"], _COUNTRIES) is True
    assert eligibility_allows("us-only", ["DE", "CA"], _COUNTRIES) is False
