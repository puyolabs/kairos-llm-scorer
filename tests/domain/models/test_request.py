# SPDX-License-Identifier: Apache-2.0
"""Tests for `ScoreRequest`'s embedding-dimension validator and `_iter_vectors`.

The validator (`_embeddings_share_one_dimension`) only works if `_iter_vectors`
reaches *every* embedding-bearing node in the request tree — postings, profile
ledger, preference domains, and the gate's role families, at any nesting depth.
These assert the iterator's reach and that a mismatch buried at any depth is
caught (→ HTTP 422).
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from scorer.domain.models import ScoreRequest
from scorer.domain.models.request import _iter_vectors


def _request_dict() -> dict:
    """A request carrying a vector at every embedding-bearing node (all dim 2).

    Eleven vectors total: posting abilities (3) + role_families (1) + domains (1),
    profile ledger (3), preferences candidate_domains (2), gate role family (1).
    """
    return {
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
            "abilities": [
                {"tag": "Go", "ordinal": 0, "gloss": "Go", "vector": [1.0, 0.0]},
                {"tag": "K8s", "ordinal": 1, "gloss": "Kubernetes", "vector": [0.0, 1.0]},
                {"tag": "PG", "ordinal": 2, "gloss": "PostgreSQL", "vector": [1.0, 1.0]},
            ],
            "role_families": [{"tag": "swe", "gloss": "software", "vector": [1.0, 0.0]}],
            "domains": [{"tag": "logi", "gloss": "logistics", "vector": [0.0, 1.0]}],
        },
        "profile": {
            "body": "x",
            "ledger": [
                {"tag": "Go", "tier": "core", "gloss": "Go", "vector": [1.0, 0.0]},
                {"tag": "K8s", "tier": "proficient", "gloss": "Kubernetes", "vector": [0.0, 1.0]},
                {"tag": "PG", "tier": "core", "gloss": "PostgreSQL", "vector": [1.0, 1.0]},
            ],
            "preferences": {
                "candidate_domains": [
                    {"tag": "fin", "gloss": "fintech", "vector": [0.0, 1.0]},
                    {"tag": "saas", "gloss": "saas", "vector": [0.5, 0.5]},
                ],
                "gate": {
                    "allowed_role_families": [
                        {"tag": "swe", "gloss": "software", "vector": [1.0, 0.0]}
                    ]
                },
            },
        },
    }


def test_iter_vectors_reaches_every_embedding_node():
    req = ScoreRequest.model_validate(_request_dict())
    # All eleven embeddings are surfaced — proves the recursion descends through
    # nested models and lists rather than stopping at the top level.
    assert len(list(_iter_vectors(req))) == 11


# Each path points at a distinct embedding-bearing node at a different depth.
@pytest.mark.parametrize(
    "path",
    [
        ("posting", "abilities", 0, "vector"),
        ("posting", "domains", 0, "vector"),
        ("profile", "ledger", 2, "vector"),  # deeply-nested LedgerEntry
        ("profile", "preferences", "candidate_domains", 1, "vector"),
        ("profile", "preferences", "gate", "allowed_role_families", 0, "vector"),
    ],
)
def test_mismatch_at_any_depth_is_caught(path):
    data = copy.deepcopy(_request_dict())
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = [1.0, 0.0, 0.0]  # dim 3 against the rest's dim 2
    with pytest.raises(ValidationError, match="dimension"):
        ScoreRequest.model_validate(data)
