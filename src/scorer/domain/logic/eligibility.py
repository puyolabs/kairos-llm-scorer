# SPDX-License-Identifier: Apache-2.0

"""Country-eligibility gating.

Decides whether a posting's hiring-eligibility rule admits the user, given
where the user can legally work. The three inputs come from three distinct
sources: the posting selects *which* rule applies, operator config defines
*what countries that rule permits*, and the user declares *where they can work*.
"""

from __future__ import annotations

from collections.abc import Mapping


def eligibility_allows(
    gate: str | None,
    work_countries: list[str],
    eligibility_countries: Mapping[str, frozenset[str]],
) -> bool:
    """Return whether the user is eligible under the posting's gate.

    Permissive by default: only excludes the user when a known gate lists a
    specific country set and none of the user's work countries fall within it.

    Args:
        gate: Eligibility rule key declared by the posting
            (``posting.eligibility_gate``), or ``None`` if the posting sets no
            rule.
        work_countries: ISO-2 countries the user can work in, from their
            profile (``profile.preferences.gate.work_countries``). Might be empty.
        eligibility_countries: Operator config mapping each gate key to the set
            of countries it permits (``ScorerTuning.eligibility_countries``).

    Returns:
        ``True`` if eligible — when there is no gate, the gate is unknown to the
        config, or the user lists no work countries; otherwise ``True`` only if
        the gate permits at least one work country. ``False`` when a
        known gate's permitted set excludes all the user's work countries.
    """
    allowed = None if gate is None else eligibility_countries.get(gate)
    if allowed is None or not work_countries:
        return True
    return any(country in allowed for country in work_countries)
