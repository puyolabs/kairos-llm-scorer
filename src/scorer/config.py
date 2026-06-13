# SPDX-License-Identifier: Apache-2.0

"""Process configuration: runtime ``Settings`` and the scorer-tuning loader.

Two distinct config surfaces meet here. ``Settings`` is the per-process runtime
knob set — secrets, model choice, decision bands — sourced from environment
variables (and ``.env``) by alias and validated once. ``load_scorer_tuning`` /
``get_scorer_tuning`` parse the operator's ``scorer.toml`` into the frozen
``ScorerTuning`` the scoring math reads. Both lookups are ``lru_cache``d so the
environment and the TOML are read once per process; ``config_dir`` resolves where
the latter lives (overridable via ``KAIROS_CONFIG_DIR``, which the test suite uses
to point at the committed example).
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .domain.logic.tuning import ScorerTuning


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables by alias.

    Each field reads a ``KAIROS_*`` (or ``ANTHROPIC_API_KEY``) env var, falling
    back to its declared default; unrecognized env vars are ignored. The decision
    bands are cross-validated (``_bands_ordered``) so an impossible threshold
    ordering fails at load rather than mid-score.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    scorer_api_keys: str = Field(default="", validation_alias="KAIROS_SCORER_API_KEYS")
    scorer_model: str = Field(default="claude-sonnet-4-6", validation_alias="KAIROS_SCORER_MODEL")
    scorer_effort: Literal["low", "medium", "high", "max"] = Field(
        default="medium", validation_alias="KAIROS_SCORER_EFFORT"
    )
    scorer_max_tokens: int = Field(default=2048, gt=0, validation_alias="KAIROS_SCORER_MAX_TOKENS")

    apply_threshold: int = Field(
        default=65, ge=0, le=100, validation_alias="KAIROS_APPLY_THRESHOLD"
    )
    maybe_threshold: int = Field(
        default=35, ge=0, le=100, validation_alias="KAIROS_MAYBE_THRESHOLD"
    )

    escalate_floor: int = Field(default=25, ge=0, le=100, validation_alias="KAIROS_ESCALATE_FLOOR")

    build_sha: str = Field(default="dev", validation_alias="KAIROS_BUILD_SHA")

    @model_validator(mode="after")
    def _bands_ordered(self) -> Settings:
        """Enforce ``apply ≥ maybe ≥ escalate_floor`` — the band/backstop invariant.

        ``apply_threshold`` must sit at or above ``maybe_threshold``, and
        ``escalate_floor`` at or below it, so the escalation backstop can only
        widen (never invert) the maybe band. Raises ``ValueError`` otherwise.
        """
        if self.apply_threshold < self.maybe_threshold:
            raise ValueError("apply_threshold must be ≥ maybe_threshold")
        if self.escalate_floor > self.maybe_threshold:
            raise ValueError("escalate_floor must be ≤ maybe_threshold (backstop invariant)")
        return self

    def accepted_api_keys(self) -> frozenset[str]:
        """Parse ``scorer_api_keys`` (a CSV) into the set of accepted bearer keys.

        Whitespace is stripped and blank entries dropped, so an empty or
        all-whitespace setting yields an empty set — which ``api`` reads as
        "auth disabled" (the open-access posture).
        """
        return frozenset(k.strip() for k in self.scorer_api_keys.split(",") if k.strip())


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide ``Settings``, built once from the environment."""
    return Settings()


def config_dir() -> Path:
    """Resolve the directory holding ``scorer.toml`` / ``screener.xml``.

    Honors ``KAIROS_CONFIG_DIR`` when set (the test suite points it at the
    committed example); otherwise defaults to the repo's top-level ``config/``.
    """
    env = os.environ.get("KAIROS_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"


def load_scorer_tuning(path: Path) -> ScorerTuning:
    """Parse a ``scorer.toml`` at ``path`` into a validated ``ScorerTuning``.

    Flattens the TOML's nested tables into the flat tuning model and composes the
    eligibility gates: each gate's permitted country set is the union of its named
    regions' country lists. Fails loud if the file is absent (the real
    ``scorer.toml`` is gitignored — provision it from the example).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"scorer tuning not found at {path}. The real scorer.toml is gitignored; "
            "provision it from config/scorer.example.toml."
        )
    d = tomllib.loads(path.read_text(encoding="utf-8"))
    regions = d["eligibility"]["regions"]
    eligibility_countries = {
        gate: frozenset().union(*(regions[name] for name in region_names))
        for gate, region_names in d["eligibility"]["gates"].items()
    }
    return ScorerTuning(
        weights=d["weights"],
        tier_credit=d["tier_credit"],
        baseline_credit=d["technical"]["baseline_credit"],
        no_abilities_t=d["technical"]["no_abilities_t"],
        exact_sim=d["relatedness"]["exact_sim"],
        related_sim_floor=d["relatedness"]["related_sim_floor"],
        domain_direct=d["domain"]["direct"],
        domain_transferable=d["domain"]["transferable"],
        domain_mismatch=d["domain"]["mismatch"],
        domain_direct_sim=d["domain"]["direct_sim"],
        domain_mismatch_sim=d["domain"]["mismatch_sim"],
        role_gate_sim=d["role"]["gate_sim"],
        seniority_ladder=d["seniority"]["ladder"],
        seniority_step=d["seniority"]["step"],
        seniority_neutral=d["seniority"]["neutral"],
        work_window_hours=d["remote"]["work_window_hours"],
        full_shift_hours=d["remote"]["full_shift_hours"],
        remote_unknown=d["remote"]["unknown"],
        region_utc_offset=d["remote"]["region_utc_offset"],
        country_utc_offset=d["remote"]["country_utc_offset"],
        engagement_mismatch_factor=d["engagement"]["mismatch_factor"],
        remote_to_arrangement=d["modality"]["remote_to_arrangement"],
        eligibility_countries=eligibility_countries,
    )


@lru_cache
def get_scorer_tuning() -> ScorerTuning:
    """Return the process-wide ``ScorerTuning``, loaded once from ``config_dir()``."""
    return load_scorer_tuning(config_dir() / "scorer.toml")
