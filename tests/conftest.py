# SPDX-License-Identifier: Apache-2.0
"""Point the scorer-tuning loader at the committed example so the suite runs
without the gitignored real config/scorer.toml (mirrors how the prompt tests
use screener.example.xml)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import scorer.config as cfg


@pytest.fixture(autouse=True, scope="session")
def _scorer_config(tmp_path_factory):
    repo_config = Path(cfg.__file__).resolve().parents[2] / "config"
    d = tmp_path_factory.mktemp("config")
    shutil.copy(repo_config / "scorer.example.toml", d / "scorer.toml")
    os.environ["KAIROS_CONFIG_DIR"] = str(d)
    cfg.get_scorer_tuning.cache_clear()
    yield


@pytest.fixture
def anyio_backend() -> str:
    """Run `@pytest.mark.anyio` tests on asyncio only (no trio dependency)."""
    return "asyncio"
