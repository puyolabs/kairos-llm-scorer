# SPDX-License-Identifier: Apache-2.0
# Build context: repo root (see docker-compose.base.yml / infra/README.md).
#
#   docker build -f infra/Dockerfile -t kairos-llm-scorer .
#
# uv builds the project into a venv at /app/.venv. The install is EDITABLE
# (default `uv sync`), so the package resolves from the copied source tree at
# runtime. Provisioned config lives under config/: the screener prompt template
# (config/screener.xml) AND the deterministic scorer's tuning (config/scorer.toml).
# Both are gitignored; provision BOTH into the build context (or bind-mount them)
# before building. The scorer tuning is fail-loud, so even the baseline (and thus
# /score and the eval) requires config/scorer.toml — the screener.xml is needed
# only for deploys that escalate to the LLM.
FROM python:3.13-slim AS runtime

# uv: fast, lockfile-aware installer. Pinned by digest-free tag — bump as needed.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1

# uv.lock is committed, so the build is reproducible: --frozen installs the locked
# versions exactly and fails if the lock is stale vs pyproject.toml (run `uv lock`
# to refresh). README.md is referenced by pyproject's `readme`, so it must be
# present for the build to succeed.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
# 0.0.0.0 so the container is reachable from the inner Caddy / load balancer.
CMD ["uvicorn", "scorer.api:app", "--host", "0.0.0.0", "--port", "8000"]
