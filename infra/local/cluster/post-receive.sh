#!/usr/bin/env bash
#
# Bare-repo post-receive hook for the cluster deploy.
#
# Lives canonically here (version-controlled). The active copy on the cluster host is
# ~/kairos-llm-scorer.git/hooks/post-receive — keep them in sync. Updating the host:
#
#   scp infra/local/cluster/post-receive.sh cluster-host:~/kairos-llm-scorer.git/hooks/post-receive
#   ssh cluster-host 'chmod +x ~/kairos-llm-scorer.git/hooks/post-receive'
#
# One-time host setup (see infra/README.md):
#   git init --bare ~/kairos-llm-scorer.git && install this hook
#   docker network create edge   # if not already created by another project
#   drop the IP screener.xml + a .env (ANTHROPIC_API_KEY, KAIROS_SCORER_API_KEYS)
#     into ~/kairos-llm-scorer once — `checkout -f` won't delete them.
#
# Flow on each push to TARGET:
#   1. checkout the branch into $WORK
#   2. docker compose build
#   3. docker compose up -d --wait   (block until the /health healthcheck passes)
#   4. deliver the edge route fragment + reload the shared edge

set -euo pipefail
# Branch this host deploys. Each host tracks ONE environment — set
# SCORER_DEPLOY_BRANCH to prd | stg | lab in the hook's environment (default stg).
# A push to any other branch is ignored (logged, no deploy).
TARGET="${SCORER_DEPLOY_BRANCH:-stg}"
WORK=$HOME/kairos-llm-scorer
GIT_DIR=$HOME/kairos-llm-scorer.git
LOG=$HOME/kairos-llm-scorer-deploy.log
# Cluster topology compose file (relative to $WORK after checkout).
COMPOSE=infra/local/cluster/docker-compose.yml

# Mirror all output to the on-disk deploy log AND stream it back to the pushing
# client (git forwards a hook's stderr to `git push` as `remote: …` lines).
exec > >(tee -a "$LOG" >&2) 2>&1
echo
echo "=== $(date -Iseconds) post-receive ==="

while read -r _old new ref; do
  branch=${ref#refs/heads/}
  if [ "$branch" != "$TARGET" ]; then
    echo ">> ignoring push to $branch (only $TARGET triggers deploy)"
    continue
  fi
  echo ">> deploying $branch ($new)"
  mkdir -p "$WORK"
  git --git-dir="$GIT_DIR" --work-tree="$WORK" checkout -f "$branch"
  cd "$WORK"

  echo ">> building image"
  docker compose -f "$COMPOSE" build

  echo ">> starting stack (waiting on /health)"
  docker compose -f "$COMPOSE" up -d --wait

  # Deliver this project's route fragment to the shared edge and reload it.
  if [ -d "$HOME/edge/conf.d" ]; then
    echo ">> updating shared edge fragment"
    cp infra/local/cluster/edge/kairos-llm-scorer.caddy "$HOME/edge/conf.d/kairos-llm-scorer.caddy"
    docker exec edge caddy reload --config /etc/caddy/Caddyfile \
      || echo ">> WARN: edge reload failed (is the edge up?)"
  fi

  docker image prune -f >/dev/null 2>&1 || true
  echo ">> deploy complete"
done
