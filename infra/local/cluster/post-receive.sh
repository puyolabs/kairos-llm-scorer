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
#   drop the IP screener.xml + a .env (ANTHROPIC_API_KEY, KAIROS_SCORER_API_KEYS)
#     into ~/kairos-llm-scorer once — `checkout -f` won't delete them.
#
# The scorer is a PURELY INTERNAL service. It joins the dedicated
# private `kairos-scorer` network (OWNED by kairos — the owner creates it; this repo
# is a MEMBER and neither creates the network nor delivers any Janus fragment) and
# gets NO Janus ingress. This hook therefore neither creates a network nor installs a
# route fragment — and it REMOVES any stale scorer fragments left on the Janus host
# by an earlier edge-served topology.
#
# Flow on each push to TARGET:
#   1. checkout the branch into $WORK
#   2. docker compose build
#   3. docker compose up -d --wait   (block until the /health healthcheck passes)
#   4. remove any stale scorer Janus fragments + reload the Janus gateway

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

  # --env-file .env: the scorer's secret-bearing vars (ANTHROPIC_API_KEY,
  # KAIROS_SCORER_API_KEYS) are interpolated in the INCLUDED base.yml. Compose's
  # auto-.env loading keys off the compose file's dir AND scopes per included
  # file, so it never finds the repo-root .env. Passing it explicitly puts the
  # values in the process env, which every included file's interpolation sees.
  echo ">> building image"
  docker compose --env-file .env -f "$COMPOSE" build

  # --remove-orphans clears containers for services no longer in the compose file
  # — e.g. the retired inner `nginx`/`caddy` proxy, which would otherwise linger
  # holding a stale network alias.
  echo ">> starting stack (waiting on /health)"
  docker compose --env-file .env -f "$COMPOSE" up -d --wait --remove-orphans

  # The scorer has NO Janus ingress. It is reached only by the kairos worker over
  # the private `kairos-scorer` network. This hook installs NO route fragment;
  # instead it REMOVES any stale scorer fragments a previous edge-served deploy
  # left on the Janus host, then reloads nginx via /reload-janus.sh (re-render +
  # `nginx -t` + reload — a bad config fails the test and never goes hot, so the
  # running config survives).
  if [ -d "$HOME/janus/conf.d" ]; then
    echo ">> removing stale scorer Janus fragments (scorer has no edge ingress)"
    rm -f "$HOME/janus/conf.d/kairos-llm-scorer.conf" \
          "$HOME/janus/conf.d/kairos-llm-scorer.caddy" \
          "$HOME/janus/http.d/kairos-llm-scorer.zones.conf"
    docker exec janus /reload-janus.sh \
      || echo ">> WARN: Janus reload failed (is Janus up?)"
  fi

  docker image prune -f >/dev/null 2>&1 || true
  echo ">> deploy complete"
done
