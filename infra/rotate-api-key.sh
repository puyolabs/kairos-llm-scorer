#!/usr/bin/env bash
# Rotate KAIROS_SCORER_API_KEYS: generate a fresh key, flush the old value out of
# the target env file, and write the new one in — without ever printing the secret.
# Only a non-sensitive fingerprint (prefix + SHA-256) is shown so you can confirm
# a match (e.g. against a deployed host) without exposing the key.
#
# Usage:
#   infra/rotate-api-key.sh [ENV_FILE]   # default: ./.env
set -euo pipefail

ENV_FILE="${1:-.env}"
VAR="KAIROS_SCORER_API_KEYS"

[ -f "$ENV_FILE" ] || { echo "error: env file not found: $ENV_FILE" >&2; exit 1; }

# 43 url-safe base64 chars (32 bytes of entropy), prefixed for provenance.
key="ksk_live_$(openssl rand -base64 32 | tr '+/' '-_' | tr -d '=' )"

# Flush the old line (if present) and write the new one — atomic via temp file.
tmp="$(mktemp)"
if grep -q "^${VAR}=" "$ENV_FILE"; then
  grep -v "^${VAR}=" "$ENV_FILE" > "$tmp"
else
  cat "$ENV_FILE" > "$tmp"
fi
printf '%s=%s\n' "$VAR" "$key" >> "$tmp"
cat "$tmp" > "$ENV_FILE"   # preserve original file perms/inode
rm -f "$tmp"

fp="$(printf '%s' "$key" | shasum -a 256 | cut -c1-8)"
echo "rotated ${VAR} in ${ENV_FILE}  | length=${#key}  | prefix=${key:0:9}…  | sha256=${fp}"
