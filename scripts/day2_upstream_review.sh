#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

printf '=== KU SPARCy Day 2 upstream review ===\n'

if [ -n "$(git status --porcelain)" ]; then
  printf '[UPSTREAM REVIEW][FAIL] Working tree is not clean before fetch:\n'
  git status --short
  exit 1
fi
printf '[PASS] Working tree is clean before upstream review\n'

printf 'Fetching official upstream without merging or resetting...\n'
git fetch upstream --tags --prune

BASELINE_SHA="$(
  awk -F': ' '/^- Baseline commit:/ {print $2; exit}' OFFICIAL_BASELINE.md
)"
if ! printf '%s' "$BASELINE_SHA" | grep -Eq '^[0-9a-f]{40}$'; then
  printf '[UPSTREAM REVIEW][FAIL] Could not read the 40-character baseline SHA\n'
  exit 1
fi
UPSTREAM_SHA="$(git rev-parse upstream/main)"
CURRENT_SHA="$(git rev-parse HEAD)"

printf 'Recorded official baseline: %s\n' "$BASELINE_SHA"
printf 'Current upstream/main:      %s\n' "$UPSTREAM_SHA"
printf 'Current KU SPARCy HEAD:     %s\n' "$CURRENT_SHA"

if ! git merge-base --is-ancestor "$BASELINE_SHA" HEAD; then
  printf '[UPSTREAM REVIEW][FAIL] Recorded official baseline is not an ancestor of team HEAD\n'
  exit 1
fi
printf '[PASS] Day 1 team history still descends from the official baseline\n'

if [ "$UPSTREAM_SHA" != "$BASELINE_SHA" ]; then
  printf '\n[UPSTREAM REVIEW][STOP] Official upstream changed after our baseline.\n'
  printf 'Do NOT merge, rebase, reset, or copy files yet. Review these commits:\n'
  git log --oneline --decorate "$BASELINE_SHA..upstream/main"
  printf '\nChanged official paths:\n'
  git diff --name-status "$BASELINE_SHA..upstream/main"
  exit 2
fi

printf '[PASS] upstream/main is unchanged from the recorded official baseline\n'

ORIGIN_URL="$(git remote get-url origin)"
ORIGIN_REPO="$(
  printf '%s\n' "$ORIGIN_URL" |
  sed -E 's#^(https://github.com/|git@github.com:)##; s#\.git$##'
)"
if ! printf '%s' "$ORIGIN_REPO" | grep -Eq '^[^/]+/[^/]+$'; then
  printf '[UPSTREAM REVIEW][FAIL] Could not parse origin repository: %s\n' "$ORIGIN_URL"
  exit 1
fi
IS_PRIVATE="$(gh repo view "$ORIGIN_REPO" --json isPrivate --jq .isPrivate)"
IS_FORK="$(gh repo view "$ORIGIN_REPO" --json isFork --jq .isFork)"
if [ "$IS_PRIVATE" = true ] && [ "$IS_FORK" = false ]; then
  printf '[PASS] origin remains a private independent repository\n'
else
  printf '[UPSTREAM REVIEW][FAIL] origin private=%s fork=%s\n' \
    "$IS_PRIVATE" "$IS_FORK"
  exit 1
fi

printf '[UPSTREAM REVIEW][PASS]\n'
