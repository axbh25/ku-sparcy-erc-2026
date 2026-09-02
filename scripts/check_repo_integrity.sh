#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

printf '=== KU SPARCy repository integrity check ===\n'

if ! git remote get-url upstream >/dev/null 2>&1; then
  printf '[FAIL] upstream remote is missing\n'
  exit 1
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  printf '[FAIL] origin remote is missing\n'
  exit 1
fi

UPSTREAM_URL="$(git remote get-url upstream)"
ORIGIN_URL="$(git remote get-url origin)"
case "$UPSTREAM_URL" in
  *dfl-rlab/erc_sim_2026.git)
    printf '[PASS] upstream points to the official repository\n'
    ;;
  *)
    printf '[FAIL] unexpected upstream URL: %s\n' "$UPSTREAM_URL"
    exit 1
    ;;
esac
case "$ORIGIN_URL" in
  *dfl-rlab/erc_sim_2026.git)
    printf '[FAIL] origin must not point to the public official repository\n'
    exit 1
    ;;
  *)
    printf '[PASS] origin is separate from the official repository\n'
    ;;
esac

# Include committed team history, unstaged files, staged files, and untracked
# files. Day 1 did not include the cached diff; Day 2 closes that blind spot.
CHANGED="$(
  git diff --name-only upstream/main...HEAD
  git diff --name-only
  git diff --cached --name-only
  git ls-files --others --exclude-standard
)"
BAD="$(
  printf '%s\n' "$CHANGED" |
  sed '/^$/d' |
  sort -u |
  grep -Ev '^(\.gitignore|OFFICIAL_BASELINE\.md|TEAM_README\.md|scripts/|team_docs/|erc_images/|src/ku_sparcy_erc/)' || true
)"
if [ -n "$BAD" ]; then
  printf '[FAIL] Official competition files were changed:\n%s\n' "$BAD"
  exit 1
fi
printf '[PASS] No official competition file has been changed\n'
printf '[PASS] All committed, staged, unstaged, and untracked changes are confined to approved KU SPARCy paths\n'
