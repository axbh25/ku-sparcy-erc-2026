#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
EXPECTED_DAY2='8c921d7ec228297efe683f7eb973ecb55973eaf0'

printf '=== KU SPARCy Day 3 upstream and repository review ===\n'

if [ -n "$(git status --porcelain)" ]; then
  printf '[UPSTREAM REVIEW][FAIL] Working tree is not clean before review:\n'
  git status --short
  exit 1
fi
printf '[PASS] working tree is clean before review\n'

printf 'Fetching origin and official upstream without merging, rebasing or resetting...\n'
git fetch origin main --prune
git fetch upstream --tags --prune

HEAD_SHA="$(git rev-parse HEAD)"
ORIGIN_SHA="$(git rev-parse origin/main)"
UPSTREAM_SHA="$(git rev-parse upstream/main)"
BASELINE_SHA="$(awk -F': ' '/^- Baseline commit:/ {print $2; exit}' OFFICIAL_BASELINE.md)"

printf 'Local HEAD:                %s\n' "$HEAD_SHA"
printf 'origin/main:               %s\n' "$ORIGIN_SHA"
printf 'Expected validated Day 2:  %s\n' "$EXPECTED_DAY2"
printf 'Recorded official baseline:%s\n' "$BASELINE_SHA"
printf 'upstream/main:             %s\n' "$UPSTREAM_SHA"

if [ "$HEAD_SHA" != "$ORIGIN_SHA" ]; then
  printf '[UPSTREAM REVIEW][FAIL] local HEAD and origin/main differ\n'
  exit 1
fi
printf '[PASS] local HEAD exactly matches origin/main\n'

if [ "$HEAD_SHA" != "$EXPECTED_DAY2" ]; then
  printf '[UPSTREAM REVIEW][STOP] origin/main is not the reviewed Day 2 commit.\n'
  printf 'Do not install Day 3 files. Review the unexpected team commits first:\n'
  git log --oneline --decorate --graph -8
  exit 2
fi
printf '[PASS] validated Day 2 commit is the current branch tip\n'

if ! git merge-base --is-ancestor "$BASELINE_SHA" HEAD; then
  printf '[UPSTREAM REVIEW][FAIL] official baseline is not an ancestor of team HEAD\n'
  exit 1
fi
printf '[PASS] team history still descends from the official baseline\n'

if [ "$UPSTREAM_SHA" != "$BASELINE_SHA" ]; then
  printf '[UPSTREAM REVIEW][STOP] official upstream changed after the reviewed baseline.\n'
  printf 'Do NOT merge, rebase, pull, reset, or copy official files.\n'
  printf '\nNew official commits:\n'
  git log --oneline --decorate "$BASELINE_SHA..upstream/main"
  printf '\nChanged official paths:\n'
  git diff --name-status "$BASELINE_SHA..upstream/main"
  exit 2
fi
printf '[PASS] upstream/main is unchanged from the official baseline\n'

LATEST_RELEASE="$(gh release view --repo dfl-rlab/erc_sim_2026 --json tagName --jq .tagName 2>/dev/null || true)"
if [ "$LATEST_RELEASE" != 'v1.0.3' ]; then
  printf '[UPSTREAM REVIEW][STOP] latest official release is %s, not reviewed v1.0.3\n' "${LATEST_RELEASE:-MISSING}"
  exit 2
fi
printf '[PASS] latest official release remains v1.0.3\n'

OPEN_PR_COUNT="$(gh pr list --repo dfl-rlab/erc_sim_2026 --state open --json number --jq length)"
if [ "$OPEN_PR_COUNT" -ne 0 ]; then
  printf '[UPSTREAM REVIEW][STOP] official repository has %s open pull request(s):\n' "$OPEN_PR_COUNT"
  gh pr list --repo dfl-rlab/erc_sim_2026 --state open
  exit 2
fi
printf '[PASS] official repository has no open pull requests\n'

ORIGIN_REPO="$(git remote get-url origin | sed -E 's#^(https://github.com/|git@github.com:)##; s#\.git$##')"
IS_PRIVATE="$(gh repo view "$ORIGIN_REPO" --json isPrivate --jq .isPrivate)"
IS_FORK="$(gh repo view "$ORIGIN_REPO" --json isFork --jq .isFork)"
if [ "$IS_PRIVATE" = true ] && [ "$IS_FORK" = false ]; then
  printf '[PASS] origin is private and independent (not a fork)\n'
else
  printf '[UPSTREAM REVIEW][FAIL] origin private=%s fork=%s\n' "$IS_PRIVATE" "$IS_FORK"
  exit 1
fi

printf '[DAY2 COMMIT BASELINE][PASS]\n'
printf '[UPSTREAM REVIEW][PASS]\n'
