#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
DAY4="4deeba6a69c98bb4be953439389e50fbc2e73494"
GRIPPER_MERGE="93554d4f9335b2ee3acb49c6b332611f6ad2a964"
REVIEWED_UPSTREAM="68e175fada5b8115b0f7bbb0e9b2b08a96504f22"

[ -z "$(git status --porcelain)" ] || {
  echo '[DAY567 BASELINE][FAIL] working tree is not clean'
  git status --short
  exit 1
}

git fetch origin --prune
git fetch upstream --tags --prune

LOCAL="$(git rev-parse HEAD)"
ORIGIN="$(git rev-parse origin/main)"
UPSTREAM="$(git rev-parse upstream/main)"

printf 'local:         %s\n' "$LOCAL"
printf 'origin/main:   %s\n' "$ORIGIN"
printf 'upstream/main: %s\n' "$UPSTREAM"

[ "$LOCAL" = "$DAY4" ] || {
  echo '[DAY567 BASELINE][FAIL] local HEAD is not validated Day 4'
  exit 1
}
[ "$ORIGIN" = "$DAY4" ] || {
  echo '[DAY567 BASELINE][FAIL] origin/main is not validated Day 4'
  exit 1
}
git merge-base --is-ancestor "$GRIPPER_MERGE" HEAD || {
  echo '[DAY567 BASELINE][FAIL] official gripper merge missing from team history'
  exit 1
}
git merge-base --is-ancestor "$GRIPPER_MERGE" upstream/main || {
  echo '[UPSTREAM REVIEW][FAIL] official gripper merge missing upstream'
  exit 1
}

CHANGED_AFTER_GRIPPER="$(git diff --name-only "$GRIPPER_MERGE..upstream/main" | sort -u)"
if [ "$UPSTREAM" != "$REVIEWED_UPSTREAM" ]; then
  echo '[UPSTREAM REVIEW][STOP] upstream/main advanced after this kit was reviewed'
  git log --oneline "$REVIEWED_UPSTREAM..upstream/main" || true
  git diff --name-status "$REVIEWED_UPSTREAM..upstream/main" || true
  exit 2
fi
if [ "$CHANGED_AFTER_GRIPPER" != 'README.md' ]; then
  echo '[UPSTREAM REVIEW][STOP] post-gripper changes are not README-only'
  printf '%s\n' "$CHANGED_AFTER_GRIPPER"
  exit 2
fi

echo '[PASS] validated Day 4 is local and on origin/main'
echo '[PASS] official gripper merge remains in both histories'
echo '[PASS] upstream changes after gripper merge are README-only'
echo '[UPSTREAM REVIEW][PASS]'
