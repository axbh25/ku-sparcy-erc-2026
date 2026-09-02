#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
MODE="${1:-check}"
if [ "$MODE" != check ] && [ "$MODE" != --restore ]; then
  printf 'Usage: %s [--restore]\n' "$0" >&2
  exit 2
fi

mapfile -t TRACKED_PYC < <(
  git ls-files |
  grep -E '^src/.*/__pycache__/.*\.pyc$' |
  grep -Ev '^src/ku_sparcy_erc/' || true
)

if [ "${#TRACKED_PYC[@]}" -eq 0 ]; then
  printf '[OFFICIAL BYTECODE][PASS] No official tracked .pyc files found\n'
  exit 0
fi

CHANGED=()
for path in "${TRACKED_PYC[@]}"; do
  if ! git diff --quiet upstream/main -- "$path" || \
     ! git diff --cached --quiet upstream/main -- "$path"; then
    CHANGED+=("$path")
  fi
done

if [ "${#CHANGED[@]}" -eq 0 ]; then
  printf '[OFFICIAL BYTECODE][PASS] %d tracked official .pyc files match upstream/main\n' \
    "${#TRACKED_PYC[@]}"
  exit 0
fi

printf '[OFFICIAL BYTECODE][FAIL] Modified tracked official bytecode:\n'
printf '  %s\n' "${CHANGED[@]}"

if [ "$MODE" = --restore ]; then
  printf 'Restoring ONLY the generated tracked .pyc paths listed above...\n'
  git restore --source=upstream/main --staged --worktree -- "${CHANGED[@]}"
  printf '[OFFICIAL BYTECODE][RESTORED]\n'
  exit 0
fi

printf 'Run exactly this guarded command to restore only those files:\n'
printf '  ./scripts/check_tracked_official_bytecode.sh --restore\n'
exit 1
