#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OFFICIAL_GRIPPER_MERGE="93554d4f9335b2ee3acb49c6b332611f6ad2a964"
OFFICIAL_GRIPPER_FIX="4e0ce63b22e205392e33e732d9dbcf66ce36eaeb"
CURRENT_REVIEWED_MAIN="b1f9b05e20f4750b59f3321d88f172cc4dbb1386"

printf '=== Day 4 continuation upstream review ===\n'

case "$(git remote get-url upstream 2>/dev/null || true)" in
  *dfl-rlab/erc_sim_2026.git)
    printf '[PASS] upstream points to dfl-rlab/erc_sim_2026\n'
    ;;
  *)
    printf '[UPSTREAM CONTINUATION REVIEW][FAIL] unexpected upstream remote\n' >&2
    exit 1
    ;;
esac

git fetch upstream main --tags --prune
UPSTREAM_SHA="$(git rev-parse upstream/main)"
printf 'upstream/main: %s\n' "$UPSTREAM_SHA"

if ! git merge-base --is-ancestor "$OFFICIAL_GRIPPER_MERGE" upstream/main; then
  printf '[UPSTREAM CONTINUATION REVIEW][STOP] 93554d4 is no longer an ancestor of upstream/main\n' >&2
  exit 1
fi
printf '[PASS] official gripper merge 93554d4 remains in upstream/main\n'

if ! git merge-base --is-ancestor "$OFFICIAL_GRIPPER_FIX" upstream/main; then
  printf '[UPSTREAM CONTINUATION REVIEW][STOP] 4e0ce63 is no longer an ancestor of upstream/main\n' >&2
  exit 1
fi
printf '[PASS] corrected gripper fix 4e0ce63 remains in upstream/main\n'

POST_MERGE_PATHS="$(
  git diff --name-only "$OFFICIAL_GRIPPER_MERGE..upstream/main" |
  sed '/^$/d' |
  sort -u
)"

if [ -z "$POST_MERGE_PATHS" ]; then
  printf '[PASS] no official changes after 93554d4\n'
elif [ "$POST_MERGE_PATHS" = 'README.md' ]; then
  printf '[PASS] changes after 93554d4 are documentation-only: README.md\n'
else
  printf '[UPSTREAM CONTINUATION REVIEW][STOP] runtime-affecting official paths changed after 93554d4:\n%s\n' "$POST_MERGE_PATHS" >&2
  git log --oneline "$OFFICIAL_GRIPPER_MERGE..upstream/main" >&2
  exit 1
fi

if [ "$UPSTREAM_SHA" = "$CURRENT_REVIEWED_MAIN" ]; then
  printf '[PASS] upstream/main matches reviewed README-only tip b1f9b05\n'
else
  printf '[NOTICE] upstream/main differs from reviewed tip b1f9b05, but only README.md changed after 93554d4\n'
fi

if git diff --quiet "$OFFICIAL_GRIPPER_MERGE" upstream/main -- \
    src/erc_description/models/book/sdf/erc_book.sdf \
    src/erc_description/urdf/tiago_pro.urdf; then
  printf '[PASS] official book and TIAGo gripper assets are unchanged since 93554d4\n'
else
  printf '[UPSTREAM CONTINUATION REVIEW][STOP] official runtime gripper assets changed after 93554d4\n' >&2
  exit 1
fi

printf '[NOTICE] This continuation check performs no merge, rebase, reset, checkout, or file restore.\n'
printf '[UPSTREAM CONTINUATION REVIEW][PASS]\n'
