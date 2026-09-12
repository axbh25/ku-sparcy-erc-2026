#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASE="4deeba6a69c98bb4be953439389e50fbc2e73494"
FILES=(
  src/ku_sparcy_erc/ku_sparcy_erc/opening_sequence.py
  src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py
  src/ku_sparcy_erc/ku_sparcy_erc/marker_detector.py
  src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py
  src/ku_sparcy_erc/ku_sparcy_erc/range_fusion.py
  src/ku_sparcy_erc/ku_sparcy_erc/day4_mission.py
  src/ku_sparcy_erc/ku_sparcy_erc/book_perception.py
  src/ku_sparcy_erc/ku_sparcy_erc/passive_book_mapping.py
  src/ku_sparcy_erc/ku_sparcy_erc/urdf_kinematics.py
  src/ku_sparcy_erc/ku_sparcy_erc/staged_grasp_experiment.py
  src/ku_sparcy_erc/config/opening_sequence.yaml
  src/ku_sparcy_erc/config/day2_perception.yaml
  src/ku_sparcy_erc/config/day3_approach.yaml
  src/ku_sparcy_erc/config/day4_books.yaml
  src/ku_sparcy_erc/config/day4_grasp.yaml
  src/ku_sparcy_erc/tools/validate_day1_result.py
  src/ku_sparcy_erc/tools/validate_day2_result.py
  src/ku_sparcy_erc/tools/validate_day3_result.py
  src/ku_sparcy_erc/tools/validate_day4_result.py
  src/ku_sparcy_erc/tools/validate_day4_grasp_result.py
)
PLANNER="src/ku_sparcy_erc/ku_sparcy_erc/grasp_planner.py"
APPROVED_DAY5_PLANNER_BLOB="cda4b5c593d635fbe593d857c6754c65e74b3a7b"

FAIL=0

if git diff --quiet "$BASE" -- "${FILES[@]}"; then
  echo '[DAY1-4 FROZEN CORE][PASS]'
else
  echo '[DAY1-4 FROZEN CORE][FAIL]'
  git diff --name-status "$BASE" -- "${FILES[@]}"
  FAIL=1
fi

ACTUAL_PLANNER_BLOB="$(git hash-object "$PLANNER")"

if [ "$ACTUAL_PLANNER_BLOB" = "$APPROVED_DAY5_PLANNER_BLOB" ]; then
  echo '[DAY5 APPROVED GRASP PLANNER][PASS]'
else
  echo '[DAY5 APPROVED GRASP PLANNER][FAIL]'
  printf 'expected: %s\nactual:   %s\n' \
    "$APPROVED_DAY5_PLANNER_BLOB" \
    "$ACTUAL_PLANNER_BLOB"
  FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
  echo '[DAY1-4 FROZEN FILES][PASS]'
else
  echo '[DAY1-4 FROZEN FILES][FAIL]'
  exit 1
fi
