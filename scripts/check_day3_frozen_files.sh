#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
DAY2_SHA='8c921d7ec228297efe683f7eb973ecb55973eaf0'

FROZEN=(
  src/ku_sparcy_erc/ku_sparcy_erc/opening_sequence.py
  src/ku_sparcy_erc/config/opening_sequence.yaml
  src/ku_sparcy_erc/tools/day1_ros_healthcheck.sh
  src/ku_sparcy_erc/tools/validate_day1_result.py
  src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py
  src/ku_sparcy_erc/ku_sparcy_erc/marker_detector.py
  src/ku_sparcy_erc/config/day2_perception.yaml
  src/ku_sparcy_erc/tools/day2_ros_healthcheck.sh
  src/ku_sparcy_erc/tools/validate_day2_result.py
  src/ku_sparcy_erc/tools/run_day2_matrix.sh
  src/ku_sparcy_erc/tools/expected_marker_layout.py
  src/ku_sparcy_erc/tools/set_robot_orientation.sh
)

if ! git cat-file -e "$DAY2_SHA^{commit}" 2>/dev/null; then
  printf '[FROZEN FILES][FAIL] Day 2 commit object is missing\n'
  exit 1
fi

if git diff --quiet "$DAY2_SHA" -- "${FROZEN[@]}" && \
   git diff --cached --quiet "$DAY2_SHA" -- "${FROZEN[@]}"; then
  printf '[DAY1 FROZEN FILES][PASS]\n'
  printf '[DAY2 FROZEN FILES][PASS]\n'
  exit 0
fi
printf '[FROZEN FILES][FAIL] a validated Day 1/Day 2 core file changed:\n'
git diff --name-status "$DAY2_SHA" -- "${FROZEN[@]}"
git diff --cached --name-status "$DAY2_SHA" -- "${FROZEN[@]}"
exit 1
