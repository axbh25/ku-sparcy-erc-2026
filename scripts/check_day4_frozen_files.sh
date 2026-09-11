#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASE='9a9768acb824862f2dd4b8de9950ffed54828987'

DAY1=(
  src/ku_sparcy_erc/ku_sparcy_erc/opening_sequence.py
  src/ku_sparcy_erc/config/opening_sequence.yaml
  src/ku_sparcy_erc/tools/day1_ros_healthcheck.sh
  src/ku_sparcy_erc/tools/validate_day1_result.py
)
DAY2=(
  src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py
  src/ku_sparcy_erc/ku_sparcy_erc/marker_detector.py
  src/ku_sparcy_erc/config/day2_perception.yaml
  src/ku_sparcy_erc/tools/day2_ros_healthcheck.sh
  src/ku_sparcy_erc/tools/validate_day2_result.py
  src/ku_sparcy_erc/tools/run_day2_matrix.sh
  src/ku_sparcy_erc/tools/expected_marker_layout.py
  src/ku_sparcy_erc/tools/set_robot_orientation.sh
)
DAY3=(
  src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py
  src/ku_sparcy_erc/ku_sparcy_erc/range_fusion.py
  src/ku_sparcy_erc/config/day3_approach.yaml
  src/ku_sparcy_erc/launch/day2_regression.launch.py
  src/ku_sparcy_erc/tools/test_day3_geometry.py
  src/ku_sparcy_erc/tools/day3_interface_probe.py
  src/ku_sparcy_erc/tools/day3_ros_healthcheck.sh
  src/ku_sparcy_erc/tools/validate_day3_result.py
  src/ku_sparcy_erc/tools/clean_day3_sim.sh
  src/ku_sparcy_erc/tools/run_day3_small_regression.sh
  src/ku_sparcy_erc/tools/summarize_day3_results.py
)

check_group() {
  local label="$1"
  shift
  local existing=()
  for path in "$@"; do
    [ -e "$path" ] && existing+=("$path")
  done
  if [ "${#existing[@]}" -eq 0 ]; then
    echo "[$label][FAIL] no expected files exist"
    exit 1
  fi
  if git diff --quiet "$BASE" -- "${existing[@]}" && \
     git diff --cached --quiet "$BASE" -- "${existing[@]}"; then
    echo "[$label][PASS]"
  else
    echo "[$label][FAIL]"
    git diff "$BASE" -- "${existing[@]}" || true
    exit 1
  fi
}

check_group 'DAY1 FROZEN FILES' "${DAY1[@]}"
check_group 'DAY2 FROZEN FILES' "${DAY2[@]}"
check_group 'DAY3 FROZEN FILES' "${DAY3[@]}"
