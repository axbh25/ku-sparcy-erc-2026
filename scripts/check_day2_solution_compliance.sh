#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

FAIL=0
pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAIL=1; }

SOLUTION_FILES=(
  src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py
  src/ku_sparcy_erc/ku_sparcy_erc/marker_detector.py
  src/ku_sparcy_erc/launch/solution.launch.py
)

if ./scripts/check_repo_integrity.sh >/tmp/ku_sparcy_compliance_integrity.txt 2>&1; then
  pass 'official competition paths are unchanged'
else
  fail 'repository integrity checker failed'
  cat /tmp/ku_sparcy_compliance_integrity.txt
fi

if ! grep -R -n -E \
    'ERC_SEED|expected_marker_layout|/world/erc_world|gz[[:space:]]+service|number_marker_col_' \
    "${SOLUTION_FILES[@]}" >/tmp/ku_sparcy_hidden_state_matches.txt; then
  pass 'competition solution does not read seed, Gazebo entity order, or hidden world state'
else
  fail 'competition solution contains forbidden validation/world-state references'
  cat /tmp/ku_sparcy_hidden_state_matches.txt
fi

if grep -q "'/erc/shelf_column_identification'" \
    src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py; then
  pass 'official ERC shelf-column topic is present'
else
  fail 'official ERC shelf-column topic is missing'
fi

if ! grep -E -n \
    '/arm_|/gripper_|torso_controller' \
    src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py \
    >/tmp/ku_sparcy_day2_arm_matches.txt; then
  pass 'Day 2 competition node sends no arm, gripper, or torso command'
else
  fail 'Day 2 node unexpectedly references an arm/gripper/torso controller'
  cat /tmp/ku_sparcy_day2_arm_matches.txt
fi

if grep -q 'Clock,.*\/clock\|create_subscription(.*Clock' \
    src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py || \
   grep -q "Clock, '/clock'" \
    src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py; then
  pass 'Day 2 source contains simulation-clock subscription'
else
  # Formatting across lines makes a direct grep brittle; use a Python AST/text
  # fallback that checks both imported Clock and the literal /clock.
  if grep -q 'from rosgraph_msgs.msg import Clock' \
      src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py && \
     grep -q "'/clock'" \
      src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py; then
    pass 'Day 2 source contains simulation-clock subscription'
  else
    fail 'simulation-clock subscription is missing'
  fi
fi

if [ "$FAIL" -eq 0 ]; then
  printf '[DAY2 SOLUTION COMPLIANCE][PASS]\n'
  exit 0
fi
printf '[DAY2 SOLUTION COMPLIANCE][FAIL]\n'
exit 1
