#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
FAIL=0
pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAIL=1; }

DAY3_FILES=(
  src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py
  src/ku_sparcy_erc/ku_sparcy_erc/range_fusion.py
  src/ku_sparcy_erc/launch/solution.launch.py
  src/ku_sparcy_erc/config/day3_approach.yaml
)
COMPETITION_FILES=(
  src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py
  src/ku_sparcy_erc/ku_sparcy_erc/marker_detector.py
  "${DAY3_FILES[@]}"
)

if ./scripts/check_repo_integrity.sh >/tmp/ku_sparcy_day3_integrity.txt 2>&1; then
  pass 'official competition paths are unchanged'
else
  fail 'repository integrity checker failed'
  cat /tmp/ku_sparcy_day3_integrity.txt
fi

if ./scripts/check_day3_frozen_files.sh >/tmp/ku_sparcy_day3_frozen.txt 2>&1; then
  pass 'validated Day 1 and Day 2 core files are unchanged'
else
  fail 'validated Day 1 or Day 2 core file changed'
  cat /tmp/ku_sparcy_day3_frozen.txt
fi

if ! grep -R -n -E \
    'ERC_SEED|expected_marker_layout|/world/erc_world|gz[[:space:]]+service|number_marker_col_|model/[A-Za-z0-9_]+/pose' \
    "${COMPETITION_FILES[@]}" >/tmp/ku_sparcy_day3_hidden_matches.txt; then
  pass 'competition code contains no seed, entity-order, service or hidden-world oracle'
else
  fail 'competition code contains a forbidden validation/world-state reference'
  cat /tmp/ku_sparcy_day3_hidden_matches.txt
fi

DAY3_MISSION='src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py'

# Day 3 is permitted to command exactly two arm trajectory topics for the
# validated collision-safe navigation travel pose.  Grippers and torso remain
# completely outside Day 3 scope.
if grep -Fq "'/arm_left_controller/joint_trajectory'" "$DAY3_MISSION" && \
   grep -Fq "'/arm_right_controller/joint_trajectory'" "$DAY3_MISSION"; then
  pass 'validated left/right travel-arm trajectory interfaces are present'
else
  fail 'validated travel-arm trajectory interface is missing'
fi

ARM_TOPIC_MATCHES="$(
  grep -oE "'/arm_[A-Za-z0-9_/]+'" "$DAY3_MISSION" \
    | sort -u \
    || true
)"

EXPECTED_ARM_TOPICS="$(
  printf '%s\n' \
    "'/arm_left_controller/joint_trajectory'" \
    "'/arm_right_controller/joint_trajectory'" \
    | sort -u
)"

if [ "$ARM_TOPIC_MATCHES" = "$EXPECTED_ARM_TOPICS" ]; then
  pass 'Day 3 references only the approved arm trajectory topics'
else
  fail 'Day 3 references an unapproved arm topic'
  printf '%s\n' "$ARM_TOPIC_MATCHES"
fi

if grep -Fq "TRAVEL_ARM_POSE_NAME = 'pal_spherical_home_arm_only'" \
      "$DAY3_MISSION" && \
   grep -Fq 'TRAVEL_ARM_LEFT_TARGET_RAD' "$DAY3_MISSION" && \
   grep -Fq 'TRAVEL_ARM_RIGHT_TARGET_RAD' "$DAY3_MISSION"; then
  pass 'validated PAL arm-only home travel pose is explicit in source'
else
  fail 'validated travel-arm pose declaration is missing'
fi

if ! grep -E -n \
    '/gripper_|gripper_controller|/torso_|torso_controller' \
    "${DAY3_FILES[@]}" \
    >/tmp/ku_sparcy_day3_forbidden_manipulation_matches.txt; then
  pass 'Day 3 sends no gripper or torso command'
else
  fail 'Day 3 unexpectedly references a gripper or torso controller'
  cat /tmp/ku_sparcy_day3_forbidden_manipulation_matches.txt
fi

if grep -Fq "'/erc/shelf_column_identification'" \
    src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py; then
  pass 'official ERC shelf-column output remains present'
else
  fail 'official ERC shelf-column output is missing'
fi

for literal in \
  '/head_front_camera/head_front_camera/color/camera_info' \
  '/head_front_camera/head_front_camera/depth/image_rect_raw' \
  '/head_front_camera/head_front_camera/depth/camera_info' \
  '/scan_front_raw' \
  'base_footprint'; do
  if grep -Fq "$literal" src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py; then
    pass "Day 3 source contains required live interface: $literal"
  else
    fail "Day 3 source is missing required live interface: $literal"
  fi
done

if ! grep -Fq '/depth/color/points' \
    src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py; then
  pass 'Day 3 uses raw depth and does not depend on reconstructed point cloud'
else
  fail 'Day 3 unexpectedly depends on reconstructed point cloud'
fi

if grep -Fq 'from ku_sparcy_erc.mission_start import MissionStart' \
      src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py && \
   grep -Fq 'class Day3Mission(MissionStart)' \
      src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py; then
  pass 'Day 3 extends the actual committed Day 2 mission node'
else
  fail 'Day 3 does not visibly inherit the Day 2 mission node'
fi

if grep -Fq 'tf2_ros' src/ku_sparcy_erc/package.xml && \
   grep -Fq "'day3_mission = ku_sparcy_erc.day3_mission:main'" \
      src/ku_sparcy_erc/setup.py; then
  pass 'Day 3 runtime dependency and executable are declared'
else
  fail 'Day 3 package metadata is incomplete'
fi

for argument in \
  shelf_column_number \
  book_colour \
  approach_motion_enabled \
  approach_distance_limit_m \
  approach_max_forward_speed_mps \
  approach_standoff_m; do
  if grep -Fq "'$argument'" src/ku_sparcy_erc/launch/solution.launch.py; then
    pass "solution launch declares $argument"
  else
    fail "solution launch is missing $argument"
  fi
done

if grep -Fq 'Clock' src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py && \
   grep -Fq "'/clock'" src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py && \
   grep -Fq 'clock_stall_wall_timeout_sec' \
      src/ku_sparcy_erc/ku_sparcy_erc/day3_mission.py; then
  pass 'simulation-time control and independent wall stall watchdog are present'
else
  fail 'simulation-time or wall-stall protection is missing'
fi

if [ "$FAIL" -eq 0 ]; then
  printf '[DAY3 SOLUTION COMPLIANCE][PASS]\n'
  exit 0
fi
printf '[DAY3 SOLUTION COMPLIANCE][FAIL]\n'
exit 1
