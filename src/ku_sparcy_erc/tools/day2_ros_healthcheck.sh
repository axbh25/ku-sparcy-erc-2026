#!/usr/bin/env bash
set -u

export PYTHONDONTWRITEBYTECODE=1
PASS_COUNT=0
FAIL_COUNT=0

pass() {
  printf '[PASS] %s\n' "$1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  printf '[FAIL] %s\n' "$1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

printf '=== KU SPARCy Day 2 ROS/perception health check ===\n'

if ros2 node list 2>/dev/null | grep -q .; then
  pass 'ROS 2 graph is visible'
else
  fail 'ROS 2 graph is empty'
fi

# Preserve the corrected Day 1 parser: controller identity is field 1 and
# active/inactive state is the final field. ANSI terminal escapes are removed.
CONTROLLERS="$(
  ros2 control list_controllers 2>&1 |
  sed -E $'s/\x1B\\[[0-9;?]*[ -\\/]*[@-~]//g' ||
  true
)"
for controller in \
  joint_state_broadcaster \
  arm_left_controller \
  arm_right_controller \
  head_controller \
  torso_controller \
  gripper_left_controller_raw \
  gripper_right_controller_raw; do
  if printf '%s\n' "$CONTROLLERS" | \
      awk -v name="$controller" \
      '$1 == name && $NF == "active" { found=1 } END { exit !found }'; then
    pass "$controller is active"
  else
    fail "$controller is not active"
  fi
done

if timeout 15 ros2 topic echo --once \
    /head_front_camera/head_front_camera/color/image_raw \
    >/tmp/ku_sparcy_day2_camera_sample.txt 2>&1; then
  pass 'RGB camera produced a live frame'
else
  fail 'RGB camera did not produce a frame within 15 wall seconds'
fi

CAMERA_TYPE="$(ros2 topic type \
  /head_front_camera/head_front_camera/color/image_raw 2>/dev/null || true)"
if [ "$CAMERA_TYPE" = 'sensor_msgs/msg/Image' ]; then
  pass 'RGB camera topic has sensor_msgs/msg/Image type'
else
  fail "RGB camera topic type was '${CAMERA_TYPE:-MISSING}'"
fi

for package in ku_sparcy_erc erc_description cv_bridge; do
  if ros2 pkg prefix "$package" >/dev/null 2>&1; then
    pass "ROS package available: $package"
  else
    fail "ROS package missing: $package"
  fi
done

EXECUTABLES="$(ros2 pkg executables ku_sparcy_erc 2>/dev/null || true)"
for executable in opening_sequence mission_start; do
  if printf '%s\n' "$EXECUTABLES" | \
      awk -v exe="$executable" '$1 == "ku_sparcy_erc" && $2 == exe {found=1} END {exit !found}'; then
    pass "ku_sparcy_erc executable installed: $executable"
  else
    fail "ku_sparcy_erc executable missing: $executable"
  fi
done

if python3 - <<'PY' >/tmp/ku_sparcy_day2_imports.txt 2>&1
import cv2
import numpy
from cv_bridge import CvBridge
from ku_sparcy_erc.marker_detector import ShelfMarkerDetector
print('OpenCV', cv2.__version__)
print('NumPy', numpy.__version__)
print('CvBridge', CvBridge)
print('Detector', ShelfMarkerDetector)
PY
then
  pass 'OpenCV, NumPy, cv_bridge and marker_detector import successfully'
else
  fail 'Python perception imports failed; inspect /tmp/ku_sparcy_day2_imports.txt'
fi

MARKER_DIR="$(ros2 pkg prefix erc_description 2>/dev/null)/share/erc_description/models/number_marker/textures"
ASSET_FAILURES=0
for digit in 1 2 3 4 5; do
  if [ -s "$MARKER_DIR/$digit.png" ]; then
    pass "official marker asset exists: $digit.png"
  else
    fail "official marker asset missing: $MARKER_DIR/$digit.png"
    ASSET_FAILURES=$((ASSET_FAILURES + 1))
  fi
done

SHOW_ARGS="$(ros2 launch ku_sparcy_erc solution.launch.py --show-args 2>&1 || true)"
for argument in \
  shelf_column_number \
  book_colour \
  phase1_fast_start \
  enable_motion \
  result_path \
  image_output_dir \
  validation_require_all_markers; do
  if printf '%s\n' "$SHOW_ARGS" | grep -q "$argument"; then
    pass "solution launch argument exists: $argument"
  else
    fail "solution launch argument missing: $argument"
  fi
done

printf '\nSUMMARY: %d PASS, %d FAIL\n' "$PASS_COUNT" "$FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
  printf '[DAY2 ROS HEALTH][PASS]\n'
  exit 0
fi
printf '[DAY2 ROS HEALTH][FAIL]\n'
exit 1
