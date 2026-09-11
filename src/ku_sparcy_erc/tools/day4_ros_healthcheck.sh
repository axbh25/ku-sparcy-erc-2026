#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1

PASS_COUNT=0
FAIL_COUNT=0
pass() { printf '[PASS] %s\n' "$1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { printf '[FAIL] %s\n' "$1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

check_live() {
  local topic="$1"
  local type="$2"
  local timeout_sec="${3:-15}"
  if timeout "$timeout_sec" ros2 topic echo --no-daemon \
      "$topic" "$type" --once >/dev/null 2>&1; then
    pass "$topic produced a typed daemonless message"
  else
    fail "$topic did not produce $type within ${timeout_sec}s"
  fi
}

printf '=== KU SPARCy Day 4 ROS health check ===\n'

GZ_COUNT="$(pgrep -fc '[g]z sim.*erc_world' || true)"
if [ "$GZ_COUNT" -eq 1 ]; then
  pass 'exactly one Gazebo erc_world process is running'
else
  fail "expected one Gazebo erc_world process, found $GZ_COUNT"
  pgrep -af '[g]z sim.*erc_world' || true
fi

check_live /clock rosgraph_msgs/msg/Clock 10
check_live /odom nav_msgs/msg/Odometry 10
check_live /joint_states sensor_msgs/msg/JointState 12
check_live /head_front_camera/head_front_camera/color/image_raw sensor_msgs/msg/Image 18
check_live /head_front_camera/head_front_camera/depth/image_rect_raw sensor_msgs/msg/Image 18
check_live /head_front_camera/head_front_camera/color/camera_info sensor_msgs/msg/CameraInfo 18
check_live /head_front_camera/head_front_camera/depth/camera_info sensor_msgs/msg/CameraInfo 18
check_live /scan_front_raw sensor_msgs/msg/LaserScan 15
check_live /head_controller/controller_state control_msgs/msg/JointTrajectoryControllerState 15
check_live /arm_left_controller/controller_state control_msgs/msg/JointTrajectoryControllerState 15
check_live /arm_right_controller/controller_state control_msgs/msg/JointTrajectoryControllerState 15
check_live /gripper_left_controller_raw/controller_state control_msgs/msg/JointTrajectoryControllerState 15
check_live /gripper_right_controller_raw/controller_state control_msgs/msg/JointTrajectoryControllerState 15

for package in ku_sparcy_erc cv_bridge controller_manager_msgs tf2_ros ros_gz_interfaces; do
  if ros2 pkg prefix "$package" >/dev/null 2>&1; then
    pass "ROS package available: $package"
  else
    fail "ROS package missing: $package"
  fi
done

EXECUTABLES="$(ros2 pkg executables ku_sparcy_erc 2>/dev/null || true)"
for executable in opening_sequence mission_start day3_mission day4_mission staged_grasp_experiment; do
  if printf '%s\n' "$EXECUTABLES" | awk -v exe="$executable" \
      '$1 == "ku_sparcy_erc" && $2 == exe {found=1} END {exit !found}'; then
    pass "installed executable: $executable"
  else
    fail "missing executable: $executable"
  fi
done

CONTACT_TYPE="$(ros2 topic type --no-daemon /contacts 2>/dev/null || true)"
if [ "$CONTACT_TYPE" = 'ros_gz_interfaces/msg/Contacts' ]; then
  pass '/contacts has the expected explicit type'
else
  fail "/contacts type unexpected: $CONTACT_TYPE"
fi

SHOW_ARGS="$(ros2 launch ku_sparcy_erc solution.launch.py --show-args 2>&1 || true)"
for argument in shelf_column_number book_colour result_path image_output_dir \
  approach_motion_enabled day4_stop_after; do
  if printf '%s\n' "$SHOW_ARGS" | grep -Fq "$argument"; then
    pass "solution launch argument exists: $argument"
  else
    fail "solution launch argument missing: $argument"
  fi
done

GRASP_ARGS="$(ros2 launch ku_sparcy_erc staged_grasp_experiment.launch.py --show-args 2>&1 || true)"
for argument in source_result_path grasp_result_path target_stage manual_approval_required; do
  if printf '%s\n' "$GRASP_ARGS" | grep -Fq "$argument"; then
    pass "staged grasp launch argument exists: $argument"
  else
    fail "staged grasp launch argument missing: $argument"
  fi
done

if python3 - <<'PY' >/tmp/ku_sparcy_day4_imports.txt 2>&1
import cv2
import numpy
from controller_manager_msgs.srv import ListHardwareInterfaces
from ku_sparcy_erc.book_perception import SelectedColumnBookDetector
from ku_sparcy_erc.day4_mission import Day4Mission
from ku_sparcy_erc.grasp_planner import build_grasp_plan
from ku_sparcy_erc.staged_grasp_experiment import StagedGraspExperiment
print(cv2.__version__, numpy.__version__, ListHardwareInterfaces, SelectedColumnBookDetector, Day4Mission, build_grasp_plan, StagedGraspExperiment)
PY
then
  pass 'Day 4 Python dependencies import successfully'
else
  fail 'Day 4 imports failed; inspect /tmp/ku_sparcy_day4_imports.txt'
fi

printf '\nSUMMARY: %d PASS, %d FAIL\n' "$PASS_COUNT" "$FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
  printf '[DAY4 ROS HEALTH][PASS]\n'
  exit 0
fi
printf '[DAY4 ROS HEALTH][FAIL]\n'
exit 1
