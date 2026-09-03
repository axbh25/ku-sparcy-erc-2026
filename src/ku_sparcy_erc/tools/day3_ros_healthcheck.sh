#!/usr/bin/env bash
# Daemon-resistant Day 3 ROS/sensor health check. Run inside erc_sim.
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

live_topic() {
  local topic="$1"
  local type="$2"
  local timeout_seconds="${3:-15}"
  if timeout "$timeout_seconds" ros2 topic echo --no-daemon \
      "$topic" "$type" --once >/dev/null 2>&1; then
    pass "$topic produced a typed daemonless sample"
  else
    fail "$topic did not produce $type within ${timeout_seconds}s"
  fi
}

check_type() {
  local topic="$1"
  local expected="$2"
  local actual
  actual="$(ros2 topic type --no-daemon "$topic" 2>/dev/null || true)"
  if [ "$actual" = "$expected" ]; then
    pass "$topic has type $expected"
  else
    fail "$topic expected $expected but got ${actual:-MISSING}"
  fi
}

printf '=== KU SPARCy Day 3 ROS/range health check ===\n'
printf 'ROS_DOMAIN_ID=%s\n' "${ROS_DOMAIN_ID:-UNSET}"

GZ_COUNT="$(pgrep -fc '[g]z sim.*erc_world' || true)"
if [ "$GZ_COUNT" -eq 1 ]; then
  pass 'exactly one Gazebo erc_world process is running'
else
  fail "expected one Gazebo erc_world process, found $GZ_COUNT"
  pgrep -af '[g]z sim.*erc_world' || true
fi

# joint_state_broadcaster is validated through its actual output. Every
# trajectory controller is validated through its typed controller_state stream,
# avoiding the poisoned ROS CLI daemon and the old output parser entirely.
live_topic /joint_states sensor_msgs/msg/JointState 15
for controller in \
  arm_left_controller \
  arm_right_controller \
  head_controller \
  torso_controller \
  gripper_left_controller_raw \
  gripper_right_controller_raw; do
  live_topic \
    "/${controller}/controller_state" \
    control_msgs/msg/JointTrajectoryControllerState \
    15
done

check_type /cmd_vel geometry_msgs/msg/Twist
check_type /odom nav_msgs/msg/Odometry
check_type /head_front_camera/head_front_camera/color/image_raw sensor_msgs/msg/Image
check_type /head_front_camera/head_front_camera/color/camera_info sensor_msgs/msg/CameraInfo
check_type /head_front_camera/head_front_camera/depth/image_rect_raw sensor_msgs/msg/Image
check_type /head_front_camera/head_front_camera/depth/camera_info sensor_msgs/msg/CameraInfo
check_type /scan_front_raw sensor_msgs/msg/LaserScan

live_topic /clock rosgraph_msgs/msg/Clock 10
live_topic /odom nav_msgs/msg/Odometry 15
live_topic \
  /head_front_camera/head_front_camera/color/image_raw \
  sensor_msgs/msg/Image 20
live_topic \
  /head_front_camera/head_front_camera/color/camera_info \
  sensor_msgs/msg/CameraInfo 20
live_topic \
  /head_front_camera/head_front_camera/depth/image_rect_raw \
  sensor_msgs/msg/Image 20
live_topic \
  /head_front_camera/head_front_camera/depth/camera_info \
  sensor_msgs/msg/CameraInfo 20
live_topic /scan_front_raw sensor_msgs/msg/LaserScan 15

EXECUTABLES="$(ros2 pkg executables ku_sparcy_erc 2>/dev/null || true)"
for executable in opening_sequence mission_start day3_mission; do
  if printf '%s\n' "$EXECUTABLES" | awk -v exe="$executable" \
      '$1 == "ku_sparcy_erc" && $2 == exe {found=1} END {exit !found}'; then
    pass "ku_sparcy_erc executable installed: $executable"
  else
    fail "ku_sparcy_erc executable missing: $executable"
  fi
done

if PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY' >/tmp/ku_sparcy_day3_imports.txt 2>&1
import cv2
import numpy
import tf2_ros
from cv_bridge import CvBridge
from ku_sparcy_erc.marker_detector import ShelfMarkerDetector
from ku_sparcy_erc.range_fusion import CameraModel
from ku_sparcy_erc.day3_mission import Day3Mission
print('OpenCV', cv2.__version__)
print('NumPy', numpy.__version__)
print('TF2', tf2_ros.Buffer)
print('CvBridge', CvBridge)
print('Detector', ShelfMarkerDetector)
print('CameraModel', CameraModel)
print('Day3Mission', Day3Mission)
PY
then
  pass 'Day 2 perception and Day 3 range/control modules import successfully'
else
  fail 'Python imports failed; inspect /tmp/ku_sparcy_day3_imports.txt'
fi

printf '\nSUMMARY: %d PASS, %d FAIL\n' "$PASS_COUNT" "$FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
  printf '[DAY3 ROS HEALTH][PASS]\n'
  exit 0
fi
printf '[DAY3 ROS HEALTH][FAIL]\n'
exit 1
