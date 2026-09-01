#!/usr/bin/env bash
set -u

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

check_topic_type() {
  local topic="$1"
  local expected="$2"
  local actual
  actual="$(ros2 topic type "$topic" 2>/dev/null || true)"
  if printf '%s\n' "$actual" | grep -Fxq "$expected"; then
    pass "$topic has type $expected"
  else
    fail "$topic type expected '$expected' but got '${actual:-MISSING}'"
  fi
}

check_live_topic() {
  local topic="$1"
  local timeout_seconds="${2:-12}"
  if timeout "$timeout_seconds" ros2 topic echo --once "$topic" >/dev/null 2>&1; then
    pass "$topic produced a live message"
  else
    fail "$topic did not produce a message within ${timeout_seconds}s"
  fi
}

printf '=== KU SPARCy Day 1 ROS health check ===\n'
printf 'ROS_DOMAIN_ID=%s\n' "${ROS_DOMAIN_ID:-UNSET}"

if ros2 node list 2>/dev/null | grep -q .; then
  pass 'ROS 2 graph is visible'
else
  fail 'ROS 2 graph is empty'
fi

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
  if printf '%s\n' "$CONTROLLERS" | awk -v name="$controller" '$1 == name && $NF == "active" { found=1 } END { exit !found }'; then
    pass "$controller is active"
  else
    fail "$controller is not active"
  fi
done

check_topic_type /clock rosgraph_msgs/msg/Clock
check_topic_type /cmd_vel geometry_msgs/msg/Twist
check_topic_type /odom nav_msgs/msg/Odometry
check_topic_type /joint_states sensor_msgs/msg/JointState
check_topic_type /scan_front_raw sensor_msgs/msg/LaserScan
check_topic_type /scan_rear_raw sensor_msgs/msg/LaserScan
check_topic_type /base_imu sensor_msgs/msg/Imu
check_topic_type /head_front_camera/head_front_camera/color/image_raw sensor_msgs/msg/Image
check_topic_type /head_front_camera/head_front_camera/depth/image_rect_raw sensor_msgs/msg/Image
check_topic_type /head_front_camera/head_front_camera/color/camera_info sensor_msgs/msg/CameraInfo
check_topic_type /contacts ros_gz_interfaces/msg/Contacts
check_topic_type /bin_contacts ros_gz_interfaces/msg/Contacts

check_live_topic /clock 8
check_live_topic /odom 12
check_live_topic /joint_states 12
check_live_topic /scan_front_raw 12
check_live_topic /scan_rear_raw 12
check_live_topic /base_imu 12
check_live_topic /head_front_camera/head_front_camera/color/image_raw 15
check_live_topic /head_front_camera/head_front_camera/depth/image_rect_raw 15

printf '\nSUMMARY: %d PASS, %d FAIL\n' "$PASS_COUNT" "$FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
  printf '[DAY1 ROS HEALTH][PASS]\n'
  exit 0
fi
printf '[DAY1 ROS HEALTH][FAIL]\n'
exit 1
