#!/usr/bin/env bash
# Test-only helper. It uses Gazebo's public UserCommands set_pose service to
# exercise the Day 2 general visual search from several initial headings.
set -euo pipefail

orientation="${1:-}"
case "$orientation" in
  phase1-left)
    # Official simulator start: +90 degrees.
    qz='0.7071067812'; qw='0.7071067812'; expected='+90'
    ;;
  shelf-facing)
    qz='0.0'; qw='1.0'; expected='0'
    ;;
  right)
    qz='-0.7071067812'; qw='0.7071067812'; expected='-90'
    ;;
  backward)
    qz='1.0'; qw='0.0'; expected='180'
    ;;
  *)
    printf 'Usage: %s {phase1-left|shelf-facing|right|backward}\n' "$0" >&2
    exit 2
    ;;
esac

# Stop residual command before teleporting.
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
  >/dev/null 2>&1 || true

request="name: 'tiago_pro', position: {x: 0.0, y: 0.0, z: 0.15}, orientation: {x: 0.0, y: 0.0, z: ${qz}, w: ${qw}}"
response="$(gz service \
  --service /world/erc_world/set_pose \
  --reqtype gz.msgs.Pose \
  --reptype gz.msgs.Boolean \
  --timeout 5000 \
  --req "$request" 2>&1)"
printf '%s\n' "$response"

if ! printf '%s\n' "$response" | grep -Eqi 'data:[[:space:]]*true|true'; then
  printf '[SET ROBOT ORIENTATION][FAIL] Gazebo did not return true\n' >&2
  exit 1
fi
sleep 2
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
  >/dev/null 2>&1 || true
printf '[SET ROBOT ORIENTATION][PASS] mode=%s expected_yaw_deg=%s\n' \
  "$orientation" "$expected"
