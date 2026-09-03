#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
FAIL=0
pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAIL=1; }

printf '=== KU SPARCy Day 3 preserved-container startup check ===\n'

if [ "$(docker inspect --format '{{.State.Running}}' erc_sim 2>/dev/null || true)" = true ]; then
  pass 'erc_sim is running'
else
  fail 'erc_sim is not running'
fi

if docker exec erc_sim test -d /opt/erc_ws/install/ku_sparcy_erc; then
  pass 'preserved installed team package exists'
else
  fail 'preserved team package install is missing'
fi
if docker exec erc_sim test -f /opt/erc_ws/src/ku_sparcy_erc/ku_sparcy_erc/mission_start.py; then
  pass 'team source bind mount is visible'
else
  fail 'team source bind mount is missing'
fi
if docker exec erc_sim bash -lc \
    "source /opt/ros/humble/setup.bash && source /opt/erc_ws/install/setup.bash && ros2 pkg prefix ku_sparcy_erc >/dev/null"; then
  pass 'ROS can resolve preserved ku_sparcy_erc install'
else
  fail 'ROS cannot resolve preserved ku_sparcy_erc install'
fi

GZ_COUNT="$(docker exec erc_sim bash -lc "pgrep -fc '[g]z sim.*erc_world' || true" | tr -d '\r')"
if [ "${GZ_COUNT:-0}" -eq 0 ]; then
  pass 'no orphan Gazebo erc_world process exists before testing'
else
  fail "found $GZ_COUNT orphan Gazebo erc_world process(es)"
  docker exec erc_sim bash -lc "pgrep -af '[g]z sim.*erc_world' || true"
fi

if ./scripts/check_repo_integrity.sh >/tmp/ku_sparcy_day3_start_integrity.txt 2>&1; then
  pass 'repository changes remain confined to approved team paths'
else
  fail 'repository integrity check failed'
  cat /tmp/ku_sparcy_day3_start_integrity.txt
fi

if [ "$FAIL" -eq 0 ]; then
  printf '[DAY3 STARTUP][PASS]\n'
  exit 0
fi
printf '[DAY3 STARTUP][FAIL]\n'
exit 1
