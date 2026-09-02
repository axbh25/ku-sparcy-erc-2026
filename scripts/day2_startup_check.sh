#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

FAIL=0
pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAIL=1; }

printf '=== KU SPARCy Day 2 startup check ===\n'

if ./scripts/check_repo_integrity.sh >/tmp/ku_sparcy_day2_start_integrity.txt 2>&1; then
  pass 'all current changes are confined to approved team paths'
else
  fail 'repository integrity check failed'
  cat /tmp/ku_sparcy_day2_start_integrity.txt
fi

docker inspect erc_sim >/dev/null 2>&1 && \
  pass 'preserved erc_sim container exists' || fail 'erc_sim container missing'

RUNNING="$(docker inspect --format '{{.State.Running}}' erc_sim 2>/dev/null || true)"
[ "$RUNNING" = true ] && pass 'erc_sim is running' || fail 'erc_sim is not running'

docker exec erc_sim test -f /opt/erc_ws/install/setup.bash >/dev/null 2>&1 && \
  pass 'preserved colcon install/setup.bash exists' || \
  fail 'preserved colcon workspace install is missing'

docker exec erc_sim test -d /opt/erc_ws/src/ku_sparcy_erc >/dev/null 2>&1 && \
  pass 'team source package is mounted in the container' || \
  fail 'team source package is not mounted'

docker exec erc_sim bash -lc '
  export PYTHONDONTWRITEBYTECODE=1
  source /opt/erc_ws/install/setup.bash
  ros2 pkg prefix ku_sparcy_erc >/dev/null
  ros2 pkg executables ku_sparcy_erc | grep -q "ku_sparcy_erc opening_sequence"
' >/dev/null 2>&1 && pass 'Day 1 ROS package is still usable' || \
  fail 'Day 1 ROS package/executable is unavailable'

if [ "$FAIL" -eq 0 ]; then
  printf '[DAY2 STARTUP][PASS]\n'
  exit 0
fi
printf '[DAY2 STARTUP][FAIL]\n'
exit 1
