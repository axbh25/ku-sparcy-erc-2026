#!/usr/bin/env bash
# Remove only lingering ERC simulation launch/server processes inside erc_sim.
set -euo pipefail

kill_pattern() {
  local signal="$1"
  local pattern="$2"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [ -n "$pids" ]; then
    kill "-$signal" $pids >/dev/null 2>&1 || true
  fi
}

kill_pattern INT '[r]os2 launch erc_bringup simulation.launch.py'
kill_pattern INT '[g]z sim.*erc_world'
sleep 3
kill_pattern TERM '[r]os2 launch erc_bringup simulation.launch.py'
kill_pattern TERM '[g]z sim.*erc_world'
sleep 2
kill_pattern KILL '[r]os2 launch erc_bringup simulation.launch.py'
kill_pattern KILL '[g]z sim.*erc_world'
sleep 1

if pgrep -f '[g]z sim.*erc_world' >/dev/null 2>&1; then
  printf '[SIM CLEANUP][FAIL] Gazebo erc_world process remains\n' >&2
  pgrep -af '[g]z sim.*erc_world' >&2 || true
  exit 1
fi
if pgrep -f '[r]os2 launch erc_bringup simulation.launch.py' >/dev/null 2>&1; then
  printf '[SIM CLEANUP][FAIL] simulation launch process remains\n' >&2
  pgrep -af '[r]os2 launch erc_bringup simulation.launch.py' >&2 || true
  exit 1
fi
printf '[SIM CLEANUP][PASS]\n'
