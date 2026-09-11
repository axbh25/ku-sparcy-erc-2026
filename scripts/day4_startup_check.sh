#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
DAY3='9a9768acb824862f2dd4b8de9950ffed54828987'
OFFICIAL='93554d4f9335b2ee3acb49c6b332611f6ad2a964'
FAIL=0
pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAIL=1; }

if git merge-base --is-ancestor "$DAY3" HEAD && git merge-base --is-ancestor "$OFFICIAL" HEAD; then
  pass 'working branch contains validated Day 3 and merged official gripper update'
else
  fail 'working branch is missing Day 3 or official upstream update ancestry'
fi

if [ "$(docker inspect --format '{{.State.Running}}' erc_sim 2>/dev/null || true)" = true ]; then
  pass 'erc_sim container is running'
else
  fail 'erc_sim container is not running'
fi

if docker exec erc_sim test -d /opt/erc_ws/src/ku_sparcy_erc; then
  pass 'team package is mounted inside the container'
else
  fail 'team package mount is missing'
fi

if docker exec erc_sim test -f /opt/erc_ws/install/setup.bash; then
  pass 'preserved colcon workspace exists'
else
  fail 'preserved colcon install workspace is missing'
fi

if ./scripts/check_repo_integrity.sh >/tmp/ku_day4_start_integrity.txt 2>&1; then
  pass 'repository integrity check passes before build'
else
  fail 'repository integrity check failed'
  cat /tmp/ku_day4_start_integrity.txt
fi

if [ "$FAIL" -eq 0 ]; then
  echo '[DAY4 STARTUP][PASS]'
  exit 0
fi
echo '[DAY4 STARTUP][FAIL]'
exit 1
