#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="team_docs/day_logs/day567_diagnostics_${STAMP}.txt"
mkdir -p team_docs/day_logs
{
  echo '=== git ==='
  git status --short
  git log --oneline --decorate -12
  git remote -v
  echo '=== docker ==='
  docker inspect --format '{{.State.Status}}' erc_sim 2>&1 || true
  docker logs --tail 250 erc_sim 2>&1 || true
  echo '=== generated results ==='
  for file in src/ku_sparcy_erc/{home_pose,day4_result,day5_result,day6_result,day7_result}.json; do
    echo "--- $file ---"
    [ -f "$file" ] && cat "$file" || echo missing
  done
  echo '=== live container ==='
  docker exec erc_sim bash -lc '
    export PYTHONDONTWRITEBYTECODE=1
    source /opt/ros/humble/setup.bash
    source /opt/erc_ws/install/setup.bash 2>/dev/null || true
    echo "-- processes --"
    pgrep -af "[g]z sim.*erc_world" || true
    echo "-- topics --"
    ros2 topic list --no-daemon 2>&1 || true
    echo "-- nodes --"
    ros2 node list --no-daemon 2>&1 || true
  ' 2>&1 || true
} > "$OUT"
echo "[DAY567 DIAGNOSTICS][PASS] $OUT"
