#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/team_docs/day_logs/day1_diagnostics_${STAMP}.txt"
mkdir -p "$(dirname "$OUT")"

{
  echo '=== KU SPARCy Day 1 diagnostics ==='
  date --iso-8601=seconds
  echo
  echo '--- HOST ---'
  uname -a
  cat /etc/os-release 2>/dev/null || true
  echo
  lscpu 2>/dev/null | sed -n '1,25p' || true
  free -h || true
  df -h "$HOME" || true
  echo "DISPLAY=${DISPLAY:-UNSET}"
  ls -la /dev/dri 2>/dev/null || true
  glxinfo -B 2>/dev/null || true
  echo
  echo '--- TOOLS ---'
  git --version 2>&1 || true
  docker --version 2>&1 || true
  docker compose version 2>&1 || true
  gh --version 2>&1 | head -n 2 || true
  echo
  echo '--- GIT (NO TOKENS) ---'
  cd "$ROOT"
  git status --short --branch 2>&1 || true
  git remote -v 2>&1 || true
  git log -5 --oneline --decorate 2>&1 || true
  echo
  echo '--- DOCKER ---'
  docker ps -a --filter name=erc_sim 2>&1 || true
  docker inspect erc_sim --format '{{json .State}}' 2>&1 || true
  echo
  echo '--- CONTAINER LOG TAIL ---'
  docker logs --tail 250 erc_sim 2>&1 || true
  echo
  echo '--- ROS SNAPSHOT ---'
  if docker ps --format '{{.Names}}' | grep -Fxq erc_sim; then
    docker exec erc_sim bash -lc '
      set +e
      source /opt/ros/humble/setup.bash
      [ -f /opt/erc_ws/install/setup.bash ] && source /opt/erc_ws/install/setup.bash
      echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-UNSET}"
      echo "[nodes]"
      timeout 10 ros2 node list
      echo "[topics]"
      timeout 10 ros2 topic list -t
      echo "[controllers]"
      timeout 10 ros2 control list_controllers
      echo "[packages]"
      ros2 pkg prefix erc_bringup
      ros2 pkg prefix ku_sparcy_erc 2>/dev/null || true
      echo "[day1 result]"
      cat /opt/erc_ws/src/ku_sparcy_erc/day1_result.json 2>/dev/null || true
    ' 2>&1 || true
  else
    echo 'erc_sim is not running; ROS snapshot unavailable.'
  fi
} > "$OUT" 2>&1

printf '[PASS] Diagnostics saved to: %s\n' "$OUT"
printf 'This file contains no GitHub token or password.\n'
