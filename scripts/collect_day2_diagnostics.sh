#!/usr/bin/env bash
set -u

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT" || exit 1
mkdir -p team_docs/day_logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="team_docs/day_logs/day2_diagnostics_${STAMP}.txt"

section() {
  printf '\n\n===== %s =====\n' "$1"
}

{
  printf 'KU SPARCy ERC 2026 Day 2 diagnostics\n'
  printf 'Generated UTC: %s\n' "$(date -u --iso-8601=seconds)"

  section 'HOST'
  uname -a 2>&1 || true
  lsb_release -a 2>&1 || true
  free -h 2>&1 || true
  df -h "$ROOT" 2>&1 || true

  section 'GIT STATUS'
  git status --short 2>&1 || true
  git log --oneline --decorate -8 2>&1 || true
  git remote -v 2>&1 || true
  printf 'HEAD=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
  printf 'upstream/main=%s\n' "$(git rev-parse upstream/main 2>/dev/null || true)"

  section 'OFFICIAL TRACKED BYTECODE DIFF'
  git diff --name-status upstream/main -- \
    ':(glob)src/**/__pycache__/*.pyc' 2>&1 || true
  git diff --cached --name-status upstream/main -- \
    ':(glob)src/**/__pycache__/*.pyc' 2>&1 || true

  section 'DOCKER'
  docker --version 2>&1 || true
  docker compose version 2>&1 || true
  docker inspect erc_sim \
    --format 'status={{.State.Status}} running={{.State.Running}} image={{.Config.Image}}' \
    2>&1 || true

  section 'CONTAINER LOG TAIL'
  docker logs --tail 250 erc_sim 2>&1 || true

  if docker inspect --format '{{.State.Running}}' erc_sim 2>/dev/null | \
      grep -qx true; then
    section 'ROS AND DAY 2 RUNTIME'
    docker exec erc_sim bash -lc '
      export PYTHONDONTWRITEBYTECODE=1
      source /opt/ros/humble/setup.bash
      [ -f /opt/erc_ws/install/setup.bash ] && \
        source /opt/erc_ws/install/setup.bash
      echo "--- nodes ---"
      ros2 node list 2>&1 || true
      echo "--- topics and types ---"
      ros2 topic list -t 2>&1 || true
      echo "--- controllers ---"
      ros2 control list_controllers 2>&1 || true
      echo "--- KU SPARCy executables ---"
      ros2 pkg executables ku_sparcy_erc 2>&1 || true
      echo "--- official column topic info ---"
      ros2 topic info -v /erc/shelf_column_identification 2>&1 || true
      echo "--- Python dependencies ---"
      python3 - <<"PY" 2>&1 || true
import cv2
import numpy
from cv_bridge import CvBridge
print("OpenCV", cv2.__version__)
print("NumPy", numpy.__version__)
print("CvBridge", CvBridge)
PY
      echo "--- Day 2 results ---"
      find /opt/erc_ws/src/ku_sparcy_erc -maxdepth 2 -type f \
        \( -name "day2_result*.json" -o -path "*/day2_results/*.json" \) \
        -printf "%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n" 2>/dev/null | sort || true
      echo "--- annotated images ---"
      find /opt/erc_ws/src/ku_sparcy_erc/erc_images -maxdepth 1 -type f \
        -name "*.png" -printf "%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n" \
        2>/dev/null | sort || true
      echo "--- default Day 2 result ---"
      cat /opt/erc_ws/src/ku_sparcy_erc/day2_result.json 2>&1 || true
      echo "--- matrix summary ---"
      cat /opt/erc_ws/src/ku_sparcy_erc/day2_results/matrix_summary.tsv \
        2>&1 || true
      echo "--- matrix log tails ---"
      for f in /tmp/ku_sparcy_day2_matrix/*.log; do
        [ -e "$f" ] || continue
        echo "### $f"
        tail -n 120 "$f" 2>&1 || true
      done
    ' 2>&1 || true
  else
    section 'ROS AND DAY 2 RUNTIME'
    printf 'erc_sim is not running; runtime ROS details unavailable.\n'
  fi
} >"$OUT"

printf '[DAY2 DIAGNOSTICS][PASS] %s\n' "$OUT"
