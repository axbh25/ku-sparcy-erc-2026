#!/usr/bin/env bash
set -u

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT" || exit 1
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="team_docs/day_logs/day3_diagnostics_${STAMP}.txt"
mkdir -p team_docs/day_logs

section() {
  printf '\n============================================================\n'
  printf '%s\n' "$1"
  printf '============================================================\n'
}

{
  section 'HOST / GIT'
  date -u --iso-8601=seconds
  uname -a
  git status --short
  git log --oneline --decorate -8
  git remote -v
  printf '\nHEAD=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
  printf 'origin/main=%s\n' "$(git rev-parse origin/main 2>/dev/null || true)"
  printf 'upstream/main=%s\n' "$(git rev-parse upstream/main 2>/dev/null || true)"

  section 'REPOSITORY CHECKS'
  ./scripts/check_repo_integrity.sh 2>&1 || true
  ./scripts/check_day3_frozen_files.sh 2>&1 || true
  ./scripts/check_day3_solution_compliance.sh 2>&1 || true
  ./scripts/check_tracked_official_bytecode.sh 2>&1 || true

  section 'DOCKER'
  docker ps -a --filter name=erc_sim
  docker inspect --format '{{json .State}}' erc_sim 2>/dev/null || true
  docker logs --tail 250 erc_sim 2>&1 || true

  if [ "$(docker inspect --format '{{.State.Running}}' erc_sim 2>/dev/null || true)" = true ]; then
    section 'CONTAINER PROCESSES'
    docker exec erc_sim bash -lc \
      "ps -eo pid,ppid,pgid,stat,etime,cmd | grep -E 'gz sim|ros2 launch|mission_start|day3_mission|controller_manager' | grep -v grep || true"

    section 'TYPED DAEMONLESS TOPIC SAMPLES'
    for specification in \
      '/clock|rosgraph_msgs/msg/Clock' \
      '/odom|nav_msgs/msg/Odometry' \
      '/head_front_camera/head_front_camera/color/camera_info|sensor_msgs/msg/CameraInfo' \
      '/head_front_camera/head_front_camera/depth/camera_info|sensor_msgs/msg/CameraInfo' \
      '/head_front_camera/head_front_camera/depth/image_rect_raw|sensor_msgs/msg/Image' \
      '/scan_front_raw|sensor_msgs/msg/LaserScan'; do
      topic="${specification%%|*}"
      type="${specification#*|}"
      printf '\n--- %s (%s) ---\n' "$topic" "$type"
      docker exec erc_sim bash -lc \
        "export PYTHONDONTWRITEBYTECODE=1; source /opt/ros/humble/setup.bash; source /opt/erc_ws/install/setup.bash; timeout 12 ros2 topic echo --no-daemon '$topic' '$type' --once" \
        2>&1 || true
    done

    section 'DAY 3 INTERFACE PROBE'
    docker exec erc_sim bash -lc \
      "export PYTHONDONTWRITEBYTECODE=1; source /opt/ros/humble/setup.bash; source /opt/erc_ws/install/setup.bash; timeout 45 python3 /opt/erc_ws/src/ku_sparcy_erc/tools/day3_interface_probe.py" \
      2>&1 || true

    section 'RUNTIME RESULTS'
    docker exec erc_sim bash -lc \
      "for f in /opt/erc_ws/src/ku_sparcy_erc/day3_result*.json /opt/erc_ws/src/ku_sparcy_erc/day3_results/*.json; do [ -f \"\$f\" ] || continue; echo --- \"\$f\"; python3 -m json.tool \"\$f\"; done" \
      2>&1 || true

    section 'AUTOMATED LOG TAILS'
    docker exec erc_sim bash -lc \
      "for f in /tmp/ku_sparcy_day3_regression/*.log; do [ -f \"\$f\" ] || continue; echo --- \"\$f\"; tail -n 160 \"\$f\"; done" \
      2>&1 || true
  fi
} >"$OUT" 2>&1

printf '[DAY3 DIAGNOSTICS][PASS] %s\n' "$OUT"
