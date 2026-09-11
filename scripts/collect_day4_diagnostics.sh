#!/usr/bin/env bash
set -u
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT" || exit 1
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="team_docs/day_logs/day4_diagnostics_${STAMP}.txt"
mkdir -p team_docs/day_logs

{
  echo '=== KU SPARCy Day 4 diagnostics ==='
  date -u --iso-8601=seconds
  echo
  echo '=== git ==='
  git status --short 2>&1 || true
  git log --oneline --decorate --graph -15 2>&1 || true
  git remote -v 2>&1 || true
  echo
  echo '=== refs ==='
  for ref in HEAD origin/main upstream/main \
      9a9768acb824862f2dd4b8de9950ffed54828987 \
      93554d4f9335b2ee3acb49c6b332611f6ad2a964 \
      4e0ce63b22e205392e33e732d9dbcf66ce36eaeb; do
    printf '%s: ' "$ref"
    git rev-parse "$ref" 2>&1 || true
  done
  echo
  echo '=== official update diff ==='
  git diff --name-status \
    9a9768acb824862f2dd4b8de9950ffed54828987..HEAD 2>&1 || true
  echo
  echo '=== container ==='
  docker inspect --format '{{json .State}}' erc_sim 2>&1 || true
  echo
  echo '=== Gazebo processes ==='
  docker exec erc_sim bash -lc "pgrep -af '[g]z sim.*erc_world' || true" 2>&1 || true
  echo
  echo '=== ROS typed topic probes ==='
  docker exec erc_sim bash -lc '
    export PYTHONDONTWRITEBYTECODE=1
    source /opt/ros/humble/setup.bash
    source /opt/erc_ws/install/setup.bash 2>/dev/null || true
    for spec in \
      "/clock rosgraph_msgs/msg/Clock" \
      "/odom nav_msgs/msg/Odometry" \
      "/joint_states sensor_msgs/msg/JointState" \
      "/head_front_camera/head_front_camera/color/image_raw sensor_msgs/msg/Image" \
      "/head_front_camera/head_front_camera/depth/image_rect_raw sensor_msgs/msg/Image" \
      "/scan_front_raw sensor_msgs/msg/LaserScan"; do
      set -- $spec
      echo "--- $1 ($2) ---"
      timeout 8 ros2 topic echo --no-daemon "$1" "$2" --once 2>&1 | head -n 25 || true
    done
    echo "--- /contacts type ---"
    ros2 topic type --no-daemon /contacts 2>&1 || true
  ' 2>&1 || true
  echo
  echo '=== official installed asset verifier ==='
  docker exec erc_sim bash -lc '
    export PYTHONDONTWRITEBYTECODE=1
    source /opt/ros/humble/setup.bash
    source /opt/erc_ws/install/setup.bash 2>/dev/null || true
    python3 /opt/erc_ws/src/ku_sparcy_erc/tools/verify_official_gripper_update.py 2>&1
  ' 2>&1 || true
  echo
  echo '=== stationary pre-approach mapping configuration ==='
  sed -n '1,260p' src/ku_sparcy_erc/config/day4_books.yaml 2>&1 || true
  echo
  echo '=== stationary pre-approach analyses ==='
  for file in src/ku_sparcy_erc/day4_results/preapproach_analysis*.json; do
    [ -f "$file" ] || continue
    echo "--- $file ---"
    python3 -m json.tool "$file" 2>&1 || true
  done
  echo
  echo '=== latest runtime JSON ==='
  for file in \
    src/ku_sparcy_erc/day4_result*.json \
    src/ku_sparcy_erc/day4_grasp_result*.json \
    src/ku_sparcy_erc/day4_results/*.json \
    src/ku_sparcy_erc/day4_grasp_results/*.json; do
    [ -f "$file" ] || continue
    echo "--- $file ---"
    python3 -m json.tool "$file" 2>&1 | tail -n 450 || true
  done
  echo
  echo '=== regression logs ==='
  docker exec erc_sim bash -lc '
    for directory in /tmp/ku_sparcy_day4_regression /tmp/ku_sparcy_day4_grasp; do
      for file in "$directory"/*.log; do
        [ -f "$file" ] || continue
        echo "--- $file ---"
        tail -n 220 "$file"
      done
    done
  ' 2>&1 || true
  echo
  echo '=== container logs ==='
  docker logs --tail 300 erc_sim 2>&1 || true
} >"$OUT"

echo "[DAY4 DIAGNOSTICS][PASS] $OUT"
