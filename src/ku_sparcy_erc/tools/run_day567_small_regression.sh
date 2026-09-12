#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
source /opt/ros/humble/setup.bash
source /opt/erc_ws/install/setup.bash
TEAM=/opt/erc_ws/src/ku_sparcy_erc
WORK=/tmp/ku_sparcy_day567_regression
mkdir -p "$WORK" "$TEAM/day5_results" "$TEAM/day6_results" "$TEAM/day7_results"

cleanup_sim() {
  "$TEAM/tools/clean_day3_sim.sh" >/dev/null 2>&1 || true
  local deadline=$((SECONDS + 20))
  while pgrep -f '[g]z sim.*erc_world' >/dev/null 2>&1 && [ "$SECONDS" -lt "$deadline" ]; do
    sleep 1
  done
  if pgrep -f '[g]z sim.*erc_world' >/dev/null 2>&1; then
    echo '[SIM CLEANUP][FAIL]'
    pgrep -af '[g]z sim.*erc_world'
    return 1
  fi
  echo '[SIM CLEANUP][PASS]'
}

wait_ready() {
  local deadline=$((SECONDS + 180))
  while [ "$SECONDS" -lt "$deadline" ]; do
    local count
    count="$(pgrep -fc '[g]z sim.*erc_world' || true)"
    if [ "$count" -eq 1 ] && \
       timeout 8 ros2 topic echo --no-daemon /clock rosgraph_msgs/msg/Clock --once >/dev/null 2>&1 && \
       timeout 8 ros2 topic echo --no-daemon /odom nav_msgs/msg/Odometry --once >/dev/null 2>&1 && \
       timeout 18 ros2 topic echo --no-daemon /head_front_camera/head_front_camera/color/image_raw sensor_msgs/msg/Image --once >/dev/null 2>&1 && \
       timeout 18 ros2 topic echo --no-daemon /head_front_camera/head_front_camera/depth/image_rect_raw sensor_msgs/msg/Image --once >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon /scan_front_raw sensor_msgs/msg/LaserScan --once >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon /scan_rear_raw sensor_msgs/msg/LaserScan --once >/dev/null 2>&1; then
      echo '[SIM READY][PASS]'
      return 0
    fi
    sleep 2
  done
  echo '[SIM READY][FAIL]'
  return 1
}

start_sim() {
  local seed="$1"
  local log="$2"
  cleanup_sim
  setsid bash -lc "
    export PYTHONDONTWRITEBYTECODE=1
    export ERC_SEED=${seed}
    source /opt/ros/humble/setup.bash
    source /opt/erc_ws/install/setup.bash
    ros2 launch erc_bringup simulation.launch.py headless:=true depth_cloud:=false
  " >"$log" 2>&1 &
  SIM_PID=$!
  wait_ready
}

stop_sim() {
  if [ -n "${SIM_PID:-}" ]; then
    kill -TERM -- "-$SIM_PID" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
    unset SIM_PID
  fi
  cleanup_sim
}

run_case() {
  local name="$1" seed="$2" marker="$3" colour="$4" row="$5" arm="$6" stop="$7"
  local sim_log="$WORK/${name}_simulation.log"
  local pipeline_log="$WORK/${name}_pipeline.log"
  local home="$WORK/${name}_home.json"
  local day4="$WORK/${name}_day4.json"
  local day5="$TEAM/day5_results/${name}.json"
  local day6="$TEAM/day6_results/${name}.json"
  local day7="$TEAM/day7_results/${name}.json"
  rm -f "$home" "$day4" "$day5" "$day6" "$day7"
  start_sim "$seed" "$sim_log"
  set +e
  timeout 2400 ros2 launch ku_sparcy_erc "day${stop#day}_pipeline.launch.py" \
    shelf_column_number:="$marker" \
    book_colour:="$colour" \
    stop_after:="$stop" \
    home_pose_path:="$home" \
    day4_result_path:="$day4" \
    day5_result_path:="$day5" \
    day6_result_path:="$day6" \
    day7_result_path:="$day7" \
    image_output_dir:="$TEAM/erc_images" \
    >"$pipeline_log" 2>&1
  local code=$?
  set -e
  if [ "$code" -ne 0 ]; then
    echo "[CASE][FAIL] $name pipeline_code=$code"
    tail -n 120 "$pipeline_log" || true
    stop_sim
    return 1
  fi
  python3 "$TEAM/tools/validate_day4_result.py" "$day4" \
    --target "$marker" --colour "$colour" --expected-row "$row" \
    --mode pregrasp --require-preapproach-lock --require-zero-holds
  python3 "$TEAM/tools/validate_day5_result.py" "$day5" --expected-arm "$arm"
  if [ "$stop" = day6 ] || [ "$stop" = day7 ]; then
    python3 "$TEAM/tools/validate_day6_result.py" "$day6"
  fi
  if [ "$stop" = day7 ]; then
    python3 "$TEAM/tools/validate_day7_result.py" "$day7"
  fi
  echo "[CASE][PASS] $name"
  stop_sim
}

trap 'stop_sim >/dev/null 2>&1 || true' EXIT

# Opposite-arm grasp checkpoint only.
run_case day5_left_seed20260902 20260902 1 red 2 left day5

# One complete pick -> return -> red-bin approach case.
run_case day7_right_seed505 505 5 blue 3 right day7

trap - EXIT

echo '[DAY567 SMALL REGRESSION][PASS]'
