#!/usr/bin/env bash
# Two-case headless Day 4 regression. The harness owns simulator startup and
# cleanup; do not run a separate manual simulator at the same time.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT='/opt/erc_ws/src/ku_sparcy_erc'
TOOLS="$ROOT/tools"
RESULT_DIR="$ROOT/day4_results"
IMAGE_DIR="$ROOT/erc_images"
LOG_DIR='/tmp/ku_sparcy_day4_regression'
mkdir -p "$RESULT_DIR" "$IMAGE_DIR" "$LOG_DIR"

if ! grep -Eq '^[[:space:]]*preapproach_scan_calibrated:[[:space:]]*true[[:space:]]*$' \
    "$ROOT/config/day4_books.yaml"; then
  echo '[DAY4 SMALL REGRESSION][FAIL] stationary pre-approach scan is not calibrated' >&2
  exit 1
fi

python3 "$TOOLS/verify_official_gripper_update.py" \
  --result "$RESULT_DIR/official_update_verification.json"

SIM_PID=''
COLUMN_MONITOR_PID=''
ROW_MONITOR_PID=''

cleanup_case() {
  set +e
  for pid_name in COLUMN_MONITOR_PID ROW_MONITOR_PID; do
    pid="${!pid_name:-}"
    if [ -n "$pid" ]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      printf -v "$pid_name" '%s' ''
    fi
  done
  if [ -n "${SIM_PID:-}" ] && kill -0 "$SIM_PID" >/dev/null 2>&1; then
    timeout 5 ros2 topic pub --once \
      /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
      >/dev/null 2>&1 || true
    kill -INT -- "-$SIM_PID" >/dev/null 2>&1 || \
      kill -INT "$SIM_PID" >/dev/null 2>&1 || true
    for _ in $(seq 1 25); do
      kill -0 "$SIM_PID" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -TERM -- "-$SIM_PID" >/dev/null 2>&1 || true
    wait "$SIM_PID" >/dev/null 2>&1 || true
  fi
  SIM_PID=''

  gz_pids="$(pgrep -f '[g]z sim.*erc_world' || true)"
  if [ -n "$gz_pids" ]; then
    kill -INT $gz_pids >/dev/null 2>&1 || true
    sleep 3
  fi
  gz_pids="$(pgrep -f '[g]z sim.*erc_world' || true)"
  if [ -n "$gz_pids" ]; then
    kill -TERM $gz_pids >/dev/null 2>&1 || true
    sleep 2
  fi
  gz_pids="$(pgrep -f '[g]z sim.*erc_world' || true)"
  if [ -n "$gz_pids" ]; then
    kill -KILL $gz_pids >/dev/null 2>&1 || true
    sleep 1
  fi
  if pgrep -f '[g]z sim.*erc_world' >/dev/null 2>&1; then
    echo '[SIM CLEANUP][FAIL] orphan Gazebo world remains' >&2
    pgrep -af '[g]z sim.*erc_world' >&2 || true
    set -e
    return 1
  fi
  sleep 2
  set -e
}
trap cleanup_case EXIT INT TERM

wait_for_simulation() {
  local deadline=$((SECONDS + 120))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ -n "$SIM_PID" ] && ! kill -0 "$SIM_PID" >/dev/null 2>&1; then
      echo '[SIM READY][FAIL] launch process exited' >&2
      return 1
    fi
    local count
    count="$(pgrep -fc '[g]z sim.*erc_world' || true)"
    if [ "$count" -eq 1 ] && \
       timeout 8 ros2 topic echo --no-daemon \
         /clock rosgraph_msgs/msg/Clock --once >/dev/null 2>&1 && \
       timeout 8 ros2 topic echo --no-daemon \
         /odom nav_msgs/msg/Odometry --once >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon \
         /head_front_camera/head_front_camera/color/image_raw \
         sensor_msgs/msg/Image --once >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon \
         /head_front_camera/head_front_camera/depth/image_rect_raw \
         sensor_msgs/msg/Image --once >/dev/null 2>&1 && \
       timeout 12 ros2 topic echo --no-daemon \
         /scan_front_raw sensor_msgs/msg/LaserScan --once >/dev/null 2>&1 && \
       timeout 12 ros2 topic echo --no-daemon \
         /arm_left_controller/controller_state \
         control_msgs/msg/JointTrajectoryControllerState --once >/dev/null 2>&1 && \
       timeout 12 ros2 topic echo --no-daemon \
         /arm_right_controller/controller_state \
         control_msgs/msg/JointTrajectoryControllerState --once >/dev/null 2>&1; then
      echo '[SIM READY][PASS]'
      return 0
    fi
    sleep 2
  done
  echo '[SIM READY][FAIL] timeout' >&2
  return 1
}

run_case() {
  local name="$1"
  local seed="$2"
  local marker="$3"
  local colour="$4"

  cleanup_case
  echo
  echo '============================================================'
  echo "CASE $name seed=$seed marker=$marker colour=$colour"
  echo '============================================================'

  local sim_log="$LOG_DIR/${name}_simulation.log"
  local solution_log="$LOG_DIR/${name}_solution.log"
  local column_log="$LOG_DIR/${name}_column.log"
  local row_log="$LOG_DIR/${name}_row.log"
  local result="$RESULT_DIR/${name}.json"
  local expected_row
  expected_row="$($TOOLS/expected_day4_layout.py \
    "$seed" "$marker" "$colour" --row-only)"

  rm -f "$result" "$result.tmp" "$column_log" "$row_log"

  ERC_SEED="$seed" setsid ros2 launch erc_bringup simulation.launch.py \
    headless:=true depth_cloud:=false >"$sim_log" 2>&1 &
  SIM_PID=$!
  if ! wait_for_simulation; then
    echo "[CASE][FAIL] simulator readiness; inspect $sim_log" >&2
    return 1
  fi
  sleep 3

  timeout 1800 ros2 topic echo --no-daemon \
    /erc/shelf_column_identification std_msgs/msg/Int32 --once \
    >"$column_log" 2>&1 &
  COLUMN_MONITOR_PID=$!

  timeout 1800 ros2 topic echo --no-daemon \
    /erc/shelf_row_identification std_msgs/msg/Int32 --once \
    >"$row_log" 2>&1 &
  ROW_MONITOR_PID=$!

  set +e
  timeout 1800 ros2 launch ku_sparcy_erc solution.launch.py \
    shelf_column_number:="$marker" \
    book_colour:="$colour" \
    phase1_fast_start:=true \
    approach_motion_enabled:=true \
    approach_distance_limit_m:=0.0 \
    day4_stop_after:=pregrasp \
    result_path:="$result" \
    image_output_dir:="$IMAGE_DIR" \
    validation_require_all_markers:=false \
    2>&1 | tee "$solution_log"
  solution_status=${PIPESTATUS[0]}
  set -e

  if [ "$solution_status" -ne 0 ]; then
    echo "[CASE][FAIL] solution exit=$solution_status; inspect $solution_log" >&2
    return 1
  fi

  set +e
  wait "$COLUMN_MONITOR_PID"
  column_status=$?
  wait "$ROW_MONITOR_PID"
  row_status=$?
  set -e
  COLUMN_MONITOR_PID=''
  ROW_MONITOR_PID=''

  if [ "$column_status" -ne 0 ] || \
     ! grep -Eq "data:[[:space:]]*$marker([[:space:]]|$)" "$column_log"; then
    echo "[COLUMN MONITOR][FAIL] expected $marker" >&2
    cat "$column_log" >&2 || true
    return 1
  fi
  echo "[COLUMN MONITOR][PASS] data: $marker"

  if [ "$row_status" -ne 0 ] || \
     ! grep -Eq "data:[[:space:]]*$expected_row([[:space:]]|$)" "$row_log"; then
    echo "[ROW MONITOR][FAIL] expected $expected_row" >&2
    cat "$row_log" >&2 || true
    return 1
  fi
  echo "[ROW MONITOR][PASS] data: $expected_row"

  if ! python3 "$TOOLS/validate_day4_result.py" "$result" \
      --target "$marker" --colour "$colour" \
      --expected-row "$expected_row" --mode pregrasp \
      --require-zero-holds --require-preapproach-lock; then
    echo "[CASE][FAIL] validator rejected $result" >&2
    return 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$seed" "$marker" "$colour" "$expected_row" "$result" \
    >>"$RESULT_DIR/regression_summary.tsv"
  echo "[CASE][PASS] $name"
}

printf 'case\tseed\tmarker\tcolour\texpected_row\tresult\n' \
  >"$RESULT_DIR/regression_summary.tsv"

run_case row_target1_red_seed20260902 20260902 1 red
run_case row_target5_blue_seed505 505 5 blue

cleanup_case
trap - EXIT INT TERM

echo '[DAY4 SMALL REGRESSION][PASS]'
