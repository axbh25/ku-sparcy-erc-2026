#!/usr/bin/env bash
# Two-case Day 3 headless regression. This harness owns simulator startup and
# cleanup; do not run a separate manual Gazebo instance at the same time.
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
SOURCE_ROOT='/opt/erc_ws/src/ku_sparcy_erc'
TOOLS="$SOURCE_ROOT/tools"
RESULT_DIR="$SOURCE_ROOT/day3_results"
IMAGE_DIR="$SOURCE_ROOT/erc_images"
LOG_DIR='/tmp/ku_sparcy_day3_regression'
mkdir -p "$RESULT_DIR" "$IMAGE_DIR" "$LOG_DIR"

SIM_PID=''
MONITOR_PID=''

cleanup_case() {
  set +e
  if [ -n "${MONITOR_PID:-}" ]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
    MONITOR_PID=''
  fi
  if [ -n "${SIM_PID:-}" ] && kill -0 "$SIM_PID" >/dev/null 2>&1; then
    timeout 5 ros2 topic pub --once \
      /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
      >/dev/null 2>&1 || true
    kill -INT -- "-$SIM_PID" >/dev/null 2>&1 || \
      kill -INT "$SIM_PID" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      kill -0 "$SIM_PID" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -TERM -- "-$SIM_PID" >/dev/null 2>&1 || true
    wait "$SIM_PID" >/dev/null 2>&1 || true
  fi
  SIM_PID=''
  "$TOOLS/clean_day3_sim.sh" >/dev/null 2>&1 || true
  if pgrep -f '[g]z sim.*erc_world' >/dev/null 2>&1; then
    printf '[SIM CLEANUP][FAIL] orphaned Gazebo process remains\n' >&2
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
    if [ -n "${SIM_PID:-}" ] && ! kill -0 "$SIM_PID" >/dev/null 2>&1; then
      printf '[SIM READY][FAIL] simulation launch exited\n' >&2
      return 1
    fi
    if timeout 8 ros2 topic echo --no-daemon \
         /clock rosgraph_msgs/msg/Clock --once >/dev/null 2>&1 && \
       timeout 8 ros2 topic echo --no-daemon \
         /odom nav_msgs/msg/Odometry --once >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon \
         /head_front_camera/head_front_camera/color/image_raw \
         sensor_msgs/msg/Image --once >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon \
         /head_front_camera/head_front_camera/depth/image_rect_raw \
         sensor_msgs/msg/Image --once >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon \
         /head_front_camera/head_front_camera/color/camera_info \
         sensor_msgs/msg/CameraInfo --once >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon \
         /head_front_camera/head_front_camera/depth/camera_info \
         sensor_msgs/msg/CameraInfo --once >/dev/null 2>&1 && \
       timeout 12 ros2 topic echo --no-daemon \
         /scan_front_raw sensor_msgs/msg/LaserScan --once >/dev/null 2>&1 && \
       timeout 12 ros2 topic echo --no-daemon \
         /head_controller/controller_state \
         control_msgs/msg/JointTrajectoryControllerState --once \
         >/dev/null 2>&1; then
      local count
      count="$(pgrep -fc '[g]z sim.*erc_world' || true)"
      if [ "$count" -eq 1 ]; then
        printf '[SIM READY][PASS]\n'
        return 0
      fi
      printf '[SIM READY][RETRY] expected one Gazebo process, found %s\n' "$count" >&2
    fi
    sleep 2
  done
  printf '[SIM READY][FAIL] timed out after 120 wall seconds\n' >&2
  return 1
}

infrastructure_starved_result() {
  local result="$1"
  python3 - "$result" <<'PY'
import json
import os
import sys

path = sys.argv[1]
if not os.path.isfile(path):
    sys.exit(1)
try:
    with open(path, encoding='utf-8') as handle:
        data = json.load(handle)
except Exception:
    sys.exit(1)
if data.get('passed') is not False:
    sys.exit(1)
reason = str(data.get('reason', ''))
if 'Gazebo /clock stopped advancing' in reason:
    print('[INFRA STARVATION] simulation clock stalled')
    sys.exit(0)
if 'Readiness timeout' in reason and any(token in reason for token in (
    'RGB', 'depth', 'LiDAR', 'CameraInfo', 'TF')):
    print(f'[INFRA STARVATION] {reason}')
    sys.exit(0)
if any(token in reason for token in (
    'remained stale', 'remained unavailable', 'target recovery timeout')):
    camera = data.get('camera_frames_during_approach', 0)
    depth = data.get('depth_frames_during_approach', 0)
    lidar = data.get('lidar_frames_during_approach', 0)
    if camera < 5 or depth < 3 or lidar < 2:
        print(
            '[INFRA STARVATION] failed run had low sensor throughput: '
            f'camera={camera} depth={depth} lidar={lidar}; reason={reason}')
        sys.exit(0)
sys.exit(1)
PY
}

run_once() {
  local name="$1"
  local seed="$2"
  local target="$3"
  cleanup_case
  printf '\n============================================================\n'
  printf 'DAY3 CASE %s seed=%s target=%s\n' "$name" "$seed" "$target"
  printf '============================================================\n'

  local sim_log="$LOG_DIR/${name}_simulation.log"
  local solution_log="$LOG_DIR/${name}_solution.log"
  local monitor_log="$LOG_DIR/${name}_column_topic.log"
  local result="$RESULT_DIR/${name}.json"
  rm -f "$result" "$result.tmp" "$monitor_log"

  ERC_SEED="$seed" setsid ros2 launch erc_bringup simulation.launch.py \
    headless:=true depth_cloud:=false >"$sim_log" 2>&1 &
  SIM_PID=$!
  if ! wait_for_simulation; then
    return 75
  fi
  sleep 3

  timeout 420 ros2 topic echo --no-daemon \
    /erc/shelf_column_identification std_msgs/msg/Int32 --once \
    >"$monitor_log" 2>&1 &
  MONITOR_PID=$!

  set +e
  timeout 420 ros2 launch ku_sparcy_erc solution.launch.py \
    shelf_column_number:="$target" \
    book_colour:=red \
    phase1_fast_start:=true \
    result_path:="$result" \
    image_output_dir:="$IMAGE_DIR" \
    validation_require_all_markers:=false \
    approach_motion_enabled:=true \
    approach_distance_limit_m:=0.0 \
    approach_max_forward_speed_mps:=0.30 \
    2>&1 | tee "$solution_log"
  local solution_status=${PIPESTATUS[0]}
  set -e

  if infrastructure_starved_result "$result"; then
    return 75
  fi
  if [ "$solution_status" -ne 0 ]; then
    printf '[CASE][FAIL] solution exit=%s; inspect %s\n' \
      "$solution_status" "$solution_log" >&2
    return 1
  fi

  set +e
  wait "$MONITOR_PID"
  local monitor_status=$?
  set -e
  MONITOR_PID=''
  if [ "$monitor_status" -ne 0 ] || \
     ! grep -Eq "data:[[:space:]]*$target([[:space:]]|$)" "$monitor_log"; then
    printf '[COLUMN TOPIC MONITOR][FAIL] expected data: %s\n' "$target" >&2
    cat "$monitor_log" >&2 || true
    return 1
  fi
  printf '[COLUMN TOPIC MONITOR][PASS] data: %s\n' "$target"

  if ! python3 "$TOOLS/validate_day3_result.py" "$result" \
      --target "$target" --mode full --min-travel-m 0.75 \
      --max-safety-stops 0 --require-lateral-command; then
    printf '[CASE][FAIL] Day 3 validator rejected %s\n' "$name" >&2
    return 1
  fi

  printf '%s\t%s\t%s\t%s\n' "$name" "$seed" "$target" "$result" \
    >>"$RESULT_DIR/day3_small_regression.tsv"
  printf '[CASE][PASS] %s\n' "$name"
}

run_case() {
  local name="$1"
  local retries=0
  local status
  while true; do
    if run_once "$@"; then
      return 0
    else
      status=$?
    fi
    if [ "$status" -eq 75 ]; then
      retries=$((retries + 1))
      if [ "$retries" -ge 3 ]; then
        printf '[CASE][FAIL] %s had three infrastructure-starved starts\n' "$name" >&2
        return 1
      fi
      printf '[CASE INFRA RECOVERY] %s clean retry %d/3\n' "$name" "$retries"
      cleanup_case
      sleep 3
      continue
    fi
    return "$status"
  done
}

printf 'case\tseed\ttarget\tresult\n' >"$RESULT_DIR/day3_small_regression.tsv"
run_case full_target1_seed20260902 20260902 1
run_case full_target5_seed505      505      5

cleanup_case
printf '[DAY3 SMALL REGRESSION][PASS]\n'
