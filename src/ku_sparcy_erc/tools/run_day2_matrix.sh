#!/usr/bin/env bash
# Automated headless Day 2 regression matrix. Run inside the preserved ERC
# container after sourcing /opt/erc_ws/install/setup.bash.
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
SOURCE_ROOT='/opt/erc_ws/src/ku_sparcy_erc'
TOOLS="$SOURCE_ROOT/tools"
RESULT_DIR="$SOURCE_ROOT/day2_results"
IMAGE_DIR="$SOURCE_ROOT/erc_images"
LOG_DIR='/tmp/ku_sparcy_day2_matrix'
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
  # Send a final zero command only while the simulator is still alive.
  # ros2 topic pub --once waits for a subscriber, so calling it before the
  # first simulation or after Gazebo exits can otherwise block forever.
  if [ -n "${SIM_PID:-}" ] && \
     kill -0 "$SIM_PID" >/dev/null 2>&1; then
    timeout 5 ros2 topic pub --once \
      /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
      >/dev/null 2>&1 || true
  fi

  if [ -n "${SIM_PID:-}" ]; then
    kill -INT -- "-$SIM_PID" >/dev/null 2>&1 || \
      kill -INT "$SIM_PID" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      kill -0 "$SIM_PID" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -TERM -- "-$SIM_PID" >/dev/null 2>&1 || true
    wait "$SIM_PID" >/dev/null 2>&1 || true
    SIM_PID=''
  fi

  # ros_gz_sim can leave the gz server orphaned after the ROS launch process
  # exits.  A surviving erc_world server causes duplicate controller_manager
  # nodes and makes the next case's readiness checks ambiguous.
  local gz_pids
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
    printf '[SIM CLEANUP][FAIL] orphaned Gazebo server remains\n' >&2
    pgrep -af '[g]z sim.*erc_world' >&2 || true
    set -e
    return 1
  fi

  sleep 2
  set -e
}
trap cleanup_case EXIT INT TERM

wait_for_simulation() {
  local deadline=$((SECONDS + 100))

  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ -n "${SIM_PID:-}" ] && \
       ! kill -0 "$SIM_PID" >/dev/null 2>&1; then
      printf '[SIM READY][FAIL] simulation launch process exited\n' >&2
      return 1
    fi

    # Do not use the ROS 2 CLI daemon here. We have verified that stale
    # daemon state can report !rclpy.ok() even while Gazebo and ROS topics
    # themselves are healthy.
    if timeout 8 ros2 topic echo --no-daemon \
         /clock rosgraph_msgs/msg/Clock --once \
         >/dev/null 2>&1 && \
       timeout 8 ros2 topic echo --no-daemon \
         /odom nav_msgs/msg/Odometry --once \
         >/dev/null 2>&1 && \
       timeout 12 ros2 topic echo --no-daemon \
         /head_controller/controller_state \
         control_msgs/msg/JointTrajectoryControllerState --once \
         >/dev/null 2>&1 && \
       timeout 15 ros2 topic echo --no-daemon \
         /head_front_camera/head_front_camera/color/image_raw \
         sensor_msgs/msg/Image --once \
         >/dev/null 2>&1; then

      printf '[SIM READY][PASS]\n'
      return 0
    fi

    sleep 2
  done

  printf '[SIM READY][FAIL] Timed out after 100 wall seconds\n' >&2
  return 1
}

camera_starved_result() {
  local result="$1"

  python3 - "$result" <<'PY2'
import json
import os
import sys

path = sys.argv[1]

if not os.path.isfile(path):
    sys.exit(1)

try:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
except Exception:
    sys.exit(1)

# Never retry a result that actually passed.
if data.get("passed") is not False:
    sys.exit(1)

# This infrastructure classification applies only to the fast opening
# sweep. General-search failures remain real validation failures.
if data.get("search_mode") != "phase1_fast":
    sys.exit(1)

frames = data.get("camera_frames_during_motion")
reason = str(data.get("reason", ""))

if not isinstance(frames, (int, float)):
    sys.exit(1)

camera_sensitive_failure = (
    "Target marker was not confirmed after opening turn" in reason
    or "Fast opening turn exceeded simulated motion_timeout_sec" in reason
)

# The official camera is nominally 30 Hz. A normal ~4.1 s opening turn
# has produced roughly 110-125 frames in healthy runs. Retry a FAILED
# fast case when fewer than 90 frames arrived. Passing low-frame cases
# are deliberately left untouched.
if camera_sensitive_failure and frames < 90:
    print(
        "[CAMERA STARVATION] "
        f"failed fast case received only {frames} RGB frames "
        f"during opening motion; reason={reason}"
    )
    sys.exit(0)

sys.exit(1)
PY2
}

run_case_once() {
  local name="$1"
  local seed="$2"
  local target="$3"
  local mode="$4"
  local orientation="$5"
  local progress_min="$6"
  local progress_max="$7"
  local require_all="$8"

  cleanup_case
  printf '\n============================================================\n'
  printf 'CASE %s seed=%s target=%s mode=%s orientation=%s\n' \
    "$name" "$seed" "$target" "$mode" "$orientation"
  printf '============================================================\n'

  local sim_log="$LOG_DIR/${name}_simulation.log"
  local solution_log="$LOG_DIR/${name}_solution.log"
  local monitor_log="$LOG_DIR/${name}_column_topic.log"
  local result="$RESULT_DIR/${name}.json"
  rm -f "$result" "$result.tmp" "$monitor_log"

  ERC_SEED="$seed" setsid ros2 launch erc_bringup simulation.launch.py \
    headless:=true depth_cloud:=false \
    >"$sim_log" 2>&1 &
  SIM_PID=$!
  printf 'Simulation PID=%s log=%s\n' "$SIM_PID" "$sim_log"
  if ! wait_for_simulation; then
    printf '[CASE INFRA][RETRY] simulator did not become stably ready\n' >&2
    return 75
  fi

  # Let controller/spawner startup load settle before the timed mission.
  sleep 3

  if [ "$orientation" != 'official' ]; then
    "$TOOLS/set_robot_orientation.sh" "$orientation"
  fi

  timeout 220 ros2 topic echo --no-daemon \
    /erc/shelf_column_identification \
    std_msgs/msg/Int32 \
    --once >"$monitor_log" 2>&1 &
  MONITOR_PID=$!

  local fast='true'
  [ "$mode" = 'general' ] && fast='false'

  set +e
  timeout 220 ros2 launch ku_sparcy_erc solution.launch.py \
    shelf_column_number:="$target" \
    book_colour:=red \
    phase1_fast_start:="$fast" \
    result_path:="$result" \
    image_output_dir:="$IMAGE_DIR" \
    validation_require_all_markers:="$require_all" \
    2>&1 | tee "$solution_log"
  local solution_status=${PIPESTATUS[0]}
  set -e

  # ros2 launch can return success even when mission_start itself has
  # already failed and written a failed JSON result. Classify that result
  # before waiting for a topic publication that can never arrive.
  if camera_starved_result "$result"; then
    if [ -n "${MONITOR_PID:-}" ]; then
      kill "$MONITOR_PID" >/dev/null 2>&1 || true
      wait "$MONITOR_PID" >/dev/null 2>&1 || true
      MONITOR_PID=''
    fi

    printf '[CASE INFRA][RETRY] RGB starvation caused failed fast mission\n' >&2
    return 75
  fi
  if [ "$solution_status" -ne 0 ]; then
    printf '[CASE][FAIL] solution exit status=%s; inspect %s\n' \
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

    if camera_starved_result "$result"; then
      printf '[CASE INFRA][RETRY] failed publication followed RGB starvation\n' >&2
      return 75
    fi

    printf '[COLUMN TOPIC MONITOR][FAIL] expected data: %s\n' "$target" >&2
    cat "$monitor_log" >&2 || true
    return 1
  fi
  printf '[COLUMN TOPIC MONITOR][PASS] data: %s\n' "$target"

  local expected_layout
  expected_layout="$($TOOLS/expected_marker_layout.py "$seed")"
  local validator=(
    python3 "$TOOLS/validate_day2_result.py" "$result"
    --target "$target" --mode "$mode"
  )
  if [ "$require_all" = 'true' ]; then
    validator+=(--require-all-markers --expected-layout "$expected_layout")
  fi
  if [ -n "$progress_min" ]; then
    validator+=(--progress-min "$progress_min")
  fi
  if [ -n "$progress_max" ]; then
    validator+=(--progress-max "$progress_max" --expect-during-motion yes)
  fi
  if ! "${validator[@]}"; then
    printf '[CASE][FAIL] validator rejected %s\n' "$name" >&2
    return 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$seed" "$target" "$mode" "$orientation" "$result" \
    >>"$RESULT_DIR/matrix_summary.tsv"
  printf '[CASE][PASS] %s\n' "$name"
}

run_case() {
  local name="$1"
  local infra_failures=0
  local status

  while true; do
    printf '[CASE START] %s infra_failures=%d/6\n' \
      "$name" "$infra_failures"

    if run_case_once "$@"; then
      return 0
    else
      status=$?
    fi

    # 75 means the solution was never fairly tested:
    # simulator readiness failed, or the RGB stream collapsed.
    # These failures must NOT count as algorithm attempts.
    if [ "$status" -eq 75 ]; then
      infra_failures=$((infra_failures + 1))

      if [ "$infra_failures" -ge 6 ]; then
        printf '[CASE][FAIL] %s infrastructure failed 6 times\n' \
          "$name" >&2
        return 1
      fi

      printf '[CASE INFRA RECOVERY] %s clean restart %d/6\n' \
        "$name" "$infra_failures"

      cleanup_case
      sleep 3
      continue
    fi

    # Any non-infrastructure failure is a genuine solution/validator
    # failure and must not be hidden by retrying it.
    return "$status"
  done
}

printf 'case\tseed\ttarget\tmode\torientation\tresult\n' \
  >"$RESULT_DIR/matrix_summary.tsv"

# Five fast-start cases cover every requested digit.
#
# Timing gates are applied only to representative seeded cases whose
# appearance timing has been validated:
#   - target 3 / seed 20260901: early detection
#   - target 1 / seed 20260902: late detection
#
# The remaining target-value cases verify recognition/publication without
# imposing brittle assumptions about the exact rotation angle at which a
# marker first becomes confidently recognizable.
run_case fast_target3_early 20260901 3 fast official '' 75 true
run_case fast_target4_late  20260901 4 fast official '' '' true
run_case fast_target1_late  20260902 1 fast official 60 '' true
run_case fast_target2_late  9        2 fast official '' '' true
run_case fast_target5_early 505      5 fast official '' '' true

# General mode is tested from all four cardinal starting headings.  The
# official Phase 1 left-facing pose is included separately from fast mode so
# the general visual search cannot hide a direction assumption.
run_case general_from_left     606 4 general phase1-left  '' '' false
run_case general_from_front    707 2 general shelf-facing '' '' false
run_case general_from_right    303 5 general right        '' '' false
run_case general_from_backward 404 3 general backward     '' '' false

cleanup_case
trap - EXIT INT TERM
printf '\n[MULTI-SEED VALIDATION][PASS]\n'
printf 'Summary: %s\n' "$RESULT_DIR/matrix_summary.tsv"
