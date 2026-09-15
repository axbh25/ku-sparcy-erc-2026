#!/usr/bin/env bash
# Run inside erc_sim AFTER starting one fresh simulator. Camera runs separately.
# No simulator restart is performed by this script.
set -eo pipefail
export HOME=/root PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
TEAM="${KU_SPARCY_TEAM:-/opt/erc_ws/src/ku_sparcy_erc}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo 'Run this script INSIDE erc_sim, not in an Ubuntu host shell.'
    exit 1
fi
source /opt/ros/humble/setup.bash
source /opt/erc_ws/install/setup.bash
MARKER="${1:-5}"
COLOUR="${2:-blue}"
case "$COLOUR" in blue|red|yellow|green) ;; *) echo 'Invalid book colour'; exit 1;; esac
mkdir -p "$TEAM/day8_results"
RUN="$(mktemp -d "$TEAM/day8_results/mesh_$(date +%Y%m%d_%H%M%S)_XXXX")"
printf '%s\n' "$RUN" > "$TEAM/day8_results/last_mesh_repair_run.txt"
STAGE=preflight
finish() {
    rc=$?
    trap - EXIT
    printf '\nTRIAL_EXIT=%s STAGE=%s\nRUN=%s\n' "$rc" "$STAGE" "$RUN"
    if [ "$rc" -ne 0 ]; then
        # A stop only; never issue a gripper command in this failure handler.
        timeout 4 ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}' >/dev/null 2>&1 || true
        [ ! -f "$RUN/day8_mesh.log" ] || tail -n 30 "$RUN/day8_mesh.log"
        echo 'Trial stopped. No automatic retry or simulator reset.'
    fi
    exit "$rc"
}
trap finish EXIT
python3 - "$TEAM" "$KIT" <<'PY'
import sys
sys.path.insert(0,sys.argv[2]);from verify_compatibility import verify
verify(sys.argv[1]);print('[CAPTURED INTERFACES][PASS]')
PY
timeout 20 ros2 topic echo /clock rosgraph_msgs/msg/Clock --once >/dev/null
HOME_RESULT="$RUN/home.json"
D4="$RUN/day4.json";D5="$RUN/day5.json";D7="$RUN/day7_fast.json";D8="$RUN/day8_mesh.json"
STAGE=day5_pick
ros2 launch ku_sparcy_erc day5_pipeline.launch.py \
    shelf_column_number:="$MARKER" book_colour:="$COLOUR" stop_after:=day5 \
    home_pose_path:="$HOME_RESULT" day4_result_path:="$D4" day5_result_path:="$D5" \
    image_output_dir:="$TEAM/erc_images" 2>&1 | tee "$RUN/day5_pipeline.log"
grep -q '\[PIPELINE\]\[PASS\] requested stop after day5' "$RUN/day5_pipeline.log"
if grep -q '\[PIPELINE\]\[FAIL\]' "$RUN/day5_pipeline.log"; then exit 1; fi
ARM="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_arm"])' "$D5")"
python3 "$TEAM/tools/validate_day5_result.py" "$D5" --expected-arm "$ARM"
STAGE=fast67
# Original physical-order route, original transport speeds and supported carry.
timeout -s INT -k 10 1500 ros2 run ku_sparcy_erc day67_fast_transport \
    --ros-args --params-file "$TEAM/config/day67_fast_transport.yaml" \
    -p use_sim_time:=true -p day4_result_path:="$D4" -p day5_result_path:="$D5" \
    -p home_pose_path:="$HOME_RESULT" -p result_path:="$D7" \
    -p image_output_dir:="$TEAM/erc_images" 2>&1 | tee "$RUN/fast67.log"
python3 - "$D7" <<'PY'
import json,math,sys
p=json.load(open(sys.argv[1]))
required=('passed','carry_completed','gravity_supported_carry_completed',
          'gravity_supported_day8_handoff','book_retained_at_result')
if not all(p.get(k) is True for k in required):raise SystemExit('FAST67 handoff did not pass')
if p.get('unintended_robot_contacts')!=0 or p.get('bin_contact_messages')!=0:raise SystemExit('Unexpected transport contacts')
for k in ('compact_carry_target_positions_rad','gravity_supported_target_positions_rad'):
    v=p.get(k)
    if not isinstance(v,list) or len(v)!=7 or not all(math.isfinite(float(x)) for x in v):raise SystemExit('Invalid joint-vector handoff: '+k)
q=p.get('gripper_actual_position_m')
if not isinstance(q,(int,float)) or not math.isfinite(q) or not -1e-6<=q<=.001:raise SystemExit('Invalid closed gripper reading')
print('[SUPPORTED FAST67 HANDOFF][PASS]')
PY
STAGE=day8_measured_mesh
# Explicit permission for one additional, fully screened, short forward dock.
# All original FAST67 speeds remain unchanged. Placement stays base-stationary.
timeout -s INT -k 10 300 python3 "$KIT/day8_mesh_node.py" \
    --ros-args --params-file "$TEAM/config/day8_place.yaml" \
    -p use_sim_time:=true -p operation:=execute -p manual_approval_required:=false \
    -p repair_allow_precision_dock:=true -p ready_wall_timeout_sec:=12.0 \
    -p gripper_hold_position_m:=0.0 -p day7_result_path:="$D7" -p result_path:="$D8" \
    2>&1 | tee "$RUN/day8_mesh.log"
STAGE=validate_deposit
python3 "$KIT/validate_mesh_result.py" "$D8" --team "$TEAM" | tee "$RUN/validation.log"
echo '[COMPLETE MESH-REPAIR TRIAL][PASS]'
