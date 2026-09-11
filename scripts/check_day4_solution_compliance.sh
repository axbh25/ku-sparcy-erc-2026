#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
FAIL=0
pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAIL=1; }

if ./scripts/check_repo_integrity.sh >/tmp/ku_day4_integrity.txt 2>&1; then
  pass 'repository integrity checker passes'
else
  fail 'repository integrity checker failed'
  cat /tmp/ku_day4_integrity.txt
fi

./scripts/check_day4_frozen_files.sh || FAIL=1

COMPETITION=(
  src/ku_sparcy_erc/ku_sparcy_erc/day4_mission.py
  src/ku_sparcy_erc/ku_sparcy_erc/book_perception.py
  src/ku_sparcy_erc/ku_sparcy_erc/passive_book_mapping.py
  src/ku_sparcy_erc/launch/solution.launch.py
)
LAB=(
  src/ku_sparcy_erc/ku_sparcy_erc/urdf_kinematics.py
  src/ku_sparcy_erc/ku_sparcy_erc/grasp_planner.py
  src/ku_sparcy_erc/ku_sparcy_erc/staged_grasp_experiment.py
  src/ku_sparcy_erc/launch/staged_grasp_experiment.launch.py
)

if grep -R -n -E \
    'ERC_SEED|expected_day4_layout|expected_marker_layout|/world/erc_world|number_marker_col_|book_col_[0-9]|gz service|entity_name' \
    "${COMPETITION[@]}" "${LAB[@]}" >/tmp/ku_day4_forbidden.txt; then
  fail 'team runtime source contains a seed/world/entity oracle reference'
  cat /tmp/ku_day4_forbidden.txt
else
  pass 'runtime source contains no seed, world-state, or spawn-name shortcut'
fi

if grep -n -E \
    '/arm_(left|right)_controller|/gripper_|torso_controller|command_interface name=.effort' \
    src/ku_sparcy_erc/ku_sparcy_erc/day4_mission.py \
    >/tmp/ku_day4_competition_manipulation.txt; then
  fail 'normal Day 4 competition node contains a new manipulation command'
  cat /tmp/ku_day4_competition_manipulation.txt
else
  pass 'normal Day 4 competition node remains perception/geometry only'
fi

DAY4=src/ku_sparcy_erc/ku_sparcy_erc/day4_mission.py
if grep -Fq 'class Day4Mission(Day3Mission)' "$DAY4"; then
  pass 'Day 4 subclasses the validated Day 3 mission'
else
  fail 'Day 4 does not subclass Day 3'
fi

if grep -Fq "ROW_TOPIC = '/erc/shelf_row_identification'" "$DAY4"; then
  pass 'official ERC shelf-row topic is present'
else
  fail 'official ERC shelf-row topic is missing'
fi

if grep -Fq "executable='day4_mission'" \
    src/ku_sparcy_erc/launch/solution.launch.py && \
   ! grep -Fq 'staged_grasp_experiment' \
    src/ku_sparcy_erc/launch/solution.launch.py; then
  pass 'competition launch remains separate from staged grasp laboratory'
else
  fail 'competition launch incorrectly includes staged grasp laboratory'
fi

if grep -Fq 'max_rgb_depth_stamp_skew_sec: 0.20' \
    src/ku_sparcy_erc/config/day3_approach.yaml && \
   grep -Fq 'depth_stale_timeout_sec: 0.55' \
    src/ku_sparcy_erc/config/day3_approach.yaml; then
  pass 'validated Day 3 RGB/depth synchronization limits remain unchanged'
else
  fail 'Day 3 synchronization limits changed'
fi

if grep -Fq 'SpatialBookMapAccumulator' "$DAY4" && \
   grep -Fq "'row_mapping_requires_single_frame': False" "$DAY4" && \
   ! grep -Fq 'detect_layout(' "$DAY4"; then
  pass 'row mapping accumulates independent colour tracks without a single-frame gate'
else
  fail 'single-frame layout architecture appears in the Day 4 mission'
fi

if grep -Fq 'INHERITED_PASSIVE_STATES = set()' "$DAY4" && \
   grep -Fq "if mapping_source == 'passive_approach':" "$DAY4" && \
   grep -Fq 'moving_head_during_approach_enabled' "$DAY4"; then
  pass 'approach-time row mapping and moving-head sweep are disabled'
else
  fail 'retired approach-time mapping architecture is still active'
fi

if grep -Fq 'def _start_approach(self, geometry' "$DAY4" && \
   grep -Fq "'POSITION_PREAPPROACH_HEAD'" "$DAY4" && \
   grep -Fq 'Day3Mission._start_approach(self, geometry)' "$DAY4"; then
  pass 'Day 4 intercepts only the pre-translation boundary and then resumes frozen Day 3'
else
  fail 'stationary pre-approach interception/resume path is missing'
fi

if grep -Fq "if self.state == 'DWELL_PREAPPROACH_HEAD':" "$DAY4" && \
   grep -Fq "mapping_source='stationary_preapproach'" "$DAY4"; then
  pass 'mapping frames are accepted only during settled stationary dwell'
else
  fail 'settled stationary mapping hook is missing'
fi

if grep -Fq 'preapproach_scan_nonzero_cmd_vel_publications' "$DAY4" && \
   grep -Fq 'def _publish_velocity(self, vx: float, vy: float, wz: float)' "$DAY4" && \
   grep -Fq 'self.stop_base()' "$DAY4"; then
  pass 'stationary scan audits base commands and actively holds zero velocity'
else
  fail 'stationary base-authority audit is missing'
fi

if python3 - "$DAY4" <<'PY' >/tmp/ku_day4_stationary_ast_check.txt
import ast
import sys

source = open(sys.argv[1], encoding='utf-8').read()
tree = ast.parse(source)
methods = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        methods[node.name] = node
required = {
    '_start_approach',
    '_position_preapproach_head_tick',
    '_wait_for_preapproach_head_tick',
    '_dwell_preapproach_head_tick',
    '_restore_preapproach_head_tick',
    '_wait_for_preapproach_restore_tick',
    '_passive_head_schedule_tick',
}
missing = required - methods.keys()
if missing:
    print('missing methods:', sorted(missing))
    raise SystemExit(1)
# The retired moving-head scheduler must contain no calls at all.
scheduler_calls = [
    node for node in ast.walk(methods['_passive_head_schedule_tick'])
    if isinstance(node, ast.Call)
]
if scheduler_calls:
    print('retired moving-head scheduler still contains calls')
    raise SystemExit(1)
# Only the explicit resume line may call the parent Day 3 approach starter.
resume_text = ast.get_source_segment(source, methods['_wait_for_preapproach_restore_tick']) or ''
if 'Day3Mission._start_approach(self, geometry)' not in resume_text:
    print('explicit frozen Day 3 resume is absent')
    raise SystemExit(1)
print('stationary pre-approach state-machine structure is valid')
PY
then
  pass 'stationary pre-approach state-machine AST check passes'
else
  fail 'stationary pre-approach state-machine AST check failed'
  cat /tmp/ku_day4_stationary_ast_check.txt
fi

if grep -Fq 'TargetBookReacquisitionFilter' "$DAY4" && \
   grep -Fq '[self.book_colour]' "$DAY4"; then
  pass 'final stand-off reacquisition detects only the requested colour'
else
  fail 'requested-colour-only close reacquisition is missing'
fi

if grep -Fq 'fallback_head_tilt_sequence_rad' "$DAY4" && \
   grep -Fq 'close_range_fallback' "$DAY4"; then
  pass 'bounded close-range multi-pose mapping remains fallback only'
else
  fail 'close-range fallback architecture is missing'
fi

if grep -Fq "'day4_gripper_commanded': False" "$DAY4" && \
   grep -Fq "'book_lift_attempted': False" "$DAY4"; then
  pass 'safe competition result records manipulation inhibition'
else
  fail 'competition manipulation-inhibit evidence is missing'
fi

STAGED=src/ku_sparcy_erc/ku_sparcy_erc/staged_grasp_experiment.py
if grep -n -E '/gripper_(left|right)_controller_raw|effort_controller|command_interface.*effort|/torso_controller' "$STAGED" >/tmp/ku_day4_lab_forbidden.txt; then
  fail 'staged laboratory uses a forbidden raw/effort/torso command path'
  cat /tmp/ku_day4_lab_forbidden.txt
else
  pass 'staged laboratory uses no raw gripper, effort-command, or torso topic'
fi

if grep -Fq "f'/gripper_{arm}_controller/joint_trajectory'" "$STAGED" && \
   grep -Fq "Bool, '/ku_sparcy/day4_continue'" "$STAGED"; then
  pass 'staged laboratory uses public position topics and explicit approval gates'
else
  fail 'public position topic or manual approval gate is missing'
fi

if grep -Fq "'effort_commanded': False" "$STAGED" && \
   grep -Fq "'gripper_command_mode': 'position'" "$STAGED" && \
   grep -Fq "'single_arm_rule_respected': True" "$STAGED"; then
  pass 'staged result records position-only, one-arm execution'
else
  fail 'staged result safety metadata is missing'
fi

if [ "$FAIL" -eq 0 ]; then
  echo '[DAY4 SOLUTION COMPLIANCE][PASS]'
  exit 0
fi

echo '[DAY4 SOLUTION COMPLIANCE][FAIL]'
exit 1
