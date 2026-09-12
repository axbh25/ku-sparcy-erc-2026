#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
STAGE="${1:-day7}"
case "$STAGE" in day5|day6|day7) ;; *) echo '[FAIL] stage must be day5, day6, or day7'; exit 2;; esac
RUNTIME=(src/ku_sparcy_erc/ku_sparcy_erc src/ku_sparcy_erc/launch src/ku_sparcy_erc/config)
FAIL=0
for forbidden in ERC_SEED expected_layout get_model_state entity_state '/world/erc_world/model'; do
  if grep -R -n -F "$forbidden" "${RUNTIME[@]}" --include='*.py' --include='*.yaml' >/tmp/day567_forbidden.txt 2>/dev/null; then
    echo "[FAIL] forbidden runtime token: $forbidden"
    cat /tmp/day567_forbidden.txt
    FAIL=1
  fi
done
if grep -R -n -E '/gripper_(left|right)_controller_raw/joint_trajectory' "${RUNTIME[@]}" --include='*.py' --include='*.yaml' >/tmp/day567_raw.txt 2>/dev/null; then
  echo '[FAIL] runtime publishes or configures a raw gripper command topic'
  cat /tmp/day567_raw.txt
  FAIL=1
else
  echo '[PASS] no raw gripper command topic in runtime source'
fi
for file in \
  src/ku_sparcy_erc/ku_sparcy_erc/day5_autonomous_pick.py \
  src/ku_sparcy_erc/ku_sparcy_erc/day6_return_home.py \
  src/ku_sparcy_erc/ku_sparcy_erc/day7_bin_approach.py; do
  [ -f "$file" ] || continue
  if grep -n -E 'create_publisher\([^)]*Effort|/effort|command_interface.*effort' "$file" >/tmp/day567_effort.txt 2>/dev/null; then
    echo "[FAIL] direct effort command found in $file"
    cat /tmp/day567_effort.txt
    FAIL=1
  fi
done
grep -q 'lift_distance_m: 0.02' src/ku_sparcy_erc/config/day5_autonomous_pick.yaml || {
  echo '[FAIL] validated 2 cm lift is not preserved'; FAIL=1; }
grep -q 'extract_distance_m: 0.10' src/ku_sparcy_erc/config/day5_autonomous_pick.yaml || {
  echo '[FAIL] validated 10 cm extraction is not preserved'; FAIL=1; }
grep -q "generate_pipeline_description('${STAGE}')" src/ku_sparcy_erc/launch/solution.launch.py || {
  echo "[FAIL] solution.launch.py does not stop at ${STAGE}"; FAIL=1; }
if [ "$STAGE" = day7 ]; then
  grep -q "head_search_poses_pan_tilt_rad" src/ku_sparcy_erc/config/day7_bin.yaml || {
    echo '[FAIL] Day 7 safe head scan missing'; FAIL=1; }
fi
if [ "$FAIL" -eq 0 ]; then
  echo "[DAY567 SOLUTION COMPLIANCE][PASS] stage=${STAGE}"
else
  echo "[DAY567 SOLUTION COMPLIANCE][FAIL] stage=${STAGE}"
  exit 1
fi
