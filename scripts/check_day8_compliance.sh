#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
import ast
from pathlib import Path
root=Path('src/ku_sparcy_erc/ku_sparcy_erc')
files=sorted(root.glob('day8_*.py'))
assert files, 'No Day 8 runtime files'
for p in files:
    s=p.read_text();tree=ast.parse(s,filename=str(p))
    for token in ('ERC_SEED','expected_marker_layout','expected_day4_layout','/gazebo/model_states',
                  '/world/erc_world','get_entity_state','get_model_state','set_entity_state'):
        assert token not in s, f'Forbidden runtime shortcut {token} in {p}'
    assert 'controller_raw/joint_trajectory' not in s, f'Raw gripper topic in {p}'
    for n in ast.walk(tree):
        if isinstance(n,ast.Attribute):
            assert not (n.attr in ('effort','efforts') and isinstance(n.ctx,ast.Store)), 'Effort assignment'
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in ('create_client',), 'Unexpected service client'
node=(root/'day8_bin_place.py').read_text()
assert "if self.op!='plan':" in node and "if self.op=='execute':" in node
assert "manual_approval_required': False" in (root/'day8_pipeline_support.py').read_text()
assert 'release_started is None' in node
assert 'ROBOT_CONTACT_STREAM_STALE' in node and 'BIN_CONTACT_STREAM_STALE' in node
assert 'OPEN_STEPWISE' in node and 'VERIFY_DEPOSIT' in node
assert "self.cmd.publish(Twist())" in node
assert '.linear.x =' not in node and '.linear.y =' not in node and '.angular.z =' not in node
assert 'actual_deposited_book_checked=True' in (root/'day8_planner.py').read_text()
for key in ('DAY7 INPUT','SUPPORT BEFORE RELEASE','SUPPORT AFTER RETRACT','FINGERS DISENGAGED'):
    assert key in Path('src/ku_sparcy_erc/tools/validate_day8_result.py').read_text()
print('[DAY8 NO HIDDEN ORACLE][PASS]')
print('[DAY8 PUBLIC POSITION ONLY][PASS]')
print('[DAY8 BASE ZERO ONLY][PASS]')
print('[DAY8 PHASE-AWARE CONTACT POLICY][PASS]')
print('[DAY8 PLAN-ONLY PUBLISHER GUARD][PASS]')
print('[DAY8 SOLUTION COMPLIANCE][PASS]')
PY
bash ./scripts/check_repo_integrity.sh
git diff --check
