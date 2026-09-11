#!/usr/bin/env bash
# Verify the official update after merging upstream/main into an integration branch.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
DAY3='9a9768acb824862f2dd4b8de9950ffed54828987'
OFFICIAL='93554d4f9335b2ee3acb49c6b332611f6ad2a964'

if [ -n "$(git status --porcelain)" ]; then
  echo '[OFFICIAL UPDATE INTEGRATION][FAIL] working tree is not clean'
  git status --short
  exit 1
fi

for commit in "$DAY3" "$OFFICIAL"; do
  if ! git merge-base --is-ancestor "$commit" HEAD; then
    echo "[OFFICIAL UPDATE INTEGRATION][FAIL] HEAD does not contain $commit"
    exit 1
  fi
done

EXPECTED=(
  'src/erc_description/models/book/sdf/erc_book.sdf'
  'src/erc_description/urdf/tiago_pro.urdf'
)
CHANGED="$(git diff --name-only "$DAY3..HEAD")"
BAD="$(printf '%s\n' "$CHANGED" | sed '/^$/d' | grep -Ev '^(src/erc_description/models/book/sdf/erc_book\.sdf|src/erc_description/urdf/tiago_pro\.urdf|\.gitignore|OFFICIAL_BASELINE\.md|TEAM_README\.md|scripts/|team_docs/|erc_images/|src/ku_sparcy_erc/)' || true)"
if [ -n "$BAD" ]; then
  echo '[OFFICIAL UPDATE INTEGRATION][FAIL] unexpected paths changed:'
  printf '%s\n' "$BAD"
  exit 1
fi
for path in "${EXPECTED[@]}"; do
  if ! printf '%s\n' "$CHANGED" | grep -Fxq "$path"; then
    echo "[OFFICIAL UPDATE INTEGRATION][FAIL] expected official file not changed: $path"
    exit 1
  fi
done

python3 - <<'PY'
import xml.etree.ElementTree as ET
from pathlib import Path
book = ET.parse(Path('src/erc_description/models/book/sdf/erc_book.sdf')).getroot()
sizes = [[float(v) for v in e.text.split()] for e in book.findall('.//geometry/box/size')]
assert sizes and all(s == [0.25, 0.02, 0.16] for s in sizes), sizes
assert all(float(e.text) == 10.0 for e in book.findall('.//friction/ode/mu'))
assert all(float(e.text) == 10.0 for e in book.findall('.//friction/ode/mu2'))
urdf = ET.parse(Path('src/erc_description/urdf/tiago_pro.urdf')).getroot()
mimic = {}
for joint in urdf.findall('joint'):
    name = joint.attrib.get('name', '')
    if name.startswith('gripper_') and joint.find('mimic') is not None:
        limit = joint.find('limit')
        assert limit is not None and 'effort' in limit.attrib, name
        mimic[name] = float(limit.attrib['effort'])

expected_mimic = {
    'gripper_left_inner_finger_left_joint': 40.0,
    'gripper_left_outer_finger_left_joint': 40.0,
    'gripper_left_fingertip_left_joint': 40.0,
    'gripper_left_finger_right_joint': 10.0,
    'gripper_left_inner_finger_right_joint': 40.0,
    'gripper_left_outer_finger_right_joint': 40.0,
    'gripper_left_fingertip_right_joint': 40.0,
    'gripper_right_inner_finger_left_joint': 40.0,
    'gripper_right_outer_finger_left_joint': 40.0,
    'gripper_right_fingertip_left_joint': 40.0,
    'gripper_right_finger_right_joint': 10.0,
    'gripper_right_inner_finger_right_joint': 40.0,
    'gripper_right_outer_finger_right_joint': 40.0,
    'gripper_right_fingertip_right_joint': 40.0,
}
assert mimic == expected_mimic, mimic
for name in ('gripper_left_finger_joint','gripper_right_finger_joint'):
    element = next(j for j in urdf.findall('.//ros2_control/joint') if j.attrib.get('name') == name)
    assert [x.attrib.get('name') for x in element.findall('command_interface')] == ['position']
    assert {'position','velocity','effort'} <= {x.attrib.get('name') for x in element.findall('state_interface')}
print('[OFFICIAL SOURCE ASSETS][PASS]')
PY

echo '[OFFICIAL UPDATE INTEGRATION][PASS]'
