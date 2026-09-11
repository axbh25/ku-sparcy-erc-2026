#!/usr/bin/env python3
"""Pure, deterministic tests for the Day 4 URDF/IK helpers."""

from __future__ import annotations

import math

import numpy as np

from ku_sparcy_erc.grasp_planner import solve_ik
from ku_sparcy_erc.urdf_kinematics import URDFKinematicModel


SYNTHETIC_URDF = r'''<?xml version="1.0"?>
<robot name="synthetic_six_dof">
  <link name="base"/>
  <link name="l1"/><link name="l2"/><link name="l3"/>
  <link name="l4"/><link name="l5"/><link name="l6"/>
  <link name="tip"/>
  <joint name="j1" type="prismatic">
    <parent link="base"/><child link="l1"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="l1"/><child link="l2"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j3" type="prismatic">
    <parent link="l2"/><child link="l3"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j4" type="revolute">
    <parent link="l3"/><child link="l4"/><axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
  <joint name="j5" type="revolute">
    <parent link="l4"/><child link="l5"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
  <joint name="j6" type="revolute">
    <parent link="l5"/><child link="l6"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
  <joint name="fixed_tip" type="fixed">
    <parent link="l6"/><child link="tip"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>
</robot>
'''


def main() -> int:
    model = URDFKinematicModel.from_xml(SYNTHETIC_URDF)
    chain = model.chain('base', 'tip')
    names = tuple(f'j{index}' for index in range(1, 7))
    expected_names = [joint.name for joint in chain]
    chain_ok = expected_names == [*names, 'fixed_tip']

    known = np.asarray([0.36, -0.22, 0.31, 0.14, -0.11, 0.23])
    target_state = model.forward(
        chain,
        {name: float(value) for name, value in zip(names, known)},
    )
    target_position = target_state.tip_transform[:3, 3]
    target_rotation = target_state.tip_transform[:3, :3]

    solution = solve_ik(
        model,
        chain,
        names,
        {},
        target_position,
        target_rotation,
        np.zeros(6),
        max_iterations=400,
        position_tolerance_m=0.002,
        orientation_tolerance_rad=0.02,
        orientation_weight=0.50,
    )
    solved_state = model.forward(
        chain,
        {name: value for name, value in zip(names, solution.positions)},
    )
    position_error = float(np.linalg.norm(
        solved_state.tip_transform[:3, 3] - target_position))
    rotation_delta = solved_state.tip_transform[:3, :3] @ target_rotation.T
    angle = math.acos(float(np.clip(
        (np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)))
    ik_ok = solution.success and position_error <= 0.003 and angle <= 0.03

    limits_ok = False
    try:
        model.forward(chain, {'j1': 2.0})
    except ValueError:
        limits_ok = True

    tests = (
        ('URDF CHAIN PARSING', chain_ok),
        ('NUMERICAL IK', ik_ok),
        ('JOINT LIMIT ENFORCEMENT', limits_ok),
    )
    for label, ok in tests:
        print(f"[{label}][{'PASS' if ok else 'FAIL'}]")
    print(
        'IK diagnostics: '
        f'solver_success={solution.success} '
        f'position_error_m={position_error:.8f} '
        f'orientation_error_rad={angle:.8f} '
        f'iterations={solution.iterations}'
    )
    if all(ok for _, ok in tests):
        print('[DAY4 KINEMATICS UNIT TEST][PASS]')
        return 0
    print('[DAY4 KINEMATICS UNIT TEST][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
