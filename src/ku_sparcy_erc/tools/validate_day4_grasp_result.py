#!/usr/bin/env python3
"""Validate one explicitly approved Day 4 staged-grasp laboratory result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2


STAGES = (
    'plan', 'pregrasp', 'open', 'no_contact',
    'contact_pose', 'close', 'extract', 'lift')
UPSTREAM = '93554d4f9335b2ee3acb49c6b332611f6ad2a964'


def gate(label: str, ok: bool, details: str = '') -> bool:
    suffix = f' {details}' if details else ''
    print(f"[{label}][{'PASS' if ok else 'FAIL'}]{suffix}")
    return ok


def image_ok(path_text: object) -> bool:
    if not isinstance(path_text, str) or not path_text:
        return False
    path = Path(path_text)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('result')
    parser.add_argument('--target-stage', choices=STAGES, required=True)
    parser.add_argument('--require-contact', action='store_true')
    parser.add_argument('--manual-retention-confirmed', action='store_true')
    args = parser.parse_args()

    with open(args.result, encoding='utf-8') as handle:
        data = json.load(handle)

    expected = list(STAGES[:STAGES.index(args.target_stage) + 1])
    completed = data.get('completed_stages')
    plan = data.get('grasp_plan')
    selected = data.get('selected_arm')
    candidate = None
    if isinstance(plan, dict) and selected in {'left', 'right'}:
        candidates = plan.get('candidate_plans')
        if isinstance(candidates, dict):
            candidate = candidates.get(selected)

    checks = []
    checks.append(gate(
        'GRASP RESULT STATUS',
        data.get('passed') is True
        and data.get('experiment') == 'staged_position_controlled_grasp'))
    checks.append(gate(
        'OFFICIAL UPSTREAM BASELINE',
        data.get('official_upstream_required_commit') == UPSTREAM))
    checks.append(gate(
        'SINGLE ARM RULE',
        selected in {'left', 'right'}
        and data.get('single_arm_rule_respected') is True))
    checks.append(gate(
        'POSITION COMMAND ONLY',
        data.get('gripper_command_mode') == 'position'
        and data.get('effort_command_interface_added') is False
        and data.get('effort_commanded') is False
        and data.get('torso_commanded') is False))
    checks.append(gate(
        'SOURCE POSE CONSISTENCY',
        isinstance(data.get('source_pose_error_m'), (int, float))
        and float(data['source_pose_error_m']) <= 0.035 + 1.0e-6
        and isinstance(data.get('source_yaw_error_deg'), (int, float))
        and abs(float(data['source_yaw_error_deg'])) <= 2.0 + 1.0e-6))
    checks.append(gate(
        'IK AND ENVELOPE PLAN',
        isinstance(plan, dict)
        and plan.get('passed') is True
        and isinstance(candidate, dict)
        and candidate.get('passed') is True))
    checks.append(gate(
        'STAGE COMPLETION',
        isinstance(completed, list)
        and completed == expected,
        f'expected={expected} actual={completed}'))
    checks.append(gate(
        'NO PREMATURE FINGERTIP CONTACT',
        data.get('premature_selected_fingertip_contacts') == 0))
    checks.append(gate(
        'NO NON-FINGERTIP ARM COLLISION',
        data.get('unintended_selected_arm_contacts') == 0))
    checks.append(gate(
        'BASE CLEARANCE',
        isinstance(data.get('front_scan_min_m'), (int, float))
        and math.isfinite(float(data['front_scan_min_m']))
        and float(data['front_scan_min_m']) >= 0.70))

    evidence = data.get('evidence_paths')
    evidence_ok = isinstance(evidence, dict) and all(
        image_ok(evidence.get(stage)) for stage in expected)
    checks.append(gate('STAGED LIVE EVIDENCE', evidence_ok))

    arm_expected = STAGES.index(args.target_stage) >= STAGES.index('pregrasp')
    gripper_expected = STAGES.index(args.target_stage) >= STAGES.index('open')
    checks.append(gate(
        'ARM COMMAND SCOPE',
        data.get('arm_commanded') is arm_expected))
    checks.append(gate(
        'GRIPPER COMMAND SCOPE',
        data.get('gripper_commanded') is gripper_expected))

    if args.require_contact:
        contact_ok = (
            STAGES.index(args.target_stage) >= STAGES.index('close')
            and isinstance(
                data.get('selected_fingertip_contact_messages'), int)
            and data['selected_fingertip_contact_messages'] > 0
        )
        checks.append(gate('FINGERTIP CONTACT EVIDENCE', contact_ok))

    if args.target_stage == 'lift':
        checks.append(gate(
            'MANUAL BOOK RETENTION',
            args.manual_retention_confirmed,
            'required because Day 4 does not yet have autonomous book-pose tracking'))

    print(json.dumps({
        'selected_arm': selected,
        'completed_stages': completed,
        'gripper_position_m': data.get('gripper_actual_position_m'),
        'gripper_effort_state_statistics': (
            data.get('gripper_effort_state_statistics')),
        'selected_fingertip_contact_messages': (
            data.get('selected_fingertip_contact_messages')),
        'unintended_selected_arm_contacts': (
            data.get('unintended_selected_arm_contacts')),
        'front_scan_min_m': data.get('front_scan_min_m'),
        'front_lidar_robust_clearance_m': (
            data.get('front_lidar_robust_clearance_m')),
        'source_pose_error_m': data.get('source_pose_error_m'),
        'source_yaw_error_deg': data.get('source_yaw_error_deg'),
        'reason': data.get('reason'),
    }, indent=2, sort_keys=True))

    if all(checks):
        print('[DAY4 STAGED GRASP RESULT][PASS]')
        return 0
    print('[DAY4 STAGED GRASP RESULT][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
