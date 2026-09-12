#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys


def gate(label: str, condition: bool) -> int:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return 0 if condition else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('result')
    parser.add_argument('--expected-arm', choices=['left', 'right'])
    args = parser.parse_args()
    with open(args.result, encoding='utf-8') as handle:
        data = json.load(handle)
    failures = 0
    failures += gate('DAY 5 RESULT STATUS', data.get('passed') is True)
    failures += gate(
        'AUTONOMOUS APPROVAL MODE',
        data.get('manual_approval_required') is False,
    )
    failures += gate(
        'VALIDATED EXTRACTION DISTANCE',
        math.isclose(float(data.get('validated_extract_distance_m', -1)), 0.10,
                     abs_tol=1e-9),
    )
    failures += gate(
        'VALIDATED TWO-CENTIMETRE LIFT',
        math.isclose(float(data.get('validated_lift_distance_m', -1)), 0.02,
                     abs_tol=1e-9),
    )
    required = {
        'plan', 'pregrasp', 'open', 'no_contact',
        'contact_pose', 'close', 'extract', 'lift',
    }
    failures += gate(
        'COMPLETE GRASP STAGE SEQUENCE',
        required.issubset(set(data.get('completed_stages', []))),
    )
    failures += gate(
        'AUTOMATIC RETENTION VERIFIED',
        data.get('retention_verified') is True,
    )
    failures += gate(
        'LIVE REQUESTED-COLOUR RETENTION EVIDENCE',
        int(data.get('retention_visual_frames', 0)) >= 3
        and isinstance(data.get('retention_visual_bbox_xywh'), list),
    )
    failures += gate(
        'FINGERTIP CONTACT EVIDENCE',
        int(data.get('selected_fingertip_contact_messages', 0)) > 0,
    )
    failures += gate(
        'NO PREMATURE FINGERTIP CONTACT',
        int(data.get('premature_selected_fingertip_contacts', -1)) == 0,
    )
    failures += gate(
        'NO NON-FINGERTIP ARM CONTACT',
        int(data.get('unintended_selected_arm_contacts', -1)) == 0,
    )
    failures += gate(
        'PUBLIC POSITION GRIPPER',
        data.get('gripper_command_mode') == 'position'
        and '_raw' not in str(data.get('gripper_public_topic'))
        and data.get('effort_commanded') is False,
    )
    failures += gate(
        'ONE ARM RULE',
        data.get('selected_arm') in {'left', 'right'}
        and data.get('single_arm_rule_respected') is True,
    )
    if args.expected_arm:
        failures += gate(
            f'EXPECTED {args.expected_arm.upper()} ARM',
            data.get('selected_arm') == args.expected_arm,
        )
    label = 'PASS' if failures == 0 else 'FAIL'
    print(f'[DAY5 ACCEPTANCE][{label}]')
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
