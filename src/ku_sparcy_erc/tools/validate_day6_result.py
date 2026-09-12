#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math


def gate(label: str, condition: bool) -> int:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return 0 if condition else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('result')
    args = parser.parse_args()
    with open(args.result, encoding='utf-8') as handle:
        data = json.load(handle)
    failures = 0
    failures += gate('DAY 6 RESULT STATUS', data.get('passed') is True)
    distance = data.get('distance_to_home_center_m')
    failures += gate(
        'RETURNED INSIDE START ZONE',
        isinstance(distance, (int, float)) and float(distance) <= 0.40,
    )
    failures += gate(
        'HELD BOOK INCLUDED IN SAFETY',
        data.get('held_book_geometry_in_safety_reasoning') is True
        and float(data.get('carried_book_swept_radius_m', 0.0)) > 0.20,
    )
    failures += gate(
        'SHELF RETREAT COMPLETED',
        float(data.get('retreat_distance_m', 0.0)) >= 0.30,
    )
    failures += gate(
        'FRONT AND REAR LIDAR LIVE',
        int(data.get('front_scan_frames', 0)) > 0
        and int(data.get('rear_scan_frames', 0)) > 0,
    )
    failures += gate(
        'BOOK RETAINED DURING RETURN',
        int(data.get('fingertip_contact_messages', 0)) > 0
        and data.get('last_fingertip_contact_sim_sec') is not None,
    )
    failures += gate(
        'ZERO UNINTENDED RETURN CONTACTS',
        int(data.get('unintended_robot_contacts', -1)) == 0,
    )
    failures += gate(
        'PUBLIC POSITION HOLD ONLY',
        data.get('gripper_command_mode') == 'position'
        and '_raw' not in str(data.get('gripper_public_topic'))
        and data.get('effort_commanded') is False
        and data.get('torso_commanded') is False,
    )
    failures += gate(
        'PATH TRACE RECORDED',
        isinstance(data.get('path_trace'), list)
        and len(data['path_trace']) >= 5,
    )
    label = 'PASS' if failures == 0 else 'FAIL'
    print(f'[DAY6 ACCEPTANCE][{label}]')
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
