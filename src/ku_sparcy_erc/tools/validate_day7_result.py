#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


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
    failures += gate('DAY 7 RESULT STATUS', data.get('passed') is True)
    detection = data.get('bin_detection')
    geometry = data.get('bin_geometry')
    failures += gate(
        'LIVE RED BIN DETECTION',
        isinstance(detection, dict)
        and float(detection.get('confidence', 0.0)) > 0.0,
    )
    failures += gate(
        'RGB-D TF BIN GEOMETRY',
        isinstance(geometry, dict)
        and data.get('bin_detection_uses_live_rgb_depth_tf') is True,
    )
    image = data.get('bin_evidence_image_path')
    failures += gate(
        'BIN EVIDENCE IMAGE',
        isinstance(image, str) and os.path.isfile(image)
        and os.path.getsize(image) > 0,
    )
    target = data.get('final_held_book_aware_standoff_m')
    final_forward = data.get('final_bin_forward_m')
    failures += gate(
        'HELD-BOOK-AWARE BIN STANDOFF',
        isinstance(target, (int, float))
        and isinstance(final_forward, (int, float))
        and abs(float(final_forward) - float(target)) <= 0.14,
    )
    failures += gate(
        'BOOK RETAINED AT BIN',
        int(data.get('fingertip_contact_messages', 0)) > 0
        and data.get('last_fingertip_contact_sim_sec') is not None,
    )
    failures += gate(
        'NO PREMATURE BIN CONTACT',
        int(data.get('bin_contact_messages', -1)) == 0,
    )
    failures += gate(
        'ZERO UNINTENDED BIN-APPROACH CONTACTS',
        int(data.get('unintended_robot_contacts', -1)) == 0,
    )
    failures += gate(
        'NO HIDDEN ORACLE',
        data.get('hidden_simulator_oracle_used') is False
        and data.get('validation_layout_oracle_used') is False,
    )
    failures += gate(
        'POSITION HOLD ONLY',
        data.get('gripper_command_mode') == 'position'
        and '_raw' not in str(data.get('gripper_public_topic'))
        and data.get('effort_commanded') is False
        and data.get('torso_commanded') is False,
    )
    label = 'PASS' if failures == 0 else 'FAIL'
    print(f'[DAY7 ACCEPTANCE][{label}]')
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
