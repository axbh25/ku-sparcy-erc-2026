#!/usr/bin/env python3
"""Validate the JSON result written by opening_sequence."""

import json
import math
import pathlib
import sys


def main() -> int:
    path = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1
        else '/opt/erc_ws/src/ku_sparcy_erc/day1_result.json'
    )
    if not path.is_file():
        print(f'[FAIL] Result file does not exist: {path}')
        return 1

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[FAIL] Result file is unreadable or invalid JSON: {exc}')
        return 1

    checks = {
        'node reported passed=true': data.get('passed') is True,
        'head command was sent': data.get('head_command_sent') is True,
        'head reached commanded upward tilt': (
            isinstance(data.get('actual_head_tilt_rad'), (int, float))
            and isinstance(data.get('head_tilt_rad'), (int, float))
            and abs(data['actual_head_tilt_rad'] - data['head_tilt_rad']) <= 0.05
        ),
        'clockwise direction was negative': (
            isinstance(data.get('signed_rotation_deg'), (int, float))
            and data['signed_rotation_deg'] < 0.0
        ),
        'absolute rotation was 90 +/- 3 degrees': (
            isinstance(data.get('absolute_rotation_deg'), (int, float))
            and abs(data['absolute_rotation_deg'] - 90.0) <= 3.0
        ),
        'final yaw error was <= 2 degrees': (
            isinstance(data.get('final_error_deg'), (int, float))
            and abs(data['final_error_deg']) <= 2.0
        ),
        'RGB camera remained live during the turn': (
            isinstance(data.get('camera_frames_during_motion'), int)
            and data['camera_frames_during_motion'] >= 1
        ),
        'turn completed in <= 10 seconds': (
            isinstance(data.get('duration_sec'), (int, float))
            and math.isfinite(data['duration_sec'])
            and data['duration_sec'] <= 10.0
        ),
        'requested shelf marker was accepted': (
            isinstance(data.get('shelf_column_number'), int)
            and 1 <= data['shelf_column_number'] <= 5
        ),
        'requested colour was accepted': (
            data.get('book_colour') in {'red', 'green', 'yellow', 'blue'}
        ),
    }

    for label, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print('\nMeasured result:')
    for key in (
        'signed_rotation_deg',
        'absolute_rotation_deg',
        'final_error_deg',
        'duration_sec',
        'camera_frames_during_motion',
        'head_tilt_rad',
        'actual_head_tilt_rad',
        'head_tilt_error_rad',
        'reason',
    ):
        print(f'  {key}: {data.get(key)}')

    if all(checks.values()):
        print('\n[DAY1 OPENING SEQUENCE][PASS]')
        return 0
    print('\n[DAY1 OPENING SEQUENCE][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
