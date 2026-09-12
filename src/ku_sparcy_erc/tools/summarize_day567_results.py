#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import sys


def load(pattern: str):
    rows = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding='utf-8') as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get('passed') is True:
            rows.append((path, data))
    return rows


def main() -> int:
    team = '/opt/erc_ws/src/ku_sparcy_erc'
    day5 = load(os.path.join(team, 'day5_results', '*.json'))
    day6 = load(os.path.join(team, 'day6_results', '*.json'))
    day7 = load(os.path.join(team, 'day7_results', '*.json'))
    print('DAY 5 PASSED:', len(day5))
    for path, data in day5:
        print(os.path.basename(path), data.get('selected_arm'),
              data.get('retention_verified'),
              data.get('validated_lift_distance_m'))
    print('DAY 6 PASSED:', len(day6))
    for path, data in day6:
        print(os.path.basename(path), data.get('distance_to_home_center_m'),
              data.get('minimum_front_clearance_m'),
              data.get('minimum_rear_clearance_m'))
    print('DAY 7 PASSED:', len(day7))
    for path, data in day7:
        print(os.path.basename(path),
              data.get('final_held_book_aware_standoff_m'),
              data.get('bin_contact_messages'))
    if len(day5) < 2 or len(day6) < 1 or len(day7) < 1:
        print('[DAY567 REGRESSION SUMMARY][FAIL]')
        return 1
    print('[DAY567 REGRESSION SUMMARY][PASS]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
