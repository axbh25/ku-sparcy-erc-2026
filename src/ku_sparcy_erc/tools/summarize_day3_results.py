#!/usr/bin/env python3
"""Summarize Day 3 full-approach JSON results without jq."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directory', nargs='?', type=Path,
        default=Path('/opt/erc_ws/src/ku_sparcy_erc/day3_results'))
    parser.add_argument('--expected-count', type=int, default=2)
    args = parser.parse_args()

    paths = sorted(args.directory.glob('*.json'))
    rows = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] {path}: {exc}')
            return 1
        rows.append((path, data))

    print('case\ttarget\tpassed\ttravel_m\tapproach_sim_s\t'
          'start_depth_m\tfinal_depth_m\tmin_lidar_m\tstale\tsensor_hold\tsafety')
    for path, data in rows:
        print('\t'.join(str(item) for item in (
            path.stem,
            data.get('shelf_column_number'),
            data.get('passed'),
            data.get('approach_distance_travelled_m'),
            data.get('approach_duration_sim_sec'),
            data.get('start_depth_forward_m'),
            data.get('final_depth_forward_m'),
            data.get('minimum_lidar_clearance_during_approach_m'),
            data.get('target_stale_events'),
            data.get('sensor_hold_events'),
            data.get('safety_stop_events'),
        )))

    full = [data for _, data in rows if data.get('approach_outcome') == 'safe_standoff']
    pass_count = sum(data.get('passed') is True for _, data in rows)
    print(f'\nresults={len(rows)} passed={pass_count} full_standoff={len(full)}')
    if full:
        travel = [float(data['approach_distance_travelled_m']) for data in full]
        duration = [float(data['approach_duration_sim_sec']) for data in full]
        final_depth = [float(data['final_depth_forward_m']) for data in full]
        print(f'mean_travel_m={statistics.fmean(travel):.3f}')
        print(f'mean_approach_sim_s={statistics.fmean(duration):.3f}')
        print(f'mean_final_depth_m={statistics.fmean(final_depth):.3f}')

    ok = (
        len(rows) == args.expected_count
        and pass_count == args.expected_count
        and len(full) == args.expected_count
        and all(int(data.get('safety_stop_events', -1)) == 0 for _, data in rows)
    )
    print(f"[DAY3 REGRESSION SUMMARY][{'PASS' if ok else 'FAIL'}]")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
