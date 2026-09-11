#!/usr/bin/env python3
"""Summarize stationary pre-approach Day 4 regression results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory', type=Path,
        default=Path('/opt/erc_ws/src/ku_sparcy_erc/day4_results'))
    parser.add_argument('--expected-count', type=int, default=2)
    args = parser.parse_args()

    files = sorted(args.directory.glob('*.json'))
    files = [
        path for path in files
        if 'official_update_verification' not in path.name
        and 'preapproach_analysis' not in path.name
    ]
    if len(files) != args.expected_count:
        print(
            f'[DAY4 REGRESSION SUMMARY][FAIL] expected '
            f'{args.expected_count} mission JSON files, found {len(files)}')
        return 1

    headings = (
        'case', 'marker', 'colour', 'row', 'mapping_source',
        'scan_poses', 'scan_sim_s', 'base_drift_m', 'fallback',
        'reacq_frames', 'book_depth_m', 'selected_arm',
        'mission_sim_s', 'day4_sim_s', 'final_lidar_m', 'holds')
    print('\t'.join(headings))
    all_ok = True

    for path in files:
        with path.open(encoding='utf-8') as handle:
            data = json.load(handle)
        reacq = data.get('target_book_reacquisition') or {}
        geometry = data.get('target_book_geometry') or {}
        depth = geometry.get('depth') or {}
        holds = sum(int(data.get(key, 0)) for key in (
            'target_stale_events', 'sensor_hold_events', 'safety_stop_events'))
        scan_start = data.get('preapproach_scan_started_sim_sec')
        scan_end = data.get('preapproach_scan_completed_sim_sec')
        scan_duration = None
        if isinstance(scan_start, (int, float)) and isinstance(scan_end, (int, float)):
            scan_duration = max(0.0, float(scan_end) - float(scan_start))
        row = [
            path.stem,
            str(data.get('shelf_column_number')),
            str(data.get('book_colour_requested')),
            str(data.get('target_book_row')),
            str(data.get('row_mapping_source')),
            str(len(data.get('preapproach_scan_attempts') or [])),
            str(scan_duration),
            str(data.get('preapproach_scan_base_translation_m')),
            str(data.get('fallback_mapping_used')),
            str(reacq.get('confirming_frames')),
            str(depth.get('depth_m')),
            str(data.get('selected_manipulation_arm')),
            str(data.get('mission_duration_sim_sec')),
            str(data.get('day4_duration_sim_sec')),
            str(data.get('final_lidar_clearance_m')),
            str(holds),
        ]
        print('\t'.join(row))
        all_ok = all_ok and (
            data.get('passed') is True
            and data.get('day') == 4
            and data.get('day4_outcome') == 'pregrasp_geometry_ready'
            and data.get('primary_row_mapping_strategy')
            == 'stationary_preapproach_scan'
            and data.get('moving_head_during_approach_enabled') is False
            and data.get('row_mapping_source') == 'stationary_preapproach'
            and data.get('preapproach_scan_map_locked_before_approach') is True
            and data.get('preapproach_scan_head_restored') is True
            and data.get('preapproach_scan_nonzero_cmd_vel_publications') == 0
            and data.get('fallback_mapping_used') is False
            and data.get('row_mapping_requires_single_frame') is False
            and isinstance(reacq.get('confirming_frames'), int)
            and reacq['confirming_frames'] >= 3
            and holds == 0
        )

    if all_ok:
        print('[DAY4 REGRESSION SUMMARY][PASS]')
        return 0
    print('[DAY4 REGRESSION SUMMARY][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
