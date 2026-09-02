#!/usr/bin/env python3
"""Print a compact quantitative summary of Day 2 JSON results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'directory', nargs='?', type=Path,
        default=Path('/opt/erc_ws/src/ku_sparcy_erc/day2_results'))
    parser.add_argument('--expected-count', type=int, default=9)
    args = parser.parse_args()

    paths = sorted(args.directory.glob('*.json'))
    if not paths:
        print(f'[DAY2 MATRIX SUMMARY][FAIL] No JSON files in {args.directory}')
        return 1

    rows = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            print(f'[FAIL] {path.name}: {exc}')
            return 1
        target = data.get('target_marker') or {}
        rows.append({
            'case': path.stem,
            'pass': data.get('passed') is True,
            'mode': data.get('search_mode'),
            'target': data.get('shelf_column_number'),
            'confidence': number(target.get('confidence')),
            'frames': target.get('confirming_frames'),
            'first_sim': number(data.get('time_to_first_detection_sim_sec')),
            'confirm_sim': number(data.get('time_to_confirmed_detection_sim_sec')),
            'progress_deg': number(data.get('rotation_progress_deg_at_confirmation')),
            'motion_sim': number(data.get('duration_sec')),
            'motion_wall': number(data.get('wall_duration_sec')),
            'detector_ms': number(data.get('detector_processing_ms_mean')),
            'published': data.get('published_column_value'),
            'image': data.get('annotated_image_saved') is True,
        })

    header = (
        'case', 'pass', 'mode', 'target', 'confidence', 'frames',
        'first_sim', 'confirm_sim', 'progress_deg', 'motion_sim',
        'motion_wall', 'detector_ms', 'published', 'image')
    print('\t'.join(header))
    for row in rows:
        print('\t'.join(str(row[key]) for key in header))

    confirms = [row['confirm_sim'] for row in rows if row['confirm_sim'] is not None]
    confidences = [row['confidence'] for row in rows if row['confidence'] is not None]
    all_pass = all(row['pass'] and row['image'] for row in rows)
    count_ok = len(rows) >= args.expected_count
    print()
    print(f'cases={len(rows)} expected_at_least={args.expected_count}')
    if confirms:
        print(f'confirmed_detection_sim_mean={statistics.fmean(confirms):.3f}')
        print(f'confirmed_detection_sim_max={max(confirms):.3f}')
    if confidences:
        print(f'confidence_mean={statistics.fmean(confidences):.3f}')
        print(f'confidence_min={min(confidences):.3f}')

    if all_pass and count_ok:
        print('[DAY2 MATRIX SUMMARY][PASS]')
        return 0
    print('[DAY2 MATRIX SUMMARY][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
