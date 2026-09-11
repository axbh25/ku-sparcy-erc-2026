#!/usr/bin/env python3
"""Offline integration test for stationary profile analyzer and apply tool."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile


COLOURS = ('red', 'green', 'yellow', 'blue')
HEIGHTS = {'red': 1.80, 'green': 1.45, 'yellow': 1.10, 'blue': 0.75}


def observation(colour: str, frame: int, stamp: float, tilt: float) -> dict:
    return {
        'colour': colour,
        'stamp_sec': stamp,
        'frame_index': frame,
        'head_tilt_rad': tilt,
        'lidar_clearance_m': 2.85,
        'detection': {
            'colour': colour,
            'confidence': 0.90,
            'bbox_xywh': [300, 40, 12, 48],
        },
        'geometry': {
            'base_point_xyz_m': [2.70, 0.03, HEIGHTS[colour]],
        },
        'odom_xyz_m': [3.00, 0.03, HEIGHTS[colour] + 0.001 * (frame % 3)],
        'roi_xywh': [200, 0, 240, 360],
    }


def main() -> int:
    tools = Path(__file__).resolve().parent
    package_root = tools.parent
    analyzer = tools / 'analyze_day4_preapproach_scan.py'
    apply_tool = tools / 'apply_day4_preapproach_scan.py'
    source_config = package_root / 'config' / 'day4_books.yaml'

    sequence = [0.35, 0.10, -0.10]
    attempts = []
    for index, tilt in enumerate(sequence):
        attempts.append({
            'pose_index': index,
            'target_tilt_rad': tilt,
            'actual_tilt_rad': tilt + 0.002,
            'settled': True,
            'dwell_sim_sec': 1.0,
            'processed_frames': 8,
            'sample_count': 8,
            'visible_frame_counts': {colour: 3 for colour in COLOURS},
            'geometry_frame_counts': {colour: 3 for colour in COLOURS},
            'accepted_frame_counts': {colour: 3 for colour in COLOURS},
        })

    observations = {colour: [] for colour in COLOURS}
    frame = 1
    # +0.35 sees red/yellow/blue; -0.10 sees red/green/blue.  No one pose
    # contains every colour, so the minimal valid sequence must contain two.
    for tilt, visible in (
        (0.35, ('red', 'yellow', 'blue')),
        (-0.10, ('red', 'green', 'blue')),
    ):
        for repeat in range(3):
            stamp = 10.0 + 0.08 * frame
            for colour in visible:
                observations[colour].append(
                    observation(colour, frame, stamp, tilt)
                )
            frame += 1

    layout = {
        'mapping_source': 'stationary_preapproach',
        'single_frame_required': False,
        'row_colours_top_to_bottom': ['red', 'green', 'yellow', 'blue'],
        'row_separations_m': [0.35, 0.35, 0.35],
    }
    result = {
        'passed': True,
        'day4_outcome': 'preapproach_visibility_profile_only',
        'primary_row_mapping_strategy': 'stationary_preapproach_scan',
        'moving_head_during_approach_enabled': False,
        'preapproach_scan_profile_only': True,
        'preapproach_scan_started': True,
        'preapproach_scan_head_restored': True,
        'preapproach_scan_approach_started_sim_sec': None,
        'approach_started_sim_sec': None,
        'preapproach_scan_nonzero_cmd_vel_publications': 0,
        'preapproach_scan_zero_cmd_vel_publications': 64,
        'preapproach_scan_base_translation_tolerance_m': 0.03,
        'preapproach_scan_base_yaw_tolerance_deg': 2.0,
        'preapproach_scan_base_translation_m': 0.002,
        'preapproach_scan_base_yaw_change_deg': 0.1,
        'preapproach_scan_sequence_used_rad': sequence,
        'preapproach_scan_attempts': attempts,
        'preapproach_scan_min_dwell_sec': 0.90,
        'preapproach_scan_min_frames_per_pose': 4,
        'preapproach_scan_head_tolerance_rad': 0.06,
        'book_layout_confirmation': layout,
        'book_map_settings': {
            'required_frames_per_colour': 3,
            'minimum_observation_span_sec': 0.05,
            'minimum_average_confidence': 0.60,
            'maximum_cluster_xy_radius_m': 0.18,
            'maximum_cluster_z_radius_m': 0.10,
            'maximum_track_xy_span_m': 0.18,
            'maximum_track_z_span_m': 0.10,
            'minimum_row_separation_m': 0.14,
            'maximum_row_separation_m': 0.55,
        },
        'preapproach_scan_observations_by_colour': observations,
    }

    with tempfile.TemporaryDirectory(prefix='day4_preapproach_test_') as temp:
        temp_path = Path(temp)
        result_path = temp_path / 'profile.json'
        analysis_path = temp_path / 'analysis.json'
        config_path = temp_path / 'day4_books.yaml'
        for attempt in attempts:
            evidence = temp_path / f"pose_{attempt['pose_index']}.png"
            evidence.write_bytes(b'PNG test evidence')
            attempt['evidence_image_path'] = str(evidence)
        result_path.write_text(json.dumps(result), encoding='utf-8')
        shutil.copy2(source_config, config_path)

        analyzed = subprocess.run(
            [str(analyzer), str(result_path), '--output', str(analysis_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        print(analyzed.stdout, end='')
        if analyzed.returncode != 0:
            print(analyzed.stderr, end='')
            print('[PRE-APPROACH ANALYZER TEST][FAIL]')
            return 1
        report = json.loads(analysis_path.read_text(encoding='utf-8'))
        recommended = report.get('recommended_head_tilt_sequence_rad')
        if recommended != [0.35, -0.10]:
            print('[PRE-APPROACH ANALYZER TEST][FAIL] unexpected sequence:', recommended)
            return 1

        applied = subprocess.run(
            [str(apply_tool), str(analysis_path), str(config_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        print(applied.stdout, end='')
        if applied.returncode != 0:
            print(applied.stderr, end='')
            print('[PRE-APPROACH APPLY TEST][FAIL]')
            return 1
        updated = config_path.read_text(encoding='utf-8')
        if 'preapproach_scan_calibrated: true' not in updated:
            print('[PRE-APPROACH APPLY TEST][FAIL] calibrated flag not set')
            return 1
        if 'preapproach_scan_head_tilt_sequence_rad: [0.35, -0.10]' not in updated:
            print('[PRE-APPROACH APPLY TEST][FAIL] sequence not applied')
            return 1

    print('[PRE-APPROACH ANALYZER TEST][PASS]')
    print('[PRE-APPROACH APPLY TEST][PASS]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
