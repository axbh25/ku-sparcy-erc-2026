#!/usr/bin/env python3
"""Validate one KU SPARCy Day 2 result JSON and annotated image."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import cv2


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def parse_layout(text: Optional[str]) -> Optional[list[int]]:
    if text is None:
        return None
    try:
        values = [int(item.strip()) for item in text.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'layout must be comma-separated integers: {exc}') from exc
    if sorted(values) != [1, 2, 3, 4, 5]:
        raise argparse.ArgumentTypeError(
            'layout must contain each marker 1,2,3,4,5 exactly once')
    return values


def add_check(checks: Dict[str, bool], label: str, condition: bool) -> None:
    checks[label] = bool(condition)


def state_names(data: Dict[str, Any]) -> list[str]:
    history = data.get('state_history')
    if not isinstance(history, list):
        return []
    return [
        str(item.get('state'))
        for item in history
        if isinstance(item, dict) and item.get('state') is not None
    ]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--target', type=int, required=True, choices=range(1, 6))
    parser.add_argument(
        '--mode', choices=('fast', 'general'), required=True)
    parser.add_argument('--expected-layout', type=parse_layout)
    parser.add_argument(
        '--expect-during-motion', choices=('yes', 'no', 'any'),
        default='any')
    parser.add_argument('--progress-min', type=float)
    parser.add_argument('--progress-max', type=float)
    parser.add_argument('--require-all-markers', action='store_true')
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.result.is_file():
        print(f'[FAIL] Missing result JSON: {args.result}')
        return 1
    try:
        data = json.loads(args.result.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[FAIL] Invalid result JSON: {exc}')
        return 1

    checks: Dict[str, bool] = {}
    add_check(checks, 'node reported passed=true', data.get('passed') is True)
    add_check(checks, 'result identifies Day 2', data.get('day') == 2)
    add_check(
        checks, 'requested target value preserved',
        data.get('shelf_column_number') == args.target)
    add_check(
        checks, 'simulation clock used for motion timing',
        data.get('duration_clock') == 'simulation')

    add_check(
        checks, 'RGB camera dimensions are 640x360',
        data.get('frame_width') == 640 and data.get('frame_height') == 360)
    add_check(
        checks, 'RGB camera produced frames',
        isinstance(data.get('camera_frames_total'), int)
        and data['camera_frames_total'] >= 3)
    add_check(
        checks, 'RGB frames were processed',
        isinstance(data.get('processed_frames'), int)
        and data['processed_frames'] >= 3)
    add_check(
        checks, 'RGB camera stayed live during motion',
        isinstance(data.get('camera_frames_during_motion'), int)
        and data['camera_frames_during_motion'] >= 1)

    target = data.get('target_marker')
    target_ok = isinstance(target, dict)
    add_check(checks, 'target has a confirmed detection record', target_ok)
    if target_ok:
        configuration = data.get('detector_configuration', {})
        min_conf = float(configuration.get(
            'confirmation_average_confidence', 0.62))
        required_frames = int(configuration.get('confirmation_frames', 3))
        bbox = target.get('bbox_xywh')
        add_check(
            checks, 'confirmed digit equals requested target',
            target.get('digit') == args.target)
        add_check(
            checks, 'temporal confidence meets configured threshold',
            finite_number(target.get('confidence'))
            and float(target['confidence']) >= min_conf)
        add_check(
            checks, 'confirmation used enough distinct frames',
            isinstance(target.get('confirming_frames'), int)
            and target['confirming_frames'] >= required_frames)
        bbox_valid = (
            isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(item, int) for item in bbox)
            and bbox[0] >= 0 and bbox[1] >= 0
            and bbox[2] > 0 and bbox[3] > 0
        )
        add_check(checks, 'target bounding box is valid', bbox_valid)
        column_bbox = data.get('target_column_bbox_xywh')
        column_bbox_valid = (
            bbox_valid
            and isinstance(column_bbox, list) and len(column_bbox) == 4
            and all(isinstance(item, int) for item in column_bbox)
            and column_bbox[0] >= 0 and column_bbox[1] >= 0
            and column_bbox[2] > bbox[2]
            and column_bbox[3] > bbox[3]
        )
        add_check(
            checks,
            'target shelf-column evidence box is valid and larger than digit',
            column_bbox_valid)
        add_check(
            checks, 'confirmation followed first accepted observation',
            finite_number(data.get('time_to_first_detection_sim_sec'))
            and finite_number(data.get('time_to_confirmed_detection_sim_sec'))
            and data['time_to_confirmed_detection_sim_sec']
            >= data['time_to_first_detection_sim_sec'])

    add_check(
        checks, 'official ERC column topic was used',
        data.get('column_result_topic')
        == '/erc/shelf_column_identification')
    add_check(
        checks, 'official ERC Int32 type was recorded',
        data.get('column_result_message_type') == 'std_msgs/msg/Int32')
    add_check(
        checks, 'published value equals requested target',
        data.get('published_column_value') == args.target)
    add_check(
        checks, 'column result was published at least once',
        isinstance(data.get('column_publication_count'), int)
        and data['column_publication_count'] >= 1)

    image_path_raw = data.get('annotated_image_path')
    image_path = Path(image_path_raw) if isinstance(image_path_raw, str) else None
    image_readable = False
    image_has_pixels = False
    if image_path is not None and image_path.is_file():
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_readable = image is not None
        image_has_pixels = (
            image_readable and image.shape[:2] == (360, 640)
            and int(image.std()) > 0)
    add_check(
        checks, 'annotated image reported saved',
        data.get('annotated_image_saved') is True)
    add_check(
        checks, 'annotated image exists under team erc_images',
        image_path is not None and image_path.is_file()
        and '/src/ku_sparcy_erc/erc_images/' in str(image_path))
    add_check(checks, 'annotated image can be decoded', image_readable)
    add_check(checks, 'annotated image is a non-empty 640x360 frame', image_has_pixels)
    add_check(
        checks, 'image filename includes simulation and UTC timestamps',
        image_path is not None
        and 'sim_' in image_path.name and 'utc_' in image_path.name)
    add_check(
        checks, 'timestamp overlays were recorded',
        data.get('annotated_image_contains_sim_timestamp') is True
        and data.get('annotated_image_contains_wall_timestamp') is True)
    add_check(
        checks, 'annotated image records a target shelf-column box',
        data.get('annotated_image_contains_target_column_box') is True)

    if args.mode == 'fast':
        add_check(
            checks, 'result used Phase 1 fast-start mode',
            data.get('search_mode') == 'phase1_fast'
            and data.get('phase1_fast_start') is True)
        add_check(
            checks, 'Day 1 clockwise rotation direction preserved',
            finite_number(data.get('signed_rotation_deg'))
            and data['signed_rotation_deg'] < 0.0)
        add_check(
            checks, 'Day 1 rotation remains 90 +/- 3 degrees',
            finite_number(data.get('absolute_rotation_deg'))
            and abs(data['absolute_rotation_deg'] - 90.0) <= 3.0)
        add_check(
            checks, 'Day 1 final yaw error remains <= 2 degrees',
            finite_number(data.get('final_error_deg'))
            and abs(data['final_error_deg']) <= 2.0)
        add_check(
            checks, 'fast motion completed within 10 simulation seconds',
            finite_number(data.get('duration_sec'))
            and data['duration_sec'] <= 10.0)
        add_check(
            checks, 'head reached the Day 1 tilt target',
            finite_number(data.get('actual_head_tilt_rad'))
            and finite_number(data.get('head_tilt_rad'))
            and abs(data['actual_head_tilt_rad'] - data['head_tilt_rad']) <= 0.05)
        names = state_names(data)
        add_check(checks, 'fast state executed', 'ROTATING_FAST' in names)
    else:
        add_check(
            checks, 'result used orientation-independent mode',
            data.get('search_mode') == 'orientation_independent'
            and data.get('phase1_fast_start') is False)
        names = state_names(data)
        add_check(
            checks, 'general visual-search state executed',
            'SEARCH_FOR_SHELF' in names)
        add_check(
            checks, 'general target-alignment state executed',
            'ALIGN_TO_TARGET' in names)
        add_check(
            checks, 'general search rotated clockwise',
            finite_number(data.get('general_search_signed_rotation_deg'))
            and data['general_search_signed_rotation_deg'] < 0.0)
        tolerance = float(data.get('general_align_tolerance_px', 30.0))
        add_check(
            checks, 'general alignment ended within pixel tolerance',
            finite_number(data.get('final_target_pixel_error_px'))
            and abs(data['final_target_pixel_error_px']) <= tolerance + 5.0)
        add_check(
            checks, 'general motion completed within 25 simulation seconds',
            finite_number(data.get('duration_sec'))
            and data['duration_sec'] <= 25.0)

    if args.expect_during_motion != 'any':
        expected = args.expect_during_motion == 'yes'
        add_check(
            checks,
            f'target confirmed during motion expected={expected}',
            data.get('target_confirmed_during_motion') is expected)

    progress = data.get('rotation_progress_deg_at_confirmation')
    if args.progress_min is not None:
        add_check(
            checks,
            f'confirmation progress >= {args.progress_min:.1f} deg',
            finite_number(progress) and progress >= args.progress_min)
    if args.progress_max is not None:
        add_check(
            checks,
            f'confirmation progress <= {args.progress_max:.1f} deg',
            finite_number(progress) and progress <= args.progress_max)

    if args.require_all_markers or args.expected_layout is not None:
        add_check(
            checks, 'all five marker values were temporally confirmed',
            data.get('confirmed_marker_count') == 5
            and sorted(int(item) for item in data.get(
                'confirmed_markers', {}).keys()) == [1, 2, 3, 4, 5])
    if args.expected_layout is not None:
        add_check(
            checks, 'observed left-to-right layout matches ERC seed oracle',
            data.get('observed_left_to_right') == args.expected_layout)
        add_check(
            checks, 'derived target physical column index matches seed oracle',
            data.get('target_column_index_left_to_right')
            == args.expected_layout.index(args.target) + 1)

    for label, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print('\nMeasured Day 2 result:')
    for key in (
        'search_mode',
        'shelf_column_number',
        'time_to_first_detection_sim_sec',
        'time_to_confirmed_detection_sim_sec',
        'target_confirmation_yaw_deg',
        'rotation_progress_deg_at_confirmation',
        'target_confirmed_during_motion',
        'signed_rotation_deg',
        'absolute_rotation_deg',
        'final_error_deg',
        'final_target_pixel_error_px',
        'duration_sec',
        'wall_duration_sec',
        'camera_frames_during_motion',
        'confirmed_marker_count',
        'observed_left_to_right',
        'target_column_bbox_xywh',
        'target_column_index_left_to_right',
        'detector_processing_ms_mean',
        'detector_processing_ms_max',
        'column_publication_count',
        'annotated_image_path',
        'reason',
    ):
        print(f'  {key}: {data.get(key)}')

    rgb_gate = (
        data.get('frame_width') == 640
        and data.get('frame_height') == 360
        and isinstance(data.get('processed_frames'), int)
        and data['processed_frames'] >= 3
        and isinstance(data.get('camera_frames_during_motion'), int)
        and data['camera_frames_during_motion'] >= 1
    )
    target_gate = (
        target_ok and target.get('digit') == args.target
        and isinstance(target.get('bbox_xywh'), list)
    )
    confidence_gate = (
        target_ok
        and finite_number(target.get('confidence'))
        and isinstance(target.get('confirming_frames'), int)
        and target['confirming_frames'] >= 3
    )
    publication_gate = (
        data.get('column_result_topic')
        == '/erc/shelf_column_identification'
        and data.get('published_column_value') == args.target
        and isinstance(data.get('column_publication_count'), int)
        and data['column_publication_count'] >= 1
    )
    image_gate = (
        data.get('annotated_image_saved') is True
        and image_path is not None and image_path.is_file()
        and image_readable and image_has_pixels
        and data.get('annotated_image_contains_target_column_box') is True
    )
    print()
    print(f"[RGB PERCEPTION][{'PASS' if rgb_gate else 'FAIL'}]")
    print(f"[TARGET MARKER DETECTION][{'PASS' if target_gate else 'FAIL'}]")
    print(f"[CONFIDENCE FILTER][{'PASS' if confidence_gate else 'FAIL'}]")
    print(f"[COLUMN RESULT PUBLICATION][{'PASS' if publication_gate else 'FAIL'}]")
    print(f"[ANNOTATED IMAGE][{'PASS' if image_gate else 'FAIL'}]")
    if args.mode == 'general':
        general_gate = (
            data.get('search_mode') == 'orientation_independent'
            and 'SEARCH_FOR_SHELF' in state_names(data)
            and 'ALIGN_TO_TARGET' in state_names(data)
            and finite_number(data.get('final_target_pixel_error_px'))
            and abs(data['final_target_pixel_error_px'])
            <= float(data.get('general_align_tolerance_px', 30.0)) + 5.0
        )
        print(
            f"[GENERAL SHELF SEARCH][{'PASS' if general_gate else 'FAIL'}]")

    if all(checks.values()):
        print('\n[DAY2 RESULT][PASS]')
        return 0
    print('\n[DAY2 RESULT][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
