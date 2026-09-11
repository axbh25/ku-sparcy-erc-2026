#!/usr/bin/env python3
"""Validate stationary pre-approach row mapping and close target geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import cv2


COLOURS = ('red', 'green', 'yellow', 'blue')


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def image_ok(path_value: Any) -> bool:
    if not isinstance(path_value, str):
        return False
    path = Path(path_value)
    if not path.is_file() or '/src/ku_sparcy_erc/erc_images/' not in str(path):
        return False
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return (
        image is not None
        and image.shape[:2] == (360, 640)
        and float(image.std()) > 0.0
        and 'sim_' in path.name
        and 'utc_' in path.name
    )


def add(checks: Dict[str, bool], label: str, value: bool) -> None:
    checks[label] = bool(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--target', type=int, required=True, choices=range(1, 6))
    parser.add_argument(
        '--colour', required=True,
        choices=('red', 'green', 'yellow', 'blue'))
    parser.add_argument('--expected-row', type=int, choices=range(1, 5))
    parser.add_argument(
        '--mode', required=True,
        choices=(
            'preapproach_visibility',
            'visibility',
            'perception',
            'geometry',
            'pregrasp',
        ))
    parser.add_argument('--require-zero-holds', action='store_true')
    parser.add_argument('--require-preapproach-lock', action='store_true')
    parser.add_argument(
        '--require-passive-lock', action='store_true',
        help='Deprecated alias for --require-preapproach-lock.')
    args = parser.parse_args()

    if args.mode == 'visibility':
        args.mode = 'preapproach_visibility'
    require_preapproach = (
        args.require_preapproach_lock or args.require_passive_lock
    )

    if not args.result.is_file():
        print(f'[FAIL] Missing Day 4 result: {args.result}')
        return 1
    try:
        data = json.loads(args.result.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[FAIL] Invalid Day 4 JSON: {exc}')
        return 1

    profile_mode = args.mode == 'preapproach_visibility'
    checks: Dict[str, bool] = {}
    add(checks, 'node reported passed=true', data.get('passed') is True)
    add(checks, 'result identifies Day 4', data.get('day') == 4)
    add(
        checks, 'official merged gripper baseline is recorded',
        data.get('official_upstream_required_commit')
        == '93554d4f9335b2ee3acb49c6b332611f6ad2a964'
        and finite(data.get('expected_official_book_spine_width_m'))
        and abs(float(data['expected_official_book_spine_width_m']) - 0.02)
        <= 1.0e-9,
    )
    add(
        checks, 'validated Day 3 implementation was reused',
        data.get('day3_mission_reused') is True
        and data.get('day3_commit_baseline')
        == '9a9768acb824862f2dd4b8de9950ffed54828987',
    )
    add(checks, 'requested shelf marker is preserved',
        data.get('shelf_column_number') == args.target)
    add(checks, 'requested book colour is preserved',
        data.get('book_colour_requested') == args.colour)
    add(checks, 'simulation time remains the mission clock',
        data.get('duration_clock') == 'simulation')

    add(
        checks, 'requested physical column was locked once',
        data.get('column_lock_active') is True
        and data.get('column_lock_mode') == 'identify_once_then_odom_lidar',
    )
    add(
        checks, 'travel arms reached the validated navigation pose before scan',
        data.get('travel_arm_pose_ready') is True
        and finite(data.get('travel_arm_max_error_rad'))
        and float(data['travel_arm_max_error_rad']) <= 0.050001,
    )
    add(
        checks, 'official column publication is preserved',
        data.get('column_result_topic') == '/erc/shelf_column_identification'
        and data.get('column_result_message_type') == 'std_msgs/msg/Int32'
        and data.get('published_column_value') == args.target
        and isinstance(data.get('column_publication_count'), int)
        and data['column_publication_count'] >= 1,
    )
    add(checks, 'Day 2 target-column image remains present',
        image_ok(data.get('annotated_image_path')))

    add(
        checks, 'stationary pre-approach strategy is authoritative',
        data.get('primary_row_mapping_strategy')
        == 'stationary_preapproach_scan',
    )
    add(
        checks, 'moving-head-during-approach is disabled',
        data.get('moving_head_during_approach_enabled') is False,
    )
    add(
        checks, 'pre-approach scan started and restored Day 3 head pose',
        data.get('preapproach_scan_started') is True
        and data.get('preapproach_scan_head_restored') is True,
    )
    add(
        checks, 'stationary scan published no nonzero base command',
        data.get('preapproach_scan_nonzero_cmd_vel_publications') == 0,
    )
    add(
        checks, 'stationary base hold produced explicit zero commands',
        isinstance(data.get('preapproach_scan_zero_cmd_vel_publications'), int)
        and data['preapproach_scan_zero_cmd_vel_publications'] >= 4,
    )
    translation = data.get('preapproach_scan_base_translation_m')
    translation_limit = data.get(
        'preapproach_scan_base_translation_tolerance_m')
    yaw_change = data.get('preapproach_scan_base_yaw_change_deg')
    yaw_limit = data.get('preapproach_scan_base_yaw_tolerance_deg')
    add(
        checks, 'base translation stayed inside stationary tolerance',
        finite(translation) and finite(translation_limit)
        and float(translation) <= float(translation_limit),
    )
    add(
        checks, 'base yaw stayed inside stationary tolerance',
        finite(yaw_change) and finite(yaw_limit)
        and float(yaw_change) <= float(yaw_limit),
    )
    attempts = data.get('preapproach_scan_attempts')
    sequence = data.get('preapproach_scan_sequence_used_rad')
    add(
        checks, 'settled scan pose evidence exists',
        isinstance(attempts, list) and len(attempts) >= 1
        and isinstance(sequence, list) and len(attempts) == len(sequence)
        and all(
            isinstance(item, dict)
            and item.get('settled') is True
            and isinstance(item.get('processed_frames'), int)
            and item['processed_frames'] >= 4
            and image_ok(item.get('evidence_image_path'))
            for item in attempts
        ),
    )
    add(
        checks, 'row mapping does not require all colours in one frame',
        data.get('row_mapping_requires_single_frame') is False,
    )

    samples = data.get('passive_mapping_visibility_samples')
    add(
        checks, 'stationary scan mapping samples were recorded',
        isinstance(samples, list) and len(samples) >= 4,
    )

    if profile_mode:
        add(
            checks, 'visibility-only outcome is correct',
            data.get('day4_outcome')
            == 'preapproach_visibility_profile_only',
        )
        add(
            checks, 'profile completed every configured head pose',
            data.get('preapproach_scan_profile_only') is True
            and isinstance(attempts, list)
            and isinstance(sequence, list)
            and len(attempts) == len(sequence),
        )
        add(
            checks, 'Day 3 translation did not start during profile',
            data.get('preapproach_scan_approach_started_sim_sec') is None
            and data.get('approach_started_sim_sec') is None,
        )
        add(
            checks, 'visibility mode did not publish a row',
            data.get('row_publication_count') == 0
            and data.get('published_row_value') is None,
        )
        add(
            checks, 'visibility mode performed no manipulation',
            data.get('day4_new_arm_trajectory_commanded') is False
            and data.get('day4_torso_commanded') is False
            and data.get('day4_gripper_commanded') is False
            and data.get('book_contact_attempted') is False
            and data.get('book_lift_attempted') is False,
        )
    else:
        add(
            checks, 'stationary scan calibration was applied',
            data.get('preapproach_scan_calibrated') is True,
        )
        add(
            checks, 'Day 3 safe stand-off completed after row mapping',
            data.get('approach_outcome') == 'safe_standoff'
            and data.get('final_geometry_source') == 'locked_column_odometry'
            and data.get('final_forward_standoff_source') == 'front_lidar',
        )
        final_lidar = data.get('final_lidar_clearance_m')
        add(
            checks, 'final LiDAR clearance remains in validated interval',
            finite(final_lidar) and 0.82 <= float(final_lidar) <= 1.18,
        )
        add(checks, 'Day 3 approach image remains present',
            image_ok(data.get('approach_annotated_image_path')))
        add(
            checks, 'head restore completed before Day 3 translation',
            finite(data.get('preapproach_scan_completed_sim_sec'))
            and finite(data.get('preapproach_scan_approach_started_sim_sec'))
            and float(data['preapproach_scan_completed_sim_sec'])
            <= float(data['preapproach_scan_approach_started_sim_sec'])
            and data.get('preapproach_scan_complete') is True,
        )

        layout = data.get('book_layout_confirmation')
        layout_ok = isinstance(layout, dict)
        add(checks, 'multi-frame spatial row map exists', layout_ok)
        if layout_ok:
            rows = layout.get('row_colours_top_to_bottom')
            tracks = layout.get('tracks_by_colour')
            add(checks, 'row map contains every colour exactly once',
                isinstance(rows, list) and sorted(rows) == sorted(COLOURS))
            add(checks, 'row map explicitly rejects a single-frame requirement',
                layout.get('single_frame_required') is False)
            add(checks, 'every colour has a three-frame spatial track',
                isinstance(tracks, dict)
                and all(
                    isinstance(tracks.get(colour), dict)
                    and isinstance(tracks[colour].get('confirming_frames'), int)
                    and tracks[colour]['confirming_frames'] >= 3
                    for colour in COLOURS
                ))
            add(checks, 'spatial row-map confidence is acceptable',
                finite(layout.get('confidence'))
                and float(layout['confidence']) >= 0.60)
            separations = layout.get('row_separations_m')
            add(checks, 'row separations are physically ordered',
                isinstance(separations, list)
                and len(separations) == 3
                and all(finite(value) and 0.14 <= float(value) <= 0.55
                        for value in separations))

        add(
            checks, 'row map source is stationary pre-approach scan',
            data.get('row_mapping_source') == 'stationary_preapproach',
        )
        add(
            checks, 'row map locked before Day 3 translation',
            data.get('preapproach_scan_map_locked_before_approach') is True
            and finite(data.get('passive_mapping_lock_sim_sec'))
            and finite(data.get('preapproach_scan_approach_started_sim_sec'))
            and float(data['passive_mapping_lock_sim_sec'])
            <= float(data['preapproach_scan_approach_started_sim_sec']),
        )
        add(
            checks, 'requested row was first published before translation',
            finite(data.get('first_row_publish_sim_sec'))
            and finite(data.get('preapproach_scan_approach_started_sim_sec'))
            and float(data['first_row_publish_sim_sec'])
            <= float(data['preapproach_scan_approach_started_sim_sec']),
        )
        if require_preapproach:
            add(
                checks, 'primary scan locked row map without close fallback',
                data.get('row_mapping_source') == 'stationary_preapproach'
                and data.get('preapproach_scan_map_locked_before_approach') is True
                and data.get('fallback_mapping_used') is False,
            )

        row = data.get('target_book_row')
        add(checks, 'target row is a valid active row',
            isinstance(row, int) and 1 <= row <= 4)
        if args.expected_row is not None:
            add(checks, 'target row matches validation-only seed oracle',
                row == args.expected_row)
        add(
            checks, 'official row topic and Int32 value are correct',
            data.get('row_result_topic') == '/erc/shelf_row_identification'
            and data.get('row_result_message_type') == 'std_msgs/msg/Int32'
            and data.get('published_row_value') == row
            and isinstance(data.get('row_publication_count'), int)
            and data['row_publication_count'] >= 1,
        )

        reacquisition = data.get('target_book_reacquisition')
        reacq_ok = isinstance(reacquisition, dict)
        add(checks, 'requested colour was reacquired at final stand-off', reacq_ok)
        if reacq_ok:
            add(checks, 'reacquisition uses requested colour and locked row',
                reacquisition.get('colour') == args.colour
                and reacquisition.get('row') == row)
            add(checks, 'close target requires multiple confirming frames',
                isinstance(reacquisition.get('confirming_frames'), int)
                and reacquisition['confirming_frames'] >= 3)
            add(checks, 'close target confidence is acceptable',
                finite(reacquisition.get('average_confidence'))
                and float(reacquisition['average_confidence']) >= 0.60)

        target = data.get('target_book_detection')
        target_ok = isinstance(target, dict)
        add(checks, 'live requested target-book detection exists', target_ok)
        if target_ok:
            bbox = target.get('bbox_xywh')
            add(checks, 'target colour and live bounding box are valid',
                target.get('colour') == args.colour
                and isinstance(bbox, list) and len(bbox) == 4
                and all(isinstance(value, int) for value in bbox)
                and bbox[0] >= 0 and bbox[1] >= 0
                and bbox[2] > 0 and bbox[3] > 0)
        add(checks, 'live close-range target-book evidence exists',
            data.get('target_book_evidence_saved') is True
            and image_ok(data.get('target_book_evidence_path')))

        if args.mode in {'geometry', 'pregrasp'}:
            geometry = data.get('target_book_geometry')
            geometry_ok = isinstance(geometry, dict)
            add(checks, 'target-book 3-D geometry exists', geometry_ok)
            if geometry_ok:
                point = geometry.get('base_point_xyz_m')
                depth = geometry.get('depth')
                add(checks, 'target-book point is finite and in front',
                    isinstance(point, list) and len(point) == 3
                    and all(finite(value) for value in point)
                    and float(point[0]) > 0.20)
                add(checks, 'book RGB/depth skew remains <= 0.20 s',
                    finite(geometry.get('rgb_depth_stamp_skew_sec'))
                    and float(geometry['rgb_depth_stamp_skew_sec']) <= 0.20)
                add(checks, 'book depth has enough valid target pixels',
                    isinstance(depth, dict)
                    and isinstance(depth.get('valid_pixels'), int)
                    and depth['valid_pixels'] >= 12
                    and finite(depth.get('valid_fraction'))
                    and float(depth['valid_fraction']) >= 0.12)
            add(checks, 'live geometry evidence exists',
                image_ok(data.get('target_book_geometry_image_path')))

        if args.mode == 'pregrasp':
            plan = data.get('pregrasp_plan')
            plan_ok = isinstance(plan, dict)
            add(checks, 'non-executing pre-grasp screen exists', plan_ok)
            if plan_ok:
                add(checks, 'exactly one candidate arm passed geometry screen',
                    plan.get('selected_arm') in {'left', 'right'}
                    and plan.get('geometric_screen_passed') is True)
                add(checks, 'pre-grasp screen performed no IK or motion',
                    plan.get('ik_solved') is False
                    and plan.get('collision_checked') is False
                    and plan.get('arm_motion_executed') is False)

    add(checks, 'one-arm rule metadata is preserved',
        data.get('single_arm_rule_respected') is True)
    add(checks, 'competition Day 4 performed no manipulation motion',
        data.get('day4_new_arm_trajectory_commanded') is False
        and data.get('day4_torso_commanded') is False
        and data.get('day4_gripper_commanded') is False
        and data.get('gripper_effort_commanded') is False
        and data.get('book_contact_attempted') is False
        and data.get('book_lift_attempted') is False)
    if args.require_zero_holds and not profile_mode:
        add(checks, 'navigation completed with zero stale/sensor/safety holds',
            data.get('target_stale_events') == 0
            and data.get('sensor_hold_events') == 0
            and data.get('safety_stop_events') == 0)

    for label, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print('\nMeasured revised Day 4 result:')
    for key in (
        'shelf_column_number',
        'book_colour_requested',
        'target_book_row',
        'row_mapping_source',
        'preapproach_scan_profile_only',
        'preapproach_scan_sequence_used_rad',
        'preapproach_scan_map_locked_before_approach',
        'preapproach_scan_base_translation_m',
        'preapproach_scan_base_yaw_change_deg',
        'preapproach_scan_approach_started_sim_sec',
        'first_row_publish_sim_sec',
        'fallback_mapping_used',
        'target_head_sequence_rad',
        'target_head_pose_index_used',
        'day4_outcome',
        'day4_duration_sim_sec',
        'day4_duration_wall_sec',
        'final_lidar_clearance_m',
        'row_publication_count',
        'target_book_evidence_path',
        'target_book_geometry_image_path',
        'selected_manipulation_arm',
        'reason',
    ):
        print(f'  {key}: {data.get(key)}')

    if profile_mode:
        gate = (
            data.get('day4_outcome') == 'preapproach_visibility_profile_only'
            and data.get('preapproach_scan_head_restored') is True
            and data.get('preapproach_scan_nonzero_cmd_vel_publications') == 0
        )
        print(
            f"[STATIONARY PRE-APPROACH VISIBILITY][{'PASS' if gate else 'FAIL'}]"
        )
    else:
        layout_gate = (
            isinstance(data.get('book_layout_confirmation'), dict)
            and data.get('row_mapping_source') == 'stationary_preapproach'
            and data.get('preapproach_scan_map_locked_before_approach') is True
        )
        reacq_gate = isinstance(data.get('target_book_reacquisition'), dict)
        row_gate = (
            isinstance(data.get('target_book_row'), int)
            and data.get('published_row_value') == data.get('target_book_row')
            and data.get('row_publication_count', 0) >= 1
        )
        image_gate = (
            data.get('target_book_evidence_saved') is True
            and image_ok(data.get('target_book_evidence_path'))
        )
        print(f"[STATIONARY ROW MAPPING][{'PASS' if layout_gate else 'FAIL'}]")
        print('[NO SINGLE-FRAME REQUIREMENT][PASS]' if
              data.get('row_mapping_requires_single_frame') is False else
              '[NO SINGLE-FRAME REQUIREMENT][FAIL]')
        print(f"[TARGET-ONLY REACQUISITION][{'PASS' if reacq_gate else 'FAIL'}]")
        print(f"[SHELF ROW PUBLICATION][{'PASS' if row_gate else 'FAIL'}]")
        print(f"[TARGET BOOK EVIDENCE][{'PASS' if image_gate else 'FAIL'}]")
        if args.mode in {'geometry', 'pregrasp'}:
            geometry_gate = isinstance(data.get('target_book_geometry'), dict)
            print(f"[BOOK 3D GEOMETRY][{'PASS' if geometry_gate else 'FAIL'}]")
        if args.mode == 'pregrasp':
            plan = data.get('pregrasp_plan')
            plan_gate = (
                isinstance(plan, dict)
                and plan.get('geometric_screen_passed') is True
            )
            print(f"[ARM SELECTION][{'PASS' if plan_gate else 'FAIL'}]")
            print(f"[PREGRASP GEOMETRY][{'PASS' if plan_gate else 'FAIL'}]")

    manipulation_inhibited = (
        data.get('day4_new_arm_trajectory_commanded') is False
        and data.get('day4_gripper_commanded') is False
        and data.get('day4_torso_commanded') is False
        and data.get('book_lift_attempted') is False
    )
    print(
        f"[MANIPULATION MOTION INHIBIT][{'PASS' if manipulation_inhibited else 'FAIL'}]"
    )

    if all(checks.values()):
        print('[DAY4 RESULT][PASS]')
        return 0
    print('[DAY4 RESULT][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
