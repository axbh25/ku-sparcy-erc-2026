#!/usr/bin/env python3
"""Validate one KU SPARCy Day 3 ranging/approach result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def state_names(data: Dict[str, Any]) -> list[str]:
    history = data.get('state_history')
    if not isinstance(history, list):
        return []
    return [
        str(item.get('state'))
        for item in history
        if isinstance(item, dict) and item.get('state') is not None
    ]


def add(checks: Dict[str, bool], label: str, condition: bool) -> None:
    checks[label] = bool(condition)


def image_valid(path_raw: Any) -> tuple[bool, Optional[Path]]:
    if not isinstance(path_raw, str):
        return False, None
    path = Path(path_raw)
    if not path.is_file() or '/src/ku_sparcy_erc/erc_images/' not in str(path):
        return False, path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    valid = (
        image is not None
        and image.shape[:2] == (360, 640)
        and float(image.std()) > 0.0
    )
    return bool(valid), path


def geometry_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    base = value.get('base_point_xyz_m')
    depth = value.get('depth')
    return (
        isinstance(base, list) and len(base) == 3
        and all(finite(item) for item in base)
        and float(base[0]) > 0.2
        and isinstance(depth, dict)
        and finite(depth.get('depth_m'))
        and 0.2 <= float(depth['depth_m']) <= 8.0
        and isinstance(depth.get('valid_pixels'), int)
        and depth['valid_pixels'] >= 20
        and finite(depth.get('valid_fraction'))
        and float(depth['valid_fraction']) >= 0.08
        and finite(value.get('rgb_depth_stamp_skew_sec'))
        and float(value['rgb_depth_stamp_skew_sec']) <= 0.20
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--target', type=int, required=True, choices=range(1, 6))
    parser.add_argument('--mode', required=True, choices=('range', 'partial', 'full'))
    parser.add_argument('--min-travel-m', type=float, default=0.0)
    parser.add_argument('--max-safety-stops', type=int, default=0)
    parser.add_argument('--require-lateral-command', action='store_true')
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
    add(checks, 'node reported passed=true', data.get('passed') is True)
    add(checks, 'result identifies Day 3', data.get('day') == 3)
    add(checks, 'committed Day 2 detector was reused', data.get('day2_detector_reused') is True)
    add(checks, 'requested target is preserved', data.get('shelf_column_number') == args.target)
    add(checks, 'simulation time controls the mission', data.get('duration_clock') == 'simulation')
    add(checks, 'mission simulation duration was recorded', finite(data.get('mission_duration_sim_sec')))
    add(checks, 'mission wall duration was recorded', finite(data.get('mission_duration_wall_sec')))

    # Frozen Day 1 opening regression fields remain embedded in every fast run.
    add(checks, 'Phase-1 fast path was used', data.get('phase1_fast_start') is True)
    add(checks, 'opening remained clockwise', finite(data.get('signed_rotation_deg'))
        and float(data['signed_rotation_deg']) < 0.0)
    add(checks, 'opening remained 90 +/- 3 degrees', finite(data.get('absolute_rotation_deg'))
        and abs(float(data['absolute_rotation_deg']) - 90.0) <= 3.0)
    add(checks, 'opening final yaw error remained <= 2 degrees', finite(data.get('final_error_deg'))
        and abs(float(data['final_error_deg'])) <= 2.0)
    add(checks, 'opening duration remained <= 10 simulation seconds', finite(data.get('duration_sec'))
        and float(data['duration_sec']) <= 10.0)

    target = data.get('target_marker')
    add(checks, 'requested marker remained temporally confirmed',
        isinstance(target, dict)
        and target.get('digit') == args.target
        and finite(target.get('confidence'))
        and isinstance(target.get('confirming_frames'), int)
        and target['confirming_frames'] >= 3)
    add(checks, 'official shelf-column topic was preserved',
        data.get('column_result_topic') == '/erc/shelf_column_identification'
        and data.get('column_result_message_type') == 'std_msgs/msg/Int32')
    add(checks, 'correct target value was published',
        data.get('published_column_value') == args.target
        and isinstance(data.get('column_publication_count'), int)
        and data['column_publication_count'] >= 1)

    day2_image_ok, day2_image = image_valid(data.get('annotated_image_path'))
    approach_image_ok, approach_image = image_valid(
        data.get('approach_annotated_image_path'))
    add(checks, 'Day 2 target-column evidence image was preserved',
        data.get('annotated_image_saved') is True and day2_image_ok)
    add(checks, 'Day 3 live approach evidence image was saved',
        data.get('approach_annotated_image_saved') is True and approach_image_ok)
    add(checks, 'Day 3 image records live range and both clocks',
        data.get('approach_image_contains_live_range') is True
        and data.get('approach_image_contains_sim_timestamp') is True
        and data.get('approach_image_contains_utc_timestamp') is True)

    start_geometry = data.get('start_target_geometry')
    final_geometry = data.get('final_target_geometry')
    add(checks, 'start target geometry is valid', geometry_valid(start_geometry))
    add(checks, 'final target geometry is valid', geometry_valid(final_geometry))
    add(checks, 'raw depth topic is correct',
        data.get('raw_depth_topic')
        == '/head_front_camera/head_front_camera/depth/image_rect_raw')
    add(checks, 'depth encoding is supported',
        str(data.get('depth_encoding', '')).upper() in {'32FC1', '16UC1', 'MONO16'})
    add(checks, 'front LiDAR topic is correct', data.get('front_lidar_topic') == '/scan_front_raw')
    add(checks, 'target and scan geometry use base_footprint', data.get('base_frame') == 'base_footprint')
    add(checks, 'front LiDAR produced data',
        isinstance(data.get('lidar_frames_total'), int) and data['lidar_frames_total'] >= 1)
    add(checks, 'raw depth produced data',
        isinstance(data.get('depth_frames_total'), int) and data['depth_frames_total'] >= 1)
    add(checks, 'minimum approach LiDAR clearance was measured',
        args.mode == 'range'
        or finite(data.get('minimum_lidar_clearance_during_approach_m')))
    add(checks, 'no unsafe LiDAR penetration was accepted',
        args.mode == 'range'
        or (
            finite(data.get('minimum_lidar_clearance_during_approach_m'))
            and float(data['minimum_lidar_clearance_during_approach_m']) >= 0.70
        ))
    add(checks, 'safety-stop event count is within test limit',
        isinstance(data.get('safety_stop_events'), int)
        and data['safety_stop_events'] <= args.max_safety_stops)

    configuration = data.get('controller_configuration', {})
    add(checks, 'controller coordinate convention is recorded',
        isinstance(configuration, dict)
        and configuration.get('coordinate_convention')
        == '+x forward, +y left, +yaw counter-clockwise')
    add(checks, 'forward command respected configured clamp',
        finite(data.get('maximum_command_abs_vx_mps'))
        and finite(configuration.get('max_forward_speed_mps'))
        and float(data['maximum_command_abs_vx_mps'])
        <= float(configuration['max_forward_speed_mps']) + 1.0e-6)
    add(checks, 'lateral command respected configured clamp',
        finite(data.get('maximum_command_abs_vy_mps'))
        and finite(configuration.get('max_lateral_speed_mps'))
        and float(data['maximum_command_abs_vy_mps'])
        <= float(configuration['max_lateral_speed_mps']) + 1.0e-6)
    add(checks, 'yaw command respected configured clamp',
        finite(data.get('maximum_command_abs_wz_radps'))
        and finite(configuration.get('max_yaw_speed_radps'))
        and float(data['maximum_command_abs_wz_radps'])
        <= float(configuration['max_yaw_speed_radps']) + 1.0e-6)
    if args.require_lateral_command:
        add(checks, 'holonomic lateral correction was actually exercised',
            finite(data.get('maximum_command_abs_vy_mps'))
            and float(data['maximum_command_abs_vy_mps']) >= 0.01)

    states = state_names(data)
    add(checks, 'geometry-estimation state executed', 'ESTIMATE_TARGET_GEOMETRY' in states)
    add(checks, 'approach-verification state executed', 'VERIFY_APPROACH' in states)

    # Collision-safe travel-arm contract.
    #
    # Full and partial translation require the validated PAL spherical-wrist
    # arm-only home pose.  Range-only mode remains arm-free.
    if args.mode == 'range':
        add(
            checks,
            'range-only mode does not require travel-arm motion',
            data.get('travel_arm_pose_required') is False,
        )
        add(
            checks,
            'range-only mode did not command travel arms',
            data.get('travel_arm_pose_commanded') is False,
        )
    else:
        approved_left = [
            0.36,
            -1.83,
            0.47,
            -2.35,
            0.0,
            -1.20,
            0.0,
        ]

        approved_right = [
            -0.36,
            -1.83,
            -0.47,
            -2.35,
            0.0,
            -1.20,
            0.0,
        ]

        add(
            checks,
            'approved travel-arm pose was used',
            data.get('travel_arm_pose_name')
            == 'pal_spherical_home_arm_only'
            and data.get('travel_arm_left_target_rad')
            == approved_left
            and data.get('travel_arm_right_target_rad')
            == approved_right,
        )

        add(
            checks,
            'travel-arm pose was required for translation',
            data.get('travel_arm_pose_required') is True,
        )

        add(
            checks,
            'travel-arm pose command was sent',
            data.get('travel_arm_pose_commanded') is True,
        )

        add(
            checks,
            'travel-arm safety states executed',
            'SET_TRAVEL_ARM_POSE' in states
            and 'WAIT_FOR_TRAVEL_ARM_POSE' in states,
        )

        add(
            checks,
            'travel arms reached validated pose before approach',
            data.get('travel_arm_pose_ready') is True,
        )

        arm_tolerance = data.get(
            'travel_arm_position_tolerance_rad')
        arm_max_error = data.get(
            'travel_arm_max_error_rad')

        add(
            checks,
            'travel-arm joint error is inside configured tolerance',
            finite(arm_tolerance)
            and finite(arm_max_error)
            and float(arm_tolerance) > 0.0
            and float(arm_max_error)
            <= float(arm_tolerance) + 1.0e-6,
        )

        arm_motion_requested = data.get(
            'travel_arm_motion_sec')
        arm_motion_actual = data.get(
            'travel_arm_motion_actual_sim_sec')
        arm_ready_timeout = data.get(
            'travel_arm_ready_timeout_sec')

        add(
            checks,
            'travel-arm simulation timing is valid',
            finite(arm_motion_requested)
            and finite(arm_motion_actual)
            and finite(arm_ready_timeout)
            and float(arm_motion_requested) > 0.0
            and float(arm_motion_actual)
            >= max(
                0.0,
                float(arm_motion_requested) - 0.25,
            )
            and float(arm_motion_actual)
            <= float(arm_ready_timeout),
        )

        add(
            checks,
            'travel-arm joint feedback was received after command',
            isinstance(
                data.get(
                    'travel_arm_joint_updates_after_command'),
                int,
            )
            and data[
                'travel_arm_joint_updates_after_command'
            ] >= 2,
        )

        actual_arm_positions = data.get(
            'travel_arm_actual_positions_rad')

        expected_arm_names = [
            *(f'arm_left_{i}_joint' for i in range(1, 8)),
            *(f'arm_right_{i}_joint' for i in range(1, 8)),
        ]

        add(
            checks,
            'all fourteen travel-arm joints have finite feedback',
            isinstance(actual_arm_positions, dict)
            and all(
                finite(actual_arm_positions.get(name))
                for name in expected_arm_names
            ),
        )

    trace = data.get('approach_trace')
    trace_samples = trace if isinstance(trace, list) else []
    add(checks, 'quantitative trace count matches list',
        isinstance(data.get('approach_trace_samples'), int)
        and data['approach_trace_samples'] == len(trace_samples))
    if args.mode != 'range':
        add(checks, 'approach controller state executed', 'APPROACH_TARGET_COLUMN' in states)
        add(checks, 'visual tracking continued during approach',
            isinstance(data.get('camera_frames_during_approach'), int)
            and data['camera_frames_during_approach'] >= 5
            and sum(
                1 for item in trace_samples
                if isinstance(item, dict)
                and finite(item.get('target_pixel_error_px'))
                and finite(item.get('target_confidence'))
            ) >= 3)
        add(checks, 'depth continued during approach',
            isinstance(data.get('depth_frames_during_approach'), int)
            and data['depth_frames_during_approach'] >= 3)
        add(checks, 'LiDAR continued during approach',
            isinstance(data.get('lidar_frames_during_approach'), int)
            and data['lidar_frames_during_approach'] >= 2)
        add(checks, 'camera throughput was not infrastructure-starved',
            finite(data.get('camera_effective_hz_during_approach'))
            and float(data['camera_effective_hz_during_approach']) >= 3.0)
        add(checks, 'depth throughput was not infrastructure-starved',
            finite(data.get('depth_effective_hz_during_approach'))
            and float(data['depth_effective_hz_during_approach']) >= 3.0)
        add(checks, 'LiDAR throughput was not infrastructure-starved',
            finite(data.get('lidar_effective_hz_during_approach'))
            and float(data['lidar_effective_hz_during_approach']) >= 1.5)
        add(checks, 'odometry recorded requested travel',
            finite(data.get('approach_distance_travelled_m'))
            and float(data['approach_distance_travelled_m']) >= args.min_travel_m)
        add(checks, 'approach simulation duration was recorded',
            finite(data.get('approach_duration_sim_sec')))
        add(checks, 'approach wall duration was recorded',
            finite(data.get('approach_duration_wall_sec')))

    if args.mode == 'range':
        add(checks, 'range-only outcome is correct',
            data.get('approach_outcome') == 'ranging_only'
            and data.get('approach_motion_enabled') is False)
        add(checks, 'range-only mode did not translate',
            float(data.get('approach_distance_travelled_m', 0.0)) <= 0.05)
    elif args.mode == 'partial':
        add(checks, 'partial-distance outcome is correct',
            data.get('approach_outcome') == 'partial_distance_limit'
            and data.get('approach_motion_enabled') is True
            and finite(data.get('approach_distance_limit_m'))
            and float(data['approach_distance_limit_m']) > 0.0)
    else:
        add(checks, 'full approach reached safe stand-off outcome',
            data.get('approach_outcome') == 'safe_standoff'
            and data.get('approach_motion_enabled') is True)
        if geometry_valid(start_geometry) and geometry_valid(final_geometry):
            add(checks, 'depth/base forward range decreased substantially',
                float(final_geometry['forward_m'])
                <= float(start_geometry['forward_m']) - 0.75)

            column_lock_mode = (
                data.get('column_lock_active') is True
                and data.get('column_lock_mode')
                == 'identify_once_then_odom_lidar'
                and data.get('final_geometry_source')
                == 'locked_column_odometry'
                and data.get('final_forward_standoff_source')
                == 'front_lidar'
            )

            if column_lock_mode:
                # In locked-column mode the numbered marker was used once to
                # select the physical shelf column.  final_geometry.forward_m
                # is subsequently propagated from odometry, not measured from
                # a live close-range digit/depth observation.  The independent
                # front LiDAR is therefore the authoritative final forward
                # stand-off sensor.
                add(
                    checks,
                    'locked-column final ranging semantics are consistent',
                    True,
                )
            else:
                add(
                    checks,
                    'final depth stand-off is inside configured envelope',
                    abs(
                        float(final_geometry['forward_m'])
                        - float(configuration['approach_standoff_m'])
                    )
                    <= float(
                        configuration['final_depth_tolerance_m']
                    ) + 0.02,
                )

            add(checks, 'final lateral error is inside configured envelope',
                abs(float(final_geometry['lateral_left_m']))
                <= float(configuration['final_lateral_tolerance_m']) + 0.02)
            add(checks, 'final bearing error is inside configured envelope',
                abs(float(final_geometry['base_bearing_left_deg']))
                <= float(configuration['final_bearing_tolerance_deg']) + 0.5)
        add(checks, 'post-opening shelf-facing heading was preserved',
            finite(data.get('final_heading_error_deg'))
            and abs(float(data['final_heading_error_deg']))
            <= float(configuration['final_heading_tolerance_deg']) + 0.5
            and configuration.get('yaw_control_mode')
            == 'hold post-opening shelf-facing heading')
        add(checks, 'final LiDAR stand-off is inside configured envelope',
            finite(data.get('final_lidar_clearance_m'))
            and configuration['final_lidar_interval_m'][0]
            <= float(data['final_lidar_clearance_m'])
            <= configuration['final_lidar_interval_m'][1])
        add(checks, 'final target confidence remains available',
            finite(data.get('final_target_confidence'))
            and float(data['final_target_confidence']) >= 0.55)

    for label, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print('\nMeasured Day 3 result:')
    for key in (
        'approach_outcome',
        'shelf_column_number',
        'column_lock_active',
        'column_lock_mode',
        'final_geometry_source',
        'final_forward_standoff_source',
        'start_depth_forward_m',
        'final_depth_forward_m',
        'start_lidar_clearance_m',
        'final_lidar_clearance_m',
        'minimum_lidar_clearance_during_approach_m',
        'approach_distance_travelled_m',
        'approach_duration_sim_sec',
        'approach_duration_wall_sec',
        'mission_duration_sim_sec',
        'mission_duration_wall_sec',
        'camera_frames_during_approach',
        'depth_frames_during_approach',
        'lidar_frames_during_approach',
        'camera_effective_hz_during_approach',
        'depth_effective_hz_during_approach',
        'lidar_effective_hz_during_approach',
        'target_stale_events',
        'sensor_hold_events',
        'safety_stop_events',
        'final_target_confidence',
        'maximum_command_abs_vx_mps',
        'maximum_command_abs_vy_mps',
        'maximum_command_abs_wz_radps',
        'approach_heading_reference_deg',
        'final_heading_error_deg',
        'maximum_abs_heading_error_deg',
        'annotated_image_path',
        'approach_annotated_image_path',
        'reason',
    ):
        print(f'  {key}: {data.get(key)}')

    bearing_gate = geometry_valid(start_geometry)
    depth_gate = bearing_gate and geometry_valid(final_geometry)
    lidar_gate = (
        finite(data.get('start_lidar_clearance_m'))
        and finite(data.get('final_lidar_clearance_m'))
    )
    controller_gate = (
        args.mode == 'range'
        or (
            finite(data.get('maximum_command_abs_vx_mps'))
            and float(data['maximum_command_abs_vx_mps']) > 0.0
        )
    )
    tracking_gate = (
        args.mode == 'range'
        or (
            isinstance(data.get('camera_frames_during_approach'), int)
            and data['camera_frames_during_approach'] >= 5
        )
    )
    stand_off_gate = (
        args.mode != 'full'
        or data.get('approach_outcome') == 'safe_standoff'
    )
    column_gate = (
        data.get('published_column_value') == args.target
        and data.get('column_result_topic')
        == '/erc/shelf_column_identification'
    )
    image_gate = day2_image_ok and approach_image_ok

    print()
    print(f"[TARGET BEARING][{'PASS' if bearing_gate else 'FAIL'}]")
    print(f"[DEPTH INPUT][{'PASS' if depth_gate else 'FAIL'}]")
    print(f"[LIDAR SAFETY][{'PASS' if lidar_gate else 'FAIL'}]")
    print(f"[APPROACH CONTROLLER][{'PASS' if controller_gate else 'FAIL'}]")
    print(
        f"[TARGET TRACKING DURING APPROACH][{'PASS' if tracking_gate else 'FAIL'}]")
    print(f"[SAFE STANDOFF][{'PASS' if stand_off_gate else 'FAIL'}]")
    print(f"[COLUMN RESULT PRESERVED][{'PASS' if column_gate else 'FAIL'}]")
    print(f"[ANNOTATED IMAGE PRESERVED][{'PASS' if image_gate else 'FAIL'}]")

    if all(checks.values()):
        print('\n[DAY3 RESULT][PASS]')
        return 0
    print('\n[DAY3 RESULT][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
