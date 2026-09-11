#!/usr/bin/env python3
"""Strictly analyze a stationary pre-approach Day 4 visibility profile.

The analyzer never accepts the retired moving-head-during-approach profile and
never weakens detector thresholds.  It verifies base stationarity, settled head
poses, multi-frame RGB/depth/TF observations, and a coherent four-colour 3-D
row map.  It then selects the smallest ordered subset of measured head poses
that independently retains sufficient spatial support for every colour.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


COLOURS: Tuple[str, ...] = ('red', 'green', 'yellow', 'blue')


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fail(message: str) -> int:
    print(f'[STATIONARY PRE-APPROACH PROFILE][FAIL] {message}')
    return 1


def unique_by_frame(observations: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    selected: Dict[int, Mapping[str, Any]] = {}
    for observation in observations:
        frame = observation.get('frame_index')
        detection = observation.get('detection')
        if not isinstance(frame, int) or not isinstance(detection, dict):
            continue
        confidence = detection.get('confidence')
        if not finite(confidence):
            continue
        current = selected.get(frame)
        current_confidence = (
            current.get('detection', {}).get('confidence')
            if isinstance(current, dict) else None
        )
        if current is None or not finite(current_confidence) or float(confidence) > float(current_confidence):
            selected[frame] = observation
    return sorted(selected.values(), key=lambda item: float(item.get('stamp_sec', 0.0)))


def compatible(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    maximum_xy_radius_m: float,
    maximum_z_radius_m: float,
) -> bool:
    a = first.get('odom_xyz_m')
    b = second.get('odom_xyz_m')
    if not (
        isinstance(a, list) and len(a) == 3
        and isinstance(b, list) and len(b) == 3
        and all(finite(value) for value in a + b)
    ):
        return False
    xy = math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    z = abs(float(a[2]) - float(b[2]))
    return xy <= maximum_xy_radius_m and z <= maximum_z_radius_m


def dominant_cluster(
    observations: Sequence[Mapping[str, Any]],
    maximum_xy_radius_m: float,
    maximum_z_radius_m: float,
) -> List[Mapping[str, Any]]:
    unique = unique_by_frame(observations)
    best: List[Mapping[str, Any]] = []
    best_confidence = -math.inf
    for seed in unique:
        cluster = [
            item for item in unique
            if compatible(seed, item, maximum_xy_radius_m, maximum_z_radius_m)
        ]
        confidence_values = [
            float(item['detection']['confidence'])
            for item in cluster
            if isinstance(item.get('detection'), dict)
            and finite(item['detection'].get('confidence'))
        ]
        average = fmean(confidence_values) if confidence_values else -math.inf
        if len(cluster) > len(best) or (
            len(cluster) == len(best) and average > best_confidence
        ):
            best = cluster
            best_confidence = average
    return sorted(best, key=lambda item: float(item.get('stamp_sec', 0.0)))


def summarize_track(
    colour: str,
    observations: Sequence[Mapping[str, Any]],
    settings: Mapping[str, float],
) -> Optional[Dict[str, Any]]:
    cluster = dominant_cluster(
        observations,
        settings['maximum_cluster_xy_radius_m'],
        settings['maximum_cluster_z_radius_m'],
    )
    if len(cluster) < int(settings['required_frames_per_colour']):
        return None

    stamps = [float(item['stamp_sec']) for item in cluster if finite(item.get('stamp_sec'))]
    if len(stamps) != len(cluster):
        return None
    span = max(stamps) - min(stamps)
    if span < settings['minimum_observation_span_sec']:
        return None

    confidences = [float(item['detection']['confidence']) for item in cluster]
    average_confidence = fmean(confidences)
    if average_confidence < settings['minimum_average_confidence']:
        return None

    points = [item.get('odom_xyz_m') for item in cluster]
    if any(
        not isinstance(point, list)
        or len(point) != 3
        or not all(finite(value) for value in point)
        for point in points
    ):
        return None

    med = tuple(median(float(point[index]) for point in points) for index in range(3))
    xy_span = max(
        math.hypot(float(point[0]) - med[0], float(point[1]) - med[1])
        for point in points
    )
    z_span = max(abs(float(point[2]) - med[2]) for point in points)
    if xy_span > settings['maximum_track_xy_span_m']:
        return None
    if z_span > settings['maximum_track_z_span_m']:
        return None

    head_tilts = [
        float(item['head_tilt_rad'])
        for item in cluster
        if finite(item.get('head_tilt_rad'))
    ]
    if len(head_tilts) != len(cluster):
        return None

    return {
        'colour': colour,
        'confirming_frames': len(cluster),
        'first_stamp_sec': min(stamps),
        'last_stamp_sec': max(stamps),
        'observation_span_sec': span,
        'average_confidence': average_confidence,
        'median_odom_xyz_m': list(med),
        'xy_span_m': xy_span,
        'z_span_m': z_span,
        'head_tilt_range_rad': [min(head_tilts), max(head_tilts)],
        'frame_indices': [int(item['frame_index']) for item in cluster],
    }


def evaluate_subset(
    pose_indices: Sequence[int],
    pose_targets: Mapping[int, float],
    observations_by_colour: Mapping[str, Sequence[Mapping[str, Any]]],
    settings: Mapping[str, float],
    head_tolerance_rad: float,
) -> Optional[Dict[str, Any]]:
    selected_targets = [pose_targets[index] for index in pose_indices]
    tracks: Dict[str, Dict[str, Any]] = {}
    for colour in COLOURS:
        observations = observations_by_colour.get(colour)
        if not isinstance(observations, list):
            return None
        selected = []
        for observation in observations:
            tilt = observation.get('head_tilt_rad')
            if not finite(tilt):
                continue
            if any(abs(float(tilt) - target) <= head_tolerance_rad for target in selected_targets):
                selected.append(observation)
        summary = summarize_track(colour, selected, settings)
        if summary is None:
            return None
        tracks[colour] = summary

    ordered = sorted(
        tracks.values(),
        key=lambda item: float(item['median_odom_xyz_m'][2]),
        reverse=True,
    )
    row_colours = [item['colour'] for item in ordered]
    if sorted(row_colours) != sorted(COLOURS):
        return None
    separations = [
        float(ordered[index]['median_odom_xyz_m'][2])
        - float(ordered[index + 1]['median_odom_xyz_m'][2])
        for index in range(3)
    ]
    if any(
        value < settings['minimum_row_separation_m']
        or value > settings['maximum_row_separation_m']
        for value in separations
    ):
        return None

    travel = abs(selected_targets[0] - 0.35)
    travel += sum(
        abs(selected_targets[index] - selected_targets[index - 1])
        for index in range(1, len(selected_targets))
    )
    travel += abs(selected_targets[-1] - 0.35)

    return {
        'pose_indices': list(pose_indices),
        'head_tilt_sequence_rad': selected_targets,
        'estimated_head_travel_rad': travel,
        'tracks_by_colour': tracks,
        'row_colours_top_to_bottom': row_colours,
        'row_separations_m': separations,
        'minimum_confirming_frames': min(
            int(item['confirming_frames']) for item in tracks.values()
        ),
        'minimum_average_confidence': min(
            float(item['average_confidence']) for item in tracks.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    try:
        data = json.loads(args.result.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return fail(f'cannot read result: {exc}')

    if data.get('passed') is not True:
        return fail('profile node did not report passed=true')
    if data.get('day4_outcome') != 'preapproach_visibility_profile_only':
        return fail('result is not a stationary pre-approach profile')
    if data.get('primary_row_mapping_strategy') != 'stationary_preapproach_scan':
        return fail('primary strategy is not stationary_preapproach_scan')
    if data.get('moving_head_during_approach_enabled') is not False:
        return fail('moving-head-during-approach must be disabled')
    if data.get('preapproach_scan_profile_only') is not True:
        return fail('profile-only flag is not true')
    if data.get('preapproach_scan_started') is not True:
        return fail('stationary scan never started')
    if data.get('preapproach_scan_head_restored') is not True:
        return fail('validated +0.35 rad Day 3 head pose was not restored')
    if data.get('preapproach_scan_approach_started_sim_sec') is not None:
        return fail('Day 3 translation started during visibility-only mode')
    if data.get('approach_started_sim_sec') is not None:
        return fail('approach_started_sim_sec must remain null in profile mode')
    if data.get('preapproach_scan_nonzero_cmd_vel_publications') != 0:
        return fail('a non-zero base command was published during the stationary scan')
    if not isinstance(data.get('preapproach_scan_zero_cmd_vel_publications'), int) or data['preapproach_scan_zero_cmd_vel_publications'] < 4:
        return fail('stationary base hold was not evidenced by zero commands')

    translation_tolerance = data.get('preapproach_scan_base_translation_tolerance_m')
    yaw_tolerance = data.get('preapproach_scan_base_yaw_tolerance_deg')
    translation = data.get('preapproach_scan_base_translation_m')
    yaw_change = data.get('preapproach_scan_base_yaw_change_deg')
    if not all(finite(value) for value in (
        translation_tolerance, yaw_tolerance, translation, yaw_change
    )):
        return fail('base stationarity measurements or tolerances are missing')
    if float(translation) > float(translation_tolerance):
        return fail(
            f'base translated {float(translation):.4f} m during scan; '
            f'limit={float(translation_tolerance):.4f} m'
        )
    if float(yaw_change) > float(yaw_tolerance):
        return fail(
            f'base yaw changed {float(yaw_change):.3f} deg during scan; '
            f'limit={float(yaw_tolerance):.3f} deg'
        )

    sequence = data.get('preapproach_scan_sequence_used_rad')
    attempts = data.get('preapproach_scan_attempts')
    if not isinstance(sequence, list) or not sequence or not all(finite(value) for value in sequence):
        return fail('profile head sequence is missing or invalid')
    if not isinstance(attempts, list) or len(attempts) != len(sequence):
        return fail('not every configured profile pose produced an attempt record')

    min_dwell = data.get('preapproach_scan_min_dwell_sec')
    min_frames = data.get('preapproach_scan_min_frames_per_pose')
    tolerance = data.get('preapproach_scan_head_tolerance_rad')
    if not finite(min_dwell) or not isinstance(min_frames, int) or not finite(tolerance):
        return fail('scan dwell/frame/tolerance settings are missing')

    pose_targets: Dict[int, float] = {}
    for expected_index, (target, attempt) in enumerate(zip(sequence, attempts)):
        if not isinstance(attempt, dict):
            return fail(f'pose attempt {expected_index} is invalid')
        index = attempt.get('pose_index')
        actual = attempt.get('actual_tilt_rad')
        recorded_target = attempt.get('target_tilt_rad')
        if index != expected_index:
            return fail(f'pose attempt index mismatch at {expected_index}')
        if attempt.get('settled') is not True:
            return fail(f'pose {expected_index} was not settled')
        if not all(finite(value) for value in (actual, recorded_target)):
            return fail(f'pose {expected_index} lacks a finite actual/target tilt')
        if abs(float(recorded_target) - float(target)) > 1.0e-6:
            return fail(f'pose {expected_index} target does not match configured sequence')
        if abs(float(actual) - float(target)) > float(tolerance):
            return fail(f'pose {expected_index} actual tilt exceeds tolerance')
        if not finite(attempt.get('dwell_sim_sec')) or float(attempt['dwell_sim_sec']) < float(min_dwell):
            return fail(f'pose {expected_index} dwell was too short')
        if not isinstance(attempt.get('processed_frames'), int) or attempt['processed_frames'] < min_frames:
            return fail(f'pose {expected_index} has too few processed frames')
        evidence = attempt.get('evidence_image_path')
        if not isinstance(evidence, str) or not Path(evidence).is_file() or Path(evidence).stat().st_size <= 0:
            return fail(f'pose {expected_index} live evidence image is missing')
        pose_targets[expected_index] = float(target)

    layout = data.get('book_layout_confirmation')
    if not isinstance(layout, dict):
        return fail('profile did not create a coherent four-colour spatial map')
    if layout.get('mapping_source') != 'stationary_preapproach':
        return fail('row map source is not stationary_preapproach')
    if layout.get('single_frame_required') is not False:
        return fail('profile incorrectly requires all colours in one frame')
    if sorted(layout.get('row_colours_top_to_bottom', [])) != sorted(COLOURS):
        return fail('confirmed row map does not contain all four colours exactly once')

    raw_settings = data.get('book_map_settings')
    if not isinstance(raw_settings, dict):
        return fail('book-map settings are missing from result')
    required_settings = {
        'required_frames_per_colour': 3,
        'minimum_observation_span_sec': 0.05,
        'minimum_average_confidence': 0.60,
        'maximum_cluster_xy_radius_m': 0.18,
        'maximum_cluster_z_radius_m': 0.10,
        'maximum_track_xy_span_m': 0.18,
        'maximum_track_z_span_m': 0.10,
        'minimum_row_separation_m': 0.14,
        'maximum_row_separation_m': 0.55,
    }
    settings: Dict[str, float] = {}
    for name, default in required_settings.items():
        value = raw_settings.get(name, default)
        if not finite(value):
            return fail(f'book-map setting {name} is invalid')
        settings[name] = float(value)
    settings['required_frames_per_colour'] = int(
        raw_settings.get('required_frames_per_colour', 3)
    )

    observations = data.get('preapproach_scan_observations_by_colour')
    if not isinstance(observations, dict):
        return fail('per-colour stationary observations are missing')
    if any(not isinstance(observations.get(colour), list) for colour in COLOURS):
        return fail('one or more colour observation lists are missing')

    candidates: List[Dict[str, Any]] = []
    all_indices = list(range(len(sequence)))
    for size in range(1, len(all_indices) + 1):
        for subset in combinations(all_indices, size):
            evaluated = evaluate_subset(
                subset,
                pose_targets,
                observations,
                settings,
                float(tolerance),
            )
            if evaluated is not None:
                candidates.append(evaluated)
        if candidates:
            break

    if not candidates:
        return fail(
            'no measured pose subset retained three-frame spatial support for '
            'all four colours; do not weaken the analyzer or HSV thresholds'
        )

    candidates.sort(key=lambda item: (
        len(item['pose_indices']),
        float(item['estimated_head_travel_rad']),
        -int(item['minimum_confirming_frames']),
        -float(item['minimum_average_confidence']),
        item['pose_indices'],
    ))
    recommendation = candidates[0]

    report = {
        'status': 'pass',
        'source_result': str(args.result),
        'strategy': 'stationary_preapproach_scan',
        'moving_head_during_approach_enabled': False,
        'profile_head_tilt_sequence_rad': [float(value) for value in sequence],
        'recommended_pose_indices': recommendation['pose_indices'],
        'recommended_head_tilt_sequence_rad': recommendation['head_tilt_sequence_rad'],
        'recommended_estimated_head_travel_rad': recommendation['estimated_head_travel_rad'],
        'row_colours_top_to_bottom': recommendation['row_colours_top_to_bottom'],
        'row_separations_m': recommendation['row_separations_m'],
        'tracks_by_colour': recommendation['tracks_by_colour'],
        'book_map_settings': settings,
        'base_translation_m': float(translation),
        'base_yaw_change_deg': float(yaw_change),
        'head_restored_to_day3_pose': True,
        'nonzero_cmd_vel_publications': 0,
        'profile_pose_attempts': attempts,
        'reason': (
            'A measured base-stationary pose subset preserves the complete '
            'multi-frame 3-D colour-to-row map without changing HSV thresholds.'
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print('[STATIONARY PRE-APPROACH PROFILE][PASS]')
    print('[BASE STATIONARITY][PASS]')
    print('[SETTLED HEAD POSES][PASS]')
    print('[NO SINGLE-FRAME REQUIREMENT][PASS]')
    print('[FOUR-COLOUR SPATIAL MAP][PASS]')
    print('[DAY3 HEAD RESTORE][PASS]')
    print('[NO APPROACH-TIME HEAD SWEEP][PASS]')
    print(
        '[MINIMAL PRE-APPROACH POSE SEQUENCE][PASS] '
        + ', '.join(f'{value:+.3f}' for value in recommendation['head_tilt_sequence_rad'])
    )
    print(f'analysis: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
