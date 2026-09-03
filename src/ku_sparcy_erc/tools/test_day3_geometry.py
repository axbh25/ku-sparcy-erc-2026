#!/usr/bin/env python3
"""CPU-only Day 3 geometry/depth/LiDAR/controller unit tests."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from ku_sparcy_erc.range_fusion import (  # noqa: E402
    BoundingBoxLike,
    CameraModel,
    ControllerSettings,
    compute_holonomic_command,
    lidar_corridor_clearance,
    map_bbox_between_cameras,
    normalize_depth_array,
    target_geometry_from_depth,
)


def check(label: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return bool(condition)


def main() -> int:
    results = []
    rgb = CameraModel(640, 360, 348.0, 348.0, 320.0, 180.0, 'rgb_optical')
    depth = CameraModel(640, 360, 348.0, 348.0, 320.0, 180.0, 'depth_optical')

    results.append(check(
        'principal point has zero horizontal bearing',
        abs(rgb.bearing_right_rad(320.0)) < 1.0e-12))
    expected = math.atan2(100.0, 348.0)
    results.append(check(
        'pixel right of centre gives positive camera bearing',
        abs(rgb.bearing_right_rad(420.0) - expected) < 1.0e-12))

    bbox = BoundingBoxLike(300, 120, 20, 40)
    mapped = map_bbox_between_cameras(bbox, rgb, depth)
    results.append(check(
        'aligned RGB bbox maps to a valid padded depth bbox',
        mapped.width > bbox.width and mapped.height > bbox.height
        and 0 <= mapped.x < 640 and 0 <= mapped.y < 360))

    synthetic = np.full((360, 640), 6.0, dtype=np.float32)
    synthetic[mapped.y:mapped.y2, mapped.x:mapped.x2] = 3.0
    synthetic[mapped.y:mapped.y + 2, mapped.x:mapped.x2] = np.inf
    synthetic[mapped.y + 4, mapped.x + 4] = 0.0
    geometry = target_geometry_from_depth(
        bbox,
        rgb,
        depth,
        synthetic,
        depth_to_base_translation_xyz=(0.10, 0.0, 1.40),
        depth_to_base_quaternion_xyzw=(-0.5, 0.5, -0.5, 0.5),
        rgb_depth_stamp_skew_sec=0.03,
        min_valid_pixels=20,
    )
    results.append(check('synthetic marker depth was estimated', geometry is not None))
    if geometry is not None:
        results.append(check(
            'robust depth rejects the 6 m background',
            abs(geometry.depth.depth_m - 3.0) <= 0.01))
        results.append(check(
            'target transformed into positive base-forward space',
            geometry.forward_m > 2.5))

    millimetres = np.array([[1000, 2500]], dtype=np.uint16)
    metres = normalize_depth_array(millimetres, '16UC1')
    results.append(check(
        '16UC1 depth is converted from millimetres to metres',
        np.allclose(metres, [[1.0, 2.5]])))

    # Identity TF: one frontal point at 1.0 m, one side point at 0.5 m outside
    # the 0.38 m corridor, and two more frontal points for robust clustering.
    ranges = np.full(181, np.inf, dtype=np.float64)
    angle_min = -math.pi / 2.0
    increment = math.pi / 180.0
    for angle_deg, distance in ((-2, 1.02), (0, 1.00), (2, 1.03), (70, 0.50)):
        index = int(round((math.radians(angle_deg) - angle_min) / increment))
        ranges[index] = distance
    lidar = lidar_corridor_clearance(
        ranges,
        angle_min,
        increment,
        0.05,
        25.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        corridor_half_width_m=0.38,
        robust_percentile=10.0,
        hard_stop_m=0.72,
    )
    results.append(check('synthetic LiDAR corridor produced clearance', lidar is not None))
    if lidar is not None:
        results.append(check(
            'side obstacle outside swept corridor is ignored',
            0.95 <= lidar.robust_clearance_m <= 1.05))
        results.append(check(
            'no false hard-stop cluster was produced',
            lidar.hard_cluster_point_count == 0))

    hard_ranges = np.full(181, np.inf, dtype=np.float64)
    for angle_deg in (-2, 0, 2, 4):
        index = int(round((math.radians(angle_deg) - angle_min) / increment))
        hard_ranges[index] = 0.60
    hard_lidar = lidar_corridor_clearance(
        hard_ranges,
        angle_min,
        increment,
        0.05,
        25.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        corridor_half_width_m=0.38,
        robust_percentile=10.0,
        hard_stop_m=0.72,
    )
    results.append(check(
        'three-or-more close corridor points trigger a hard-stop cluster',
        hard_lidar is not None and hard_lidar.hard_cluster_point_count >= 3))

    if geometry is not None:
        # Replace the base point with a target 2.0 m forward and 0.20 m left.
        geometry = type(geometry)(
            camera_bearing_right_rad=-0.10,
            base_bearing_left_rad=math.atan2(0.20, 2.0),
            optical_point_xyz_m=geometry.optical_point_xyz_m,
            base_point_xyz_m=(2.0, 0.20, 2.2),
            depth=geometry.depth,
            rgb_bbox_xywh=geometry.rgb_bbox_xywh,
            depth_bbox_xywh=geometry.depth_bbox_xywh,
            rgb_depth_stamp_skew_sec=geometry.rgb_depth_stamp_skew_sec,
        )
        settings = ControllerSettings()
        command = compute_holonomic_command(
            geometry, 2.0, settings, heading_error_rad=0.08)
        results.append(check(
            'target left produces positive holonomic lateral velocity',
            command.vy_mps > 0.0))
        results.append(check(
            'positive shelf-heading error produces positive yaw correction',
            command.wz_radps > 0.0))
        results.append(check(
            'forward command respects configured bound',
            0.0 < command.vx_mps <= settings.max_forward_speed_mps))
        results.append(check(
            'holonomic path cap prevents outrunning lateral correction',
            command.vx_mps <= command.lateral_path_speed_cap_mps + 1.0e-9))
        close_command = compute_holonomic_command(
            geometry, 0.80, settings, heading_error_rad=0.08)
        results.append(check(
            'LiDAR slowdown reduces forward speed near the hard envelope',
            close_command.vx_mps < command.vx_mps))

    if all(results):
        print('\n[DAY3 GEOMETRY UNIT TEST][PASS]')
        print('[CAMERA INTRINSICS][PASS]')
        print('[TARGET BEARING][PASS]')
        print('[DEPTH SAMPLING][PASS]')
        print('[LIDAR SAFETY MATH][PASS]')
        print('[APPROACH CONTROLLER MATH][PASS]')
        return 0
    print('\n[DAY3 GEOMETRY UNIT TEST][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
