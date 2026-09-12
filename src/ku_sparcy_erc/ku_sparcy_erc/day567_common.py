#!/usr/bin/env python3
"""Small shared helpers for the accelerated KU SPARCy Day 5-7 sprint.

This module is team-owned.  It does not read Gazebo entity state, seed values,
or hidden randomized layout information.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def load_passed_json(path: str, *, label: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f'{label} is not a JSON object: {path}')
    if data.get('passed') is not True:
        raise ValueError(f'{label} did not pass: {path}')
    return data


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (float(w) * float(z) + float(x) * float(y))
    cosy = 1.0 - 2.0 * (float(y) ** 2 + float(z) ** 2)
    return math.atan2(siny, cosy)


def yaw_from_odom(msg: Any) -> float:
    q = msg.pose.pose.orientation
    return yaw_from_quaternion(q.x, q.y, q.z, q.w)


def quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = x*x + y*y + z*z + w*w
    if norm <= 1.0e-16:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    xx, yy, zz = x*x*scale, y*y*scale, z*z*scale
    xy, xz, yz = x*y*scale, x*z*scale, y*z*scale
    wx, wy, wz = w*x*scale, w*y*scale, w*z*scale
    return np.array([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ], dtype=np.float64)


def transform_point(
    point_xyz: Sequence[float],
    translation_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> Tuple[float, float, float]:
    point = np.asarray(point_xyz, dtype=np.float64)
    translation = np.asarray(translation_xyz, dtype=np.float64)
    transformed = quaternion_matrix(quaternion_xyzw) @ point + translation
    return tuple(float(value) for value in transformed)


def vector_base_to_odom(
    forward_m: float,
    left_m: float,
    robot_xy: Sequence[float],
    robot_yaw_rad: float,
) -> Tuple[float, float]:
    x, y = (float(value) for value in robot_xy)
    c, s = math.cos(float(robot_yaw_rad)), math.sin(float(robot_yaw_rad))
    return (
        x + c * float(forward_m) - s * float(left_m),
        y + s * float(forward_m) + c * float(left_m),
    )


def vector_odom_to_base(
    target_xy: Sequence[float],
    robot_xy: Sequence[float],
    robot_yaw_rad: float,
) -> Tuple[float, float]:
    tx, ty = (float(value) for value in target_xy)
    rx, ry = (float(value) for value in robot_xy)
    dx, dy = tx - rx, ty - ry
    c, s = math.cos(float(robot_yaw_rad)), math.sin(float(robot_yaw_rad))
    return c * dx + s * dy, -s * dx + c * dy


def clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def limit_planar(vx: float, vy: float, maximum: float) -> Tuple[float, float]:
    magnitude = math.hypot(float(vx), float(vy))
    if magnitude <= float(maximum) or magnitude <= 1.0e-12:
        return float(vx), float(vy)
    scale = float(maximum) / magnitude
    return float(vx) * scale, float(vy) * scale


def book_half_diagonal_m(
    dimensions_xyz_m: Sequence[float] = (0.25, 0.02, 0.16),
) -> float:
    values = [0.5 * float(value) for value in dimensions_xyz_m]
    return math.sqrt(sum(value * value for value in values))


def contact_collision_names(contact: Any) -> Tuple[str, str]:
    first = str(getattr(getattr(contact, 'collision1', None), 'name', ''))
    second = str(getattr(getattr(contact, 'collision2', None), 'name', ''))
    return first, second


def selected_fingertip_tokens(arm: str) -> Tuple[str, str]:
    side = str(arm).strip().lower()
    return (
        f'gripper_{side}_fingertip_left_link',
        f'gripper_{side}_fingertip_right_link',
    )


def meaningful_robot_collision(name: str) -> bool:
    """Return true for robot structures that should not contact arena objects.

    Wheel-ground contacts are intentionally excluded.  Collision names are used
    only as direct contact-sensor evidence, never as a hidden object-pose oracle.

    ``base_link`` must be scoped to the TIAGo model because arena objects such
    as books also contain names like ``book_base_link``.
    """
    text = str(name)

    robot_link_tokens = (
        'arm_left_',
        'arm_right_',
        'gripper_left_',
        'gripper_right_',
        'torso_',
        'head_',
    )

    if any(token in text for token in robot_link_tokens):
        return True

    return (
        text.startswith('tiago_pro::base_link::')
        or '::tiago_pro::base_link::' in text
    )


def any_token(text: str, tokens: Iterable[str]) -> bool:
    return any(token in str(text) for token in tokens)


def directional_lidar_clearance(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    laser_to_base_translation_xyz: Sequence[float],
    laser_to_base_quaternion_xyzw: Sequence[float],
    *,
    direction_sign: int,
    corridor_half_width_m: float,
    robust_percentile: float,
    hard_stop_m: float,
):
    """Return a ``LidarClearance`` in the forward or rear base corridor.

    ``direction_sign=+1`` measures +X, while ``-1`` measures -X.  Import is
    local to avoid a circular dependency at module import time.
    """
    from ku_sparcy_erc.range_fusion import LidarClearance

    sign = 1 if int(direction_sign) >= 0 else -1
    array = np.asarray(list(ranges), dtype=np.float64)
    if array.size == 0:
        return None
    angles = float(angle_min) + np.arange(array.size) * float(angle_increment)
    valid = (
        np.isfinite(array)
        & (array >= float(range_min))
        & (array <= float(range_max))
    )
    if not np.any(valid):
        return None
    radii = array[valid]
    selected_angles = angles[valid]
    laser = np.column_stack((
        radii * np.cos(selected_angles),
        radii * np.sin(selected_angles),
        np.zeros_like(radii),
    ))
    rotation = quaternion_matrix(laser_to_base_quaternion_xyzw)
    translation = np.asarray(laser_to_base_translation_xyz, dtype=np.float64)
    base = (rotation @ laser.T).T + translation
    longitudinal = float(sign) * base[:, 0]
    lateral = base[:, 1]
    corridor = (
        (longitudinal > 0.0)
        & (np.abs(lateral) <= float(corridor_half_width_m))
    )
    values = longitudinal[corridor]
    if values.size == 0:
        return None
    percentile = min(50.0, max(0.0, float(robust_percentile)))
    return LidarClearance(
        robust_clearance_m=float(np.percentile(values, percentile)),
        minimum_clearance_m=float(np.min(values)),
        corridor_point_count=int(values.size),
        hard_cluster_point_count=int(np.count_nonzero(
            values <= float(hard_stop_m))),
        corridor_half_width_m=float(corridor_half_width_m),
        percentile=float(percentile),
    )
