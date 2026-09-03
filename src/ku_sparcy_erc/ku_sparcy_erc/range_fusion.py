#!/usr/bin/env python3
"""Pure geometry, depth, LiDAR, and bounded-control helpers for Day 3.

This module deliberately contains no ROS node logic.  The competition node
supplies live CameraInfo, Image, LaserScan, TF, and odometry observations; these
helpers turn them into testable numeric estimates and bounded commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundingBoxLike:
    """Small dependency-free pixel rectangle used by range helpers."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + 0.5 * self.width

    @property
    def center_y(self) -> float:
        return self.y + 0.5 * self.height

    def clipped(self, frame_width: int, frame_height: int) -> "BoundingBoxLike":
        x1 = min(max(0, int(self.x)), frame_width)
        y1 = min(max(0, int(self.y)), frame_height)
        x2 = min(max(x1, int(self.x2)), frame_width)
        y2 = min(max(y1, int(self.y2)), frame_height)
        return BoundingBoxLike(x1, y1, x2 - x1, y2 - y1)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class CameraModel:
    """Rectified pinhole intrinsics in pixel units."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str = ''

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError('Camera dimensions must be positive')
        for name, value in (
            ('fx', self.fx), ('fy', self.fy),
            ('cx', self.cx), ('cy', self.cy),
        ):
            if not math.isfinite(value):
                raise ValueError(f'Camera intrinsic {name} must be finite')
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError('Camera focal lengths must be positive')

    @classmethod
    def from_arrays(
        cls,
        width: int,
        height: int,
        k: Sequence[float],
        p: Sequence[float],
        frame_id: str = '',
        prefer_projection: bool = False,
    ) -> "CameraModel":
        """Create a model from sensor_msgs/CameraInfo numeric arrays.

        For rectified streams, ``P`` is preferred when requested and usable;
        otherwise the standard ``K`` matrix is used.
        """
        use_p = (
            prefer_projection
            and len(p) >= 12
            and float(p[0]) > 0.0
            and float(p[5]) > 0.0
        )
        if use_p:
            model = cls(
                int(width), int(height),
                float(p[0]), float(p[5]),
                float(p[2]), float(p[6]),
                str(frame_id),
            )
        else:
            if len(k) < 9:
                raise ValueError('CameraInfo K array must contain nine values')
            model = cls(
                int(width), int(height),
                float(k[0]), float(k[4]),
                float(k[2]), float(k[5]),
                str(frame_id),
            )
        model.validate()
        return model

    def bearing_right_rad(self, pixel_u: float) -> float:
        """Horizontal optical bearing; positive means image/right."""
        return math.atan2(float(pixel_u) - self.cx, self.fx)

    def normalized_ray(self, pixel_u: float, pixel_v: float) -> np.ndarray:
        """Return [X-right/Z, Y-down/Z, 1] in the optical convention."""
        return np.array([
            (float(pixel_u) - self.cx) / self.fx,
            (float(pixel_v) - self.cy) / self.fy,
            1.0,
        ], dtype=np.float64)


@dataclass(frozen=True)
class DepthEstimate:
    depth_m: float
    valid_pixels: int
    total_pixels: int
    valid_fraction: float
    median_absolute_deviation_m: float
    foreground_pixels: int
    sample_bbox_xywh: list[int]
    pixel_u: float
    pixel_v: float

    def as_dict(self) -> dict[str, object]:
        return {
            'depth_m': round(float(self.depth_m), 6),
            'valid_pixels': int(self.valid_pixels),
            'total_pixels': int(self.total_pixels),
            'valid_fraction': round(float(self.valid_fraction), 6),
            'median_absolute_deviation_m': round(
                float(self.median_absolute_deviation_m), 6),
            'foreground_pixels': int(self.foreground_pixels),
            'sample_bbox_xywh': list(self.sample_bbox_xywh),
            'pixel_u': round(float(self.pixel_u), 3),
            'pixel_v': round(float(self.pixel_v), 3),
        }


@dataclass(frozen=True)
class TargetGeometry:
    camera_bearing_right_rad: float
    base_bearing_left_rad: float
    optical_point_xyz_m: Tuple[float, float, float]
    base_point_xyz_m: Tuple[float, float, float]
    depth: DepthEstimate
    rgb_bbox_xywh: list[int]
    depth_bbox_xywh: list[int]
    rgb_depth_stamp_skew_sec: float

    @property
    def forward_m(self) -> float:
        return float(self.base_point_xyz_m[0])

    @property
    def lateral_left_m(self) -> float:
        return float(self.base_point_xyz_m[1])

    @property
    def planar_range_m(self) -> float:
        return math.hypot(self.forward_m, self.lateral_left_m)

    def as_dict(self) -> dict[str, object]:
        return {
            'camera_bearing_right_deg': round(
                math.degrees(self.camera_bearing_right_rad), 6),
            'base_bearing_left_deg': round(
                math.degrees(self.base_bearing_left_rad), 6),
            'optical_point_xyz_m': [
                round(float(value), 6) for value in self.optical_point_xyz_m
            ],
            'base_point_xyz_m': [
                round(float(value), 6) for value in self.base_point_xyz_m
            ],
            'forward_m': round(self.forward_m, 6),
            'lateral_left_m': round(self.lateral_left_m, 6),
            'planar_range_m': round(self.planar_range_m, 6),
            'depth': self.depth.as_dict(),
            'rgb_bbox_xywh': list(self.rgb_bbox_xywh),
            'depth_bbox_xywh': list(self.depth_bbox_xywh),
            'rgb_depth_stamp_skew_sec': round(
                float(self.rgb_depth_stamp_skew_sec), 6),
        }


def map_bbox_between_cameras(
    bbox: BoundingBoxLike,
    source: CameraModel,
    destination: CameraModel,
    pad_height_fraction_x: float = 0.85,
    pad_height_fraction_y: float = 0.45,
) -> BoundingBoxLike:
    """Map an aligned RGB rectangle into depth pixels through normalized rays.

    The official simulation uses matched RGB/depth resolution and FoV, so this
    is normally an identity mapping plus a plate-context pad.  The normalized
    mapping also tolerates small differences in CameraInfo intrinsics.
    """
    source.validate()
    destination.validate()
    pad_x = max(2.0, pad_height_fraction_x * float(bbox.height))
    pad_y = max(2.0, pad_height_fraction_y * float(bbox.height))
    u1 = float(bbox.x) - pad_x
    v1 = float(bbox.y) - pad_y
    u2 = float(bbox.x2) + pad_x
    v2 = float(bbox.y2) + pad_y

    def map_pixel(u: float, v: float) -> Tuple[float, float]:
        xn = (u - source.cx) / source.fx
        yn = (v - source.cy) / source.fy
        return (
            destination.fx * xn + destination.cx,
            destination.fy * yn + destination.cy,
        )

    mapped = [map_pixel(u1, v1), map_pixel(u2, v2)]
    x1 = int(math.floor(min(item[0] for item in mapped)))
    y1 = int(math.floor(min(item[1] for item in mapped)))
    x2 = int(math.ceil(max(item[0] for item in mapped)))
    y2 = int(math.ceil(max(item[1] for item in mapped)))
    return BoundingBoxLike(
        x1, y1, max(1, x2 - x1), max(1, y2 - y1)
    ).clipped(destination.width, destination.height)


def normalize_depth_array(depth: np.ndarray, encoding: str) -> np.ndarray:
    """Return float32 metres for common ROS depth encodings."""
    array = np.asarray(depth)
    normalized_encoding = str(encoding).upper()
    if normalized_encoding in {'16UC1', 'MONO16'} or array.dtype == np.uint16:
        return array.astype(np.float32) * 0.001
    return array.astype(np.float32, copy=False)


def robust_foreground_depth(
    depth_m: np.ndarray,
    sample_bbox: BoundingBoxLike,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_valid_pixels: int = 20,
    foreground_band_m: float = 0.22,
    mad_multiplier: float = 3.5,
) -> Optional[DepthEstimate]:
    """Estimate the nearest coherent surface in a marker-centred depth patch.

    The patch intentionally includes part of the white marker plate.  A lower
    quantile anchors the nearest coherent surface, a finite depth band rejects
    the wall behind the marker, and a median/MAD pass suppresses edge outliers.
    """
    if depth_m.ndim != 2:
        raise ValueError('Depth array must be two-dimensional')
    height, width = depth_m.shape
    box = sample_bbox.clipped(width, height)
    if box.width <= 0 or box.height <= 0:
        return None
    patch = depth_m[box.y:box.y2, box.x:box.x2]
    total = int(patch.size)
    valid_mask = (
        np.isfinite(patch)
        & (patch >= float(min_depth_m))
        & (patch <= float(max_depth_m))
    )
    values = patch[valid_mask].astype(np.float64)
    valid_count = int(values.size)
    if valid_count < int(min_valid_pixels):
        return None

    near_anchor = float(np.percentile(values, 20.0))
    foreground = values[values <= near_anchor + float(foreground_band_m)]
    if foreground.size < int(min_valid_pixels):
        foreground = values

    median = float(np.median(foreground))
    absolute = np.abs(foreground - median)
    mad = float(np.median(absolute))
    tolerance = max(0.015, float(mad_multiplier) * 1.4826 * mad)
    inliers = foreground[absolute <= tolerance]
    if inliers.size >= int(min_valid_pixels):
        foreground = inliers
        median = float(np.median(foreground))
        mad = float(np.median(np.abs(foreground - median)))

    return DepthEstimate(
        depth_m=median,
        valid_pixels=valid_count,
        total_pixels=total,
        valid_fraction=float(valid_count) / float(max(1, total)),
        median_absolute_deviation_m=mad,
        foreground_pixels=int(foreground.size),
        sample_bbox_xywh=box.as_list(),
        pixel_u=box.x + 0.5 * box.width,
        pixel_v=box.y + 0.5 * box.height,
    )


def quaternion_to_rotation_matrix(
    x: float, y: float, z: float, w: float,
) -> np.ndarray:
    """Return a 3x3 rotation matrix for a normalized quaternion."""
    q = np.array([float(x), float(y), float(z), float(w)], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError('Quaternion norm is zero')
    x, y, z, w = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_points(
    points_xyz: np.ndarray,
    translation_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Apply target<-source rigid transform to N x 3 points."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_xyz must have shape (N, 3)')
    if len(translation_xyz) != 3 or len(quaternion_xyzw) != 4:
        raise ValueError('Transform components have invalid lengths')
    rotation = quaternion_to_rotation_matrix(*quaternion_xyzw)
    translation = np.asarray(translation_xyz, dtype=np.float64).reshape(1, 3)
    return points @ rotation.T + translation


def target_geometry_from_depth(
    rgb_bbox: BoundingBoxLike,
    rgb_model: CameraModel,
    depth_model: CameraModel,
    depth_m: np.ndarray,
    depth_to_base_translation_xyz: Sequence[float],
    depth_to_base_quaternion_xyzw: Sequence[float],
    rgb_depth_stamp_skew_sec: float,
    min_valid_pixels: int = 20,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> Optional[TargetGeometry]:
    """Build a live base-frame target observation from a marker bbox and depth."""
    depth_bbox = map_bbox_between_cameras(rgb_bbox, rgb_model, depth_model)
    estimate = robust_foreground_depth(
        depth_m,
        depth_bbox,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        min_valid_pixels=min_valid_pixels,
    )
    if estimate is None:
        return None

    # Sample at the mapped centre; depth Z is along the optical +Z axis.
    u = estimate.pixel_u
    v = estimate.pixel_v
    z = estimate.depth_m
    optical = np.array([
        (u - depth_model.cx) * z / depth_model.fx,
        (v - depth_model.cy) * z / depth_model.fy,
        z,
    ], dtype=np.float64)
    base = transform_points(
        optical,
        depth_to_base_translation_xyz,
        depth_to_base_quaternion_xyzw,
    )[0]
    base_bearing = math.atan2(float(base[1]), float(base[0]))
    camera_bearing = rgb_model.bearing_right_rad(rgb_bbox.center_x)
    return TargetGeometry(
        camera_bearing_right_rad=camera_bearing,
        base_bearing_left_rad=base_bearing,
        optical_point_xyz_m=(
            float(optical[0]), float(optical[1]), float(optical[2])),
        base_point_xyz_m=(float(base[0]), float(base[1]), float(base[2])),
        depth=estimate,
        rgb_bbox_xywh=rgb_bbox.as_list(),
        depth_bbox_xywh=depth_bbox.as_list(),
        rgb_depth_stamp_skew_sec=float(rgb_depth_stamp_skew_sec),
    )


@dataclass(frozen=True)
class LidarClearance:
    robust_clearance_m: float
    minimum_clearance_m: float
    corridor_point_count: int
    hard_cluster_point_count: int
    corridor_half_width_m: float
    percentile: float

    def as_dict(self) -> dict[str, object]:
        return {
            'robust_clearance_m': round(float(self.robust_clearance_m), 6),
            'minimum_clearance_m': round(float(self.minimum_clearance_m), 6),
            'corridor_point_count': int(self.corridor_point_count),
            'hard_cluster_point_count': int(self.hard_cluster_point_count),
            'corridor_half_width_m': round(
                float(self.corridor_half_width_m), 6),
            'percentile': round(float(self.percentile), 3),
        }


def lidar_corridor_clearance(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    laser_to_base_translation_xyz: Sequence[float],
    laser_to_base_quaternion_xyzw: Sequence[float],
    corridor_half_width_m: float,
    robust_percentile: float = 10.0,
    hard_stop_m: float = 0.72,
) -> Optional[LidarClearance]:
    """Measure forward clearance in a base-frame swept-body corridor."""
    range_array = np.asarray(list(ranges), dtype=np.float64)
    if range_array.size == 0:
        return None
    angles = float(angle_min) + np.arange(range_array.size) * float(angle_increment)
    valid = (
        np.isfinite(range_array)
        & (range_array >= float(range_min))
        & (range_array <= float(range_max))
    )
    if not np.any(valid):
        return None
    r = range_array[valid]
    a = angles[valid]
    laser_points = np.column_stack((
        r * np.cos(a),
        r * np.sin(a),
        np.zeros_like(r),
    ))
    base_points = transform_points(
        laser_points,
        laser_to_base_translation_xyz,
        laser_to_base_quaternion_xyzw,
    )
    forward = base_points[:, 0]
    lateral = base_points[:, 1]
    corridor = (
        (forward > 0.0)
        & (np.abs(lateral) <= float(corridor_half_width_m))
    )
    values = forward[corridor]
    if values.size == 0:
        return None
    percentile = min(50.0, max(0.0, float(robust_percentile)))
    robust = float(np.percentile(values, percentile))
    minimum = float(np.min(values))
    hard_cluster = int(np.count_nonzero(values <= float(hard_stop_m)))
    return LidarClearance(
        robust_clearance_m=robust,
        minimum_clearance_m=minimum,
        corridor_point_count=int(values.size),
        hard_cluster_point_count=hard_cluster,
        corridor_half_width_m=float(corridor_half_width_m),
        percentile=percentile,
    )


@dataclass(frozen=True)
class ControllerSettings:
    target_standoff_m: float = 1.05
    max_forward_speed_mps: float = 0.30
    min_forward_speed_mps: float = 0.06
    max_lateral_speed_mps: float = 0.14
    max_yaw_speed_radps: float = 0.20
    forward_kp: float = 0.45
    lateral_kp: float = 0.75
    yaw_kp: float = 0.90
    lateral_deadband_m: float = 0.025
    yaw_deadband_rad: float = math.radians(1.5)
    lidar_slowdown_m: float = 1.50
    lidar_hard_stop_m: float = 0.72
    lidar_command_margin_m: float = 0.12
    max_planar_speed_mps: float = 0.32

    def validate(self) -> None:
        if self.target_standoff_m <= self.lidar_hard_stop_m:
            raise ValueError('Standoff must exceed the LiDAR hard stop')
        if not 0.0 < self.min_forward_speed_mps <= self.max_forward_speed_mps:
            raise ValueError('Forward speed limits are invalid')
        if self.max_lateral_speed_mps <= 0.0 or self.max_yaw_speed_radps <= 0.0:
            raise ValueError('Lateral and yaw speed limits must be positive')
        if self.lidar_slowdown_m <= self.lidar_hard_stop_m:
            raise ValueError('LiDAR slowdown distance must exceed hard stop')


@dataclass(frozen=True)
class ApproachCommand:
    vx_mps: float
    vy_mps: float
    wz_radps: float
    depth_forward_error_m: float
    lateral_error_m: float
    yaw_heading_error_rad: float
    target_bearing_rad: float
    lidar_speed_cap_mps: float
    lateral_path_speed_cap_mps: float

    def as_dict(self) -> dict[str, float]:
        return {
            'vx_mps': round(float(self.vx_mps), 6),
            'vy_mps': round(float(self.vy_mps), 6),
            'wz_radps': round(float(self.wz_radps), 6),
            'depth_forward_error_m': round(
                float(self.depth_forward_error_m), 6),
            'lateral_error_m': round(float(self.lateral_error_m), 6),
            'yaw_heading_error_deg': round(
                math.degrees(self.yaw_heading_error_rad), 6),
            'target_bearing_deg': round(
                math.degrees(self.target_bearing_rad), 6),
            'lidar_speed_cap_mps': round(
                float(self.lidar_speed_cap_mps), 6),
            'lateral_path_speed_cap_mps': round(
                float(self.lateral_path_speed_cap_mps), 6),
        }


def _signed_deadband(value: float, deadband: float) -> float:
    return 0.0 if abs(value) <= deadband else value


def compute_holonomic_command(
    geometry: TargetGeometry,
    lidar_clearance_m: float,
    settings: ControllerSettings,
    heading_error_rad: float = 0.0,
) -> ApproachCommand:
    """Compute bounded body-frame +x/+y/+yaw command.

    ROS base convention is +x forward, +y left, and +yaw counter-clockwise.
    A target left of the robot produces positive ``vy``. ``wz`` holds the
    shelf-facing reference heading supplied by the caller, so the mecanum base
    strafes to the column instead of turning onto a diagonal final pose.
    """
    settings.validate()
    forward_error = geometry.forward_m - settings.target_standoff_m
    lateral_error = _signed_deadband(
        geometry.lateral_left_m, settings.lateral_deadband_m)
    yaw_heading_error = _signed_deadband(
        float(heading_error_rad), settings.yaw_deadband_rad)

    raw_vx = max(0.0, settings.forward_kp * forward_error)
    if raw_vx > 0.0 and forward_error > 0.08:
        raw_vx = max(settings.min_forward_speed_mps, raw_vx)
    raw_vx = min(settings.max_forward_speed_mps, raw_vx)

    # Do not charge forward while severely misaligned.  Lateral correction
    # remains available because the base is holonomic.
    alignment_scale = max(
        0.20,
        math.cos(min(abs(geometry.base_bearing_left_rad), math.radians(75.0))),
    )
    raw_vx *= alignment_scale

    # Keep a far edge marker inside the 87-degree camera FoV and complete the
    # lateral displacement before reaching the forward stand-off plane.  This
    # is the key holonomic path cap: vy/vx is matched to the remaining lateral
    # and forward errors instead of blindly charging straight ahead.
    lateral_path_cap = settings.max_forward_speed_mps
    lateral_demand = abs(geometry.lateral_left_m)
    if lateral_demand > settings.lateral_deadband_m and forward_error > 0.05:
        lateral_path_cap = min(
            settings.max_forward_speed_mps,
            0.90 * settings.max_lateral_speed_mps
            * forward_error / lateral_demand,
        )
    raw_vx = min(raw_vx, max(0.0, lateral_path_cap))

    available = float(lidar_clearance_m) - (
        settings.lidar_hard_stop_m + settings.lidar_command_margin_m)
    slowdown_span = max(
        1.0e-6,
        settings.lidar_slowdown_m
        - settings.lidar_hard_stop_m
        - settings.lidar_command_margin_m,
    )
    lidar_fraction = max(0.0, min(1.0, available / slowdown_span))
    lidar_cap = settings.max_forward_speed_mps * lidar_fraction
    vx = min(raw_vx, lidar_cap)

    vy = max(
        -settings.max_lateral_speed_mps,
        min(settings.max_lateral_speed_mps,
            settings.lateral_kp * lateral_error),
    )
    wz = max(
        -settings.max_yaw_speed_radps,
        min(settings.max_yaw_speed_radps,
            settings.yaw_kp * yaw_heading_error),
    )

    planar = math.hypot(vx, vy)
    if planar > settings.max_planar_speed_mps:
        scale = settings.max_planar_speed_mps / planar
        vx *= scale
        vy *= scale

    return ApproachCommand(
        vx_mps=vx,
        vy_mps=vy,
        wz_radps=wz,
        depth_forward_error_m=forward_error,
        lateral_error_m=lateral_error,
        yaw_heading_error_rad=yaw_heading_error,
        target_bearing_rad=geometry.base_bearing_left_rad,
        lidar_speed_cap_mps=lidar_cap,
        lateral_path_speed_cap_mps=lateral_path_cap,
    )


def limit_rate(
    requested: float,
    previous: float,
    maximum_rate_per_sec: float,
    dt_sec: float,
) -> float:
    """Symmetric acceleration/rate limiter."""
    if dt_sec <= 0.0 or maximum_rate_per_sec <= 0.0:
        return float(requested)
    maximum_delta = maximum_rate_per_sec * dt_sec
    delta = max(-maximum_delta, min(maximum_delta, requested - previous))
    return previous + delta
