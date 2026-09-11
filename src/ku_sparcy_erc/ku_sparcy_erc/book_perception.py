#!/usr/bin/env python3
"""CPU-only selected-column book perception and 3-D geometry helpers.

The competition node supplies a live BGR image, a selected-column region of
interest, RGB/depth CameraInfo models, synchronized raw depth, and TF.  This
module performs only deterministic pixel and geometry operations; it never
reads an ERC seed, Gazebo entity name, model pose, or expected layout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from ku_sparcy_erc.range_fusion import (
    BoundingBoxLike,
    CameraModel,
    transform_points,
)


BOOK_COLOURS: Tuple[str, ...] = ('red', 'green', 'yellow', 'blue')


@dataclass(frozen=True)
class BookDetectorSettings:
    """Thresholds for coloured spine detection inside the selected column."""

    min_saturation: int = 90
    min_value: int = 55
    min_area_px: int = 18
    min_height_px: int = 12
    max_height_fraction: float = 0.55
    min_width_px: int = 2
    max_width_fraction: float = 0.30
    min_vertical_aspect: float = 1.20
    min_fill_fraction: float = 0.22
    min_colour_purity: float = 0.55
    minimum_row_separation_px: float = 18.0
    min_detection_confidence: float = 0.58

    def validate(self) -> None:
        if not 0 <= self.min_saturation <= 255:
            raise ValueError('min_saturation must be in [0, 255]')
        if not 0 <= self.min_value <= 255:
            raise ValueError('min_value must be in [0, 255]')
        if self.min_area_px < 4:
            raise ValueError('min_area_px must be at least 4')
        if self.min_height_px < 4 or self.min_width_px < 1:
            raise ValueError('minimum spine dimensions are too small')
        if self.min_vertical_aspect <= 0.0:
            raise ValueError('min_vertical_aspect must be positive')
        for name, value in (
            ('min_fill_fraction', self.min_fill_fraction),
            ('min_colour_purity', self.min_colour_purity),
            ('min_detection_confidence', self.min_detection_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1]')


@dataclass(frozen=True)
class BookDetection:
    """One coloured-book spine candidate in one live RGB frame."""

    colour: str
    confidence: float
    bbox: BoundingBoxLike
    colour_purity: float
    fill_fraction: float
    vertical_aspect: float

    @property
    def center_x(self) -> float:
        return self.bbox.center_x

    @property
    def center_y(self) -> float:
        return self.bbox.center_y

    def as_dict(self) -> Dict[str, object]:
        return {
            'colour': self.colour,
            'confidence': round(float(self.confidence), 6),
            'bbox_xywh': self.bbox.as_list(),
            'colour_purity': round(float(self.colour_purity), 6),
            'fill_fraction': round(float(self.fill_fraction), 6),
            'vertical_aspect': round(float(self.vertical_aspect), 6),
        }


@dataclass(frozen=True)
class BookLayoutObservation:
    """One coherent frame containing exactly one book of every colour."""

    stamp_sec: float
    frame_index: int
    row_colours_top_to_bottom: Tuple[str, str, str, str]
    detections_by_colour: Mapping[str, BookDetection]
    confidence: float

    def as_dict(self) -> Dict[str, object]:
        return {
            'stamp_sec': round(float(self.stamp_sec), 9),
            'frame_index': int(self.frame_index),
            'row_colours_top_to_bottom': list(self.row_colours_top_to_bottom),
            'confidence': round(float(self.confidence), 6),
            'detections_by_colour': {
                colour: detection.as_dict()
                for colour, detection in self.detections_by_colour.items()
            },
        }


@dataclass(frozen=True)
class ConfirmedBookLayout:
    """Temporally confirmed coherent selected-column colour layout."""

    first_stamp_sec: float
    confirmed_stamp_sec: float
    last_seen_stamp_sec: float
    confirming_frames: int
    confidence: float
    row_colours_top_to_bottom: Tuple[str, str, str, str]
    detections_by_colour: Mapping[str, BookDetection]

    def row_for_colour(self, colour: str) -> int:
        normalized = str(colour).strip().lower()
        return self.row_colours_top_to_bottom.index(normalized) + 1

    def as_dict(self) -> Dict[str, object]:
        return {
            'first_stamp_sec': round(float(self.first_stamp_sec), 9),
            'confirmed_stamp_sec': round(float(self.confirmed_stamp_sec), 9),
            'last_seen_stamp_sec': round(float(self.last_seen_stamp_sec), 9),
            'confirming_frames': int(self.confirming_frames),
            'confidence': round(float(self.confidence), 6),
            'row_colours_top_to_bottom': list(self.row_colours_top_to_bottom),
            'detections_by_colour': {
                colour: detection.as_dict()
                for colour, detection in self.detections_by_colour.items()
            },
        }


@dataclass(frozen=True)
class BookTemporalSettings:
    required_frames: int = 3
    window_sec: float = 0.90
    minimum_span_sec: float = 0.05
    maximum_center_jump_px: float = 35.0
    minimum_average_confidence: float = 0.62

    def validate(self) -> None:
        if self.required_frames < 2:
            raise ValueError('required_frames must be at least 2')
        if self.window_sec <= 0.0:
            raise ValueError('window_sec must be positive')
        if not 0.0 <= self.minimum_span_sec <= self.window_sec:
            raise ValueError('minimum_span_sec must be within the window')
        if self.maximum_center_jump_px <= 0.0:
            raise ValueError('maximum_center_jump_px must be positive')
        if not 0.0 < self.minimum_average_confidence <= 1.0:
            raise ValueError('minimum_average_confidence must be in (0, 1]')


@dataclass(frozen=True)
class BookDepthEstimate:
    depth_m: float
    valid_pixels: int
    candidate_pixels: int
    valid_fraction: float
    median_absolute_deviation_m: float
    rgb_sample_bbox_xywh: List[int]
    depth_sample_bbox_xywh: List[int]
    depth_pixel_u: float
    depth_pixel_v: float

    def as_dict(self) -> Dict[str, object]:
        return {
            'depth_m': round(float(self.depth_m), 6),
            'valid_pixels': int(self.valid_pixels),
            'candidate_pixels': int(self.candidate_pixels),
            'valid_fraction': round(float(self.valid_fraction), 6),
            'median_absolute_deviation_m': round(
                float(self.median_absolute_deviation_m), 6),
            'rgb_sample_bbox_xywh': list(self.rgb_sample_bbox_xywh),
            'depth_sample_bbox_xywh': list(self.depth_sample_bbox_xywh),
            'depth_pixel_u': round(float(self.depth_pixel_u), 3),
            'depth_pixel_v': round(float(self.depth_pixel_v), 3),
        }


@dataclass(frozen=True)
class BookGeometry:
    colour: str
    row: int
    optical_point_xyz_m: Tuple[float, float, float]
    base_point_xyz_m: Tuple[float, float, float]
    camera_bearing_right_rad: float
    base_bearing_left_rad: float
    rgb_depth_stamp_skew_sec: float
    depth: BookDepthEstimate
    bbox_xywh: List[int]

    @property
    def forward_m(self) -> float:
        return float(self.base_point_xyz_m[0])

    @property
    def lateral_left_m(self) -> float:
        return float(self.base_point_xyz_m[1])

    @property
    def height_m(self) -> float:
        return float(self.base_point_xyz_m[2])

    def as_dict(self) -> Dict[str, object]:
        return {
            'colour': self.colour,
            'row': int(self.row),
            'optical_point_xyz_m': [
                round(float(value), 6) for value in self.optical_point_xyz_m
            ],
            'base_point_xyz_m': [
                round(float(value), 6) for value in self.base_point_xyz_m
            ],
            'forward_m': round(self.forward_m, 6),
            'lateral_left_m': round(self.lateral_left_m, 6),
            'height_m': round(self.height_m, 6),
            'camera_bearing_right_deg': round(
                math.degrees(self.camera_bearing_right_rad), 6),
            'base_bearing_left_deg': round(
                math.degrees(self.base_bearing_left_rad), 6),
            'rgb_depth_stamp_skew_sec': round(
                float(self.rgb_depth_stamp_skew_sec), 6),
            'depth': self.depth.as_dict(),
            'bbox_xywh': list(self.bbox_xywh),
        }


@dataclass(frozen=True)
class ArmReachCandidate:
    arm: str
    shoulder_frame: str
    shoulder_xyz_m: Tuple[float, float, float]
    distance_to_pregrasp_m: float
    reachable: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            'arm': self.arm,
            'shoulder_frame': self.shoulder_frame,
            'shoulder_xyz_m': [
                round(float(value), 6) for value in self.shoulder_xyz_m
            ],
            'distance_to_pregrasp_m': round(
                float(self.distance_to_pregrasp_m), 6),
            'reachable': bool(self.reachable),
            'reason': self.reason,
        }


@dataclass(frozen=True)
class PregraspPlan:
    selected_arm: str
    book_surface_point_base_xyz_m: Tuple[float, float, float]
    pregrasp_point_base_xyz_m: Tuple[float, float, float]
    tentative_grasp_point_base_xyz_m: Tuple[float, float, float]
    retreat_point_base_xyz_m: Tuple[float, float, float]
    approach_axis_base_xyz: Tuple[float, float, float]
    arm_candidates: Mapping[str, ArmReachCandidate]
    geometric_screen_passed: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            'selected_arm': self.selected_arm,
            'book_surface_point_base_xyz_m': [
                round(float(value), 6)
                for value in self.book_surface_point_base_xyz_m
            ],
            'pregrasp_point_base_xyz_m': [
                round(float(value), 6)
                for value in self.pregrasp_point_base_xyz_m
            ],
            'tentative_grasp_point_base_xyz_m': [
                round(float(value), 6)
                for value in self.tentative_grasp_point_base_xyz_m
            ],
            'retreat_point_base_xyz_m': [
                round(float(value), 6)
                for value in self.retreat_point_base_xyz_m
            ],
            'approach_axis_base_xyz': [
                round(float(value), 6)
                for value in self.approach_axis_base_xyz
            ],
            'arm_candidates': {
                arm: candidate.as_dict()
                for arm, candidate in self.arm_candidates.items()
            },
            'geometric_screen_passed': bool(self.geometric_screen_passed),
            'reason': self.reason,
            'ik_solved': False,
            'collision_checked': False,
            'arm_motion_executed': False,
        }


def colour_mask(hsv: np.ndarray, colour: str, settings: BookDetectorSettings) -> np.ndarray:
    """Return a binary HSV mask for one competition book colour."""
    name = str(colour).strip().lower()
    s = settings.min_saturation
    v = settings.min_value
    if name == 'red':
        low = cv2.inRange(hsv, np.array([0, s, v]), np.array([12, 255, 255]))
        high = cv2.inRange(hsv, np.array([168, s, v]), np.array([179, 255, 255]))
        return cv2.bitwise_or(low, high)
    ranges = {
        'yellow': (18, 40),
        'green': (38, 88),
        'blue': (90, 138),
    }
    if name not in ranges:
        raise ValueError(f'unsupported book colour: {colour}')
    lower_h, upper_h = ranges[name]
    return cv2.inRange(
        hsv,
        np.array([lower_h, s, v]),
        np.array([upper_h, 255, 255]),
    )


def selected_column_roi(
    frame_width: int,
    frame_height: int,
    camera_fx: float,
    camera_cx: float,
    lidar_clearance_m: float,
    physical_width_m: float = 0.78,
    minimum_half_width_px: int = 40,
    maximum_half_width_px: int = 148,
    center_u: Optional[float] = None,
) -> BoundingBoxLike:
    """Create a sensor-centred selected-column ROI after Day 3 alignment.

    Day 3 has already aligned the base with the locked physical column.  The
    ROI therefore uses the live principal point and LiDAR stand-off, not a
    randomized shelf lookup.  Its physical width covers the allowed horizontal
    book jitter while excluding adjacent columns at the validated stand-off.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError('frame dimensions must be positive')
    if not math.isfinite(camera_fx) or camera_fx <= 0.0:
        raise ValueError('camera_fx must be positive')
    if not math.isfinite(lidar_clearance_m) or lidar_clearance_m <= 0.0:
        raise ValueError('lidar_clearance_m must be positive')
    half_width = 0.5 * camera_fx * physical_width_m / lidar_clearance_m
    half_width = int(round(max(
        float(minimum_half_width_px),
        min(float(maximum_half_width_px), half_width),
    )))
    selected_center = camera_cx if center_u is None else float(center_u)
    if not math.isfinite(selected_center):
        raise ValueError('selected-column pixel centre must be finite')
    center = int(round(selected_center))
    x1 = max(0, center - half_width)
    x2 = min(frame_width, center + half_width)
    y1 = max(0, int(round(0.02 * frame_height)))
    y2 = min(frame_height, int(round(0.98 * frame_height)))
    return BoundingBoxLike(x1, y1, max(1, x2 - x1), max(1, y2 - y1))


class SelectedColumnBookDetector:
    """Detect one red, green, yellow, and blue spine in a selected-column ROI."""

    def __init__(self, settings: Optional[BookDetectorSettings] = None) -> None:
        self.settings = settings or BookDetectorSettings()
        self.settings.validate()

    def _candidates_for_colour(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        roi: BoundingBoxLike,
        colour: str,
    ) -> List[BookDetection]:
        settings = self.settings
        full_mask = colour_mask(hsv, colour, settings)
        roi_mask = full_mask[roi.y:roi.y2, roi.x:roi.x2].copy()
        if roi_mask.size == 0:
            return []
        roi_mask = cv2.morphologyEx(
            roi_mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
            iterations=1,
        )
        roi_mask = cv2.morphologyEx(
            roi_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maximum_height = max(
            settings.min_height_px + 1,
            int(round(frame_bgr.shape[0] * settings.max_height_fraction)),
        )
        maximum_width = max(
            settings.min_width_px + 1,
            int(round(roi.width * settings.max_width_fraction)),
        )
        candidates: List[BookDetection] = []
        for contour in contours:
            x_local, y_local, width, height = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            if area < settings.min_area_px:
                continue
            if not (
                settings.min_width_px <= width <= maximum_width
                and settings.min_height_px <= height <= maximum_height
            ):
                continue
            aspect = float(height) / float(max(1, width))
            if aspect < settings.min_vertical_aspect:
                continue
            fill = area / float(max(1, width * height))
            if fill < settings.min_fill_fraction:
                continue
            patch = roi_mask[
                y_local:y_local + height,
                x_local:x_local + width,
            ]
            purity = float(np.count_nonzero(patch)) / float(max(1, patch.size))
            if purity < settings.min_colour_purity:
                continue
            height_score = min(1.0, float(height) / 55.0)
            aspect_score = min(1.0, aspect / 5.0)
            confidence = (
                0.45 * purity
                + 0.25 * min(1.0, fill / 0.70)
                + 0.20 * height_score
                + 0.10 * aspect_score
            )
            if confidence < settings.min_detection_confidence:
                continue
            bbox = BoundingBoxLike(
                roi.x + x_local,
                roi.y + y_local,
                width,
                height,
            ).clipped(frame_bgr.shape[1], frame_bgr.shape[0])
            candidates.append(BookDetection(
                colour=colour,
                confidence=float(confidence),
                bbox=bbox,
                colour_purity=float(purity),
                fill_fraction=float(fill),
                vertical_aspect=float(aspect),
            ))
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates

    def detect_colours(
        self,
        frame_bgr: np.ndarray,
        column_roi: BoundingBoxLike,
        colours: Sequence[str] = BOOK_COLOURS,
    ) -> Tuple[Dict[str, List[BookDetection]], Dict[str, object]]:
        """Return independent colour candidates without requiring a layout."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError('frame is empty')
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError('expected a BGR image')
        roi = column_roi.clipped(frame_bgr.shape[1], frame_bgr.shape[0])
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        candidates_by_colour: Dict[str, List[BookDetection]] = {}
        candidate_counts: Dict[str, int] = {}
        top_candidates: Dict[str, List[Dict[str, object]]] = {}
        for colour in colours:
            name = str(colour).strip().lower()
            if name not in BOOK_COLOURS:
                raise ValueError(f'unsupported book colour: {colour}')
            candidates = self._candidates_for_colour(
                frame_bgr, hsv, roi, name)
            candidates_by_colour[name] = candidates
            candidate_counts[name] = len(candidates)
            top_candidates[name] = [
                candidate.as_dict() for candidate in candidates[:3]
            ]
        diagnostics: Dict[str, object] = {
            'column_roi_xywh': roi.as_list(),
            'candidate_counts': candidate_counts,
            'top_candidates': top_candidates,
            'visible_colours': [
                colour for colour, values in candidates_by_colour.items()
                if values
            ],
        }
        return candidates_by_colour, diagnostics

    def detect_layout(
        self,
        frame_bgr: np.ndarray,
        column_roi: BoundingBoxLike,
        stamp_sec: float,
        frame_index: int,
    ) -> Tuple[Optional[BookLayoutObservation], Dict[str, object]]:
        candidates_by_colour, diagnostics = self.detect_colours(
            frame_bgr,
            column_roi,
            BOOK_COLOURS,
        )
        strongest: Dict[str, BookDetection] = {
            colour: candidates[0]
            for colour, candidates in candidates_by_colour.items()
            if candidates
        }
        diagnostics['complete_colour_set'] = len(strongest) == 4
        if len(strongest) != 4:
            return None, diagnostics

        ordered = sorted(strongest.values(), key=lambda item: item.center_y)
        separations = [
            ordered[index + 1].center_y - ordered[index].center_y
            for index in range(3)
        ]
        diagnostics['row_separations_px'] = [
            round(float(value), 3) for value in separations
        ]
        if any(
            value < self.settings.minimum_row_separation_px
            for value in separations
        ):
            diagnostics['rejected_reason'] = 'row centres are not sufficiently separated'
            return None, diagnostics

        row_colours = tuple(item.colour for item in ordered)
        if sorted(row_colours) != sorted(BOOK_COLOURS):
            diagnostics['rejected_reason'] = 'layout does not contain one of each colour'
            return None, diagnostics
        confidence = float(np.mean([item.confidence for item in ordered]))
        observation = BookLayoutObservation(
            stamp_sec=float(stamp_sec),
            frame_index=int(frame_index),
            row_colours_top_to_bottom=row_colours,  # type: ignore[arg-type]
            detections_by_colour=dict(strongest),
            confidence=confidence,
        )
        diagnostics['row_colours_top_to_bottom'] = list(row_colours)
        diagnostics['layout_confidence'] = round(confidence, 6)
        return observation, diagnostics


class TemporalBookLayoutFilter:
    """Confirm one coherent row-colour layout across multiple live frames."""

    def __init__(self, settings: Optional[BookTemporalSettings] = None) -> None:
        self.settings = settings or BookTemporalSettings()
        self.settings.validate()
        self._history: Deque[BookLayoutObservation] = deque()
        self._confirmed: Optional[ConfirmedBookLayout] = None

    def reset(self) -> None:
        self._history.clear()
        self._confirmed = None

    def _geometry_consistent(
        self,
        previous: BookLayoutObservation,
        current: BookLayoutObservation,
    ) -> bool:
        if previous.row_colours_top_to_bottom != current.row_colours_top_to_bottom:
            return False
        for colour in BOOK_COLOURS:
            before = previous.detections_by_colour[colour]
            after = current.detections_by_colour[colour]
            distance = math.hypot(
                after.center_x - before.center_x,
                after.center_y - before.center_y,
            )
            if distance > self.settings.maximum_center_jump_px:
                return False
        return True

    def update(
        self,
        observation: Optional[BookLayoutObservation],
        current_stamp_sec: float,
    ) -> Optional[ConfirmedBookLayout]:
        while (
            self._history
            and current_stamp_sec - self._history[0].stamp_sec
            > self.settings.window_sec
        ):
            self._history.popleft()
        if observation is None:
            return self._confirmed
        if self._history and not self._geometry_consistent(
            self._history[-1], observation
        ):
            self._history.clear()
        self._history.append(observation)
        while (
            self._history
            and current_stamp_sec - self._history[0].stamp_sec
            > self.settings.window_sec
        ):
            self._history.popleft()

        unique_frames = {item.frame_index for item in self._history}
        if len(unique_frames) < self.settings.required_frames:
            return self._confirmed
        span = self._history[-1].stamp_sec - self._history[0].stamp_sec
        if span < self.settings.minimum_span_sec:
            return self._confirmed
        average = float(np.mean([item.confidence for item in self._history]))
        if average < self.settings.minimum_average_confidence:
            return self._confirmed
        latest = self._history[-1]
        self._confirmed = ConfirmedBookLayout(
            first_stamp_sec=self._history[0].stamp_sec,
            confirmed_stamp_sec=latest.stamp_sec,
            last_seen_stamp_sec=latest.stamp_sec,
            confirming_frames=len(unique_frames),
            confidence=average,
            row_colours_top_to_bottom=latest.row_colours_top_to_bottom,
            detections_by_colour=dict(latest.detections_by_colour),
        )
        return self._confirmed

    @property
    def confirmed(self) -> Optional[ConfirmedBookLayout]:
        return self._confirmed


def _map_pixels(
    u_rgb: np.ndarray,
    v_rgb: np.ndarray,
    rgb_model: CameraModel,
    depth_model: CameraModel,
) -> Tuple[np.ndarray, np.ndarray]:
    xn = (u_rgb.astype(np.float64) - rgb_model.cx) / rgb_model.fx
    yn = (v_rgb.astype(np.float64) - rgb_model.cy) / rgb_model.fy
    u_depth = np.rint(depth_model.fx * xn + depth_model.cx).astype(np.int32)
    v_depth = np.rint(depth_model.fy * yn + depth_model.cy).astype(np.int32)
    return u_depth, v_depth


def estimate_book_geometry(
    frame_bgr: np.ndarray,
    target: BookDetection,
    row: int,
    rgb_model: CameraModel,
    depth_model: CameraModel,
    depth_m: np.ndarray,
    depth_to_base_translation_xyz: Sequence[float],
    depth_to_base_quaternion_xyzw: Sequence[float],
    rgb_depth_stamp_skew_sec: float,
    detector_settings: Optional[BookDetectorSettings] = None,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    min_valid_pixels: int = 12,
    min_valid_fraction: float = 0.12,
) -> Optional[BookGeometry]:
    """Estimate the coloured book-spine surface point using synchronized depth."""
    settings = detector_settings or BookDetectorSettings()
    settings.validate()
    rgb_model.validate()
    depth_model.validate()
    if depth_m.ndim != 2:
        raise ValueError('depth_m must be a two-dimensional array')
    bbox = target.bbox.clipped(frame_bgr.shape[1], frame_bgr.shape[0])
    if bbox.width <= 0 or bbox.height <= 0:
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = colour_mask(hsv, target.colour, settings)
    # Inset the rectangle to avoid shelf borders and anti-aliased edges.
    inset_x = max(0, int(round(0.12 * bbox.width)))
    inset_y = max(1, int(round(0.12 * bbox.height)))
    sample = BoundingBoxLike(
        bbox.x + inset_x,
        bbox.y + inset_y,
        max(1, bbox.width - 2 * inset_x),
        max(1, bbox.height - 2 * inset_y),
    ).clipped(frame_bgr.shape[1], frame_bgr.shape[0])
    patch_mask = mask[sample.y:sample.y2, sample.x:sample.x2]
    ys, xs = np.nonzero(patch_mask > 0)
    candidate_pixels = int(xs.size)
    if candidate_pixels < min_valid_pixels:
        # Fall back to all central bbox pixels if aliasing made the colour mask
        # too thin. The depth robustifier below still rejects invalid ranges.
        yy, xx = np.mgrid[sample.y:sample.y2, sample.x:sample.x2]
        v_rgb = yy.reshape(-1)
        u_rgb = xx.reshape(-1)
        candidate_pixels = int(u_rgb.size)
    else:
        u_rgb = xs.astype(np.int32) + sample.x
        v_rgb = ys.astype(np.int32) + sample.y

    u_depth, v_depth = _map_pixels(
        u_rgb, v_rgb, rgb_model, depth_model)
    inside = (
        (u_depth >= 0)
        & (u_depth < depth_model.width)
        & (v_depth >= 0)
        & (v_depth < depth_model.height)
    )
    u_depth = u_depth[inside]
    v_depth = v_depth[inside]
    if u_depth.size < min_valid_pixels:
        return None
    values = depth_m[v_depth, u_depth].astype(np.float64)
    valid = (
        np.isfinite(values)
        & (values >= float(min_depth_m))
        & (values <= float(max_depth_m))
    )
    values = values[valid]
    valid_u = u_depth[valid]
    valid_v = v_depth[valid]
    valid_count = int(values.size)
    if valid_count < min_valid_pixels:
        return None
    valid_fraction = valid_count / float(max(1, candidate_pixels))
    if valid_fraction < min_valid_fraction:
        return None

    # The book spine is the nearest coherent coloured surface. A 20th-percentile
    # anchor rejects shelf/background samples if any leaked into the thin mask.
    anchor = float(np.percentile(values, 20.0))
    foreground_selector = values <= anchor + 0.08
    foreground = values[foreground_selector]
    fg_u = valid_u[foreground_selector]
    fg_v = valid_v[foreground_selector]
    if foreground.size < min_valid_pixels:
        foreground = values
        fg_u = valid_u
        fg_v = valid_v
    median = float(np.median(foreground))
    absolute = np.abs(foreground - median)
    mad = float(np.median(absolute))
    tolerance = max(0.008, 3.5 * 1.4826 * mad)
    inlier = absolute <= tolerance
    if int(np.count_nonzero(inlier)) >= min_valid_pixels:
        foreground = foreground[inlier]
        fg_u = fg_u[inlier]
        fg_v = fg_v[inlier]
        median = float(np.median(foreground))
        mad = float(np.median(np.abs(foreground - median)))

    pixel_u = float(np.median(fg_u))
    pixel_v = float(np.median(fg_v))
    x_right = (pixel_u - depth_model.cx) * median / depth_model.fx
    y_down = (pixel_v - depth_model.cy) * median / depth_model.fy
    optical = np.array([[x_right, y_down, median]], dtype=np.float64)
    base = transform_points(
        optical,
        depth_to_base_translation_xyz,
        depth_to_base_quaternion_xyzw,
    )[0]
    depth_bbox = BoundingBoxLike(
        int(np.min(fg_u)),
        int(np.min(fg_v)),
        max(1, int(np.max(fg_u) - np.min(fg_u) + 1)),
        max(1, int(np.max(fg_v) - np.min(fg_v) + 1)),
    ).clipped(depth_model.width, depth_model.height)
    estimate = BookDepthEstimate(
        depth_m=median,
        valid_pixels=valid_count,
        candidate_pixels=candidate_pixels,
        valid_fraction=valid_fraction,
        median_absolute_deviation_m=mad,
        rgb_sample_bbox_xywh=sample.as_list(),
        depth_sample_bbox_xywh=depth_bbox.as_list(),
        depth_pixel_u=pixel_u,
        depth_pixel_v=pixel_v,
    )
    return BookGeometry(
        colour=target.colour,
        row=int(row),
        optical_point_xyz_m=(float(x_right), float(y_down), float(median)),
        base_point_xyz_m=(float(base[0]), float(base[1]), float(base[2])),
        camera_bearing_right_rad=rgb_model.bearing_right_rad(target.center_x),
        base_bearing_left_rad=math.atan2(
            float(base[1]), max(1.0e-9, float(base[0]))),
        rgb_depth_stamp_skew_sec=float(rgb_depth_stamp_skew_sec),
        depth=estimate,
        bbox_xywh=target.bbox.as_list(),
    )


def build_pregrasp_plan(
    geometry: BookGeometry,
    shoulder_positions_base: Mapping[str, Tuple[float, float, float]],
    pregrasp_offset_m: float = 0.18,
    tentative_contact_offset_m: float = 0.025,
    retreat_offset_m: float = 0.30,
    minimum_reach_m: float = 0.20,
    maximum_reach_m: float = 1.10,
) -> PregraspPlan:
    """Build a geometry-only, non-executing pre-grasp candidate.

    This is deliberately not an IK solution and not a collision check.  It
    chooses one arm by live shoulder-frame distance and records the candidate
    points that Day 5 must validate with a motion planner before commanding an
    arm.
    """
    surface = np.asarray(geometry.base_point_xyz_m, dtype=np.float64)
    if surface.shape != (3,) or not np.all(np.isfinite(surface)):
        raise ValueError('book surface point is invalid')
    pregrasp = surface.copy()
    pregrasp[0] -= float(pregrasp_offset_m)
    tentative = surface.copy()
    tentative[0] -= float(tentative_contact_offset_m)
    retreat = surface.copy()
    retreat[0] -= float(retreat_offset_m)

    candidates: Dict[str, ArmReachCandidate] = {}
    for arm in ('left', 'right'):
        frame = f'arm_{arm}_1_link'
        shoulder = shoulder_positions_base.get(arm)
        if shoulder is None:
            candidates[arm] = ArmReachCandidate(
                arm=arm,
                shoulder_frame=frame,
                shoulder_xyz_m=(math.nan, math.nan, math.nan),
                distance_to_pregrasp_m=math.inf,
                reachable=False,
                reason='shoulder TF unavailable',
            )
            continue
        shoulder_array = np.asarray(shoulder, dtype=np.float64)
        distance = float(np.linalg.norm(pregrasp - shoulder_array))
        reachable = (
            minimum_reach_m <= distance <= maximum_reach_m
            and pregrasp[0] > 0.20
            and 0.30 <= pregrasp[2] <= 1.90
        )
        reason = (
            'within conservative geometric reach screen'
            if reachable
            else 'outside conservative geometric reach screen'
        )
        candidates[arm] = ArmReachCandidate(
            arm=arm,
            shoulder_frame=frame,
            shoulder_xyz_m=tuple(float(value) for value in shoulder_array),
            distance_to_pregrasp_m=distance,
            reachable=reachable,
            reason=reason,
        )

    reachable_candidates = [
        candidate for candidate in candidates.values() if candidate.reachable
    ]
    if reachable_candidates:
        # Stable deterministic tie break: prefer right when distances are equal.
        selected = min(
            reachable_candidates,
            key=lambda item: (
                round(item.distance_to_pregrasp_m, 6),
                0 if item.arm == 'right' else 1,
            ),
        )
        passed = True
        reason = (
            f'{selected.arm} arm has the shortest live shoulder-to-pregrasp '
            'distance among geometrically reachable candidates'
        )
    else:
        selected = min(
            candidates.values(),
            key=lambda item: (
                item.distance_to_pregrasp_m,
                0 if item.arm == 'right' else 1,
            ),
        )
        passed = False
        reason = 'neither arm passed the conservative geometric reach screen'

    return PregraspPlan(
        selected_arm=selected.arm,
        book_surface_point_base_xyz_m=tuple(float(value) for value in surface),
        pregrasp_point_base_xyz_m=tuple(float(value) for value in pregrasp),
        tentative_grasp_point_base_xyz_m=tuple(float(value) for value in tentative),
        retreat_point_base_xyz_m=tuple(float(value) for value in retreat),
        approach_axis_base_xyz=(1.0, 0.0, 0.0),
        arm_candidates=candidates,
        geometric_screen_passed=passed,
        reason=reason,
    )


def draw_book_evidence(
    frame_bgr: np.ndarray,
    column_roi: BoundingBoxLike,
    layout: ConfirmedBookLayout,
    target_colour: str,
    target_row: int,
    lines: Sequence[str],
) -> np.ndarray:
    """Draw the selected column, all four books, and target-book evidence."""
    output = frame_bgr.copy()
    palette = {
        'red': (0, 0, 255),
        'green': (0, 220, 0),
        'yellow': (0, 220, 255),
        'blue': (255, 100, 0),
    }
    cv2.rectangle(
        output,
        (column_roi.x, column_roi.y),
        (column_roi.x2, column_roi.y2),
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        'SELECTED PHYSICAL SHELF COLUMN',
        (column_roi.x + 4, min(output.shape[0] - 8, column_roi.y2 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    row_lookup = {
        colour: index + 1
        for index, colour in enumerate(layout.row_colours_top_to_bottom)
    }
    for colour, detection in layout.detections_by_colour.items():
        box = detection.bbox
        is_target = colour == target_colour
        line_colour = (255, 255, 255) if is_target else palette[colour]
        thickness = 4 if is_target else 2
        cv2.rectangle(
            output,
            (box.x, box.y),
            (box.x2, box.y2),
            line_colour,
            thickness,
        )
        label = (
            f'TARGET {colour.upper()} ROW {target_row}'
            if is_target
            else f'{colour} row {row_lookup[colour]}'
        )
        cv2.putText(
            output,
            label,
            (max(2, box.x - 5), max(18, box.y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            line_colour,
            2 if is_target else 1,
            cv2.LINE_AA,
        )

    y = 22
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(
            output,
            (3, y - text_height - 5),
            (11 + text_width, y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output,
            line,
            (7, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 21
    return output


def draw_target_book_evidence(
    frame_bgr: np.ndarray,
    column_roi: BoundingBoxLike,
    target: BookDetection,
    target_row: int,
    row_colours_top_to_bottom: Sequence[str],
    per_colour_support: Mapping[str, int],
    lines: Sequence[str],
) -> np.ndarray:
    """Draw one close-range target book plus the accumulated row-map summary.

    Unlike ``draw_book_evidence``, this function does not project historical
    detections from different camera poses onto one image.  Only the current
    live target detection is boxed.  The multi-frame colour-to-row map appears
    as text metadata, preserving geometric honesty in the scoring image.
    """
    output = frame_bgr.copy()
    roi = column_roi.clipped(output.shape[1], output.shape[0])
    cv2.rectangle(
        output,
        (roi.x, roi.y),
        (roi.x2, roi.y2),
        (255, 255, 255),
        2,
    )
    cv2.putText(
        output,
        'LOCKED PHYSICAL SHELF COLUMN',
        (max(3, roi.x + 4), max(18, roi.y + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    box = target.bbox.clipped(output.shape[1], output.shape[0])
    cv2.rectangle(
        output,
        (box.x, box.y),
        (box.x2, box.y2),
        (255, 255, 255),
        4,
    )
    cv2.putText(
        output,
        f'TARGET {target.colour.upper()} - ROW {int(target_row)}',
        (max(2, box.x - 8), max(20, box.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    map_text = ' | '.join(
        f'R{index + 1}:{colour}'
        for index, colour in enumerate(row_colours_top_to_bottom)
    )
    support_text = ' '.join(
        f'{colour[0].upper()}:{int(per_colour_support.get(colour, 0))}'
        for colour in BOOK_COLOURS
    )
    all_lines = list(lines) + [
        f'accumulated row map: {map_text}',
        f'per-colour confirming frames: {support_text}',
    ]
    y = 22
    for line in all_lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            output,
            (3, y - text_height - 5),
            (11 + text_width, y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output,
            line,
            (7, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return output
