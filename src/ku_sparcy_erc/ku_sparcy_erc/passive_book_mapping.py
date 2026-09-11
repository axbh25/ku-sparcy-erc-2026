#!/usr/bin/env python3
"""Multi-frame selected-column book mapping for KU SPARCy Day 4.

The classes in this module are ROS-independent.  They combine independently
observed coloured book spines across different RGB frames, robot distances, and
bounded head poses.  A row map is accepted only after every competition colour
has a spatially consistent three-dimensional track.  No single frame is
required to contain all four books.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ku_sparcy_erc.book_perception import BookDetection, BookGeometry


BOOK_COLOURS: Tuple[str, ...] = ('red', 'green', 'yellow', 'blue')


@dataclass(frozen=True)
class SpatialBookObservation:
    """One depth-validated coloured-book observation in the odometry frame."""

    colour: str
    stamp_sec: float
    frame_index: int
    head_tilt_rad: float
    lidar_clearance_m: float
    detection: BookDetection
    geometry: BookGeometry
    odom_xyz_m: Tuple[float, float, float]
    roi_xywh: Tuple[int, int, int, int]

    def as_dict(self) -> Dict[str, object]:
        return {
            'colour': self.colour,
            'stamp_sec': round(float(self.stamp_sec), 9),
            'frame_index': int(self.frame_index),
            'head_tilt_rad': round(float(self.head_tilt_rad), 6),
            'lidar_clearance_m': round(float(self.lidar_clearance_m), 6),
            'detection': self.detection.as_dict(),
            'geometry': self.geometry.as_dict(),
            'odom_xyz_m': [
                round(float(value), 6) for value in self.odom_xyz_m
            ],
            'roi_xywh': [int(value) for value in self.roi_xywh],
        }


@dataclass(frozen=True)
class SpatialBookMapSettings:
    """Robust spatial-track and row-map acceptance thresholds."""

    required_frames_per_colour: int = 3
    minimum_observation_span_sec: float = 0.05
    minimum_average_confidence: float = 0.60
    maximum_cluster_xy_radius_m: float = 0.18
    maximum_cluster_z_radius_m: float = 0.10
    maximum_track_xy_span_m: float = 0.18
    maximum_track_z_span_m: float = 0.10
    minimum_row_separation_m: float = 0.14
    maximum_row_separation_m: float = 0.55
    minimum_book_height_m: float = 0.25
    maximum_book_height_m: float = 2.20
    maximum_observations_per_colour: int = 50

    def validate(self) -> None:
        if self.required_frames_per_colour < 2:
            raise ValueError('required_frames_per_colour must be at least 2')
        if self.minimum_observation_span_sec < 0.0:
            raise ValueError('minimum_observation_span_sec cannot be negative')
        if not 0.0 < self.minimum_average_confidence <= 1.0:
            raise ValueError('minimum_average_confidence must be in (0, 1]')
        for name, value in (
            ('maximum_cluster_xy_radius_m', self.maximum_cluster_xy_radius_m),
            ('maximum_cluster_z_radius_m', self.maximum_cluster_z_radius_m),
            ('maximum_track_xy_span_m', self.maximum_track_xy_span_m),
            ('maximum_track_z_span_m', self.maximum_track_z_span_m),
            ('minimum_row_separation_m', self.minimum_row_separation_m),
            ('maximum_row_separation_m', self.maximum_row_separation_m),
        ):
            if value <= 0.0:
                raise ValueError(f'{name} must be positive')
        if self.minimum_row_separation_m >= self.maximum_row_separation_m:
            raise ValueError('row separation interval is invalid')
        if self.minimum_book_height_m >= self.maximum_book_height_m:
            raise ValueError('book-height interval is invalid')
        if self.maximum_observations_per_colour < self.required_frames_per_colour:
            raise ValueError('observation capacity is too small')


@dataclass(frozen=True)
class ColourTrackSummary:
    colour: str
    confirming_frames: int
    first_stamp_sec: float
    last_stamp_sec: float
    average_confidence: float
    median_odom_xyz_m: Tuple[float, float, float]
    xy_span_m: float
    z_span_m: float
    clearance_range_m: Tuple[float, float]
    head_tilt_range_rad: Tuple[float, float]
    representative_detection: BookDetection
    representative_geometry: BookGeometry
    representative_frame_index: int
    representative_stamp_sec: float

    def as_dict(self) -> Dict[str, object]:
        return {
            'colour': self.colour,
            'confirming_frames': int(self.confirming_frames),
            'first_stamp_sec': round(float(self.first_stamp_sec), 9),
            'last_stamp_sec': round(float(self.last_stamp_sec), 9),
            'observation_span_sec': round(
                float(self.last_stamp_sec - self.first_stamp_sec), 9),
            'average_confidence': round(float(self.average_confidence), 6),
            'median_odom_xyz_m': [
                round(float(value), 6) for value in self.median_odom_xyz_m
            ],
            'xy_span_m': round(float(self.xy_span_m), 6),
            'z_span_m': round(float(self.z_span_m), 6),
            'clearance_range_m': [
                round(float(value), 6) for value in self.clearance_range_m
            ],
            'head_tilt_range_rad': [
                round(float(value), 6) for value in self.head_tilt_range_rad
            ],
            'representative_detection': self.representative_detection.as_dict(),
            'representative_geometry': self.representative_geometry.as_dict(),
            'representative_frame_index': int(self.representative_frame_index),
            'representative_stamp_sec': round(
                float(self.representative_stamp_sec), 9),
        }


@dataclass(frozen=True)
class ConfirmedSpatialBookMap:
    """One accepted multi-frame top-to-bottom colour map."""

    first_stamp_sec: float
    confirmed_stamp_sec: float
    row_colours_top_to_bottom: Tuple[str, str, str, str]
    confidence: float
    tracks_by_colour: Mapping[str, ColourTrackSummary]
    row_separations_m: Tuple[float, float, float]
    mapping_source: str

    @property
    def confirming_frames(self) -> int:
        return min(
            track.confirming_frames for track in self.tracks_by_colour.values()
        )

    @property
    def total_observations(self) -> int:
        return sum(
            track.confirming_frames for track in self.tracks_by_colour.values()
        )

    @property
    def detections_by_colour(self) -> Mapping[str, BookDetection]:
        return {
            colour: track.representative_detection
            for colour, track in self.tracks_by_colour.items()
        }

    def row_for_colour(self, colour: str) -> int:
        name = str(colour).strip().lower()
        return self.row_colours_top_to_bottom.index(name) + 1

    def locked_odom_xyz_for_colour(
        self, colour: str,
    ) -> Tuple[float, float, float]:
        return self.tracks_by_colour[str(colour).strip().lower()].median_odom_xyz_m

    def as_dict(self) -> Dict[str, object]:
        return {
            'first_stamp_sec': round(float(self.first_stamp_sec), 9),
            'confirmed_stamp_sec': round(float(self.confirmed_stamp_sec), 9),
            'confirming_frames': int(self.confirming_frames),
            'total_observations': int(self.total_observations),
            'confidence': round(float(self.confidence), 6),
            'row_colours_top_to_bottom': list(self.row_colours_top_to_bottom),
            'row_separations_m': [
                round(float(value), 6) for value in self.row_separations_m
            ],
            'mapping_source': self.mapping_source,
            'single_frame_required': False,
            'tracks_by_colour': {
                colour: track.as_dict()
                for colour, track in self.tracks_by_colour.items()
            },
            # Compatibility field used by the evidence renderer and prior
            # validators.  These detections are representatives from each
            # spatial track and are not asserted to come from one image.
            'detections_by_colour': {
                colour: detection.as_dict()
                for colour, detection in self.detections_by_colour.items()
            },
        }


class SpatialBookMapAccumulator:
    """Accumulate independent colour observations and confirm one row map."""

    def __init__(
        self,
        settings: Optional[SpatialBookMapSettings] = None,
    ) -> None:
        self.settings = settings or SpatialBookMapSettings()
        self.settings.validate()
        self._observations: Dict[str, List[SpatialBookObservation]] = {
            colour: [] for colour in BOOK_COLOURS
        }
        self._confirmed: Optional[ConfirmedSpatialBookMap] = None
        self.rejected_observations = 0
        self.last_rejection_reason = ''

    def reset(self) -> None:
        for values in self._observations.values():
            values.clear()
        self._confirmed = None
        self.rejected_observations = 0
        self.last_rejection_reason = ''

    @property
    def confirmed(self) -> Optional[ConfirmedSpatialBookMap]:
        return self._confirmed

    @property
    def observations(self) -> Mapping[str, Sequence[SpatialBookObservation]]:
        return self._observations

    def add(
        self,
        observation: SpatialBookObservation,
        mapping_source: str,
    ) -> Optional[ConfirmedSpatialBookMap]:
        colour = str(observation.colour).strip().lower()
        if colour not in self._observations:
            self.rejected_observations += 1
            self.last_rejection_reason = f'unsupported colour {colour}'
            return self._confirmed
        xyz = np.asarray(observation.odom_xyz_m, dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            self.rejected_observations += 1
            self.last_rejection_reason = 'non-finite odometry-frame point'
            return self._confirmed
        if not (
            self.settings.minimum_book_height_m
            <= float(xyz[2])
            <= self.settings.maximum_book_height_m
        ):
            self.rejected_observations += 1
            self.last_rejection_reason = 'book height outside plausible range'
            return self._confirmed

        values = self._observations[colour]
        # One detection per colour and RGB frame is enough; replacing a weaker
        # duplicate prevents camera callback duplication from inflating support.
        for index, existing in enumerate(values):
            if existing.frame_index == observation.frame_index:
                if observation.detection.confidence > existing.detection.confidence:
                    values[index] = observation
                return self._try_confirm(mapping_source)
        values.append(observation)
        if len(values) > self.settings.maximum_observations_per_colour:
            del values[:-self.settings.maximum_observations_per_colour]
        return self._try_confirm(mapping_source)

    def _compatible(
        self,
        first: SpatialBookObservation,
        second: SpatialBookObservation,
    ) -> bool:
        a = np.asarray(first.odom_xyz_m, dtype=np.float64)
        b = np.asarray(second.odom_xyz_m, dtype=np.float64)
        return (
            float(np.linalg.norm(a[:2] - b[:2]))
            <= self.settings.maximum_cluster_xy_radius_m
            and abs(float(a[2] - b[2]))
            <= self.settings.maximum_cluster_z_radius_m
        )

    def _dominant_cluster(
        self,
        observations: Sequence[SpatialBookObservation],
    ) -> List[SpatialBookObservation]:
        if not observations:
            return []
        best: List[SpatialBookObservation] = []
        best_confidence = -math.inf
        for seed in observations:
            cluster = [
                item for item in observations
                if self._compatible(seed, item)
            ]
            unique: Dict[int, SpatialBookObservation] = {}
            for item in cluster:
                current = unique.get(item.frame_index)
                if (
                    current is None
                    or item.detection.confidence > current.detection.confidence
                ):
                    unique[item.frame_index] = item
            cluster = list(unique.values())
            confidence = float(np.mean([
                item.detection.confidence for item in cluster
            ])) if cluster else -math.inf
            if (
                len(cluster) > len(best)
                or (
                    len(cluster) == len(best)
                    and confidence > best_confidence
                )
            ):
                best = cluster
                best_confidence = confidence
        best.sort(key=lambda item: item.stamp_sec)
        return best

    def _summary_for_colour(
        self,
        colour: str,
    ) -> Optional[ColourTrackSummary]:
        cluster = self._dominant_cluster(self._observations[colour])
        settings = self.settings
        if len(cluster) < settings.required_frames_per_colour:
            return None
        stamps = np.asarray([item.stamp_sec for item in cluster], dtype=np.float64)
        if float(stamps.max() - stamps.min()) < settings.minimum_observation_span_sec:
            return None
        confidences = np.asarray([
            item.detection.confidence for item in cluster
        ], dtype=np.float64)
        average_confidence = float(confidences.mean())
        if average_confidence < settings.minimum_average_confidence:
            return None
        points = np.asarray([item.odom_xyz_m for item in cluster], dtype=np.float64)
        median = np.median(points, axis=0)
        xy_span = float(np.max(np.linalg.norm(points[:, :2] - median[:2], axis=1)))
        z_span = float(np.max(np.abs(points[:, 2] - median[2])))
        if xy_span > settings.maximum_track_xy_span_m:
            return None
        if z_span > settings.maximum_track_z_span_m:
            return None
        representative = max(
            cluster,
            key=lambda item: (
                item.detection.confidence,
                -abs(float(item.geometry.height_m - median[2])),
            ),
        )
        clearances = [item.lidar_clearance_m for item in cluster]
        head_tilts = [item.head_tilt_rad for item in cluster]
        return ColourTrackSummary(
            colour=colour,
            confirming_frames=len({item.frame_index for item in cluster}),
            first_stamp_sec=float(stamps.min()),
            last_stamp_sec=float(stamps.max()),
            average_confidence=average_confidence,
            median_odom_xyz_m=tuple(float(value) for value in median),
            xy_span_m=xy_span,
            z_span_m=z_span,
            clearance_range_m=(float(min(clearances)), float(max(clearances))),
            head_tilt_range_rad=(float(min(head_tilts)), float(max(head_tilts))),
            representative_detection=representative.detection,
            representative_geometry=representative.geometry,
            representative_frame_index=representative.frame_index,
            representative_stamp_sec=representative.stamp_sec,
        )

    def _try_confirm(
        self,
        mapping_source: str,
    ) -> Optional[ConfirmedSpatialBookMap]:
        if self._confirmed is not None:
            return self._confirmed
        tracks: Dict[str, ColourTrackSummary] = {}
        for colour in BOOK_COLOURS:
            summary = self._summary_for_colour(colour)
            if summary is None:
                return None
            tracks[colour] = summary
        ordered = sorted(
            tracks.values(),
            key=lambda item: item.median_odom_xyz_m[2],
            reverse=True,
        )
        row_colours = tuple(item.colour for item in ordered)
        if sorted(row_colours) != sorted(BOOK_COLOURS):
            return None
        separations = tuple(
            float(
                ordered[index].median_odom_xyz_m[2]
                - ordered[index + 1].median_odom_xyz_m[2]
            )
            for index in range(3)
        )
        if any(
            value < self.settings.minimum_row_separation_m
            or value > self.settings.maximum_row_separation_m
            for value in separations
        ):
            self.last_rejection_reason = (
                'four colour tracks exist but vertical row spacing is implausible'
            )
            return None
        first_stamp = min(track.first_stamp_sec for track in tracks.values())
        confirmed_stamp = max(track.last_stamp_sec for track in tracks.values())
        confidence = float(np.mean([
            track.average_confidence for track in tracks.values()
        ]))
        self._confirmed = ConfirmedSpatialBookMap(
            first_stamp_sec=first_stamp,
            confirmed_stamp_sec=confirmed_stamp,
            row_colours_top_to_bottom=row_colours,  # type: ignore[arg-type]
            confidence=confidence,
            tracks_by_colour=tracks,
            row_separations_m=separations,  # type: ignore[arg-type]
            mapping_source=str(mapping_source),
        )
        return self._confirmed

    def support_summary(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for colour in BOOK_COLOURS:
            cluster = self._dominant_cluster(self._observations[colour])
            summary = self._summary_for_colour(colour)
            result[colour] = {
                'raw_observations': len(self._observations[colour]),
                'dominant_cluster_frames': len({item.frame_index for item in cluster}),
                'track_confirmed': summary is not None,
                'track': summary.as_dict() if summary is not None else None,
            }
        return result


@dataclass(frozen=True)
class TargetReacquisitionSettings:
    required_frames: int = 3
    window_sec: float = 0.90
    minimum_span_sec: float = 0.05
    maximum_locked_xy_error_m: float = 0.18
    maximum_locked_z_error_m: float = 0.12
    maximum_pixel_jump_px: float = 50.0
    minimum_average_confidence: float = 0.60

    def validate(self) -> None:
        if self.required_frames < 2:
            raise ValueError('required_frames must be at least 2')
        if self.window_sec <= 0.0:
            raise ValueError('window_sec must be positive')
        if not 0.0 <= self.minimum_span_sec <= self.window_sec:
            raise ValueError('minimum_span_sec must be inside the window')
        if self.maximum_locked_xy_error_m <= 0.0:
            raise ValueError('maximum_locked_xy_error_m must be positive')
        if self.maximum_locked_z_error_m <= 0.0:
            raise ValueError('maximum_locked_z_error_m must be positive')
        if self.maximum_pixel_jump_px <= 0.0:
            raise ValueError('maximum_pixel_jump_px must be positive')
        if not 0.0 < self.minimum_average_confidence <= 1.0:
            raise ValueError('minimum_average_confidence must be in (0, 1]')


@dataclass(frozen=True)
class ReacquiredTargetBook:
    colour: str
    row: int
    first_stamp_sec: float
    confirmed_stamp_sec: float
    confirming_frames: int
    average_confidence: float
    locked_odom_xyz_m: Tuple[float, float, float]
    median_observed_odom_xyz_m: Tuple[float, float, float]
    representative_detection: BookDetection
    representative_geometry: BookGeometry
    representative_frame_index: int
    representative_stamp_sec: float

    def as_dict(self) -> Dict[str, object]:
        return {
            'colour': self.colour,
            'row': int(self.row),
            'first_stamp_sec': round(float(self.first_stamp_sec), 9),
            'confirmed_stamp_sec': round(float(self.confirmed_stamp_sec), 9),
            'confirming_frames': int(self.confirming_frames),
            'average_confidence': round(float(self.average_confidence), 6),
            'locked_odom_xyz_m': [
                round(float(value), 6) for value in self.locked_odom_xyz_m
            ],
            'median_observed_odom_xyz_m': [
                round(float(value), 6)
                for value in self.median_observed_odom_xyz_m
            ],
            'representative_detection': self.representative_detection.as_dict(),
            'representative_geometry': self.representative_geometry.as_dict(),
            'representative_frame_index': int(self.representative_frame_index),
            'representative_stamp_sec': round(
                float(self.representative_stamp_sec), 9),
        }


class TargetBookReacquisitionFilter:
    """Confirm only the requested colour near its locked odometry position."""

    def __init__(
        self,
        colour: str,
        row: int,
        locked_odom_xyz_m: Sequence[float],
        settings: Optional[TargetReacquisitionSettings] = None,
    ) -> None:
        self.colour = str(colour).strip().lower()
        self.row = int(row)
        self.locked_odom_xyz_m = tuple(float(value) for value in locked_odom_xyz_m)
        if len(self.locked_odom_xyz_m) != 3:
            raise ValueError('locked target position must contain three values')
        self.settings = settings or TargetReacquisitionSettings()
        self.settings.validate()
        self._history: List[SpatialBookObservation] = []
        self._confirmed: Optional[ReacquiredTargetBook] = None
        self.rejection_count = 0
        self.last_rejection_reason = ''

    @property
    def confirmed(self) -> Optional[ReacquiredTargetBook]:
        return self._confirmed

    def reset(self) -> None:
        self._history.clear()
        self._confirmed = None
        self.rejection_count = 0
        self.last_rejection_reason = ''

    def update(
        self,
        observation: SpatialBookObservation,
        current_stamp_sec: float,
    ) -> Optional[ReacquiredTargetBook]:
        if self._confirmed is not None:
            return self._confirmed
        if observation.colour != self.colour:
            self.rejection_count += 1
            self.last_rejection_reason = 'wrong colour'
            return None
        locked = np.asarray(self.locked_odom_xyz_m, dtype=np.float64)
        observed = np.asarray(observation.odom_xyz_m, dtype=np.float64)
        xy_error = float(np.linalg.norm(observed[:2] - locked[:2]))
        z_error = abs(float(observed[2] - locked[2]))
        if xy_error > self.settings.maximum_locked_xy_error_m:
            self.rejection_count += 1
            self.last_rejection_reason = (
                f'locked XY error {xy_error:.3f} m exceeds threshold'
            )
            return None
        if z_error > self.settings.maximum_locked_z_error_m:
            self.rejection_count += 1
            self.last_rejection_reason = (
                f'locked Z error {z_error:.3f} m exceeds threshold'
            )
            return None
        while (
            self._history
            and current_stamp_sec - self._history[0].stamp_sec
            > self.settings.window_sec
        ):
            del self._history[0]
        if self._history:
            previous = self._history[-1]
            jump = math.hypot(
                observation.detection.center_x - previous.detection.center_x,
                observation.detection.center_y - previous.detection.center_y,
            )
            if jump > self.settings.maximum_pixel_jump_px:
                self._history.clear()
        self._history = [
            item for item in self._history
            if item.frame_index != observation.frame_index
        ]
        self._history.append(observation)
        if len({item.frame_index for item in self._history}) < self.settings.required_frames:
            return None
        span = self._history[-1].stamp_sec - self._history[0].stamp_sec
        if span < self.settings.minimum_span_sec:
            return None
        confidence = float(np.mean([
            item.detection.confidence for item in self._history
        ]))
        if confidence < self.settings.minimum_average_confidence:
            return None
        points = np.asarray([item.odom_xyz_m for item in self._history])
        median = np.median(points, axis=0)
        representative = max(
            self._history,
            key=lambda item: item.detection.confidence,
        )
        self._confirmed = ReacquiredTargetBook(
            colour=self.colour,
            row=self.row,
            first_stamp_sec=self._history[0].stamp_sec,
            confirmed_stamp_sec=self._history[-1].stamp_sec,
            confirming_frames=len({item.frame_index for item in self._history}),
            average_confidence=confidence,
            locked_odom_xyz_m=self.locked_odom_xyz_m,
            median_observed_odom_xyz_m=tuple(float(value) for value in median),
            representative_detection=representative.detection,
            representative_geometry=representative.geometry,
            representative_frame_index=representative.frame_index,
            representative_stamp_sec=representative.stamp_sec,
        )
        return self._confirmed
