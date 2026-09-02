#!/usr/bin/env python3
"""Deterministic shelf-number perception for KU SPARCy ERC 2026.

The official simulator renders each shelf marker from a known black-on-white
PNG texture.  This module uses those official textures only as visual
references; it never reads the randomized marker order or any hidden Gazebo
state.  Detection is performed exclusively on live camera pixels.

The pipeline is intentionally lightweight and CPU-only:

1. Search a configurable upper-image region for dark connected components.
2. Reject components that do not have a bright marker-plate neighbourhood.
3. Normalize each glyph and compare it with augmented official templates using
   HOG, normalized pixels, and contour shape.
4. Apply confidence and best-vs-second-best margin thresholds.
5. Confirm a digit only after spatially consistent observations in several
   distinct frames.

The detector contains no ROS imports, which makes it independently testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import math
import os
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned pixel bounding box."""

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

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_list(self) -> List[int]:
        return [self.x, self.y, self.width, self.height]

    def clipped(self, frame_width: int, frame_height: int) -> "BoundingBox":
        x1 = min(max(0, self.x), frame_width)
        y1 = min(max(0, self.y), frame_height)
        x2 = min(max(x1, self.x2), frame_width)
        y2 = min(max(y1, self.y2), frame_height)
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)

    def iou(self, other: "BoundingBox") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection
        return float(intersection) / float(union) if union > 0 else 0.0

    def center_distance(self, other: "BoundingBox") -> float:
        return math.hypot(
            self.center_x - other.center_x,
            self.center_y - other.center_y,
        )


@dataclass(frozen=True)
class MarkerDetection:
    """One accepted single-frame marker classification."""

    digit: int
    confidence: float
    second_best_digit: int
    second_best_confidence: float
    margin: float
    bbox: BoundingBox
    bright_ring_fraction: float
    neutral_dark_fraction: float
    classifier_scores: Mapping[int, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            'digit': self.digit,
            'confidence': round(float(self.confidence), 6),
            'second_best_digit': self.second_best_digit,
            'second_best_confidence': round(
                float(self.second_best_confidence), 6),
            'margin': round(float(self.margin), 6),
            'bbox_xywh': self.bbox.as_list(),
            'bright_ring_fraction': round(
                float(self.bright_ring_fraction), 6),
            'neutral_dark_fraction': round(
                float(self.neutral_dark_fraction), 6),
            'classifier_scores': {
                str(key): round(float(value), 6)
                for key, value in self.classifier_scores.items()
            },
        }


@dataclass(frozen=True)
class ConfirmedMarker:
    """Temporally confirmed marker observation."""

    digit: int
    confidence: float
    confirming_frames: int
    first_stamp_sec: float
    confirmed_stamp_sec: float
    last_seen_stamp_sec: float
    bbox: BoundingBox

    def as_dict(self) -> Dict[str, object]:
        return {
            'digit': self.digit,
            'confidence': round(float(self.confidence), 6),
            'confirming_frames': int(self.confirming_frames),
            'first_stamp_sec': round(float(self.first_stamp_sec), 9),
            'confirmed_stamp_sec': round(float(self.confirmed_stamp_sec), 9),
            'last_seen_stamp_sec': round(float(self.last_seen_stamp_sec), 9),
            'bbox_xywh': self.bbox.as_list(),
        }


@dataclass(frozen=True)
class DetectorSettings:
    """Single-frame detector thresholds."""

    roi_top_fraction: float = 0.04
    roi_bottom_fraction: float = 0.68
    dark_threshold: int = 115
    min_digit_height_px: int = 7
    max_digit_height_fraction: float = 0.23
    min_digit_width_px: int = 2
    max_digit_width_fraction: float = 0.16
    min_aspect_ratio: float = 0.06
    max_aspect_ratio: float = 1.20
    min_contour_area_px: float = 5.0
    min_fill_ratio: float = 0.055
    max_fill_ratio: float = 0.90
    bright_pixel_threshold: int = 155
    min_bright_ring_fraction: float = 0.48
    min_neutral_dark_fraction: float = 0.65
    min_classifier_confidence: float = 0.58
    min_classifier_margin: float = 0.025
    max_accepted_detections: int = 12

    def validate(self) -> None:
        if not 0.0 <= self.roi_top_fraction < self.roi_bottom_fraction <= 1.0:
            raise ValueError('ROI fractions must satisfy 0 <= top < bottom <= 1')
        if not 0 <= self.dark_threshold <= 255:
            raise ValueError('dark_threshold must be between 0 and 255')
        if self.min_digit_height_px < 2 or self.min_digit_width_px < 1:
            raise ValueError('minimum digit dimensions are too small')
        if not 0.0 <= self.min_neutral_dark_fraction <= 1.0:
            raise ValueError('min_neutral_dark_fraction must be in [0, 1]')
        if self.min_classifier_confidence <= 0.0:
            raise ValueError('min_classifier_confidence must be positive')
        if self.min_classifier_margin < 0.0:
            raise ValueError('min_classifier_margin cannot be negative')


@dataclass(frozen=True)
class TemporalSettings:
    """Temporal confirmation thresholds."""

    required_frames: int = 3
    window_sec: float = 0.75
    min_span_sec: float = 0.04
    min_average_confidence: float = 0.62
    maximum_center_jump_px: float = 55.0
    center_jump_height_multiplier: float = 3.5
    fresh_track_timeout_sec: float = 1.00

    def validate(self) -> None:
        if self.required_frames < 2:
            raise ValueError('required_frames must be at least 2')
        if self.window_sec <= 0.0:
            raise ValueError('window_sec must be positive')
        if self.min_span_sec < 0.0 or self.min_span_sec > self.window_sec:
            raise ValueError('min_span_sec must be in [0, window_sec]')
        if not 0.0 < self.min_average_confidence <= 1.0:
            raise ValueError('min_average_confidence must be in (0, 1]')


@dataclass(frozen=True)
class _TemplateVariant:
    hog: np.ndarray
    pixels: np.ndarray
    contour: np.ndarray


class DigitTemplateLibrary:
    """Classify glyph masks against augmented official marker textures."""

    DIGITS: Tuple[int, ...] = (1, 2, 3, 4, 5)
    CANVAS_WIDTH = 48
    CANVAS_HEIGHT = 72
    CANVAS_MARGIN = 5

    def __init__(self, template_directory: str) -> None:
        if not os.path.isdir(template_directory):
            raise FileNotFoundError(
                f'Official marker template directory not found: '
                f'{template_directory}')
        self.template_directory = os.path.abspath(template_directory)
        self._hog = cv2.HOGDescriptor(
            (self.CANVAS_WIDTH, self.CANVAS_HEIGHT),
            (16, 16),
            (8, 8),
            (8, 8),
            9,
        )
        self._variants: Dict[int, List[_TemplateVariant]] = {}
        self._hog_matrices: Dict[int, np.ndarray] = {}
        self._pixel_matrices: Dict[int, np.ndarray] = {}
        self._base_contours: Dict[int, np.ndarray] = {}
        self.template_paths: Dict[int, str] = {}
        self._load_templates()

    @staticmethod
    def _minority_binary(gray: np.ndarray) -> np.ndarray:
        """Return the less-common side of an Otsu split as foreground."""
        if gray.ndim != 2:
            raise ValueError('Expected a single-channel grayscale image')
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, normal = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverse = cv2.bitwise_not(normal)

        normal_fraction = float(np.count_nonzero(normal)) / float(normal.size)
        inverse_fraction = float(np.count_nonzero(inverse)) / float(inverse.size)
        mask = normal if normal_fraction < inverse_fraction else inverse

        # Remove tiny isolated compression/anti-alias specks while retaining a
        # thin digit "1".
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        if count <= 1:
            return mask
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        minimum = max(2, int(round(largest * 0.015)))
        cleaned = np.zeros_like(mask)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= minimum:
                cleaned[labels == label] = 255
        return cleaned

    @classmethod
    def canonicalize(cls, mask: np.ndarray) -> np.ndarray:
        """Place a tight glyph mask on a fixed canvas without distortion."""
        if mask.ndim != 2:
            raise ValueError('Glyph mask must be single-channel')
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        points = cv2.findNonZero(binary)
        if points is None:
            raise ValueError('Glyph mask is empty')
        x, y, width, height = cv2.boundingRect(points)
        if width <= 0 or height <= 0:
            raise ValueError('Glyph mask has an invalid bounding rectangle')
        glyph = binary[y:y + height, x:x + width]

        usable_width = cls.CANVAS_WIDTH - 2 * cls.CANVAS_MARGIN
        usable_height = cls.CANVAS_HEIGHT - 2 * cls.CANVAS_MARGIN
        scale = min(usable_width / width, usable_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        resized = cv2.resize(
            glyph, (resized_width, resized_height),
            interpolation=interpolation)
        resized = np.where(resized >= 127, 255, 0).astype(np.uint8)

        canvas = np.zeros(
            (cls.CANVAS_HEIGHT, cls.CANVAS_WIDTH), dtype=np.uint8)
        x_offset = (cls.CANVAS_WIDTH - resized_width) // 2
        y_offset = (cls.CANVAS_HEIGHT - resized_height) // 2
        canvas[
            y_offset:y_offset + resized_height,
            x_offset:x_offset + resized_width,
        ] = resized
        return canvas

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError('No contour in glyph mask')
        return max(contours, key=cv2.contourArea)

    def _descriptor(self, canvas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hog = self._hog.compute(canvas).reshape(-1).astype(np.float32)
        hog_norm = float(np.linalg.norm(hog))
        if hog_norm > 1e-12:
            hog /= hog_norm

        soft = cv2.GaussianBlur(
            canvas.astype(np.float32) / 255.0, (5, 5), 0).reshape(-1)
        soft -= float(soft.mean())
        pixel_norm = float(np.linalg.norm(soft))
        if pixel_norm > 1e-12:
            soft /= pixel_norm
        return hog, soft.astype(np.float32)

    @classmethod
    def _warp_variant(
        cls,
        base: np.ndarray,
        angle_deg: float,
        horizontal_scale: float,
        shear: float,
    ) -> np.ndarray:
        center_x = (cls.CANVAS_WIDTH - 1) / 2.0
        center_y = (cls.CANVAS_HEIGHT - 1) / 2.0

        rotation = cv2.getRotationMatrix2D(
            (center_x, center_y), angle_deg, 1.0)
        rotated = cv2.warpAffine(
            base,
            rotation,
            (cls.CANVAS_WIDTH, cls.CANVAS_HEIGHT),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        affine = np.array(
            [
                [horizontal_scale, shear,
                 center_x * (1.0 - horizontal_scale) - shear * center_y],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            rotated,
            affine,
            (cls.CANVAS_WIDTH, cls.CANVAS_HEIGHT),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.where(warped > 0, 255, 0).astype(np.uint8)

    def _make_variants(self, base: np.ndarray) -> List[_TemplateVariant]:
        variants: List[_TemplateVariant] = []
        unique_hashes = set()
        for angle in (-12.0, -6.0, 0.0, 6.0, 12.0):
            for horizontal_scale in (0.82, 1.00, 1.18):
                for shear in (-0.16, 0.0, 0.16):
                    warped = self._warp_variant(
                        base, angle, horizontal_scale, shear)
                    for morphology in (-1, 0, 1):
                        if morphology < 0:
                            variant = cv2.erode(
                                warped, np.ones((2, 2), np.uint8), iterations=1)
                        elif morphology > 0:
                            variant = cv2.dilate(
                                warped, np.ones((2, 2), np.uint8), iterations=1)
                        else:
                            variant = warped
                        if np.count_nonzero(variant) < 3:
                            continue
                        digest = hash(variant.tobytes())
                        if digest in unique_hashes:
                            continue
                        unique_hashes.add(digest)
                        hog, pixels = self._descriptor(variant)
                        contour = self._largest_contour(variant)
                        variants.append(_TemplateVariant(hog, pixels, contour))
        if not variants:
            raise RuntimeError('No template variants were generated')
        return variants

    def _load_templates(self) -> None:
        for digit in self.DIGITS:
            path = os.path.join(self.template_directory, f'{digit}.png')
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(
                    f'Could not load official marker template: {path}')
            if image.ndim == 3 and image.shape[2] == 4:
                # Composite transparent pixels over white before thresholding.
                bgr = image[:, :, :3].astype(np.float32)
                alpha = image[:, :, 3:4].astype(np.float32) / 255.0
                composited = bgr * alpha + 255.0 * (1.0 - alpha)
                gray = cv2.cvtColor(
                    composited.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            elif image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            mask = self._minority_binary(gray)
            canonical = self.canonicalize(mask)
            variants = self._make_variants(canonical)
            self._variants[digit] = variants
            self._hog_matrices[digit] = np.vstack(
                [variant.hog for variant in variants])
            self._pixel_matrices[digit] = np.vstack(
                [variant.pixels for variant in variants])
            self._base_contours[digit] = self._largest_contour(canonical)
            self.template_paths[digit] = path

    def classify(
        self, glyph_mask: np.ndarray,
    ) -> Tuple[int, float, int, float, float, Dict[int, float]]:
        """Classify a binary glyph mask.

        Returns ``best_digit, best_score, second_digit, second_score, margin,
        all_scores``. Scores are normalized to approximately [0, 1].
        """
        canonical = self.canonicalize(glyph_mask)
        candidate_hog, candidate_pixels = self._descriptor(canonical)
        candidate_contour = self._largest_contour(canonical)

        scores: Dict[int, float] = {}
        for digit in self.DIGITS:
            hog_similarity = float(
                np.max(self._hog_matrices[digit] @ candidate_hog))
            pixel_correlation = float(
                np.max(self._pixel_matrices[digit] @ candidate_pixels))
            pixel_similarity = max(0.0, min(1.0, 0.5 * (pixel_correlation + 1.0)))

            shape_distance = float(cv2.matchShapes(
                candidate_contour,
                self._base_contours[digit],
                cv2.CONTOURS_MATCH_I1,
                0.0,
            ))
            shape_similarity = math.exp(-2.5 * max(0.0, shape_distance))

            combined = (
                0.58 * max(0.0, min(1.0, hog_similarity))
                + 0.27 * pixel_similarity
                + 0.15 * shape_similarity
            )
            scores[digit] = max(0.0, min(1.0, combined))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_digit, best_score = ranked[0]
        second_digit, second_score = ranked[1]
        return (
            best_digit,
            best_score,
            second_digit,
            second_score,
            best_score - second_score,
            scores,
        )


class ShelfMarkerDetector:
    """Find and classify marker digits in one BGR camera frame."""

    def __init__(
        self,
        template_directory: str,
        settings: Optional[DetectorSettings] = None,
    ) -> None:
        self.settings = settings or DetectorSettings()
        self.settings.validate()
        self.templates = DigitTemplateLibrary(template_directory)

    @staticmethod
    def _bright_ring_fraction(
        gray: np.ndarray,
        bbox: BoundingBox,
        threshold: int,
    ) -> float:
        frame_height, frame_width = gray.shape[:2]
        horizontal_pad = max(7, int(round(0.90 * bbox.height)))
        vertical_pad = max(5, int(round(0.45 * bbox.height)))
        outer = BoundingBox(
            bbox.x - horizontal_pad,
            bbox.y - vertical_pad,
            bbox.width + 2 * horizontal_pad,
            bbox.height + 2 * vertical_pad,
        ).clipped(frame_width, frame_height)
        if outer.area <= bbox.area:
            return 0.0

        patch = gray[outer.y:outer.y2, outer.x:outer.x2]
        if patch.size == 0:
            return 0.0
        ring = np.ones(patch.shape, dtype=bool)
        inner_x1 = max(0, bbox.x - outer.x - 1)
        inner_y1 = max(0, bbox.y - outer.y - 1)
        inner_x2 = min(patch.shape[1], bbox.x2 - outer.x + 1)
        inner_y2 = min(patch.shape[0], bbox.y2 - outer.y + 1)
        ring[inner_y1:inner_y2, inner_x1:inner_x2] = False
        ring_values = patch[ring]
        if ring_values.size == 0:
            return 0.0
        return float(np.count_nonzero(ring_values >= threshold)) / float(
            ring_values.size)

    @staticmethod
    def _nms(detections: Sequence[MarkerDetection]) -> List[MarkerDetection]:
        accepted: List[MarkerDetection] = []
        for detection in sorted(
            detections, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for prior in accepted:
                if detection.bbox.iou(prior.bbox) > 0.35:
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(detection)
        return accepted

    def _reconcile_complete_marker_set(
        self,
        detections: Sequence[MarkerDetection],
    ) -> Tuple[List[MarkerDetection], Optional[Dict[str, object]]]:
        """Resolve one weak duplicate when all five shelf markers are visible.

        The ERC shelf contains exactly one each of digits 1..5. Normal
        single-marker classification remains untouched during partial views.
        Reconciliation is used only when:

        - exactly five spatially distinct candidates are present,
        - the independent labels are not already a unique 1..5 set,
        - the best global one-to-one assignment changes exactly one label,
        - that candidate was locally ambiguous,
        - the assigned class was already close to its local best score, and
        - the best global assignment is meaningfully better than the runner-up.

        No seed, Gazebo entity name, randomized ordering, or world state is
        consulted.
        """
        if len(detections) != 5:
            return list(detections), None

        digits = (1, 2, 3, 4, 5)
        ordered = sorted(
            detections,
            key=lambda detection: detection.bbox.center_x,
        )

        independent = tuple(detection.digit for detection in ordered)

        # Nothing to repair if the independent classifier already produced
        # exactly one of each official digit.
        if tuple(sorted(independent)) == digits:
            return list(detections), None

        assignments = []

        for permutation in itertools.permutations(digits):
            log_score = 0.0

            for detection, digit in zip(ordered, permutation):
                score = float(detection.classifier_scores[digit])
                log_score += math.log(max(score, 1.0e-9))

            assignments.append((log_score, permutation))

        assignments.sort(key=lambda item: item[0], reverse=True)

        if len(assignments) < 2:
            return list(detections), None

        best_log_score, best_assignment = assignments[0]
        second_log_score, second_assignment = assignments[1]
        assignment_margin = best_log_score - second_log_score

        changed = [
            index
            for index, (before, after) in enumerate(
                zip(independent, best_assignment)
            )
            if before != after
        ]

        # Be deliberately conservative. We are fixing one weak duplicate,
        # not replacing the normal classifier with a global permutation solver.
        if len(changed) != 1:
            return list(detections), None

        changed_index = changed[0]
        detection = ordered[changed_index]
        assigned_digit = int(best_assignment[changed_index])

        local_best_score = float(
            detection.classifier_scores[detection.digit]
        )
        assigned_score = float(
            detection.classifier_scores[assigned_digit]
        )

        # Guards derived from classifier behavior, not from any seeded layout:
        #
        # 1. global assignment must be clearly preferable;
        # 2. local decision must itself have been weak;
        # 3. replacement class must already have been close to the local best.
        if assignment_margin < 0.12:
            return list(detections), None

        if detection.margin > 0.05:
            return list(detections), None

        if (local_best_score - assigned_score) > 0.10:
            return list(detections), None

        # Convert the assigned raw class score to the same local confidence
        # scale used by the normal classifier.
        local_assigned_confidence = max(
            0.0,
            min(
                1.0,
                0.92 * assigned_score
                + 0.08 * detection.bright_ring_fraction,
            ),
        )

        # Four unchanged marker decisions provide contextual support for the
        # one weak candidate. Blend that support with the candidate's own
        # assigned-class score; this remains a confidence metric, not a claim
        # of statistical probability.
        unchanged_confidences = [
            float(item.confidence)
            for index, item in enumerate(ordered)
            if index != changed_index
            and item.digit == best_assignment[index]
        ]

        if len(unchanged_confidences) != 4:
            return list(detections), None

        context_confidence = float(np.mean(unchanged_confidences))

        reconciled_confidence = 0.5 * (
            local_assigned_confidence + context_confidence
        )

        alternatives = sorted(
            (
                (float(score), int(digit))
                for digit, score in detection.classifier_scores.items()
                if int(digit) != assigned_digit
            ),
            reverse=True,
        )

        if not alternatives:
            return list(detections), None

        second_score, second_digit = alternatives[0]

        second_confidence = max(
            0.0,
            min(
                1.0,
                0.92 * second_score
                + 0.08 * detection.bright_ring_fraction,
            ),
        )

        reconciled_margin = (
            reconciled_confidence - second_confidence
        )

        # The contextual decision still has to satisfy the same downstream
        # acceptance gates as an ordinary detection.
        if (
            reconciled_confidence
            < self.settings.min_classifier_confidence
            or reconciled_margin
            < self.settings.min_classifier_margin
        ):
            return list(detections), None

        corrected = MarkerDetection(
            digit=assigned_digit,
            confidence=float(reconciled_confidence),
            second_best_digit=int(second_digit),
            second_best_confidence=float(second_confidence),
            margin=float(reconciled_margin),
            bbox=detection.bbox,
            bright_ring_fraction=detection.bright_ring_fraction,
            neutral_dark_fraction=detection.neutral_dark_fraction,
            classifier_scores=detection.classifier_scores,
        )

        replacements = {
            id(detection): corrected,
        }

        reconciled = [
            replacements.get(id(item), item)
            for item in detections
        ]

        diagnostics = {
            'applied': True,
            'independent_left_to_right': list(independent),
            'reconciled_left_to_right': list(best_assignment),
            'changed_index_left_to_right': int(changed_index),
            'changed_from_digit': int(detection.digit),
            'changed_to_digit': int(assigned_digit),
            'local_best_score': round(local_best_score, 6),
            'assigned_score': round(assigned_score, 6),
            'local_assigned_confidence': round(
                local_assigned_confidence, 6
            ),
            'context_confidence': round(context_confidence, 6),
            'reconciled_confidence': round(
                reconciled_confidence, 6
            ),
            'reconciled_margin': round(reconciled_margin, 6),
            'best_assignment_log_score': round(
                best_log_score, 6
            ),
            'second_assignment_log_score': round(
                second_log_score, 6
            ),
            'assignment_margin': round(
                assignment_margin, 6
            ),
            'second_assignment': list(second_assignment),
        }

        return reconciled, diagnostics

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[List[MarkerDetection], Dict[str, object]]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError('Camera frame is empty')
        colour_frame: Optional[np.ndarray] = None
        if frame_bgr.ndim == 2:
            gray = frame_bgr
        elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
            colour_frame = frame_bgr
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(
                f'Unsupported camera frame shape: {frame_bgr.shape}')

        frame_height, frame_width = gray.shape[:2]
        roi_y1 = int(round(frame_height * self.settings.roi_top_fraction))
        roi_y2 = int(round(frame_height * self.settings.roi_bottom_fraction))
        roi_y1 = min(max(0, roi_y1), frame_height - 1)
        roi_y2 = min(max(roi_y1 + 1, roi_y2), frame_height)
        roi = gray[roi_y1:roi_y2, :]

        _, dark = cv2.threshold(
            roi,
            self.settings.dark_threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            np.ones((2, 2), dtype=np.uint8),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minimum_height = self.settings.min_digit_height_px
        maximum_height = max(
            minimum_height + 1,
            int(round(frame_height * self.settings.max_digit_height_fraction)),
        )
        minimum_width = self.settings.min_digit_width_px
        maximum_width = max(
            minimum_width + 1,
            int(round(frame_width * self.settings.max_digit_width_fraction)),
        )

        classified_candidates: List[MarkerDetection] = []
        diagnostic_candidates: List[Dict[str, object]] = []

        for contour in contours:
            x, y_local, width, height = cv2.boundingRect(contour)
            y = y_local + roi_y1
            bbox = BoundingBox(x, y, width, height)
            if not (
                minimum_width <= width <= maximum_width
                and minimum_height <= height <= maximum_height
            ):
                continue
            aspect = float(width) / float(height)
            if not (
                self.settings.min_aspect_ratio
                <= aspect
                <= self.settings.max_aspect_ratio
            ):
                continue
            contour_area = float(cv2.contourArea(contour))
            if contour_area < self.settings.min_contour_area_px:
                continue
            fill_ratio = contour_area / float(max(1, width * height))
            if not (
                self.settings.min_fill_ratio
                <= fill_ratio
                <= self.settings.max_fill_ratio
            ):
                continue

            bright_fraction = self._bright_ring_fraction(
                gray, bbox, self.settings.bright_pixel_threshold)
            if bright_fraction < self.settings.min_bright_ring_fraction:
                continue

            neutral_fraction = 1.0
            if colour_frame is not None:
                colour_patch = colour_frame[y:y + height, x:x + width]
                gray_patch = gray[y:y + height, x:x + width]
                dark_pixels = gray_patch <= self.settings.dark_threshold
                if not np.any(dark_pixels):
                    continue
                channel_max = np.max(colour_patch, axis=2)
                channel_min = np.min(colour_patch, axis=2)
                neutral_dark = (
                    dark_pixels
                    & ((channel_max - channel_min) <= 45)
                    & (channel_max <= 165)
                )
                neutral_fraction = float(np.count_nonzero(neutral_dark)) / float(
                    np.count_nonzero(dark_pixels))
                if neutral_fraction < self.settings.min_neutral_dark_fraction:
                    continue

            glyph = dark[
                y_local:y_local + height,
                x:x + width,
            ].copy()
            try:
                (
                    best_digit,
                    best_score,
                    second_digit,
                    second_score,
                    margin,
                    all_scores,
                ) = self.templates.classify(glyph)
            except (ValueError, cv2.error):
                continue

            # Context contributes only a small amount; digit shape remains the
            # dominant signal.
            confidence = max(
                0.0,
                min(1.0, 0.92 * best_score + 0.08 * bright_fraction),
            )
            second_confidence = max(
                0.0,
                min(1.0, 0.92 * second_score + 0.08 * bright_fraction),
            )
            adjusted_margin = confidence - second_confidence
            diagnostic_candidates.append({
                'digit': int(best_digit),
                'confidence': round(confidence, 6),
                'margin': round(adjusted_margin, 6),
                'bbox_xywh': bbox.as_list(),
                'bright_ring_fraction': round(bright_fraction, 6),
                'neutral_dark_fraction': round(neutral_fraction, 6),
            })

            classified_candidates.append(MarkerDetection(
                digit=int(best_digit),
                confidence=float(confidence),
                second_best_digit=int(second_digit),
                second_best_confidence=float(second_confidence),
                margin=float(adjusted_margin),
                bbox=bbox,
                bright_ring_fraction=float(bright_fraction),
                neutral_dark_fraction=float(neutral_fraction),
                classifier_scores=all_scores,
            ))

        classified_detections = self._nms(classified_candidates)

        reconciled_detections, unique_set_reconciliation = (
            self._reconcile_complete_marker_set(classified_detections)
        )

        detections = [
            detection
            for detection in reconciled_detections
            if (
                detection.confidence
                >= self.settings.min_classifier_confidence
                and detection.margin
                >= self.settings.min_classifier_margin
            )
        ]

        detections = detections[
            :self.settings.max_accepted_detections
        ]

        diagnostic_candidates.sort(
            key=lambda item: float(item['confidence']), reverse=True)
        diagnostics: Dict[str, object] = {
            'frame_width': frame_width,
            'frame_height': frame_height,
            'roi_y_range': [roi_y1, roi_y2],
            'raw_contour_count': len(contours),
            'classified_candidate_count': len(diagnostic_candidates),
            'accepted_detection_count': len(detections),
            'top_candidates': diagnostic_candidates[:12],
            'unique_set_reconciliation': unique_set_reconciliation,
        }
        return detections, diagnostics


@dataclass(frozen=True)
class _TimedObservation:
    frame_index: int
    stamp_sec: float
    detection: MarkerDetection


class TemporalMarkerFilter:
    """Confirm only repeated, spatially consistent marker observations."""

    def __init__(self, settings: Optional[TemporalSettings] = None) -> None:
        self.settings = settings or TemporalSettings()
        self.settings.validate()
        self._history: Dict[int, Deque[_TimedObservation]] = {
            digit: deque() for digit in DigitTemplateLibrary.DIGITS
        }
        self._confirmed: Dict[int, ConfirmedMarker] = {}

    def _spatially_consistent(
        self,
        previous: MarkerDetection,
        current: MarkerDetection,
    ) -> bool:
        allowed_jump = max(
            self.settings.maximum_center_jump_px,
            self.settings.center_jump_height_multiplier
            * max(previous.bbox.height, current.bbox.height),
        )
        return (
            previous.bbox.iou(current.bbox) >= 0.02
            or previous.bbox.center_distance(current.bbox) <= allowed_jump
        )

    @staticmethod
    def _median_box(observations: Sequence[_TimedObservation]) -> BoundingBox:
        return BoundingBox(
            int(round(float(np.median([
                observation.detection.bbox.x
                for observation in observations
            ])))),
            int(round(float(np.median([
                observation.detection.bbox.y
                for observation in observations
            ])))),
            int(round(float(np.median([
                observation.detection.bbox.width
                for observation in observations
            ])))),
            int(round(float(np.median([
                observation.detection.bbox.height
                for observation in observations
            ])))),
        )

    def update(
        self,
        detections: Iterable[MarkerDetection],
        stamp_sec: float,
        frame_index: int,
    ) -> List[ConfirmedMarker]:
        """Update tracks and return markers newly confirmed on this frame."""
        if not math.isfinite(stamp_sec):
            raise ValueError('stamp_sec must be finite')

        # At most one observation per digit per frame; use the strongest one.
        strongest: Dict[int, MarkerDetection] = {}
        for detection in detections:
            prior = strongest.get(detection.digit)
            if prior is None or detection.confidence > prior.confidence:
                strongest[detection.digit] = detection

        new_confirmations: List[ConfirmedMarker] = []
        for digit in DigitTemplateLibrary.DIGITS:
            history = self._history[digit]
            while history and stamp_sec - history[0].stamp_sec > self.settings.window_sec:
                history.popleft()

            detection = strongest.get(digit)
            if detection is None:
                continue
            if history and not self._spatially_consistent(
                history[-1].detection, detection):
                history.clear()

            history.append(_TimedObservation(
                frame_index=frame_index,
                stamp_sec=stamp_sec,
                detection=detection,
            ))
            while history and stamp_sec - history[0].stamp_sec > self.settings.window_sec:
                history.popleft()

            unique_frames = {observation.frame_index for observation in history}
            if len(unique_frames) < self.settings.required_frames:
                continue
            span = history[-1].stamp_sec - history[0].stamp_sec
            if span < self.settings.min_span_sec:
                continue
            average_confidence = float(np.mean([
                observation.detection.confidence
                for observation in history
            ]))
            if average_confidence < self.settings.min_average_confidence:
                continue

            snapshot = ConfirmedMarker(
                digit=digit,
                confidence=average_confidence,
                confirming_frames=len(unique_frames),
                first_stamp_sec=history[0].stamp_sec,
                confirmed_stamp_sec=history[-1].stamp_sec,
                last_seen_stamp_sec=stamp_sec,
                bbox=self._median_box(list(history)),
            )
            was_confirmed = digit in self._confirmed
            self._confirmed[digit] = snapshot
            if not was_confirmed:
                new_confirmations.append(snapshot)

        # Refresh last-seen and geometry for already-confirmed tracks even when
        # they are not newly confirmed.
        for digit, detection in strongest.items():
            prior = self._confirmed.get(digit)
            if prior is not None:
                history = list(self._history[digit])
                confidence = float(np.mean([
                    observation.detection.confidence
                    for observation in history
                ])) if history else prior.confidence
                self._confirmed[digit] = ConfirmedMarker(
                    digit=digit,
                    confidence=confidence,
                    confirming_frames=max(
                        prior.confirming_frames,
                        len({item.frame_index for item in history}),
                    ),
                    first_stamp_sec=prior.first_stamp_sec,
                    confirmed_stamp_sec=prior.confirmed_stamp_sec,
                    last_seen_stamp_sec=stamp_sec,
                    bbox=detection.bbox,
                )

        return new_confirmations

    def confirmed(self, digit: int) -> Optional[ConfirmedMarker]:
        return self._confirmed.get(digit)

    def all_confirmed(self) -> Dict[int, ConfirmedMarker]:
        return dict(self._confirmed)

    def fresh_confirmed(self, stamp_sec: float) -> Dict[int, ConfirmedMarker]:
        return {
            digit: marker
            for digit, marker in self._confirmed.items()
            if stamp_sec - marker.last_seen_stamp_sec
            <= self.settings.fresh_track_timeout_sec
        }

    def latest_detection(self, digit: int) -> Optional[MarkerDetection]:
        history = self._history.get(digit)
        if not history:
            return None
        return history[-1].detection


def draw_detection_overlay(
    frame_bgr: np.ndarray,
    detections: Sequence[MarkerDetection],
    target_digit: int,
    confirmed_target: Optional[ConfirmedMarker],
    target_column_bbox: Optional[BoundingBox],
    lines: Sequence[str],
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with readable evidence overlays."""
    output = frame_bgr.copy()
    for detection in detections:
        box = detection.bbox
        is_target = detection.digit == target_digit
        colour = (0, 220, 0) if is_target else (255, 180, 0)
        thickness = 3 if is_target else 1
        cv2.rectangle(
            output,
            (box.x, box.y),
            (box.x2, box.y2),
            colour,
            thickness,
        )
        label = f'{detection.digit} {detection.confidence:.2f}'
        cv2.putText(
            output,
            label,
            (box.x, max(18, box.y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )

    if target_column_bbox is not None:
        box = target_column_bbox
        cv2.rectangle(
            output,
            (box.x, box.y),
            (box.x2, box.y2),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            output,
            f'TARGET SHELF COLUMN {target_digit}',
            (box.x + 4, min(output.shape[0] - 8, box.y2 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if confirmed_target is not None:
        box = confirmed_target.bbox
        cv2.rectangle(
            output,
            (box.x, box.y),
            (box.x2, box.y2),
            (0, 255, 0),
            4,
        )

    y = 24
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(
            output,
            (4, y - text_height - 5),
            (12 + text_width, y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 23
    return output
