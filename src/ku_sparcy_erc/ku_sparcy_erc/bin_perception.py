#!/usr/bin/env python3
"""Deterministic red collection-bin detector for Day 7."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BinDetection:
    bbox_xywh: Tuple[int, int, int, int]
    confidence: float
    area_px: float
    fill_fraction: float
    aspect_ratio: float

    @property
    def center(self) -> Tuple[float, float]:
        x, y, width, height = self.bbox_xywh
        return x + 0.5 * width, y + 0.5 * height

    def as_dict(self) -> Dict[str, object]:
        return {
            'bbox_xywh': list(self.bbox_xywh),
            'confidence': round(float(self.confidence), 6),
            'area_px': round(float(self.area_px), 3),
            'fill_fraction': round(float(self.fill_fraction), 6),
            'aspect_ratio': round(float(self.aspect_ratio), 6),
        }


@dataclass(frozen=True)
class BinDetectorSettings:
    min_saturation: int = 110
    min_value: int = 55
    min_area_px: int = 450
    min_width_px: int = 34
    min_height_px: int = 18
    min_aspect_ratio: float = 1.10
    max_aspect_ratio: float = 5.0
    min_fill_fraction: float = 0.30
    roi_top_fraction: float = 0.15
    roi_bottom_fraction: float = 0.98

    def validate(self) -> None:
        if not 0 <= self.min_saturation <= 255:
            raise ValueError('min_saturation invalid')
        if not 0 <= self.min_value <= 255:
            raise ValueError('min_value invalid')
        if self.min_area_px < 50:
            raise ValueError('min_area_px too small')
        if not 0.0 <= self.roi_top_fraction < self.roi_bottom_fraction <= 1.0:
            raise ValueError('ROI fractions invalid')


def detect_red_bin(
    frame_bgr: np.ndarray,
    settings: BinDetectorSettings,
    *,
    excluded_center_px: Optional[Sequence[float]] = None,
    excluded_radius_px: float = 80.0,
) -> List[BinDetection]:
    settings.validate()
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    height, width = frame_bgr.shape[:2]
    y0 = int(round(settings.roi_top_fraction * height))
    y1 = int(round(settings.roi_bottom_fraction * height))
    roi = frame_bgr[y0:y1, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low_a = np.array([0, settings.min_saturation, settings.min_value], dtype=np.uint8)
    high_a = np.array([12, 255, 255], dtype=np.uint8)
    low_b = np.array([168, settings.min_saturation, settings.min_value], dtype=np.uint8)
    high_b = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low_a, high_a) | cv2.inRange(hsv, low_b, high_b)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: List[BinDetection] = []
    excluded = (
        tuple(float(v) for v in excluded_center_px)
        if excluded_center_px is not None else None)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < settings.min_area_px:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        y += y0
        if box_width < settings.min_width_px or box_height < settings.min_height_px:
            continue
        aspect = float(box_width) / max(1.0, float(box_height))
        if not settings.min_aspect_ratio <= aspect <= settings.max_aspect_ratio:
            continue
        fill = area / max(1.0, float(box_width * box_height))
        if fill < settings.min_fill_fraction:
            continue
        center = (x + 0.5 * box_width, y + 0.5 * box_height)
        if excluded is not None:
            if math.hypot(center[0] - excluded[0], center[1] - excluded[1]) <= excluded_radius_px:
                continue
        area_score = min(1.0, area / max(1.0, 0.08 * width * height))
        fill_score = min(1.0, fill / 0.65)
        aspect_score = max(0.0, 1.0 - abs(aspect - 2.0) / 2.0)
        confidence = 0.45 * area_score + 0.35 * fill_score + 0.20 * aspect_score
        results.append(BinDetection(
            bbox_xywh=(int(x), int(y), int(box_width), int(box_height)),
            confidence=float(confidence),
            area_px=area,
            fill_fraction=fill,
            aspect_ratio=aspect,
        ))
    results.sort(key=lambda item: (item.confidence, item.area_px), reverse=True)
    return results


def consistent_detection(
    history: Sequence[Tuple[float, BinDetection]],
    *,
    required_frames: int,
    window_sec: float,
    center_tolerance_px: float,
    size_ratio_tolerance: float = 0.45,
) -> Optional[BinDetection]:
    if len(history) < required_frames:
        return None
    newest_stamp = float(history[-1][0])
    recent = [
        (stamp, detection)
        for stamp, detection in history
        if newest_stamp - float(stamp) <= float(window_sec)
    ]
    if len(recent) < required_frames:
        return None
    candidates = recent[-required_frames:]
    reference = candidates[-1][1]
    rx, ry = reference.center
    rw, rh = reference.bbox_xywh[2], reference.bbox_xywh[3]
    for _, detection in candidates[:-1]:
        cx, cy = detection.center
        if math.hypot(cx - rx, cy - ry) > center_tolerance_px:
            return None
        width, height = detection.bbox_xywh[2], detection.bbox_xywh[3]
        if abs(width - rw) / max(1.0, rw) > size_ratio_tolerance:
            return None
        if abs(height - rh) / max(1.0, rh) > size_ratio_tolerance:
            return None
    boxes = np.asarray(
        [item[1].bbox_xywh for item in candidates], dtype=np.float64)
    confidences = [item[1].confidence for item in candidates]
    areas = [item[1].area_px for item in candidates]
    fills = [item[1].fill_fraction for item in candidates]
    aspects = [item[1].aspect_ratio for item in candidates]
    median_box = np.median(boxes, axis=0)
    return BinDetection(
        bbox_xywh=tuple(int(round(value)) for value in median_box),
        confidence=float(sum(confidences) / len(confidences)),
        area_px=float(sum(areas) / len(areas)),
        fill_fraction=float(sum(fills) / len(fills)),
        aspect_ratio=float(sum(aspects) / len(aspects)),
    )
