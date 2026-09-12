#!/usr/bin/env python3
"""Fast pure-Python checks for the Day 5-7 sprint additions."""

from __future__ import annotations

import math
import sys

import cv2
import numpy as np

try:
    from ku_sparcy_erc.bin_perception import (
        BinDetectorSettings,
        consistent_detection,
        detect_red_bin,
    )
except ModuleNotFoundError as exc:
    if exc.name != 'ku_sparcy_erc.bin_perception':
        raise
    BIN_PERCEPTION_AVAILABLE = False
else:
    BIN_PERCEPTION_AVAILABLE = True
from ku_sparcy_erc.day567_common import (
    book_half_diagonal_m,
    directional_lidar_clearance,
    vector_base_to_odom,
    vector_odom_to_base,
)


def gate(label: str, passed: bool) -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    return bool(passed)


def main() -> int:
    failures = 0

    # Day 7-specific bin checks run only after the Day 7 overlay exists.
    if BIN_PERCEPTION_AVAILABLE:
        frame = np.full((360, 640, 3), 225, dtype=np.uint8)
        cv2.rectangle(frame, (330, 185), (520, 280), (0, 0, 255), -1)
        cv2.rectangle(frame, (110, 100), (126, 260), (0, 0, 255), -1)
        settings = BinDetectorSettings()
        detections = detect_red_bin(frame, settings)
        failures += not gate(
            'BIN DETECTOR REJECTS NARROW BOOK DISTRACTOR',
            len(detections) == 1 and detections[0].bbox_xywh[2] > 150,
        )
        history = [
            (1.0 + index * 0.10, detections[0])
            for index in range(3)
        ]
        failures += not gate(
            'BIN TEMPORAL CONFIRMATION',
            consistent_detection(
                history,
                required_frames=3,
                window_sec=0.5,
                center_tolerance_px=10.0,
            ) is not None,
        )
    else:
        print('[SKIP] BIN DETECTOR REJECTS NARROW BOOK DISTRACTOR '
              '(Day 7 overlay not installed)')
        print('[SKIP] BIN TEMPORAL CONFIRMATION '
              '(Day 7 overlay not installed)')

    # Forward and rear corridor semantics with identity TF.
    ranges = [2.0] * 9
    angle_min = -math.pi
    angle_increment = math.pi / 4.0
    front = directional_lidar_clearance(
        ranges,
        angle_min,
        angle_increment,
        0.05,
        25.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        direction_sign=1,
        corridor_half_width_m=0.50,
        robust_percentile=10.0,
        hard_stop_m=0.55,
    )
    rear = directional_lidar_clearance(
        ranges,
        angle_min,
        angle_increment,
        0.05,
        25.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        direction_sign=-1,
        corridor_half_width_m=0.50,
        robust_percentile=10.0,
        hard_stop_m=0.55,
    )
    failures += not gate(
        'DIRECTIONAL FRONT LIDAR',
        front is not None and abs(front.robust_clearance_m - 2.0) < 1e-6,
    )
    failures += not gate(
        'DIRECTIONAL REAR LIDAR',
        rear is not None and abs(rear.robust_clearance_m - 2.0) < 1e-6,
    )

    # Odom/base round trip and book envelope.
    odom = vector_base_to_odom(2.0, 0.5, (1.0, -2.0), 0.7)
    forward, left = vector_odom_to_base(odom, (1.0, -2.0), 0.7)
    failures += not gate(
        'ODOM/BASE ROUND TRIP',
        abs(forward - 2.0) < 1e-9 and abs(left - 0.5) < 1e-9,
    )
    half_diag = book_half_diagonal_m()
    failures += not gate(
        'HELD BOOK HALF DIAGONAL',
        0.147 < half_diag < 0.150,
    )

    if failures:
        print(f'[DAY567 OFFLINE TEST][FAIL] failures={failures}')
        return 1
    print('[DAY567 OFFLINE TEST][PASS]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
