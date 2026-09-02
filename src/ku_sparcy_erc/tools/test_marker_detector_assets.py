#!/usr/bin/env python3
"""CPU-only smoke test against the official marker texture assets.

This test does not replace live Gazebo validation.  It verifies that all five
official assets can be loaded and that the detector survives scale, small
rotation, blur, and frame-to-frame jitter before temporally confirming 1-5.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from ku_sparcy_erc.marker_detector import (
    DetectorSettings,
    ShelfMarkerDetector,
    TemporalMarkerFilter,
    TemporalSettings,
)


def foreground_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f'could not load {path}')
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, normal = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverse = cv2.bitwise_not(normal)
    mask = normal if np.count_nonzero(normal) < np.count_nonzero(inverse) else inverse
    points = cv2.findNonZero(mask)
    if points is None:
        raise RuntimeError(f'asset has no foreground: {path}')
    x, y, width, height = cv2.boundingRect(points)
    return mask[y:y + height, x:x + width]


def render_frame(
    texture_dir: Path,
    order: list[int],
    jitter: int,
) -> np.ndarray:
    frame = np.full((360, 640, 3), 190, dtype=np.uint8)
    # A simplified shelf top: white marker plates, grey shelf face.
    cv2.rectangle(frame, (55, 105), (585, 255), (235, 235, 235), -1)
    cv2.rectangle(frame, (55, 165), (585, 180), (205, 205, 205), -1)
    x_centres = [105, 210, 320, 430, 535]
    for index, digit in enumerate(order):
        plate = np.full((42, 48, 3), 248, dtype=np.uint8)
        mask = foreground_mask(texture_dir / f'{digit}.png')
        target_height = 21 + ((index + jitter) % 3)
        scale = target_height / float(mask.shape[0])
        target_width = max(3, int(round(mask.shape[1] * scale)))
        glyph = cv2.resize(
            mask, (target_width, target_height), interpolation=cv2.INTER_AREA)
        glyph = np.where(glyph > 90, 255, 0).astype(np.uint8)
        x0 = (plate.shape[1] - target_width) // 2
        y0 = (plate.shape[0] - target_height) // 2
        plate[y0:y0 + target_height, x0:x0 + target_width][glyph > 0] = 8

        angle = (-7.0, -3.0, 0.0, 4.0, 7.0)[index]
        matrix = cv2.getRotationMatrix2D(
            ((plate.shape[1] - 1) / 2.0, (plate.shape[0] - 1) / 2.0),
            angle,
            1.0,
        )
        plate = cv2.warpAffine(
            plate,
            matrix,
            (plate.shape[1], plate.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(248, 248, 248),
        )
        if (index + jitter) % 2 == 0:
            plate = cv2.GaussianBlur(plate, (3, 3), 0.45)
        x = x_centres[index] - plate.shape[1] // 2 + jitter - 1
        y = 112 + ((index + jitter) % 2)
        frame[y:y + plate.shape[0], x:x + plate.shape[1]] = plate
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--template-dir', type=Path)
    parser.add_argument('--save-frame', type=Path)
    args = parser.parse_args()

    texture_dir = args.template_dir
    if texture_dir is None:
        from ament_index_python.packages import get_package_share_directory
        texture_dir = Path(get_package_share_directory('erc_description')) / \
            'models' / 'number_marker' / 'textures'

    missing = [
        str(texture_dir / f'{digit}.png')
        for digit in range(1, 6)
        if not (texture_dir / f'{digit}.png').is_file()
    ]
    if missing:
        print('[MARKER ASSET SELF-TEST][FAIL] Missing official assets:')
        print('\n'.join(missing))
        return 1

    print(f'Official marker directory: {texture_dir}')
    for digit in range(1, 6):
        image = cv2.imread(str(texture_dir / f'{digit}.png'), cv2.IMREAD_UNCHANGED)
        print(f'[PASS] {digit}.png loaded shape={image.shape}')

    detector = ShelfMarkerDetector(
        str(texture_dir),
        DetectorSettings(
            min_classifier_margin=0.010,
            min_classifier_confidence=0.55,
            min_bright_ring_fraction=0.42,
        ),
    )
    temporal = TemporalMarkerFilter(TemporalSettings(
        required_frames=3,
        window_sec=0.75,
        min_span_sec=0.04,
        min_average_confidence=0.58,
    ))

    order = [3, 5, 2, 1, 4]
    latest = None
    for frame_index in range(5):
        latest = render_frame(texture_dir, order, frame_index)
        detections, diagnostics = detector.detect(latest)
        temporal.update(
            detections,
            stamp_sec=10.0 + 0.10 * frame_index,
            frame_index=frame_index,
        )
        detected = sorted(item.digit for item in detections)
        confirmed_now = sorted(temporal.all_confirmed())
        print(
            f'[FRAME {frame_index}] accepted={detected} '
            f'confirmed={confirmed_now} '
            f'candidates={diagnostics["classified_candidate_count"]}')
        if frame_index < 2 and confirmed_now:
            print(
                '[MARKER ASSET SELF-TEST][FAIL] A marker was accepted before '
                'three distinct frames')
            return 1

    confirmed = temporal.all_confirmed()
    observed = sorted(confirmed)
    if args.save_frame is not None and latest is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_frame), latest)
        print(f'Synthetic test frame: {args.save_frame}')

    if observed != [1, 2, 3, 4, 5]:
        print(
            '[MARKER ASSET SELF-TEST][FAIL] Expected confirmed 1-5, got '
            f'{observed}')
        return 1
    print('[PASS] all five official marker values survived transformed rendering')
    print('[PASS] temporal confirmation required multiple distinct frames')
    print('[MARKER ASSET SELF-TEST][PASS]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
