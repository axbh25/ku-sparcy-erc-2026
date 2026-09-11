#!/usr/bin/env python3
"""Offline tests for stationary pre-approach row mapping and target reacquisition."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ku_sparcy_erc.book_perception import (
    BookDepthEstimate,
    BookDetection,
    BookDetectorSettings,
    BookGeometry,
    SelectedColumnBookDetector,
    build_pregrasp_plan,
    estimate_book_geometry,
    selected_column_roi,
)
from ku_sparcy_erc.passive_book_mapping import (
    SpatialBookMapAccumulator,
    SpatialBookMapSettings,
    SpatialBookObservation,
    TargetBookReacquisitionFilter,
    TargetReacquisitionSettings,
)
from ku_sparcy_erc.range_fusion import BoundingBoxLike, CameraModel


def check(label: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return bool(condition)


def synthetic_partial_frame(colours: tuple[str, ...]) -> np.ndarray:
    image = np.full((360, 640, 3), 205, dtype=np.uint8)
    cv2.rectangle(image, (220, 8), (420, 352), (238, 238, 238), -1)
    definitions = {
        'green': ((0, 215, 0), 30),
        'red': ((0, 0, 235), 99),
        'blue': ((235, 0, 0), 168),
        'yellow': ((0, 225, 225), 237),
    }
    for index, colour in enumerate(colours):
        bgr, y = definitions[colour]
        x = 292 + 12 * index
        cv2.rectangle(image, (x, y), (x + 13, y + 48), bgr, -1)
    return cv2.GaussianBlur(image, (3, 3), 0)


def fake_detection(colour: str, row: int, confidence: float = 0.90) -> BookDetection:
    return BookDetection(
        colour=colour,
        confidence=confidence,
        bbox=BoundingBoxLike(300, 25 + (row - 1) * 69, 13, 49),
        colour_purity=0.96,
        fill_fraction=0.88,
        vertical_aspect=49.0 / 13.0,
    )


def fake_geometry(colour: str, row: int, x: float, y: float, z: float) -> BookGeometry:
    detection = fake_detection(colour, row)
    depth = BookDepthEstimate(
        depth_m=x,
        valid_pixels=60,
        candidate_pixels=65,
        valid_fraction=60.0 / 65.0,
        median_absolute_deviation_m=0.003,
        rgb_sample_bbox_xywh=detection.bbox.as_list(),
        depth_sample_bbox_xywh=detection.bbox.as_list(),
        depth_pixel_u=detection.center_x,
        depth_pixel_v=detection.center_y,
    )
    return BookGeometry(
        colour=colour,
        row=row,
        optical_point_xyz_m=(0.0, 0.0, x),
        base_point_xyz_m=(x, y, z),
        camera_bearing_right_rad=0.0,
        base_bearing_left_rad=math.atan2(y, x),
        rgb_depth_stamp_skew_sec=0.02,
        depth=depth,
        bbox_xywh=detection.bbox.as_list(),
    )


def main() -> int:
    checks: list[bool] = []
    model = CameraModel(640, 360, 350.0, 350.0, 320.0, 180.0, 'camera')
    roi = selected_column_roi(
        640,
        360,
        model.fx,
        model.cx,
        2.40,
        physical_width_m=0.78,
        minimum_half_width_px=40,
        maximum_half_width_px=150,
    )
    checks.append(check(
        'medium-range selected-column ROI contains the principal point',
        roi.x <= model.cx <= roi.x2,
    ))
    checks.append(check(
        'medium-range selected-column ROI remains narrower than adjacent columns',
        80 <= roi.width <= 150,
    ))

    detector_settings = BookDetectorSettings()
    detector = SelectedColumnBookDetector(detector_settings)
    partial = synthetic_partial_frame(('green', 'red'))
    candidates, diagnostics = detector.detect_colours(
        partial,
        roi,
        ('red', 'green', 'yellow', 'blue'),
    )
    checks.append(check(
        'independent colour detection accepts a partial-view frame',
        bool(candidates['green']) and bool(candidates['red']),
    ))
    observation, layout_diagnostics = detector.detect_layout(
        partial,
        roi,
        stamp_sec=1.0,
        frame_index=1,
    )
    checks.append(check(
        'partial-view frame does not pretend to be a four-colour layout',
        observation is None
        and layout_diagnostics.get('complete_colour_set') is False,
    ))

    settings = SpatialBookMapSettings(
        required_frames_per_colour=3,
        minimum_observation_span_sec=0.05,
        minimum_average_confidence=0.60,
        maximum_cluster_xy_radius_m=0.18,
        maximum_cluster_z_radius_m=0.10,
        maximum_track_xy_span_m=0.18,
        maximum_track_z_span_m=0.10,
        minimum_row_separation_m=0.14,
        maximum_row_separation_m=0.55,
    )
    accumulator = SpatialBookMapAccumulator(settings)
    row_map = {
        'green': (1, 1.80),
        'red': (2, 1.45),
        'blue': (3, 1.10),
        'yellow': (4, 0.75),
    }
    confirmation = None
    frame_index = 0

    # Frames 1-3 contain only the top two colours; frames 4-6 contain only the
    # bottom two.  No single frame contains the complete physical colour set.
    for frame_group, visible in enumerate(
        (('green', 'red'), ('blue', 'yellow')),
        start=0,
    ):
        for repeat in range(3):
            frame_index += 1
            stamp = 2.0 + 0.08 * frame_index
            for colour in visible:
                row, height = row_map[colour]
                geometry = fake_geometry(
                    colour,
                    row,
                    x=2.20 - 0.10 * frame_group + 0.002 * repeat,
                    y=0.04 + 0.003 * repeat,
                    z=height + 0.002 * repeat,
                )
                confirmation = accumulator.add(
                    SpatialBookObservation(
                        colour=colour,
                        stamp_sec=stamp,
                        frame_index=frame_index,
                        head_tilt_rad=0.35 - 0.20 * frame_group,
                        lidar_clearance_m=2.60 - 0.55 * frame_group,
                        detection=fake_detection(colour, row),
                        geometry=geometry,
                        odom_xyz_m=(2.90, 0.04, height + 0.002 * repeat),
                        roi_xywh=tuple(roi.as_list()),
                    ),
                    mapping_source='stationary_preapproach',
                )

    checks.append(check(
        'spatial accumulator confirms without a complete single frame',
        confirmation is not None
        and confirmation.as_dict().get('single_frame_required') is False,
    ))
    checks.append(check(
        'every colour has at least three independent confirming frames',
        confirmation is not None
        and confirmation.confirming_frames >= 3
        and confirmation.total_observations >= 12,
    ))
    checks.append(check(
        'multi-frame 3-D row order is correct',
        confirmation is not None
        and confirmation.row_colours_top_to_bottom
        == ('green', 'red', 'blue', 'yellow'),
    ))
    checks.append(check(
        'requested red book maps to active row 2',
        confirmation is not None and confirmation.row_for_colour('red') == 2,
    ))

    assert confirmation is not None
    locked_red = confirmation.locked_odom_xyz_for_colour('red')
    reacquisition = TargetBookReacquisitionFilter(
        'red',
        2,
        locked_red,
        TargetReacquisitionSettings(required_frames=3),
    )
    reacquired = None
    for index, stamp in enumerate((5.00, 5.08, 5.16), start=100):
        geometry = fake_geometry('red', 2, 1.10, 0.03, 1.45)
        reacquired = reacquisition.update(
            SpatialBookObservation(
                colour='red',
                stamp_sec=stamp,
                frame_index=index,
                head_tilt_rad=-0.10,
                lidar_clearance_m=1.17,
                detection=fake_detection('red', 2, confidence=0.92),
                geometry=geometry,
                odom_xyz_m=(locked_red[0] + 0.01, locked_red[1], locked_red[2]),
                roi_xywh=tuple(roi.as_list()),
            ),
            current_stamp_sec=stamp,
        )
    checks.append(check(
        'close-range requested-colour-only reacquisition requires three frames',
        reacquired is not None and reacquired.confirming_frames >= 3,
    ))
    checks.append(check(
        'reacquired target remains near its locked odometry row position',
        reacquired is not None
        and abs(reacquired.median_observed_odom_xyz_m[2] - locked_red[2]) <= 0.02,
    ))

    # Retain the real depth helper test used by the production node.
    full_frame = synthetic_partial_frame(('red',))
    red_candidates, _ = detector.detect_colours(full_frame, roi, ('red',))
    target = red_candidates['red'][0]
    depth_image = np.full((360, 640), 1.30, dtype=np.float32)
    box = target.bbox
    depth_image[box.y:box.y2, box.x:box.x2] = 1.10
    geometry = estimate_book_geometry(
        full_frame,
        target,
        2,
        model,
        model,
        depth_image,
        (0.0, 0.0, 1.20),
        (0.5, -0.5, 0.5, -0.5),
        rgb_depth_stamp_skew_sec=0.02,
        detector_settings=detector_settings,
    )
    checks.append(check(
        'requested-colour depth samples the 1.10 m surface',
        geometry is not None and abs(geometry.depth.depth_m - 1.10) <= 0.02,
    ))
    checks.append(check(
        'RGB/depth skew remains quantitative and below 0.20 s',
        geometry is not None
        and math.isclose(geometry.rgb_depth_stamp_skew_sec, 0.02),
    ))

    assert geometry is not None
    plan = build_pregrasp_plan(
        geometry,
        {
            'left': (0.05, 0.25, 1.10),
            'right': (0.05, -0.25, 1.10),
        },
        maximum_reach_m=1.10,
    )
    checks.append(check(
        'one arm passes the non-executing geometry screen',
        plan.selected_arm in {'left', 'right'}
        and plan.geometric_screen_passed,
    ))
    checks.append(check(
        'pre-grasp remains behind the tentative contact point',
        plan.pregrasp_point_base_xyz_m[0]
        < plan.tentative_grasp_point_base_xyz_m[0],
    ))

    if all(checks):
        print('\n[STATIONARY PRE-APPROACH TRACKS][PASS]')
        print('[NO SINGLE-FRAME REQUIREMENT][PASS]')
        print('[SPATIAL ROW MAP][PASS]')
        print('[TARGET-ONLY REACQUISITION][PASS]')
        print('[BOOK DEPTH GEOMETRY][PASS]')
        print('[PREGRASP GEOMETRY MATH][PASS]')
        print('[DAY4 STATIONARY-MAPPING UNIT TEST][PASS]')
        return 0

    print('\n[DAY4 STATIONARY-MAPPING UNIT TEST][FAIL]')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
