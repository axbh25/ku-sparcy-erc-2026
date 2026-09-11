#!/usr/bin/env python3
"""Day 4: stationary pre-approach row mapping and close reacquisition.

This node subclasses the validated Day 3 mission without modifying it.  Day 1
opening, Day 2 shelf-number perception, the early dual-arm navigation tuck,
and Day 3 locked-column odometry/LiDAR approach remain authoritative.

After the requested physical shelf column is locked and both navigation arms
are ready, Day 4 deliberately holds the base stationary and performs a bounded
head scan of that already selected column.  Independent colour observations
are paired with raw depth, transformed into the odometry frame, and accumulated
across settled head poses.  No frame must contain all four colours.  Once the
colour-to-row map is locked, the requested row is published, the validated
Day 3 +0.35 rad head pose is restored and verified, and the unchanged Day 3
approach starts.  At final stand-off only the requested colour is reacquired for
precise 3-D geometry.  The old moving-head-during-approach scheduler is disabled.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ku_sparcy_erc.book_perception import (
    BookDetection,
    BookDetectorSettings,
    BookGeometry,
    PregraspPlan,
    SelectedColumnBookDetector,
    build_pregrasp_plan,
    draw_target_book_evidence,
    estimate_book_geometry,
    selected_column_roi,
)
from ku_sparcy_erc.day3_mission import Day3Mission
from ku_sparcy_erc.passive_book_mapping import (
    BOOK_COLOURS,
    ConfirmedSpatialBookMap,
    ReacquiredTargetBook,
    SpatialBookMapAccumulator,
    SpatialBookMapSettings,
    SpatialBookObservation,
    TargetBookReacquisitionFilter,
    TargetReacquisitionSettings,
)
from ku_sparcy_erc.range_fusion import (
    BoundingBoxLike,
    quaternion_to_rotation_matrix,
)


class Day4Mission(Day3Mission):
    """Continue from Day 3 using a stationary pre-approach row scan."""

    ROW_TOPIC = '/erc/shelf_row_identification'
    # No Day 4 mapping occurs while Day 3 is translating.  The set is kept
    # empty as an explicit architectural invariant and compatibility marker.
    INHERITED_PASSIVE_STATES = set()
    PREAPPROACH_SCAN_STATES = {
        'POSITION_PREAPPROACH_HEAD',
        'WAIT_FOR_PREAPPROACH_HEAD',
        'DWELL_PREAPPROACH_HEAD',
        'RESTORE_PREAPPROACH_HEAD',
        'WAIT_FOR_PREAPPROACH_RESTORE',
        'VERIFY_PREAPPROACH_VISIBILITY',
    }
    DAY4_STATES = PREAPPROACH_SCAN_STATES | {
        'POSITION_FALLBACK_HEAD',
        'WAIT_FOR_FALLBACK_HEAD',
        'SCAN_FALLBACK_MAPPING',
        'POSITION_TARGET_HEAD',
        'WAIT_FOR_TARGET_HEAD',
        'REACQUIRE_TARGET_BOOK',
        'ESTIMATE_BOOK_GEOMETRY',
        'SELECT_MANIPULATION_ARM',
        'VERIFY_BOOK_EVIDENCE',
    }

    def __init__(self) -> None:
        super().__init__()

        # The bounded stationary head sequence must be measured before it is
        # accepted as the competition row-mapping path.  Profile mode stops
        # before Day 3 translation and writes pose-wise visibility evidence.
        self.declare_parameter('day4_stop_after', 'pregrasp')
        self.declare_parameter('book_perception_enabled', True)
        self.declare_parameter('passive_mapping_window_calibrated', False)
        self.declare_parameter('passive_mapping_start_clearance_m', 2.80)
        self.declare_parameter('passive_mapping_stop_clearance_m', 1.55)
        self.declare_parameter('visibility_profile_start_clearance_m', 3.20)
        self.declare_parameter('visibility_profile_stop_clearance_m', 1.08)
        self.declare_parameter('visibility_profile_bin_width_m', 0.25)
        self.declare_parameter('passive_mapping_process_period_sec', 0.12)
        self.declare_parameter('passive_mapping_max_samples', 650)

        # Primary Day 4 architecture: a base-stationary scan after column lock
        # and travel-arm readiness, but before Day 3 translation.  Visibility
        # mode first profiles all bounded poses.  Normal operation is refused
        # until the analyzer has written a calibrated minimal pose sequence.
        self.declare_parameter('preapproach_scan_calibrated', False)
        self.declare_parameter(
            'preapproach_scan_profile_head_tilt_sequence_rad',
            [0.35, 0.10, -0.10],
        )
        self.declare_parameter(
            'preapproach_scan_head_tilt_sequence_rad', [0.35, -0.10])
        self.declare_parameter('preapproach_scan_head_motion_sec', 0.80)
        self.declare_parameter('preapproach_scan_head_ready_timeout_sec', 14.0)
        self.declare_parameter('preapproach_scan_head_tolerance_rad', 0.06)
        self.declare_parameter('preapproach_scan_settle_samples', 4)
        self.declare_parameter('preapproach_scan_min_dwell_sec', 0.90)
        self.declare_parameter('preapproach_scan_min_frames_per_pose', 4)
        self.declare_parameter('preapproach_scan_pose_timeout_sec', 3.50)
        self.declare_parameter('preapproach_scan_total_timeout_sec', 50.0)
        self.declare_parameter('preapproach_scan_restore_timeout_sec', 14.0)
        self.declare_parameter(
            'preapproach_scan_base_translation_tolerance_m', 0.03)
        self.declare_parameter(
            'preapproach_scan_base_yaw_tolerance_deg', 2.0)

        # Deprecated compatibility parameters from the approach-time
        # experiment.  The scheduler is now permanently disabled; these values
        # remain loadable so an existing /tmp YAML cannot crash startup.
        self.declare_parameter('passive_mapping_head_motion_enabled', False)
        self.declare_parameter(
            'passive_mapping_head_tilt_sequence_rad', [0.35, 0.10, -0.10])
        self.declare_parameter('passive_mapping_head_dwell_sec', 0.90)
        self.declare_parameter('passive_mapping_head_restore_margin_m', 0.18)
        self.declare_parameter('passive_mapping_head_tolerance_rad', 0.06)

        # Selected-column ROI follows the locked physical column's current
        # bearing, rather than always assuming it lies at image centre.
        self.declare_parameter('book_column_physical_width_m', 0.78)
        self.declare_parameter('book_column_roi_min_half_width_px', 40)
        self.declare_parameter('book_column_roi_max_half_width_px', 150)
        self.declare_parameter('book_mapping_max_column_lateral_error_m', 0.48)

        # Existing detector thresholds are preserved.  The architecture—not
        # colour tuning—is changed after the failed close-range all-in-one scan.
        self.declare_parameter('book_min_saturation', 90)
        self.declare_parameter('book_min_value', 55)
        self.declare_parameter('book_min_area_px', 18)
        self.declare_parameter('book_min_height_px', 12)
        self.declare_parameter('book_min_vertical_aspect', 1.20)
        self.declare_parameter('book_min_fill_fraction', 0.22)
        self.declare_parameter('book_min_colour_purity', 0.55)
        self.declare_parameter('book_min_detection_confidence', 0.58)
        self.declare_parameter('book_min_row_separation_px', 18.0)

        # Multi-frame, multi-distance, multi-head-pose spatial mapping.
        self.declare_parameter('book_map_frames_per_colour', 3)
        self.declare_parameter('book_map_min_span_sec', 0.05)
        self.declare_parameter('book_map_average_confidence', 0.60)
        self.declare_parameter('book_map_cluster_xy_radius_m', 0.18)
        self.declare_parameter('book_map_cluster_z_radius_m', 0.10)
        self.declare_parameter('book_map_track_xy_span_m', 0.18)
        self.declare_parameter('book_map_track_z_span_m', 0.10)
        self.declare_parameter('book_map_min_row_separation_m', 0.14)
        self.declare_parameter('book_map_max_row_separation_m', 0.55)

        # Fallback is intentionally base-stationary and accumulates independent
        # colours across poses.  It does not restore the old single-frame rule.
        self.declare_parameter('fallback_scan_total_timeout_sec', 15.0)
        self.declare_parameter('fallback_scan_pose_timeout_sec', 2.8)
        self.declare_parameter(
            'fallback_head_tilt_sequence_rad', [0.20, 0.00, -0.20, -0.40, -0.60])

        # Close-range requested-colour-only reacquisition.
        self.declare_parameter('target_head_motion_sec', 0.80)
        self.declare_parameter('target_head_ready_timeout_sec', 7.5)
        self.declare_parameter('target_head_tolerance_rad', 0.05)
        self.declare_parameter('target_reacquisition_pose_timeout_sec', 3.2)
        self.declare_parameter('target_reacquisition_total_timeout_sec', 14.0)
        self.declare_parameter('target_reacquisition_frames', 3)
        self.declare_parameter('target_reacquisition_window_sec', 0.90)
        self.declare_parameter('target_reacquisition_min_span_sec', 0.05)
        self.declare_parameter('target_reacquisition_max_xy_error_m', 0.18)
        self.declare_parameter('target_reacquisition_max_z_error_m', 0.12)
        self.declare_parameter('target_reacquisition_max_pixel_jump_px', 50.0)
        self.declare_parameter('target_reacquisition_average_confidence', 0.60)

        # Raw-depth and pre-grasp settings retain Day 3 synchronization limits.
        self.declare_parameter('book_depth_min_valid_pixels', 12)
        self.declare_parameter('book_depth_min_valid_fraction', 0.12)
        self.declare_parameter('book_geometry_timeout_sec', 2.0)
        self.declare_parameter('pregrasp_offset_m', 0.18)
        self.declare_parameter('tentative_grasp_offset_m', 0.025)
        self.declare_parameter('retreat_offset_m', 0.30)
        self.declare_parameter('pregrasp_min_reach_m', 0.20)
        self.declare_parameter('pregrasp_max_reach_m', 1.10)
        self.declare_parameter('book_result_hold_sec', 0.35)

        self._read_day4_parameters()
        self._validate_day4_parameters()

        detector_settings = BookDetectorSettings(
            min_saturation=self.book_min_saturation,
            min_value=self.book_min_value,
            min_area_px=self.book_min_area_px,
            min_height_px=self.book_min_height_px,
            min_vertical_aspect=self.book_min_vertical_aspect,
            min_fill_fraction=self.book_min_fill_fraction,
            min_colour_purity=self.book_min_colour_purity,
            min_detection_confidence=self.book_min_detection_confidence,
            minimum_row_separation_px=self.book_min_row_separation_px,
        )
        self.book_detector_settings = detector_settings
        self.book_detector = SelectedColumnBookDetector(detector_settings)
        self.book_map_accumulator = SpatialBookMapAccumulator(
            SpatialBookMapSettings(
                required_frames_per_colour=self.book_map_frames_per_colour,
                minimum_observation_span_sec=self.book_map_min_span_sec,
                minimum_average_confidence=self.book_map_average_confidence,
                maximum_cluster_xy_radius_m=self.book_map_cluster_xy_radius_m,
                maximum_cluster_z_radius_m=self.book_map_cluster_z_radius_m,
                maximum_track_xy_span_m=self.book_map_track_xy_span_m,
                maximum_track_z_span_m=self.book_map_track_z_span_m,
                minimum_row_separation_m=self.book_map_min_row_separation_m,
                maximum_row_separation_m=self.book_map_max_row_separation_m,
            )
        )

        result_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.row_pub = self.create_publisher(Int32, self.ROW_TOPIC, result_qos)
        self.book_debug_pub = self.create_publisher(
            String, '/ku_sparcy/book_debug', 10)

        self.day4_started = False
        self.day4_started_sim_sec: Optional[float] = None
        self.day4_started_wall_monotonic: Optional[float] = None
        self.day4_inherited_day3_reason = ''
        self.day3_approach_head_actual_tilt_snapshot_rad: Optional[float] = None
        self.day4_outcome = 'not_started'
        self.day4_failure_classification = ''

        # Passive mapping state exists before Day 4's post-stand-off states.
        self.passive_mapping_started = False
        self.passive_mapping_started_sim_sec: Optional[float] = None
        self.passive_mapping_last_process_stamp_sec: Optional[float] = None
        self.passive_mapping_processed_frames = 0
        self.passive_mapping_visibility_samples: List[Dict[str, object]] = []
        self.passive_mapping_lock_sim_sec: Optional[float] = None
        self.passive_mapping_lock_lidar_clearance_m: Optional[float] = None
        self.passive_mapping_locked_during_approach = False
        self.passive_mapping_cmd_vel_publications = 0
        self.passive_mapping_head_commands: List[Dict[str, object]] = []
        self.passive_mapping_head_index = 0
        self.passive_mapping_head_last_command_sim_sec: Optional[float] = None
        self.passive_mapping_head_target_rad: Optional[float] = None
        self.passive_mapping_head_restored = False


        # Stationary pre-approach scan state.  Day3Mission._start_approach() is
        # called only after this scan restores and verifies the +0.35 rad head.
        self.pending_approach_geometry = None
        self.preapproach_scan_started = False
        self.preapproach_scan_complete = False
        self.preapproach_scan_started_sim_sec: Optional[float] = None
        self.preapproach_scan_completed_sim_sec: Optional[float] = None
        self.preapproach_scan_sequence_rad: List[float] = []
        self.preapproach_scan_index = 0
        self.preapproach_scan_head_target_rad: Optional[float] = None
        self.preapproach_scan_head_command_sim_sec: Optional[float] = None
        self.preapproach_scan_frames_at_command = 0
        self.preapproach_scan_settle_count = 0
        self.preapproach_scan_dwell_started_sim_sec: Optional[float] = None
        self.preapproach_scan_dwell_processed_start = 0
        self.preapproach_scan_dwell_sample_start = 0
        self.preapproach_scan_attempts: List[Dict[str, object]] = []
        self.preapproach_scan_head_commands: List[Dict[str, object]] = []
        self.preapproach_scan_head_restored = False
        self.preapproach_scan_restore_command_sim_sec: Optional[float] = None
        self.preapproach_scan_restore_frames_at_command = 0
        self.preapproach_scan_restore_settle_count = 0
        self.preapproach_scan_map_locked_before_approach = False
        self.preapproach_scan_zero_cmd_vel_publications = 0
        self.preapproach_scan_nonzero_cmd_vel_publications = 0
        self.preapproach_scan_start_odom_xy: Optional[Tuple[float, float]] = None
        self.preapproach_scan_end_odom_xy: Optional[Tuple[float, float]] = None
        self.preapproach_scan_start_yaw_rad: Optional[float] = None
        self.preapproach_scan_end_yaw_rad: Optional[float] = None
        self.preapproach_scan_base_translation_m: Optional[float] = None
        self.preapproach_scan_base_yaw_change_deg: Optional[float] = None
        self.preapproach_scan_start_lidar_clearance_m: Optional[float] = None
        self.preapproach_scan_approach_started_sim_sec: Optional[float] = None
        self.preapproach_scan_profile_only = False

        # A separate history prevents modification of Day 3's tested marker
        # depth association.  Entries are (simulation stamp, float32 depth).
        self.book_depth_history: List[Tuple[float, np.ndarray]] = []
        self.book_depth_history_max_frames = 40

        self.book_column_roi: Optional[BoundingBoxLike] = None
        self.book_processed_frames = 0
        self.book_latest_diagnostics: Dict[str, object] = {}
        self.book_layout_confirmation: Optional[ConfirmedSpatialBookMap] = None
        self.row_mapping_source = 'not_locked'
        self.target_book_row: Optional[int] = None
        self.target_book_locked_odom_xyz_m: Optional[Tuple[float, float, float]] = None
        self.row_publication_count = 0
        self.first_row_publish_sim_sec: Optional[float] = None
        self.last_row_publish_sim_sec: Optional[float] = None
        self.last_row_publish_wall_monotonic: Optional[float] = None

        # Fallback mapping state.
        self.fallback_mapping_used = False
        self.fallback_head_index = 0
        self.fallback_head_command_sim_sec: Optional[float] = None
        self.fallback_head_frames_at_command = 0
        self.fallback_head_pose_started_sim_sec: Optional[float] = None
        self.fallback_started_sim_sec: Optional[float] = None
        self.book_scan_attempts: List[Dict[str, object]] = []

        # Requested-colour close-range reacquisition.
        self.target_head_sequence_rad: List[float] = []
        self.target_head_index = 0
        self.target_head_command_sim_sec: Optional[float] = None
        self.target_head_frames_at_command = 0
        self.target_head_pose_started_sim_sec: Optional[float] = None
        self.target_reacquisition_started_sim_sec: Optional[float] = None
        self.target_reacquisition_filter: Optional[
            TargetBookReacquisitionFilter] = None
        self.target_reacquisition_confirmation: Optional[
            ReacquiredTargetBook] = None
        self.target_reacquisition_processed_frames = 0
        self.target_reacquisition_diagnostics: Dict[str, object] = {}
        self.target_book_detection: Optional[BookDetection] = None
        self.target_book_confirmation_frame_bgr: Optional[np.ndarray] = None
        self.target_book_rgb_stamp_sec: Optional[float] = None
        self.target_book_evidence_path: Optional[str] = None
        self.target_book_evidence_saved = False
        self.target_book_geometry: Optional[BookGeometry] = None
        self.target_book_geometry_image_path: Optional[str] = None
        self.pregrasp_plan: Optional[PregraspPlan] = None
        self.book_geometry_started_sim_sec: Optional[float] = None
        self.book_verification_started_sim_sec: Optional[float] = None

        self.get_logger().info(
            '[DAY4] stationary pre-approach book-row mapping ready: '
            f'colour={self.book_colour} stop_after={self.day4_stop_after} '
            f'scan_calibrated={self.preapproach_scan_calibrated} '
            f'profile_poses={self.preapproach_scan_profile_head_tilt_sequence_rad} '
            f'competition_poses={self.preapproach_scan_head_tilt_sequence_rad}'
        )

    def _read_day4_parameters(self) -> None:
        get = self.get_parameter
        self.day4_stop_after = str(get('day4_stop_after').value).strip().lower()
        if self.day4_stop_after == 'visibility':
            self.day4_stop_after = 'preapproach_visibility'
        self.book_perception_enabled = bool(get('book_perception_enabled').value)
        self.passive_mapping_window_calibrated = bool(
            get('passive_mapping_window_calibrated').value)
        self.passive_mapping_start_clearance_m = float(
            get('passive_mapping_start_clearance_m').value)
        self.passive_mapping_stop_clearance_m = float(
            get('passive_mapping_stop_clearance_m').value)
        self.visibility_profile_start_clearance_m = float(
            get('visibility_profile_start_clearance_m').value)
        self.visibility_profile_stop_clearance_m = float(
            get('visibility_profile_stop_clearance_m').value)
        self.visibility_profile_bin_width_m = float(
            get('visibility_profile_bin_width_m').value)
        self.passive_mapping_process_period_sec = float(
            get('passive_mapping_process_period_sec').value)
        self.passive_mapping_max_samples = int(
            get('passive_mapping_max_samples').value)
        self.preapproach_scan_calibrated = bool(
            get('preapproach_scan_calibrated').value)
        self.preapproach_scan_profile_head_tilt_sequence_rad = [
            float(value)
            for value in get(
                'preapproach_scan_profile_head_tilt_sequence_rad').value
        ]
        self.preapproach_scan_head_tilt_sequence_rad = [
            float(value)
            for value in get(
                'preapproach_scan_head_tilt_sequence_rad').value
        ]
        self.preapproach_scan_head_motion_sec = float(
            get('preapproach_scan_head_motion_sec').value)
        self.preapproach_scan_head_ready_timeout_sec = float(
            get('preapproach_scan_head_ready_timeout_sec').value)
        self.preapproach_scan_head_tolerance_rad = float(
            get('preapproach_scan_head_tolerance_rad').value)
        self.preapproach_scan_settle_samples = int(
            get('preapproach_scan_settle_samples').value)
        self.preapproach_scan_min_dwell_sec = float(
            get('preapproach_scan_min_dwell_sec').value)
        self.preapproach_scan_min_frames_per_pose = int(
            get('preapproach_scan_min_frames_per_pose').value)
        self.preapproach_scan_pose_timeout_sec = float(
            get('preapproach_scan_pose_timeout_sec').value)
        self.preapproach_scan_total_timeout_sec = float(
            get('preapproach_scan_total_timeout_sec').value)
        self.preapproach_scan_restore_timeout_sec = float(
            get('preapproach_scan_restore_timeout_sec').value)
        self.preapproach_scan_base_translation_tolerance_m = float(
            get('preapproach_scan_base_translation_tolerance_m').value)
        self.preapproach_scan_base_yaw_tolerance_deg = float(
            get('preapproach_scan_base_yaw_tolerance_deg').value)
        self.passive_mapping_head_motion_enabled = bool(
            get('passive_mapping_head_motion_enabled').value)
        self.passive_mapping_head_tilt_sequence_rad = [
            float(value)
            for value in get('passive_mapping_head_tilt_sequence_rad').value
        ]
        self.passive_mapping_head_dwell_sec = float(
            get('passive_mapping_head_dwell_sec').value)
        self.passive_mapping_head_restore_margin_m = float(
            get('passive_mapping_head_restore_margin_m').value)
        self.passive_mapping_head_tolerance_rad = float(
            get('passive_mapping_head_tolerance_rad').value)
        self.book_column_physical_width_m = float(
            get('book_column_physical_width_m').value)
        self.book_column_roi_min_half_width_px = int(
            get('book_column_roi_min_half_width_px').value)
        self.book_column_roi_max_half_width_px = int(
            get('book_column_roi_max_half_width_px').value)
        self.book_mapping_max_column_lateral_error_m = float(
            get('book_mapping_max_column_lateral_error_m').value)
        self.book_min_saturation = int(get('book_min_saturation').value)
        self.book_min_value = int(get('book_min_value').value)
        self.book_min_area_px = int(get('book_min_area_px').value)
        self.book_min_height_px = int(get('book_min_height_px').value)
        self.book_min_vertical_aspect = float(
            get('book_min_vertical_aspect').value)
        self.book_min_fill_fraction = float(
            get('book_min_fill_fraction').value)
        self.book_min_colour_purity = float(
            get('book_min_colour_purity').value)
        self.book_min_detection_confidence = float(
            get('book_min_detection_confidence').value)
        self.book_min_row_separation_px = float(
            get('book_min_row_separation_px').value)
        self.book_map_frames_per_colour = int(
            get('book_map_frames_per_colour').value)
        self.book_map_min_span_sec = float(get('book_map_min_span_sec').value)
        self.book_map_average_confidence = float(
            get('book_map_average_confidence').value)
        self.book_map_cluster_xy_radius_m = float(
            get('book_map_cluster_xy_radius_m').value)
        self.book_map_cluster_z_radius_m = float(
            get('book_map_cluster_z_radius_m').value)
        self.book_map_track_xy_span_m = float(
            get('book_map_track_xy_span_m').value)
        self.book_map_track_z_span_m = float(
            get('book_map_track_z_span_m').value)
        self.book_map_min_row_separation_m = float(
            get('book_map_min_row_separation_m').value)
        self.book_map_max_row_separation_m = float(
            get('book_map_max_row_separation_m').value)
        self.fallback_scan_total_timeout_sec = float(
            get('fallback_scan_total_timeout_sec').value)
        self.fallback_scan_pose_timeout_sec = float(
            get('fallback_scan_pose_timeout_sec').value)
        self.fallback_head_tilt_sequence_rad = [
            float(value)
            for value in get('fallback_head_tilt_sequence_rad').value
        ]
        self.target_head_motion_sec = float(get('target_head_motion_sec').value)
        self.target_head_ready_timeout_sec = float(
            get('target_head_ready_timeout_sec').value)
        self.target_head_tolerance_rad = float(
            get('target_head_tolerance_rad').value)
        self.target_reacquisition_pose_timeout_sec = float(
            get('target_reacquisition_pose_timeout_sec').value)
        self.target_reacquisition_total_timeout_sec = float(
            get('target_reacquisition_total_timeout_sec').value)
        self.target_reacquisition_frames = int(
            get('target_reacquisition_frames').value)
        self.target_reacquisition_window_sec = float(
            get('target_reacquisition_window_sec').value)
        self.target_reacquisition_min_span_sec = float(
            get('target_reacquisition_min_span_sec').value)
        self.target_reacquisition_max_xy_error_m = float(
            get('target_reacquisition_max_xy_error_m').value)
        self.target_reacquisition_max_z_error_m = float(
            get('target_reacquisition_max_z_error_m').value)
        self.target_reacquisition_max_pixel_jump_px = float(
            get('target_reacquisition_max_pixel_jump_px').value)
        self.target_reacquisition_average_confidence = float(
            get('target_reacquisition_average_confidence').value)
        self.book_depth_min_valid_pixels = int(
            get('book_depth_min_valid_pixels').value)
        self.book_depth_min_valid_fraction = float(
            get('book_depth_min_valid_fraction').value)
        self.book_geometry_timeout_sec = float(
            get('book_geometry_timeout_sec').value)
        self.pregrasp_offset_m = float(get('pregrasp_offset_m').value)
        self.tentative_grasp_offset_m = float(
            get('tentative_grasp_offset_m').value)
        self.retreat_offset_m = float(get('retreat_offset_m').value)
        self.pregrasp_min_reach_m = float(get('pregrasp_min_reach_m').value)
        self.pregrasp_max_reach_m = float(get('pregrasp_max_reach_m').value)
        self.book_result_hold_sec = float(get('book_result_hold_sec').value)

    def _validate_day4_parameters(self) -> None:
        if self.day4_stop_after not in {
            'preapproach_visibility', 'perception', 'geometry', 'pregrasp'
        }:
            raise ValueError(
                'day4_stop_after must be preapproach_visibility, perception, '
                'geometry, or pregrasp')
        if not self.book_perception_enabled:
            raise ValueError('Day 4 requires book_perception_enabled:=true')
        if (
            self.day4_stop_after != 'preapproach_visibility'
            and not self.preapproach_scan_calibrated
        ):
            raise ValueError(
                'preapproach_scan_calibrated is false; first run the stationary '
                'preapproach_visibility experiment and apply its recommended '
                'minimal head-pose sequence')
        for name, value in (
            ('passive_mapping_process_period_sec', self.passive_mapping_process_period_sec),
            ('preapproach_scan_head_motion_sec', self.preapproach_scan_head_motion_sec),
            ('preapproach_scan_head_ready_timeout_sec', self.preapproach_scan_head_ready_timeout_sec),
            ('preapproach_scan_min_dwell_sec', self.preapproach_scan_min_dwell_sec),
            ('preapproach_scan_pose_timeout_sec', self.preapproach_scan_pose_timeout_sec),
            ('preapproach_scan_total_timeout_sec', self.preapproach_scan_total_timeout_sec),
            ('preapproach_scan_restore_timeout_sec', self.preapproach_scan_restore_timeout_sec),
            ('fallback_scan_total_timeout_sec', self.fallback_scan_total_timeout_sec),
            ('fallback_scan_pose_timeout_sec', self.fallback_scan_pose_timeout_sec),
            ('target_head_motion_sec', self.target_head_motion_sec),
            ('target_head_ready_timeout_sec', self.target_head_ready_timeout_sec),
            ('target_reacquisition_pose_timeout_sec', self.target_reacquisition_pose_timeout_sec),
            ('target_reacquisition_total_timeout_sec', self.target_reacquisition_total_timeout_sec),
            ('book_geometry_timeout_sec', self.book_geometry_timeout_sec),
        ):
            if value <= 0.0:
                raise ValueError(f'{name} must be positive')
        if self.passive_mapping_max_samples < 20:
            raise ValueError('passive_mapping_max_samples is too small')
        if self.preapproach_scan_settle_samples < 2:
            raise ValueError('preapproach_scan_settle_samples must be at least 2')
        if self.preapproach_scan_min_frames_per_pose < 2:
            raise ValueError(
                'preapproach_scan_min_frames_per_pose must be at least 2')
        if not 0.0 < self.preapproach_scan_head_tolerance_rad <= 0.20:
            raise ValueError('preapproach scan head tolerance is invalid')
        if not 0.0 < self.preapproach_scan_base_translation_tolerance_m <= 0.10:
            raise ValueError('preapproach base translation tolerance is invalid')
        if not 0.0 < self.preapproach_scan_base_yaw_tolerance_deg <= 5.0:
            raise ValueError('preapproach base yaw tolerance is invalid')
        if not self.preapproach_scan_profile_head_tilt_sequence_rad:
            raise ValueError('preapproach profile head sequence cannot be empty')
        if not self.preapproach_scan_head_tilt_sequence_rad:
            raise ValueError('preapproach competition head sequence cannot be empty')
        if not self.fallback_head_tilt_sequence_rad:
            raise ValueError('fallback head sequence cannot be empty')
        for sequence_name, values in (
            ('preapproach_scan_profile_head_tilt_sequence_rad',
             self.preapproach_scan_profile_head_tilt_sequence_rad),
            ('preapproach_scan_head_tilt_sequence_rad',
             self.preapproach_scan_head_tilt_sequence_rad),
            ('fallback_head_tilt_sequence_rad', self.fallback_head_tilt_sequence_rad),
        ):
            if any(
                not math.isfinite(value) or not -1.0 <= value <= 0.35
                for value in values
            ):
                raise ValueError(f'{sequence_name} contains an invalid value')
        if not (
            0 < self.book_column_roi_min_half_width_px
            <= self.book_column_roi_max_half_width_px
        ):
            raise ValueError('book column ROI bounds are invalid')
        if self.book_mapping_max_column_lateral_error_m <= 0.0:
            raise ValueError('book mapping lateral gate must be positive')
        if self.book_depth_min_valid_pixels < 4:
            raise ValueError('book_depth_min_valid_pixels must be at least 4')
        if not 0.0 < self.book_depth_min_valid_fraction <= 1.0:
            raise ValueError('book_depth_min_valid_fraction must be in (0, 1]')
        if not 0.0 < self.pregrasp_min_reach_m < self.pregrasp_max_reach_m:
            raise ValueError('pregrasp reach interval is invalid')

    def _finish(self, passed: bool, reason: str) -> None:
        """Intercept successful Day 3 stand-off and enter close Day 4 work."""
        if (
            passed
            and not self.day4_started
            and self.state == 'VERIFY_APPROACH'
            and self.approach_outcome == 'safe_standoff'
            and self.approach_image_saved
        ):
            self.stop_base()
            self.day4_started = True
            self.day4_started_sim_sec = self.sim_time_sec
            self.day4_started_wall_monotonic = time.monotonic()
            self.day4_inherited_day3_reason = reason
            self.day3_approach_head_actual_tilt_snapshot_rad = (
                float(self.actual_head_tilt)
                if self.actual_head_tilt is not None else None
            )
            self.day4_outcome = 'running'

            if self.book_layout_confirmation is not None:
                self._publish_row_result(force=True)
                self._prepare_target_reacquisition()
                self._day4_transition(
                    'POSITION_TARGET_HEAD',
                    'Stationary pre-approach colour-to-row map already locked; '
                    'reacquiring only the requested colour at final stand-off.',
                )
                return

            # The close-range multi-pose mapper remains a fallback only.  It is
            # entered only when the primary stationary pre-approach scan failed
            # to lock a row map before unchanged Day 3 translation began.
            self.fallback_mapping_used = True
            self.fallback_started_sim_sec = self.sim_time_sec
            self.fallback_head_index = 0
            self._day4_transition(
                'POSITION_FALLBACK_HEAD',
                'Primary stationary pre-approach scan did not lock a row map; '
                'using the bounded close-range fallback while the base remains '
                'at the validated Day 3 stand-off.',
            )
            return
        super()._finish(passed, reason)

    def _day4_transition(self, state: str, detail: str) -> None:
        self._transition(state, detail)
        self.get_logger().info(f'[DAY4] STATE -> {state}: {detail}')

    def _depth_callback(self, msg: Image) -> None:
        before = self.depth_frames_total
        super()._depth_callback(msg)
        if (
            self.depth_frames_total > before
            and self.latest_depth_m is not None
            and self.latest_depth_stamp_sec is not None
        ):
            stamp = float(self.latest_depth_stamp_sec)
            if (
                not self.book_depth_history
                or abs(self.book_depth_history[-1][0] - stamp) > 1.0e-9
            ):
                self.book_depth_history.append((stamp, self.latest_depth_m.copy()))
                if len(self.book_depth_history) > self.book_depth_history_max_frames:
                    del self.book_depth_history[:-self.book_depth_history_max_frames]

    def _publish_debug(
        self,
        detections: Sequence[Any],
        new_confirmations: Sequence[Any],
        stamp_sec: float,
    ) -> None:
        super()._publish_debug(detections, new_confirmations, stamp_sec)
        if self.done or self.latest_frame_bgr is None:
            return
        stamp = float(stamp_sec)

        # Mapping is intentionally disabled while Day 3 is translating.  Only
        # settled frames from the base-stationary pre-approach dwell are used.
        if self.state == 'DWELL_PREAPPROACH_HEAD':
            self._process_mapping_frame(
                self.latest_frame_bgr,
                stamp,
                mapping_source='stationary_preapproach',
            )
            return

        if self.state == 'SCAN_FALLBACK_MAPPING':
            self._process_mapping_frame(
                self.latest_frame_bgr,
                stamp,
                mapping_source='close_range_fallback',
            )
            return

        if self.state == 'REACQUIRE_TARGET_BOOK':
            self._process_reacquisition_frame(self.latest_frame_bgr, stamp)

    def _mapping_clearance_bounds(self) -> Tuple[float, float]:
        if self.day4_stop_after == 'visibility':
            return (
                self.visibility_profile_start_clearance_m,
                self.visibility_profile_stop_clearance_m,
            )
        return (
            self.passive_mapping_start_clearance_m,
            self.passive_mapping_stop_clearance_m,
        )

    def _clearance_in_mapping_window(self) -> bool:
        if self.latest_lidar is None:
            return False
        start, stop = self._mapping_clearance_bounds()
        clearance = float(self.latest_lidar.robust_clearance_m)
        return stop <= clearance <= start

    def _locked_column_center_u(self) -> Optional[float]:
        if self.color_model is None:
            return None
        geometry = self._locked_target_geometry_now()
        if geometry is None:
            return float(self.color_model.cx)
        # Base +y is left while optical +x is image right.
        return float(
            self.color_model.cx
            - self.color_model.fx * math.tan(geometry.base_bearing_left_rad)
        )

    def _current_book_column_roi(self) -> Optional[BoundingBoxLike]:
        if (
            self.color_model is None
            or self.latest_lidar is None
            or self.frame_width is None
            or self.frame_height is None
        ):
            return None
        center_u = self._locked_column_center_u()
        try:
            return selected_column_roi(
                self.frame_width,
                self.frame_height,
                self.color_model.fx,
                self.color_model.cx,
                self.latest_lidar.robust_clearance_m,
                physical_width_m=self.book_column_physical_width_m,
                minimum_half_width_px=self.book_column_roi_min_half_width_px,
                maximum_half_width_px=self.book_column_roi_max_half_width_px,
                center_u=center_u,
            )
        except ValueError:
            return None

    def _nearest_book_depth(
        self,
        rgb_stamp_sec: float,
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], str]:
        if not self.book_depth_history:
            return None, None, None, 'raw-depth history is empty'
        stamp, depth = min(
            self.book_depth_history,
            key=lambda item: abs(item[0] - rgb_stamp_sec),
        )
        skew = abs(float(stamp) - float(rgb_stamp_sec))
        if skew > self.max_rgb_depth_skew:
            return None, stamp, skew, (
                f'nearest raw-depth frame skew {skew:.3f}s exceeds '
                f'{self.max_rgb_depth_skew:.3f}s'
            )
        age = self._age(stamp)
        if age is None or age > self.depth_stale_timeout:
            return None, stamp, skew, (
                f'nearest synchronized raw-depth frame is stale; age={age}'
            )
        return depth, stamp, skew, ''

    def _geometry_for_detection(
        self,
        frame_bgr: np.ndarray,
        detection: BookDetection,
        row: int,
        stamp_sec: float,
    ) -> Tuple[Optional[BookGeometry], str]:
        if self.color_model is None or self.depth_model is None:
            return None, 'RGB/depth CameraInfo is unavailable'
        depth, _, skew, reason = self._nearest_book_depth(stamp_sec)
        if depth is None or skew is None:
            return None, reason
        transform = self._lookup_transform_components(self.depth_model.frame_id)
        if transform is None:
            return None, 'depth optical frame to base_footprint TF unavailable'
        try:
            geometry = estimate_book_geometry(
                frame_bgr,
                detection,
                row,
                self.color_model,
                self.depth_model,
                depth,
                transform[0],
                transform[1],
                rgb_depth_stamp_skew_sec=skew,
                detector_settings=self.book_detector_settings,
                min_depth_m=self.depth_min_m,
                max_depth_m=self.depth_max_m,
                min_valid_pixels=self.book_depth_min_valid_pixels,
                min_valid_fraction=self.book_depth_min_valid_fraction,
            )
        except (ValueError, cv2.error) as exc:
            return None, str(exc)
        if geometry is None:
            return None, 'not enough coherent raw-depth pixels on colour candidate'
        if geometry.forward_m <= 0.20:
            return None, (
                f'candidate is not safely in front of base: '
                f'x={geometry.forward_m:.3f}m'
            )
        return geometry, ''

    def _base_xyz_to_odom(
        self,
        base_xyz: Sequence[float],
    ) -> Optional[Tuple[float, float, float]]:
        if self.current_odom_xy is None or self.current_yaw is None:
            return None
        x_base, y_base, z_base = [float(value) for value in base_xyz]
        robot_x, robot_y = self.current_odom_xy
        yaw = float(self.current_yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            float(robot_x + cos_yaw * x_base - sin_yaw * y_base),
            float(robot_y + sin_yaw * x_base + cos_yaw * y_base),
            float(z_base),
        )

    def _odom_xyz_to_base(
        self,
        odom_xyz: Sequence[float],
    ) -> Optional[Tuple[float, float, float]]:
        if self.current_odom_xy is None or self.current_yaw is None:
            return None
        x_odom, y_odom, z_odom = [float(value) for value in odom_xyz]
        robot_x, robot_y = self.current_odom_xy
        dx = x_odom - robot_x
        dy = y_odom - robot_y
        yaw = float(self.current_yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            float(cos_yaw * dx + sin_yaw * dy),
            float(-sin_yaw * dx + cos_yaw * dy),
            float(z_odom),
        )

    def _head_is_settled_for_mapping(self) -> bool:
        target = None
        tolerance = self.passive_mapping_head_tolerance_rad
        if self.state in self.PREAPPROACH_SCAN_STATES:
            target = self.preapproach_scan_head_target_rad
            tolerance = self.preapproach_scan_head_tolerance_rad
        elif self.passive_mapping_head_target_rad is not None:
            target = self.passive_mapping_head_target_rad
        if target is None:
            return True
        return (
            self.actual_head_tilt is not None
            and abs(float(self.actual_head_tilt) - float(target)) <= tolerance
        )

    def _process_mapping_frame(
        self,
        frame_bgr: np.ndarray,
        stamp_sec: float,
        mapping_source: str,
    ) -> None:
        # Historical approach-time mapping is intentionally disabled.
        if mapping_source == 'passive_approach':
            return
        if (
            mapping_source == 'stationary_preapproach'
            and not self._head_is_settled_for_mapping()
        ):
            return
        if (
            self.passive_mapping_last_process_stamp_sec is not None
            and stamp_sec - self.passive_mapping_last_process_stamp_sec
            < self.passive_mapping_process_period_sec
        ):
            return
        self.passive_mapping_last_process_stamp_sec = stamp_sec
        if not self.passive_mapping_started:
            self.passive_mapping_started = True
            self.passive_mapping_started_sim_sec = stamp_sec

        roi = self._current_book_column_roi()
        if roi is None or self.latest_lidar is None:
            return
        self.book_column_roi = roi
        self.book_processed_frames += 1
        self.passive_mapping_processed_frames += 1

        try:
            candidates_by_colour, diagnostics = self.book_detector.detect_colours(
                frame_bgr,
                roi,
                BOOK_COLOURS,
            )
        except (ValueError, cv2.error) as exc:
            self.book_latest_diagnostics = {'error': str(exc)}
            return

        locked_column = self._locked_target_geometry_now()
        accepted_colours: List[str] = []
        geometry_colours: List[str] = []
        rejected: Dict[str, str] = {}
        frame_index = self.book_processed_frames
        clearance = float(self.latest_lidar.robust_clearance_m)
        head_tilt = float(
            self.actual_head_tilt
            if self.actual_head_tilt is not None
            else self.approach_head_tilt_rad
        )

        for colour in BOOK_COLOURS:
            candidates = candidates_by_colour.get(colour, [])
            accepted = False
            for candidate in candidates[:3]:
                geometry, reason = self._geometry_for_detection(
                    frame_bgr,
                    candidate,
                    row=0,
                    stamp_sec=stamp_sec,
                )
                if geometry is None:
                    rejected[colour] = reason
                    continue
                geometry_colours.append(colour)
                if (
                    locked_column is not None
                    and abs(
                        geometry.lateral_left_m
                        - locked_column.lateral_left_m
                    ) > self.book_mapping_max_column_lateral_error_m
                ):
                    rejected[colour] = (
                        'candidate lies outside locked-column lateral envelope'
                    )
                    continue
                odom_xyz = self._base_xyz_to_odom(geometry.base_point_xyz_m)
                if odom_xyz is None:
                    rejected[colour] = 'odometry pose unavailable'
                    continue
                observation = SpatialBookObservation(
                    colour=colour,
                    stamp_sec=stamp_sec,
                    frame_index=frame_index,
                    head_tilt_rad=head_tilt,
                    lidar_clearance_m=clearance,
                    detection=candidate,
                    geometry=geometry,
                    odom_xyz_m=odom_xyz,
                    roi_xywh=tuple(roi.as_list()),
                )
                confirmation = self.book_map_accumulator.add(
                    observation,
                    mapping_source=mapping_source,
                )
                accepted_colours.append(colour)
                accepted = True
                if (
                    confirmation is not None
                    and self.book_layout_confirmation is None
                ):
                    self._accept_spatial_map(confirmation)
                break
            if not accepted and colour not in rejected and candidates:
                rejected[colour] = 'no candidate passed the geometry gate'

        sample = {
            'sim_stamp_sec': round(float(stamp_sec), 6),
            'state': self.state,
            'mapping_source': mapping_source,
            'lidar_clearance_m': round(clearance, 6),
            'head_tilt_rad': round(head_tilt, 6),
            'column_roi_xywh': roi.as_list(),
            'candidate_counts': diagnostics.get('candidate_counts', {}),
            'visible_colours': diagnostics.get('visible_colours', []),
            'geometry_colours': sorted(set(geometry_colours)),
            'accepted_colours': sorted(set(accepted_colours)),
            'rejected_reasons': rejected,
            'map_locked': self.book_layout_confirmation is not None,
        }
        self.passive_mapping_visibility_samples.append(sample)
        if len(self.passive_mapping_visibility_samples) > self.passive_mapping_max_samples:
            del self.passive_mapping_visibility_samples[0]
        self.book_latest_diagnostics = {
            'frame': sample,
            'support': self.book_map_accumulator.support_summary(),
            'last_accumulator_rejection': (
                self.book_map_accumulator.last_rejection_reason),
        }
        self._publish_book_debug({
            'mode': 'mapping',
            'sample': sample,
            'support': self.book_map_accumulator.support_summary(),
            'confirmed_map': (
                self.book_layout_confirmation.as_dict()
                if self.book_layout_confirmation is not None else None
            ),
        })

    def _accept_spatial_map(
        self,
        confirmation: ConfirmedSpatialBookMap,
    ) -> None:
        self.book_layout_confirmation = confirmation
        self.row_mapping_source = confirmation.mapping_source
        self.target_book_row = confirmation.row_for_colour(self.book_colour)
        self.target_book_locked_odom_xyz_m = (
            confirmation.locked_odom_xyz_for_colour(self.book_colour)
        )
        self.passive_mapping_lock_sim_sec = self.sim_time_sec
        self.passive_mapping_lock_lidar_clearance_m = (
            float(self.latest_lidar.robust_clearance_m)
            if self.latest_lidar is not None else None
        )
        self.passive_mapping_locked_during_approach = False
        self.preapproach_scan_map_locked_before_approach = (
            self.approach_started_sim_sec is None
            and confirmation.mapping_source == 'stationary_preapproach'
        )
        if self.day4_stop_after != 'preapproach_visibility':
            self._publish_row_result(force=True)
        self.get_logger().info(
            '[DAY4] MULTI-FRAME ROW MAP LOCKED: '
            f'source={confirmation.mapping_source} '
            f'rows={confirmation.row_colours_top_to_bottom} '
            f'target={self.book_colour} row={self.target_book_row} '
            f'min_frames_per_colour={confirmation.confirming_frames} '
            f'clearance={self.passive_mapping_lock_lidar_clearance_m}'
        )
        if self.state == 'SCAN_FALLBACK_MAPPING':
            self._prepare_target_reacquisition()
            self._day4_transition(
                'POSITION_TARGET_HEAD',
                'Fallback accumulator completed the row map; reacquiring only '
                'the requested colour in its locked row.',
            )

    def _publish_book_debug(self, payload: Dict[str, object]) -> None:
        if self.book_debug_pub.get_subscription_count() == 0:
            return
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.book_debug_pub.publish(message)

    def _publish_row_result(self, force: bool = False) -> None:
        if (
            self.target_book_row is None
            or self.day4_stop_after == 'preapproach_visibility'
        ):
            return
        now_wall = time.monotonic()
        if not force:
            if (
                self.sim_time_sec is not None
                and self.last_row_publish_sim_sec is not None
                and self.sim_time_sec - self.last_row_publish_sim_sec
                < self.publish_period_sec
            ):
                return
            if (
                self.sim_time_sec is None
                and self.last_row_publish_wall_monotonic is not None
                and now_wall - self.last_row_publish_wall_monotonic
                < self.publish_period_sec
            ):
                return
        message = Int32()
        message.data = int(self.target_book_row)
        self.row_pub.publish(message)
        self.row_publication_count += 1
        if self.first_row_publish_sim_sec is None:
            self.first_row_publish_sim_sec = self.sim_time_sec
        self.last_row_publish_sim_sec = self.sim_time_sec
        self.last_row_publish_wall_monotonic = now_wall
        if force or self.row_publication_count == 1:
            self.get_logger().info(
                f'[DAY4] Published row {self.target_book_row} on {self.ROW_TOPIC}'
            )

    def _head_trajectory(
        self,
        tilt_rad: float,
        duration_sec: float,
    ) -> JointTrajectory:
        message = JointTrajectory()
        message.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(self.head_pan), float(tilt_rad)]
        seconds = int(duration_sec)
        nanoseconds = int(round((duration_sec - seconds) * 1.0e9))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds
        message.points = [point]
        return message

    def _publish_head_without_base_command(
        self,
        tilt_rad: float,
        duration_sec: float,
        reason: str,
    ) -> None:
        # This method deliberately does not call stop_base() and never publishes
        # /cmd_vel. Day 3 remains the only base controller during its approach.
        self.head_pub.publish(self._head_trajectory(tilt_rad, duration_sec))
        self.passive_mapping_head_commands.append({
            'sim_stamp_sec': self.sim_time_sec,
            'target_tilt_rad': float(tilt_rad),
            'reason': reason,
            'cmd_vel_publications': 0,
        })

    def _passive_head_schedule_tick(self) -> None:
        """Deprecated: moving the head during Day 3 approach is forbidden."""
        return

    def _preapproach_profile_mode(self) -> bool:
        return self.day4_stop_after == 'preapproach_visibility'

    def _publish_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Audit every base command issued while the stationary scan is active."""
        if (
            getattr(self, 'preapproach_scan_started', False)
            and getattr(self, 'state', '') in self.PREAPPROACH_SCAN_STATES
        ):
            if max(abs(float(vx)), abs(float(vy)), abs(float(wz))) <= 1.0e-9:
                self.preapproach_scan_zero_cmd_vel_publications += 1
            else:
                self.preapproach_scan_nonzero_cmd_vel_publications += 1
        super()._publish_velocity(vx, vy, wz)

    def _hold_base_for_preapproach_scan(self) -> None:
        """Publish only zero velocity while the stationary scan owns the hold."""
        self.stop_base()

    def _preapproach_base_motion(self) -> Tuple[Optional[float], Optional[float]]:
        translation = None
        yaw_change_deg = None
        if (
            self.preapproach_scan_start_odom_xy is not None
            and self.current_odom_xy is not None
        ):
            translation = math.hypot(
                float(self.current_odom_xy[0])
                - float(self.preapproach_scan_start_odom_xy[0]),
                float(self.current_odom_xy[1])
                - float(self.preapproach_scan_start_odom_xy[1]),
            )
        if (
            self.preapproach_scan_start_yaw_rad is not None
            and self.current_yaw is not None
        ):
            from ku_sparcy_erc.mission_start import normalize_angle
            yaw_change_deg = abs(math.degrees(normalize_angle(
                float(self.current_yaw)
                - float(self.preapproach_scan_start_yaw_rad)
            )))
        return translation, yaw_change_deg

    def _preapproach_base_stationary(self) -> bool:
        translation, yaw_change_deg = self._preapproach_base_motion()
        if translation is not None:
            self.preapproach_scan_base_translation_m = float(translation)
        if yaw_change_deg is not None:
            self.preapproach_scan_base_yaw_change_deg = float(yaw_change_deg)
        return (
            translation is not None
            and yaw_change_deg is not None
            and translation
            <= self.preapproach_scan_base_translation_tolerance_m
            and yaw_change_deg
            <= self.preapproach_scan_base_yaw_tolerance_deg
        )

    def _start_approach(self, geometry: Any) -> None:
        """Intercept Day 3 once arms are ready; scan before translation."""
        if self.preapproach_scan_complete:
            super()._start_approach(geometry)
            return
        if self.sim_time_sec is None:
            self._finish(False, 'Simulation clock unavailable before scan')
            return
        if not self.column_lock_active:
            self._finish(False, 'Pre-approach scan entered before column lock')
            return
        if not self.travel_arm_pose_ready:
            self._finish(False, 'Pre-approach scan entered before travel arms ready')
            return
        if self.current_odom_xy is None or self.current_yaw is None:
            self._finish(False, 'Odometry unavailable before pre-approach scan')
            return
        self.pending_approach_geometry = geometry
        self.preapproach_scan_started = True
        self.preapproach_scan_profile_only = self._preapproach_profile_mode()
        self.preapproach_scan_started_sim_sec = float(self.sim_time_sec)
        self.preapproach_scan_start_odom_xy = tuple(self.current_odom_xy)
        self.preapproach_scan_start_yaw_rad = float(self.current_yaw)
        self.preapproach_scan_start_lidar_clearance_m = (
            float(self.latest_lidar.robust_clearance_m)
            if self.latest_lidar is not None else None
        )
        self.preapproach_scan_sequence_rad = list(
            self.preapproach_scan_profile_head_tilt_sequence_rad
            if self.preapproach_scan_profile_only
            else self.preapproach_scan_head_tilt_sequence_rad
        )
        self.preapproach_scan_index = 0
        self.preapproach_scan_attempts = []
        self.preapproach_scan_head_commands = []
        self.preapproach_scan_head_restored = False
        self.preapproach_scan_map_locked_before_approach = False
        self.book_map_accumulator.reset()
        self.book_layout_confirmation = None
        self.row_mapping_source = 'not_locked'
        self.target_book_row = None
        self.target_book_locked_odom_xyz_m = None
        self.passive_mapping_visibility_samples = []
        self.passive_mapping_processed_frames = 0
        self.passive_mapping_last_process_stamp_sec = None
        self.passive_mapping_started = False
        self._hold_base_for_preapproach_scan()
        self._day4_transition(
            'POSITION_PREAPPROACH_HEAD',
            'Column locked and both travel arms ready; holding the base '
            'stationary for bounded pre-approach row mapping.',
        )

    def _preapproach_total_timeout_exceeded(self) -> bool:
        elapsed = self._simulation_elapsed(self.preapproach_scan_started_sim_sec)
        return (
            elapsed is not None
            and elapsed > self.preapproach_scan_total_timeout_sec
        )

    def _position_preapproach_head_tick(self) -> None:
        self._hold_base_for_preapproach_scan()
        if self._preapproach_total_timeout_exceeded():
            self._finish(False, 'Stationary pre-approach scan exceeded total timeout')
            return
        if self.preapproach_scan_index >= len(self.preapproach_scan_sequence_rad):
            self._day4_transition(
                'RESTORE_PREAPPROACH_HEAD',
                'All configured stationary scan poses completed; restoring the '
                'validated Day 3 approach-head pose.',
            )
            return
        target = float(
            self.preapproach_scan_sequence_rad[self.preapproach_scan_index])
        self.head_pub.publish(self._head_trajectory(
            target, self.preapproach_scan_head_motion_sec))
        self.preapproach_scan_head_target_rad = target
        self.preapproach_scan_head_command_sim_sec = self.sim_time_sec
        self.preapproach_scan_frames_at_command = self.camera_frames_total
        self.preapproach_scan_settle_count = 0
        self.preapproach_scan_head_commands.append({
            'sim_stamp_sec': self.sim_time_sec,
            'pose_index': self.preapproach_scan_index,
            'target_tilt_rad': target,
            'reason': 'base-stationary pre-approach scan pose',
            'cmd_vel_mode': 'zero_only',
        })
        self._day4_transition(
            'WAIT_FOR_PREAPPROACH_HEAD',
            f'Waiting for stationary scan pose {self.preapproach_scan_index}: '
            f'tilt={target:+.3f}rad.',
        )

    def _wait_for_preapproach_head_tick(self) -> None:
        self._hold_base_for_preapproach_scan()
        if not self._preapproach_base_stationary():
            self._finish(False, 'Base moved beyond tolerance during pre-approach scan')
            return
        elapsed = self._simulation_elapsed(
            self.preapproach_scan_head_command_sim_sec)
        if elapsed is None:
            return
        if elapsed > self.preapproach_scan_head_ready_timeout_sec:
            self._finish(
                False,
                'Pre-approach head did not settle before simulation-time timeout; '
                f'target={self.preapproach_scan_head_target_rad} '
                f'actual={self.actual_head_tilt}',
            )
            return
        position_ok = self._head_is_settled_for_mapping()
        fresh_frames = (
            self.camera_frames_total
            >= self.preapproach_scan_frames_at_command + 2
        )
        if position_ok and fresh_frames:
            self.preapproach_scan_settle_count += 1
        else:
            self.preapproach_scan_settle_count = 0
        if self.preapproach_scan_settle_count < self.preapproach_scan_settle_samples:
            return
        self.preapproach_scan_dwell_started_sim_sec = self.sim_time_sec
        self.preapproach_scan_dwell_processed_start = (
            self.passive_mapping_processed_frames)
        self.preapproach_scan_dwell_sample_start = len(
            self.passive_mapping_visibility_samples)
        self._day4_transition(
            'DWELL_PREAPPROACH_HEAD',
            'Head is settled; accepting RGB/depth/TF colour observations while '
            'the base remains stationary.',
        )

    def _save_preapproach_pose_image(
        self,
        pose_index: int,
        target_tilt_rad: float,
        accepted_counts: Dict[str, int],
        geometry_counts: Dict[str, int],
        visible_counts: Dict[str, int],
    ) -> Optional[str]:
        """Save one live diagnostic image for a settled stationary pose."""
        if self.latest_frame_bgr is None:
            return None
        frame = self.latest_frame_bgr.copy()
        if self.book_column_roi is not None:
            roi = self.book_column_roi
            cv2.rectangle(
                frame,
                (int(roi.x), int(roi.y)),
                (int(roi.x2), int(roi.y2)),
                (255, 255, 255),
                2,
            )
        lines = [
            'KU SPARCy DAY 4 - STATIONARY PRE-APPROACH PROFILE',
            f'pose {pose_index}: target={target_tilt_rad:+.3f} rad '
            f'actual={self.actual_head_tilt if self.actual_head_tilt is not None else "unknown"}',
            f'visible: {visible_counts}',
            f'depth geometry: {geometry_counts}',
            f'accepted spatial tracks: {accepted_counts}',
            f'sim={self.sim_time_sec if self.sim_time_sec is not None else 0.0:.3f}',
        ]
        overlay_height = min(frame.shape[0], 20 + 22 * len(lines))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (10, 22 + 21 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        os.makedirs(self.image_output_dir, exist_ok=True)
        utc_now = datetime.now(timezone.utc)
        sim_stamp = self.sim_time_sec or 0.0
        filename = (
            f'day4_preapproach_pose_{pose_index}_'
            f'tilt_{target_tilt_rad:+.2f}_sim_{sim_stamp:012.3f}_'
            f'utc_{utc_now.strftime("%Y%m%dT%H%M%S_%fZ")}.png'
        ).replace('+', 'p').replace('-', 'm')
        final_path = os.path.join(self.image_output_dir, filename)
        temporary = final_path + '.tmp.png'
        if not cv2.imwrite(temporary, frame):
            return None
        os.replace(temporary, final_path)
        return final_path

    def _record_preapproach_pose_attempt(self) -> None:
        samples = self.passive_mapping_visibility_samples[
            self.preapproach_scan_dwell_sample_start:
        ]
        counts = {colour: 0 for colour in BOOK_COLOURS}
        geometry_counts = {colour: 0 for colour in BOOK_COLOURS}
        visible_counts = {colour: 0 for colour in BOOK_COLOURS}
        for sample in samples:
            for colour in sample.get('accepted_colours', []):
                if colour in counts:
                    counts[colour] += 1
            for colour in sample.get('geometry_colours', []):
                if colour in geometry_counts:
                    geometry_counts[colour] += 1
            for colour in sample.get('visible_colours', []):
                if colour in visible_counts:
                    visible_counts[colour] += 1
        elapsed = self._simulation_elapsed(
            self.preapproach_scan_dwell_started_sim_sec)
        target_tilt = float(self.preapproach_scan_head_target_rad or 0.0)
        evidence_path = self._save_preapproach_pose_image(
            self.preapproach_scan_index,
            target_tilt,
            counts,
            geometry_counts,
            visible_counts,
        )
        self.preapproach_scan_attempts.append({
            'pose_index': int(self.preapproach_scan_index),
            'target_tilt_rad': target_tilt,
            'actual_tilt_rad': (
                float(self.actual_head_tilt)
                if self.actual_head_tilt is not None else None),
            'settled': self._head_is_settled_for_mapping(),
            'dwell_sim_sec': float(elapsed or 0.0),
            'processed_frames': int(
                self.passive_mapping_processed_frames
                - self.preapproach_scan_dwell_processed_start),
            'visible_frame_counts': visible_counts,
            'geometry_frame_counts': geometry_counts,
            'accepted_frame_counts': counts,
            'sample_count': len(samples),
            'evidence_image_path': evidence_path,
            'map_locked_by_end_of_pose': (
                self.book_layout_confirmation is not None),
            'support_after_pose': self.book_map_accumulator.support_summary(),
        })

    def _dwell_preapproach_head_tick(self) -> None:
        self._hold_base_for_preapproach_scan()
        if not self._preapproach_base_stationary():
            self._finish(False, 'Base moved beyond tolerance during scan dwell')
            return
        if not self._head_is_settled_for_mapping():
            self._finish(False, 'Head left tolerance during scan dwell')
            return
        elapsed = self._simulation_elapsed(
            self.preapproach_scan_dwell_started_sim_sec)
        if elapsed is None:
            return
        processed = (
            self.passive_mapping_processed_frames
            - self.preapproach_scan_dwell_processed_start
        )
        ready = (
            elapsed >= self.preapproach_scan_min_dwell_sec
            and processed >= self.preapproach_scan_min_frames_per_pose
        )
        if ready:
            self._record_preapproach_pose_attempt()
            # Experimental visibility mode always completes all profile poses.
            # Competition mode may stop as soon as a full spatial row map locks.
            if (
                self.book_layout_confirmation is not None
                and not self.preapproach_scan_profile_only
            ):
                self._day4_transition(
                    'RESTORE_PREAPPROACH_HEAD',
                    'Spatial row map locked; ending the stationary scan early '
                    'and restoring the validated Day 3 head pose.',
                )
                return
            self.preapproach_scan_index += 1
            self._day4_transition(
                'POSITION_PREAPPROACH_HEAD',
                'Stationary dwell completed; commanding the next bounded head '
                'pose without moving the base.',
            )
            return
        if elapsed > self.preapproach_scan_pose_timeout_sec:
            self._finish(
                False,
                'Pre-approach scan pose produced too few settled RGB/depth '
                f'observations; processed={processed}',
            )

    def _restore_preapproach_head_tick(self) -> None:
        self._hold_base_for_preapproach_scan()
        target = float(self.approach_head_tilt_rad)
        self.head_pub.publish(self._head_trajectory(
            target, self.preapproach_scan_head_motion_sec))
        self.preapproach_scan_head_target_rad = target
        self.preapproach_scan_restore_command_sim_sec = self.sim_time_sec
        self.preapproach_scan_restore_frames_at_command = self.camera_frames_total
        self.preapproach_scan_restore_settle_count = 0
        self.preapproach_scan_head_commands.append({
            'sim_stamp_sec': self.sim_time_sec,
            'pose_index': None,
            'target_tilt_rad': target,
            'reason': 'restore validated Day 3 approach-head pose before motion',
            'cmd_vel_mode': 'zero_only',
        })
        self._day4_transition(
            'WAIT_FOR_PREAPPROACH_RESTORE',
            f'Restoring and verifying Day 3 head tilt {target:+.3f}rad before '
            'any shelf translation.',
        )

    def _wait_for_preapproach_restore_tick(self) -> None:
        self._hold_base_for_preapproach_scan()
        if not self._preapproach_base_stationary():
            self._finish(False, 'Base moved beyond tolerance while restoring head')
            return
        elapsed = self._simulation_elapsed(
            self.preapproach_scan_restore_command_sim_sec)
        if elapsed is None:
            return
        if elapsed > self.preapproach_scan_restore_timeout_sec:
            self._finish(False, 'Validated Day 3 head pose was not restored in time')
            return
        position_ok = self._head_is_settled_for_mapping()
        fresh_frames = (
            self.camera_frames_total
            >= self.preapproach_scan_restore_frames_at_command + 2
        )
        if position_ok and fresh_frames:
            self.preapproach_scan_restore_settle_count += 1
        else:
            self.preapproach_scan_restore_settle_count = 0
        if (
            self.preapproach_scan_restore_settle_count
            < self.preapproach_scan_settle_samples
        ):
            return
        self.preapproach_scan_head_restored = True
        self.preapproach_scan_completed_sim_sec = self.sim_time_sec
        self.preapproach_scan_end_odom_xy = (
            tuple(self.current_odom_xy)
            if self.current_odom_xy is not None else None)
        self.preapproach_scan_end_yaw_rad = (
            float(self.current_yaw)
            if self.current_yaw is not None else None)
        self._preapproach_base_stationary()
        if self.preapproach_scan_profile_only:
            self.book_verification_started_sim_sec = self.sim_time_sec
            self._day4_transition(
                'VERIFY_PREAPPROACH_VISIBILITY',
                'Stationary profile complete; preserving all pose-wise evidence '
                'for the strict analyzer without starting Day 3 translation.',
            )
            return
        self.preapproach_scan_complete = True
        self.preapproach_scan_approach_started_sim_sec = self.sim_time_sec
        geometry = self.pending_approach_geometry
        if geometry is None:
            self._finish(False, 'Pending Day 3 approach geometry was lost')
            return
        # Bypass this subclass override and enter the frozen Day 3 approach.
        Day3Mission._start_approach(self, geometry)

    def _verify_preapproach_visibility_tick(self) -> None:
        self._hold_base_for_preapproach_scan()
        elapsed = self._simulation_elapsed(self.book_verification_started_sim_sec)
        if elapsed is None or elapsed < self.book_result_hold_sec:
            return
        if len(self.preapproach_scan_attempts) != len(
            self.preapproach_scan_sequence_rad
        ):
            super()._finish(
                False,
                'Pre-approach visibility profile did not complete every pose',
            )
            return
        if not self.preapproach_scan_head_restored:
            super()._finish(False, 'Pre-approach profile did not restore head')
            return
        if not self._preapproach_base_stationary():
            super()._finish(False, 'Base moved during pre-approach visibility profile')
            return
        self.day4_outcome = 'preapproach_visibility_profile_only'
        super()._finish(
            True,
            'Base-stationary pre-approach visibility profile completed; the '
            'strict analyzer must decide whether a row map and minimal pose '
            'sequence are acceptable.',
        )

    def _position_fallback_head_tick(self) -> None:
        self.stop_base()
        if self.fallback_head_index >= len(self.fallback_head_tilt_sequence_rad):
            self.day4_failure_classification = 'book_mapping_failure'
            self._finish(
                False,
                'Stationary pre-approach mapping and bounded close-range fallback '
                'were exhausted without four spatially confirmed colour tracks',
            )
            return
        target = self.fallback_head_tilt_sequence_rad[self.fallback_head_index]
        self.head_pub.publish(self._head_trajectory(target, self.target_head_motion_sec))
        self.fallback_head_command_sim_sec = self.sim_time_sec
        self.fallback_head_frames_at_command = self.camera_frames_total
        self.book_scan_attempts.append({
            'mode': 'close_range_fallback_mapping',
            'index': self.fallback_head_index,
            'target_tilt_rad': target,
            'command_sim_sec': self.sim_time_sec,
            'outcome': 'commanded',
        })
        self._day4_transition(
            'WAIT_FOR_FALLBACK_HEAD',
            'Waiting for bounded fallback head pose and fresh RGB frames.',
        )

    def _wait_for_fallback_head_tick(self) -> None:
        self.stop_base()
        if self.fallback_head_command_sim_sec is None:
            self._finish(False, 'Fallback head wait entered before command')
            return
        elapsed = self._simulation_elapsed(self.fallback_head_command_sim_sec)
        if elapsed is None:
            return
        target = self.fallback_head_tilt_sequence_rad[self.fallback_head_index]
        position_ok = (
            self.actual_head_tilt is not None
            and abs(float(self.actual_head_tilt) - target)
            <= self.target_head_tolerance_rad
        )
        frames_ok = self.camera_frames_total >= self.fallback_head_frames_at_command + 2
        if position_ok and frames_ok:
            self.fallback_head_pose_started_sim_sec = self.sim_time_sec
            self._day4_transition(
                'SCAN_FALLBACK_MAPPING',
                'Accumulating independent colours at this pose; a single '
                'four-colour frame is not required.',
            )
            return
        if elapsed > self.target_head_ready_timeout_sec:
            self._finish(
                False,
                'Fallback head failed to settle before simulation-time timeout',
            )

    def _scan_fallback_mapping_tick(self) -> None:
        self.stop_base()
        self._publish_row_result()
        if self.book_layout_confirmation is not None:
            return
        total = self._simulation_elapsed(self.fallback_started_sim_sec)
        if total is not None and total > self.fallback_scan_total_timeout_sec:
            self._finish(
                False,
                'Close-range fallback mapping exceeded total simulation-time timeout',
            )
            return
        elapsed = self._simulation_elapsed(self.fallback_head_pose_started_sim_sec)
        if elapsed is None:
            return
        if elapsed > self.fallback_scan_pose_timeout_sec:
            if self.book_scan_attempts:
                self.book_scan_attempts[-1]['outcome'] = 'partial_support_only'
                self.book_scan_attempts[-1]['support'] = (
                    self.book_map_accumulator.support_summary())
            self.fallback_head_index += 1
            self._day4_transition(
                'POSITION_FALLBACK_HEAD',
                'Moving to the next fallback pose while retaining all '
                'spatially consistent observations.',
            )

    def _locked_book_base_xyz_now(self) -> Optional[Tuple[float, float, float]]:
        if self.target_book_locked_odom_xyz_m is None:
            return None
        return self._odom_xyz_to_base(self.target_book_locked_odom_xyz_m)

    def _predict_target_head_tilt(self) -> float:
        current = float(
            self.actual_head_tilt
            if self.actual_head_tilt is not None
            else self.approach_head_tilt_rad
        )
        point_base = self._locked_book_base_xyz_now()
        if point_base is None or self.depth_model is None:
            return max(-0.85, min(0.30, current))
        transform = self._lookup_transform_components(self.depth_model.frame_id)
        if transform is None:
            return max(-0.85, min(0.30, current))
        translation = np.asarray(transform[0], dtype=np.float64)
        rotation = quaternion_to_rotation_matrix(*transform[1])
        point = np.asarray(point_base, dtype=np.float64)
        optical = rotation.T @ (point - translation)
        if optical[2] <= 0.05:
            return max(-0.85, min(0.30, current))
        vertical_angle_down = math.atan2(float(optical[1]), float(optical[2]))
        predicted = current - vertical_angle_down
        return max(-0.85, min(0.30, predicted))

    @staticmethod
    def _unique_tilts(values: Sequence[float]) -> List[float]:
        result: List[float] = []
        for value in values:
            clamped = max(-0.85, min(0.30, float(value)))
            if all(abs(clamped - existing) > 0.025 for existing in result):
                result.append(clamped)
        return result

    def _prepare_target_reacquisition(self) -> None:
        if self.target_book_row is None or self.target_book_locked_odom_xyz_m is None:
            self._finish(False, 'Target reacquisition requested before row map lock')
            return
        predicted = self._predict_target_head_tilt()
        self.target_head_sequence_rad = self._unique_tilts([
            predicted,
            predicted - 0.12,
            predicted + 0.12,
            predicted - 0.24,
            predicted + 0.24,
        ])
        self.target_head_index = 0
        self.target_reacquisition_started_sim_sec = self.sim_time_sec
        self.target_reacquisition_filter = TargetBookReacquisitionFilter(
            self.book_colour,
            self.target_book_row,
            self.target_book_locked_odom_xyz_m,
            TargetReacquisitionSettings(
                required_frames=self.target_reacquisition_frames,
                window_sec=self.target_reacquisition_window_sec,
                minimum_span_sec=self.target_reacquisition_min_span_sec,
                maximum_locked_xy_error_m=(
                    self.target_reacquisition_max_xy_error_m),
                maximum_locked_z_error_m=(
                    self.target_reacquisition_max_z_error_m),
                maximum_pixel_jump_px=(
                    self.target_reacquisition_max_pixel_jump_px),
                minimum_average_confidence=(
                    self.target_reacquisition_average_confidence),
            ),
        )

    def _position_target_head_tick(self) -> None:
        self.stop_base()
        if not self.target_head_sequence_rad:
            self._prepare_target_reacquisition()
            if not self.target_head_sequence_rad:
                return
        if self.target_head_index >= len(self.target_head_sequence_rad):
            self.day4_failure_classification = 'target_reacquisition_failure'
            self._finish(
                False,
                'Requested colour could not be reacquired in its locked row '
                'after all bounded target head poses',
            )
            return
        target = self.target_head_sequence_rad[self.target_head_index]
        self.head_pub.publish(self._head_trajectory(target, self.target_head_motion_sec))
        self.target_head_command_sim_sec = self.sim_time_sec
        self.target_head_frames_at_command = self.camera_frames_total
        self.book_scan_attempts.append({
            'mode': 'requested_colour_reacquisition',
            'index': self.target_head_index,
            'target_tilt_rad': target,
            'command_sim_sec': self.sim_time_sec,
            'outcome': 'commanded',
        })
        self._day4_transition(
            'WAIT_FOR_TARGET_HEAD',
            'Waiting for the geometry-predicted target-book head pose.',
        )

    def _wait_for_target_head_tick(self) -> None:
        self.stop_base()
        if self.target_head_command_sim_sec is None:
            self._finish(False, 'Target-head wait entered before command')
            return
        elapsed = self._simulation_elapsed(self.target_head_command_sim_sec)
        if elapsed is None:
            return
        target = self.target_head_sequence_rad[self.target_head_index]
        position_ok = (
            self.actual_head_tilt is not None
            and abs(float(self.actual_head_tilt) - target)
            <= self.target_head_tolerance_rad
        )
        frames_ok = self.camera_frames_total >= self.target_head_frames_at_command + 2
        if position_ok and frames_ok:
            self.target_head_pose_started_sim_sec = self.sim_time_sec
            if self.target_reacquisition_filter is not None:
                self.target_reacquisition_filter.reset()
            self._day4_transition(
                'REACQUIRE_TARGET_BOOK',
                'Reacquiring only the requested colour near its locked '
                'odometry/row position.',
            )
            return
        if elapsed > self.target_head_ready_timeout_sec:
            self._finish(
                False,
                'Target-book head failed to settle before simulation-time timeout',
            )

    def _process_reacquisition_frame(
        self,
        frame_bgr: np.ndarray,
        stamp_sec: float,
    ) -> None:
        if self.target_reacquisition_filter is None or self.latest_lidar is None:
            return
        if (
            self.passive_mapping_last_process_stamp_sec is not None
            and stamp_sec - self.passive_mapping_last_process_stamp_sec
            < self.passive_mapping_process_period_sec
        ):
            return
        self.passive_mapping_last_process_stamp_sec = stamp_sec
        roi = self._current_book_column_roi()
        if roi is None:
            return
        self.book_column_roi = roi
        self.book_processed_frames += 1
        self.target_reacquisition_processed_frames += 1
        try:
            candidates_by_colour, diagnostics = self.book_detector.detect_colours(
                frame_bgr,
                roi,
                [self.book_colour],
            )
        except (ValueError, cv2.error) as exc:
            self.target_reacquisition_diagnostics = {'error': str(exc)}
            return
        frame_index = self.book_processed_frames
        accepted = 0
        rejection_reasons: List[str] = []
        for candidate in candidates_by_colour.get(self.book_colour, [])[:3]:
            geometry, reason = self._geometry_for_detection(
                frame_bgr,
                candidate,
                row=int(self.target_book_row or 0),
                stamp_sec=stamp_sec,
            )
            if geometry is None:
                rejection_reasons.append(reason)
                continue
            odom_xyz = self._base_xyz_to_odom(geometry.base_point_xyz_m)
            if odom_xyz is None:
                rejection_reasons.append('odometry pose unavailable')
                continue
            observation = SpatialBookObservation(
                colour=self.book_colour,
                stamp_sec=stamp_sec,
                frame_index=frame_index,
                head_tilt_rad=float(self.actual_head_tilt or 0.0),
                lidar_clearance_m=float(self.latest_lidar.robust_clearance_m),
                detection=candidate,
                geometry=geometry,
                odom_xyz_m=odom_xyz,
                roi_xywh=tuple(roi.as_list()),
            )
            confirmation = self.target_reacquisition_filter.update(
                observation,
                current_stamp_sec=stamp_sec,
            )
            accepted += 1
            if confirmation is not None:
                self._accept_target_reacquisition(
                    confirmation,
                    observation,
                    frame_bgr,
                )
                break
        self.target_reacquisition_diagnostics = {
            'sim_stamp_sec': round(stamp_sec, 6),
            'head_tilt_rad': self.actual_head_tilt,
            'column_roi_xywh': roi.as_list(),
            'detector': diagnostics,
            'geometry_candidates_accepted': accepted,
            'rejection_reasons': rejection_reasons,
            'filter_rejections': self.target_reacquisition_filter.rejection_count,
            'filter_last_rejection': (
                self.target_reacquisition_filter.last_rejection_reason),
        }
        self._publish_book_debug({
            'mode': 'target_reacquisition',
            'diagnostics': self.target_reacquisition_diagnostics,
            'confirmed': (
                self.target_reacquisition_confirmation.as_dict()
                if self.target_reacquisition_confirmation is not None else None
            ),
        })

    def _accept_target_reacquisition(
        self,
        confirmation: ReacquiredTargetBook,
        current_observation: SpatialBookObservation,
        frame_bgr: np.ndarray,
    ) -> None:
        if self.target_reacquisition_confirmation is not None:
            return
        self.target_reacquisition_confirmation = confirmation
        # Evidence must box the current live frame, not a representative bbox
        # copied from a different frame in the temporal history.
        self.target_book_detection = current_observation.detection
        self.target_book_geometry = current_observation.geometry
        self.target_book_confirmation_frame_bgr = frame_bgr.copy()
        self.target_book_rgb_stamp_sec = current_observation.stamp_sec
        if self.book_scan_attempts:
            self.book_scan_attempts[-1]['outcome'] = 'target_reacquired'
            self.book_scan_attempts[-1]['confirmation_sim_sec'] = self.sim_time_sec
        try:
            self.target_book_evidence_path = self._save_book_image(
                frame_bgr,
                geometry=None,
                suffix='identification',
            )
            self.target_book_evidence_saved = True
        except (OSError, cv2.error) as exc:
            self._finish(False, f'Could not save live target-book evidence: {exc}')
            return
        self.get_logger().info(
            '[DAY4] TARGET BOOK REACQUIRED: '
            f'colour={self.book_colour} row={self.target_book_row} '
            f'frames={confirmation.confirming_frames} '
            f'confidence={confirmation.average_confidence:.3f}'
        )
        if self.day4_stop_after == 'perception':
            self.day4_outcome = 'row_identification_and_target_reacquisition'
            self.book_verification_started_sim_sec = self.sim_time_sec
            self._day4_transition(
                'VERIFY_BOOK_EVIDENCE',
                'Row was mapped before approach/fallback and requested colour '
                'was reacquired at close range.',
            )
            return
        self.book_geometry_started_sim_sec = self.sim_time_sec
        self._day4_transition(
            'ESTIMATE_BOOK_GEOMETRY',
            'Requested-colour reacquisition already contains synchronized '
            'raw-depth/TF geometry; saving final geometry evidence.',
        )

    def _reacquire_target_book_tick(self) -> None:
        self.stop_base()
        self._publish_row_result()
        if self.target_reacquisition_confirmation is not None:
            return
        total = self._simulation_elapsed(self.target_reacquisition_started_sim_sec)
        if total is not None and total > self.target_reacquisition_total_timeout_sec:
            self._finish(
                False,
                'Requested-colour close-range reacquisition exceeded total timeout',
            )
            return
        elapsed = self._simulation_elapsed(self.target_head_pose_started_sim_sec)
        if elapsed is None:
            return
        if elapsed > self.target_reacquisition_pose_timeout_sec:
            if self.book_scan_attempts:
                self.book_scan_attempts[-1]['outcome'] = 'not_reacquired'
                self.book_scan_attempts[-1]['diagnostics'] = (
                    self.target_reacquisition_diagnostics)
            self.target_head_index += 1
            self._day4_transition(
                'POSITION_TARGET_HEAD',
                'Requested colour was not confirmed at this predicted pose; '
                'trying the next bounded offset.',
            )

    def _estimate_book_geometry_tick(self) -> None:
        self.stop_base()
        self._publish_row_result()
        if (
            self.target_book_geometry is None
            or self.target_book_detection is None
            or self.target_book_confirmation_frame_bgr is None
        ):
            self._finish(False, 'Book geometry state is missing reacquired inputs')
            return
        if self.target_book_geometry_image_path is None:
            try:
                self.target_book_geometry_image_path = self._save_book_image(
                    self.target_book_confirmation_frame_bgr,
                    geometry=self.target_book_geometry,
                    suffix='geometry',
                )
            except (OSError, cv2.error) as exc:
                self._finish(
                    False,
                    f'Could not save target-book geometry image: {exc}',
                )
                return
        if self.day4_stop_after == 'geometry':
            self.day4_outcome = 'book_geometry_ready_after_reacquisition'
            self.book_verification_started_sim_sec = self.sim_time_sec
            self._day4_transition(
                'VERIFY_BOOK_EVIDENCE',
                'Requested row, close target reacquisition, and precise 3-D '
                'book geometry validated.',
            )
            return
        self._day4_transition(
            'SELECT_MANIPULATION_ARM',
            'Precise close-range book point established; selecting one arm '
            'without commanding manipulation motion.',
        )

    def _select_manipulation_arm_tick(self) -> None:
        self.stop_base()
        self._publish_row_result()
        if self.target_book_geometry is None:
            self._finish(False, 'Arm selection entered without book geometry')
            return
        shoulder_positions: Dict[str, Tuple[float, float, float]] = {}
        for arm in ('left', 'right'):
            transform = self._lookup_transform_components(f'arm_{arm}_1_link')
            if transform is not None:
                shoulder_positions[arm] = transform[0]
        try:
            plan = build_pregrasp_plan(
                self.target_book_geometry,
                shoulder_positions,
                pregrasp_offset_m=self.pregrasp_offset_m,
                tentative_contact_offset_m=self.tentative_grasp_offset_m,
                retreat_offset_m=self.retreat_offset_m,
                minimum_reach_m=self.pregrasp_min_reach_m,
                maximum_reach_m=self.pregrasp_max_reach_m,
            )
        except ValueError as exc:
            self._finish(False, f'Pre-grasp geometry failed: {exc}')
            return
        self.pregrasp_plan = plan
        if not plan.geometric_screen_passed:
            self._finish(
                False,
                'No arm passed the conservative geometry-only pre-grasp '
                'reachability screen; no arm motion was attempted',
            )
            return
        self.day4_outcome = 'pregrasp_geometry_ready'
        self.book_verification_started_sim_sec = self.sim_time_sec
        self._day4_transition(
            'VERIFY_BOOK_EVIDENCE',
            f'{plan.selected_arm} arm selected geometrically; staged IK and '
            'manipulation remain separate from solution.launch.py.',
        )

    def _verify_visibility_profile_tick(self) -> None:
        self._finish(
            False,
            'Deprecated approach-time visibility state was entered; use '
            'preapproach_visibility mode instead',
        )

    def _verify_book_evidence_tick(self) -> None:
        self.stop_base()
        self._publish_row_result()
        elapsed = self._simulation_elapsed(self.book_verification_started_sim_sec)
        if elapsed is None or elapsed < self.book_result_hold_sec:
            return
        if self.book_layout_confirmation is None:
            self._finish(False, 'Multi-frame colour-to-row map is missing')
            return
        if self.target_book_row is None or not 1 <= self.target_book_row <= 4:
            self._finish(False, 'Target shelf row is invalid')
            return
        if self.row_publication_count < 1:
            self._finish(False, 'Official target-row result was not published')
            return
        if self.target_reacquisition_confirmation is None:
            self._finish(False, 'Requested colour was not reacquired at close range')
            return
        if not self.target_book_evidence_saved:
            self._finish(False, 'Live close-range target-book evidence is missing')
            return
        if self.day4_stop_after in {'geometry', 'pregrasp'}:
            if self.target_book_geometry is None:
                self._finish(False, 'Target-book 3-D geometry is missing')
                return
        if self.day4_stop_after == 'pregrasp':
            if self.pregrasp_plan is None or not self.pregrasp_plan.geometric_screen_passed:
                self._finish(False, 'Pre-grasp geometric screen did not pass')
                return
        self._finish(
            True,
            'Requested book row mapped before approach without a single-frame requirement, '
            'official row published, requested colour reacquired at close '
            'range, and safe non-executing pre-grasp geometry validated',
        )

    def _save_book_image(
        self,
        frame_bgr: np.ndarray,
        geometry: Optional[BookGeometry],
        suffix: str,
    ) -> str:
        if (
            self.book_layout_confirmation is None
            or self.target_book_row is None
            or self.book_column_roi is None
            or self.target_book_detection is None
        ):
            raise OSError('book evidence prerequisites are missing')
        os.makedirs(self.image_output_dir, exist_ok=True)
        utc_now = datetime.now(timezone.utc)
        sim_stamp = self.sim_time_sec or 0.0
        filename = (
            f'target_book_{self.book_colour}_row_{self.target_book_row}_'
            f'{suffix}_sim_{sim_stamp:012.3f}_'
            f'utc_{utc_now.strftime("%Y%m%dT%H%M%S_%fZ")}.png'
        )
        final_path = os.path.join(self.image_output_dir, filename)
        temporary = final_path + '.tmp.png'
        support = {
            colour: int(details['dominant_cluster_frames'])
            for colour, details in self.book_map_accumulator.support_summary().items()
        }
        reacquisition_frames = (
            self.target_reacquisition_confirmation.confirming_frames
            if self.target_reacquisition_confirmation is not None
            else 0
        )
        lines = [
            'KU SPARCy ERC 2026 - DAY 4 LIVE TARGET-BOOK EVIDENCE',
            f'selected shelf marker: {self.shelf_column} | '
            f'target colour: {self.book_colour}',
            f'locked target row: {self.target_book_row} | '
            f'mapping source: {self.row_mapping_source}',
            'row mapping: stationary pre-approach multi-frame 3-D accumulation; '
            'single-frame layout NOT required',
            f'close reacquisition frames: {reacquisition_frames}',
            f'head tilt: '
            f'{self.actual_head_tilt if self.actual_head_tilt is not None else "unknown"}',
        ]
        if geometry is not None:
            lines.extend([
                f'book base xyz: ({geometry.forward_m:.3f}, '
                f'{geometry.lateral_left_m:+.3f}, {geometry.height_m:.3f}) m',
                f'RGB/depth skew: {geometry.rgb_depth_stamp_skew_sec:.3f} s | '
                f'depth pixels: {geometry.depth.valid_pixels}',
            ])
        lines.extend([
            f'sim timestamp: {sim_stamp:.6f} s',
            f'UTC timestamp: {utc_now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}',
            f'Day 3 base remained stopped until the stationary row scan restored {self.approach_head_tilt_rad:+.2f} rad',
        ])
        annotated = draw_target_book_evidence(
            frame_bgr,
            self.book_column_roi,
            self.target_book_detection,
            self.target_book_row,
            self.book_layout_confirmation.row_colours_top_to_bottom,
            support,
            lines,
        )
        if not cv2.imwrite(temporary, annotated):
            raise OSError(f'cv2.imwrite returned false for {temporary}')
        os.replace(temporary, final_path)
        return final_path

    def _control_tick(self) -> None:
        if self.done:
            return
        self._publish_row_result()
        if self.state not in self.DAY4_STATES:
            # The moving-head-during-approach scheduler is intentionally gone.
            super()._control_tick()
            return
        if (
            self.sim_time_sec is not None
            and time.monotonic() - self.last_sim_clock_advance_wall_monotonic
            > self.clock_stall_wall_timeout_sec
        ):
            self._finish(
                False,
                'Gazebo /clock stopped advancing beyond wall-time watchdog',
            )
            return
        self._publish_column_result()
        if self.state == 'POSITION_PREAPPROACH_HEAD':
            self._position_preapproach_head_tick()
        elif self.state == 'WAIT_FOR_PREAPPROACH_HEAD':
            self._wait_for_preapproach_head_tick()
        elif self.state == 'DWELL_PREAPPROACH_HEAD':
            self._dwell_preapproach_head_tick()
        elif self.state == 'RESTORE_PREAPPROACH_HEAD':
            self._restore_preapproach_head_tick()
        elif self.state == 'WAIT_FOR_PREAPPROACH_RESTORE':
            self._wait_for_preapproach_restore_tick()
        elif self.state == 'VERIFY_PREAPPROACH_VISIBILITY':
            self._verify_preapproach_visibility_tick()
        elif self.state == 'VERIFY_VISIBILITY_PROFILE':
            self._verify_visibility_profile_tick()
        elif self.state == 'POSITION_FALLBACK_HEAD':
            self._position_fallback_head_tick()
        elif self.state == 'WAIT_FOR_FALLBACK_HEAD':
            self._wait_for_fallback_head_tick()
        elif self.state == 'SCAN_FALLBACK_MAPPING':
            self._scan_fallback_mapping_tick()
        elif self.state == 'POSITION_TARGET_HEAD':
            self._position_target_head_tick()
        elif self.state == 'WAIT_FOR_TARGET_HEAD':
            self._wait_for_target_head_tick()
        elif self.state == 'REACQUIRE_TARGET_BOOK':
            self._reacquire_target_book_tick()
        elif self.state == 'ESTIMATE_BOOK_GEOMETRY':
            self._estimate_book_geometry_tick()
        elif self.state == 'SELECT_MANIPULATION_ARM':
            self._select_manipulation_arm_tick()
        elif self.state == 'VERIFY_BOOK_EVIDENCE':
            self._verify_book_evidence_tick()
        else:
            self._finish(False, f'Unknown Day 4 state: {self.state}')

    def _build_result(self, passed: bool, reason: str) -> Dict[str, Any]:
        current_head_tilt = self.actual_head_tilt
        if self.day3_approach_head_actual_tilt_snapshot_rad is not None:
            self.actual_head_tilt = self.day3_approach_head_actual_tilt_snapshot_rad
        try:
            result = super()._build_result(passed, reason)
        finally:
            self.actual_head_tilt = current_head_tilt
        now_wall = time.monotonic()
        day4_sim_duration = None
        if self.day4_started_sim_sec is not None and self.sim_time_sec is not None:
            day4_sim_duration = max(0.0, self.sim_time_sec - self.day4_started_sim_sec)
        day4_wall_duration = None
        if self.day4_started_wall_monotonic is not None:
            day4_wall_duration = max(
                0.0, now_wall - self.day4_started_wall_monotonic)
        target_detection = (
            self.target_book_detection.as_dict()
            if self.target_book_detection is not None else None
        )
        result.update({
            'day': 4,
            'day3_commit_baseline': (
                '9a9768acb824862f2dd4b8de9950ffed54828987'),
            'day3_mission_reused': True,
            'day4_started': self.day4_started,
            'day4_stop_after': self.day4_stop_after,
            'day4_outcome': self.day4_outcome,
            'day4_failure_classification': self.day4_failure_classification,
            'day4_inherited_day3_reason': self.day4_inherited_day3_reason,
            'day3_approach_head_actual_tilt_snapshot_rad': (
                self.day3_approach_head_actual_tilt_snapshot_rad),
            'day4_duration_sim_sec': day4_sim_duration,
            'day4_duration_wall_sec': day4_wall_duration,
            'official_upstream_required_commit': (
                '93554d4f9335b2ee3acb49c6b332611f6ad2a964'),
            'expected_official_book_spine_width_m': 0.02,
            'selected_physical_column_locked': self.column_lock_active,
            'selected_column_geometry_source': 'locked_column_odometry',
            'selected_column_forward_source': 'front_lidar',
            'day4_final_base_odom_xy': (
                list(self.current_odom_xy)
                if self.current_odom_xy is not None else None),
            'day4_final_base_yaw_rad': (
                float(self.current_yaw)
                if self.current_yaw is not None else None),
            'book_colour_requested': self.book_colour,
            'book_row_numbering': 'top_to_bottom_across_four_active_rows',
            'row_mapping_requires_single_frame': False,
            'row_mapping_source': self.row_mapping_source,
            'passive_mapping_window_calibrated': (
                self.passive_mapping_window_calibrated),
            'passive_mapping_start_clearance_m': (
                self.passive_mapping_start_clearance_m),
            'passive_mapping_stop_clearance_m': (
                self.passive_mapping_stop_clearance_m),
            'visibility_profile_start_clearance_m': (
                self.visibility_profile_start_clearance_m),
            'visibility_profile_stop_clearance_m': (
                self.visibility_profile_stop_clearance_m),
            'visibility_profile_bin_width_m': self.visibility_profile_bin_width_m,
            'passive_mapping_started': self.passive_mapping_started,
            'passive_mapping_started_sim_sec': (
                self.passive_mapping_started_sim_sec),
            'passive_mapping_processed_frames': (
                self.passive_mapping_processed_frames),
            'passive_mapping_visibility_samples': (
                self.passive_mapping_visibility_samples),
            'passive_mapping_support': (
                self.book_map_accumulator.support_summary()),
            'passive_mapping_lock_sim_sec': self.passive_mapping_lock_sim_sec,
            'passive_mapping_lock_lidar_clearance_m': (
                self.passive_mapping_lock_lidar_clearance_m),
            'passive_mapping_locked_during_approach': False,
            'passive_mapping_cmd_vel_publications': (
                self.passive_mapping_cmd_vel_publications),
            'passive_mapping_head_motion_enabled': (
                self.passive_mapping_head_motion_enabled),
            'passive_mapping_head_commands': self.passive_mapping_head_commands,
            'primary_row_mapping_strategy': 'stationary_preapproach_scan',
            'moving_head_during_approach_enabled': False,
            'preapproach_scan_calibrated': self.preapproach_scan_calibrated,
            'preapproach_scan_started': self.preapproach_scan_started,
            'preapproach_scan_complete': self.preapproach_scan_complete,
            'preapproach_scan_profile_only': self.preapproach_scan_profile_only,
            'preapproach_scan_started_sim_sec': self.preapproach_scan_started_sim_sec,
            'preapproach_scan_completed_sim_sec': self.preapproach_scan_completed_sim_sec,
            'preapproach_scan_approach_started_sim_sec': (
                self.preapproach_scan_approach_started_sim_sec),
            'preapproach_scan_profile_head_tilt_sequence_rad': list(
                self.preapproach_scan_profile_head_tilt_sequence_rad),
            'preapproach_scan_head_tilt_sequence_rad': list(
                self.preapproach_scan_head_tilt_sequence_rad),
            'preapproach_scan_sequence_used_rad': list(
                self.preapproach_scan_sequence_rad),
            'preapproach_scan_head_commands': self.preapproach_scan_head_commands,
            'preapproach_scan_attempts': self.preapproach_scan_attempts,
            'preapproach_scan_head_restored': self.preapproach_scan_head_restored,
            'preapproach_scan_map_locked_before_approach': (
                self.preapproach_scan_map_locked_before_approach),
            'preapproach_scan_start_lidar_clearance_m': (
                self.preapproach_scan_start_lidar_clearance_m),
            'preapproach_scan_start_odom_xy': (
                list(self.preapproach_scan_start_odom_xy)
                if self.preapproach_scan_start_odom_xy is not None else None),
            'preapproach_scan_end_odom_xy': (
                list(self.preapproach_scan_end_odom_xy)
                if self.preapproach_scan_end_odom_xy is not None else None),
            'preapproach_scan_base_translation_m': (
                self.preapproach_scan_base_translation_m),
            'preapproach_scan_base_yaw_change_deg': (
                self.preapproach_scan_base_yaw_change_deg),
            'preapproach_scan_zero_cmd_vel_publications': (
                self.preapproach_scan_zero_cmd_vel_publications),
            'preapproach_scan_nonzero_cmd_vel_publications': (
                self.preapproach_scan_nonzero_cmd_vel_publications),
            'preapproach_scan_head_motion_sec': (
                self.preapproach_scan_head_motion_sec),
            'preapproach_scan_head_ready_timeout_sec': (
                self.preapproach_scan_head_ready_timeout_sec),
            'preapproach_scan_head_tolerance_rad': (
                self.preapproach_scan_head_tolerance_rad),
            'preapproach_scan_settle_samples': (
                self.preapproach_scan_settle_samples),
            'preapproach_scan_min_dwell_sec': (
                self.preapproach_scan_min_dwell_sec),
            'preapproach_scan_min_frames_per_pose': (
                self.preapproach_scan_min_frames_per_pose),
            'preapproach_scan_total_timeout_sec': (
                self.preapproach_scan_total_timeout_sec),
            'preapproach_scan_restore_timeout_sec': (
                self.preapproach_scan_restore_timeout_sec),
            'preapproach_scan_base_translation_tolerance_m': (
                self.preapproach_scan_base_translation_tolerance_m),
            'preapproach_scan_base_yaw_tolerance_deg': (
                self.preapproach_scan_base_yaw_tolerance_deg),
            'preapproach_scan_observations_by_colour': {
                colour: [item.as_dict() for item in observations]
                for colour, observations
                in self.book_map_accumulator.observations.items()
            },
            'book_map_settings': {
                'required_frames_per_colour': self.book_map_frames_per_colour,
                'minimum_observation_span_sec': self.book_map_min_span_sec,
                'minimum_average_confidence': self.book_map_average_confidence,
                'maximum_cluster_xy_radius_m': self.book_map_cluster_xy_radius_m,
                'maximum_cluster_z_radius_m': self.book_map_cluster_z_radius_m,
                'maximum_track_xy_span_m': self.book_map_track_xy_span_m,
                'maximum_track_z_span_m': self.book_map_track_z_span_m,
                'minimum_row_separation_m': self.book_map_min_row_separation_m,
                'maximum_row_separation_m': self.book_map_max_row_separation_m,
            },
            'fallback_mapping_used': self.fallback_mapping_used,
            'book_column_roi_xywh': (
                self.book_column_roi.as_list()
                if self.book_column_roi is not None else None),
            'book_processed_frames': self.book_processed_frames,
            'book_scan_attempts': self.book_scan_attempts,
            'book_latest_diagnostics': self.book_latest_diagnostics,
            'book_layout_confirmation': (
                self.book_layout_confirmation.as_dict()
                if self.book_layout_confirmation is not None else None),
            'target_book_row': self.target_book_row,
            'target_book_locked_odom_xyz_m': (
                list(self.target_book_locked_odom_xyz_m)
                if self.target_book_locked_odom_xyz_m is not None else None),
            'target_book_reacquisition': (
                self.target_reacquisition_confirmation.as_dict()
                if self.target_reacquisition_confirmation is not None else None),
            'target_reacquisition_processed_frames': (
                self.target_reacquisition_processed_frames),
            'target_reacquisition_diagnostics': (
                self.target_reacquisition_diagnostics),
            'target_head_sequence_rad': list(self.target_head_sequence_rad),
            'target_head_pose_index_used': self.target_head_index,
            'target_book_detection': target_detection,
            'row_result_topic': self.ROW_TOPIC,
            'row_result_message_type': 'std_msgs/msg/Int32',
            'published_row_value': (
                self.target_book_row
                if self.row_publication_count > 0 else None),
            'row_publication_count': self.row_publication_count,
            'first_row_publish_sim_sec': self.first_row_publish_sim_sec,
            'last_row_publish_sim_sec': self.last_row_publish_sim_sec,
            'target_book_evidence_saved': self.target_book_evidence_saved,
            'target_book_evidence_path': self.target_book_evidence_path,
            'target_book_geometry_image_path': (
                self.target_book_geometry_image_path),
            'target_book_geometry': (
                self.target_book_geometry.as_dict()
                if self.target_book_geometry is not None else None),
            'pregrasp_plan': (
                self.pregrasp_plan.as_dict()
                if self.pregrasp_plan is not None else None),
            'selected_manipulation_arm': (
                self.pregrasp_plan.selected_arm
                if self.pregrasp_plan is not None else None),
            'single_arm_rule_respected': True,
            'day4_new_arm_trajectory_commanded': False,
            'day4_torso_commanded': False,
            'day4_gripper_commanded': False,
            'gripper_effort_commanded': False,
            'book_contact_attempted': False,
            'book_lift_attempted': False,
            'ik_solved': False,
            'collision_checked_pregrasp': False,
            'last_book_geometry_failure_reason': (
                self.last_geometry_failure_reason),
        })
        return result


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[Day4Mission] = None
    exit_code = 1
    try:
        node = Day4Mission()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.10)
        if node.done and node.passed:
            exit_code = 0
    except KeyboardInterrupt:
        if node is not None:
            node.stop_base()
            node.get_logger().warning(
                '[DAY4] Interrupted; repeated base stop commands sent')
        exit_code = 130
    except Exception as exc:
        if node is not None:
            node.stop_base()
            node.get_logger().error(
                f'[DAY4][FAILED] Unhandled exception: {exc}')
        else:
            print(f'[DAY4][FAILED] Could not initialize node: {exc}')
        exit_code = 1
    finally:
        if node is not None:
            node.stop_base()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
