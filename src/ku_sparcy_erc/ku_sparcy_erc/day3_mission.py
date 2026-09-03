#!/usr/bin/env python3
"""Day 3: validated Day 2 perception plus depth/LiDAR shelf approach.

The class subclasses the committed Day 2 ``MissionStart`` node rather than
copying or replacing its detector.  The inherited opening motion, temporal
marker confirmation, official Int32 publication, evidence image, complete-set
reconciliation, coherent-frame validation logic, and general-search diagnostic
therefore remain the single source of truth.

Day 3 adds only team-owned subscriptions and states for:

* live CameraInfo-based target bearing,
* aligned raw-depth sampling around the requested marker,
* TF-based projection into ``base_footprint``,
* independent front-LiDAR swept-corridor safety,
* bounded simultaneous +x/+y/+yaw holonomic approach,
* stale-target and stale-sensor holds,
* hard safety stop, quantitative trace, and final approach evidence.

After the requested column is locked, Day 3 commands only the two arm
controllers to a validated PAL spherical-wrist home travel pose before any
shelf translation.  No gripper, end-effector, torso, or lifting-body command is
created. Robot-control and freshness deadlines use Gazebo simulation time;
wall time is retained only for startup watchdogs and real-performance
measurements.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cv_bridge import CvBridgeError
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros

from ku_sparcy_erc.marker_detector import draw_detection_overlay
from ku_sparcy_erc.mission_start import MissionStart, normalize_angle
from ku_sparcy_erc.range_fusion import (
    BoundingBoxLike,
    CameraModel,
    ControllerSettings,
    LidarClearance,
    TargetGeometry,
    compute_holonomic_command,
    lidar_corridor_clearance,
    limit_rate,
    normalize_depth_array,
    target_geometry_from_depth,
)


class Day3Mission(MissionStart):
    """Extend the proven opening/perception node with safe shelf approach."""

    COLOR_INFO_TOPIC = (
        '/head_front_camera/head_front_camera/color/camera_info')
    DEPTH_TOPIC = (
        '/head_front_camera/head_front_camera/depth/image_rect_raw')
    DEPTH_INFO_TOPIC = (
        '/head_front_camera/head_front_camera/depth/camera_info')
    FRONT_SCAN_TOPIC = '/scan_front_raw'
    BASE_FRAME = 'base_footprint'

    # Validated stationary using PAL's official TIAGo Pro spherical-wrist
    # final "home" arm configuration.  Only arm joints are commanded:
    # torso and grippers remain untouched.
    TRAVEL_ARM_POSE_NAME = 'pal_spherical_home_arm_only'

    TRAVEL_ARM_LEFT_JOINTS = (
        'arm_left_1_joint',
        'arm_left_2_joint',
        'arm_left_3_joint',
        'arm_left_4_joint',
        'arm_left_5_joint',
        'arm_left_6_joint',
        'arm_left_7_joint',
    )

    TRAVEL_ARM_RIGHT_JOINTS = (
        'arm_right_1_joint',
        'arm_right_2_joint',
        'arm_right_3_joint',
        'arm_right_4_joint',
        'arm_right_5_joint',
        'arm_right_6_joint',
        'arm_right_7_joint',
    )

    TRAVEL_ARM_LEFT_TARGET_RAD = (
        0.36,
        -1.83,
        0.47,
        -2.35,
        0.0,
        -1.20,
        0.0,
    )

    TRAVEL_ARM_RIGHT_TARGET_RAD = (
        -0.36,
        -1.83,
        -0.47,
        -2.35,
        0.0,
        -1.20,
        0.0,
    )

    def __init__(self) -> None:
        # MissionStart declares and initializes every frozen Day 2 interface.
        super().__init__()

        # Day 3 execution mode and output controls.
        self.declare_parameter('approach_motion_enabled', True)
        self.declare_parameter('approach_distance_limit_m', 0.0)
        self.declare_parameter('approach_result_hold_sec', 0.35)
        self.declare_parameter('clock_stall_wall_timeout_sec', 20.0)

        # Day-3-only shelf-viewing camera pose.
        self.declare_parameter('approach_head_tilt_rad', 0.40)
        self.declare_parameter('approach_head_motion_sec', 0.80)
        self.declare_parameter(
            'approach_head_position_tolerance_rad', 0.05)
        self.declare_parameter('approach_head_ready_timeout_sec', 3.0)

        # Collision-safe arm travel pose.  These settings apply only after
        # the requested shelf column is locked and only when translation is
        # enabled.
        self.declare_parameter('travel_arm_motion_sec', 5.0)
        self.declare_parameter(
            'travel_arm_position_tolerance_rad', 0.05)
        self.declare_parameter('travel_arm_ready_timeout_sec', 8.0)

        # Raw-depth and synchronization constraints.
        self.declare_parameter('depth_min_m', 0.2)
        self.declare_parameter('depth_max_m', 8.0)
        self.declare_parameter('depth_min_valid_pixels', 20)
        self.declare_parameter('depth_min_valid_fraction', 0.08)
        self.declare_parameter('max_rgb_depth_stamp_skew_sec', 0.20)
        self.declare_parameter('depth_stale_timeout_sec', 0.55)

        # Front-LiDAR swept-body corridor.
        self.declare_parameter('lidar_corridor_half_width_m', 0.38)
        self.declare_parameter('lidar_robust_percentile', 10.0)
        self.declare_parameter('lidar_min_corridor_points', 4)
        self.declare_parameter('lidar_hard_cluster_points', 3)
        self.declare_parameter('lidar_stale_timeout_sec', 0.55)
        self.declare_parameter('lidar_slowdown_m', 1.50)
        self.declare_parameter('lidar_hard_stop_m', 0.72)
        self.declare_parameter('lidar_hard_release_m', 0.82)
        self.declare_parameter('lidar_command_margin_m', 0.12)

        # Holonomic controller.  Day 1 angular constants remain untouched;
        # these values apply only after the opening/perception stage finishes.
        self.declare_parameter('approach_standoff_m', 1.05)
        self.declare_parameter('approach_max_forward_speed_mps', 0.30)
        self.declare_parameter('approach_min_forward_speed_mps', 0.06)
        self.declare_parameter('approach_max_lateral_speed_mps', 0.14)
        self.declare_parameter('approach_max_yaw_speed_radps', 0.20)
        self.declare_parameter('approach_forward_kp', 0.45)
        self.declare_parameter('approach_lateral_kp', 0.75)
        self.declare_parameter('approach_yaw_kp', 0.90)
        self.declare_parameter('approach_lateral_deadband_m', 0.025)
        self.declare_parameter('approach_yaw_deadband_deg', 1.5)
        self.declare_parameter('approach_max_planar_speed_mps', 0.32)
        self.declare_parameter('approach_linear_accel_limit_mps2', 0.80)
        self.declare_parameter('approach_yaw_accel_limit_radps2', 1.00)

        # Final acceptance envelope and safe recovery deadlines.
        self.declare_parameter('approach_timeout_sec', 30.0)
        self.declare_parameter('target_recovery_timeout_sec', 2.0)
        self.declare_parameter('sensor_hold_timeout_sec', 2.0)
        self.declare_parameter('safety_hold_timeout_sec', 2.5)
        self.declare_parameter('final_depth_tolerance_m', 0.12)
        self.declare_parameter('final_lidar_min_m', 0.82)
        self.declare_parameter('final_lidar_max_m', 1.18)
        self.declare_parameter('final_lateral_tolerance_m', 0.12)
        self.declare_parameter('final_bearing_tolerance_deg', 4.0)
        self.declare_parameter('final_heading_tolerance_deg', 3.0)
        self.declare_parameter('final_settle_samples', 6)

        # Quantitative trace controls.
        self.declare_parameter('trace_period_sec', 0.20)
        self.declare_parameter('max_trace_samples', 500)

        self._read_day3_parameters()
        self._validate_day3_parameters()
        self._validate_approach_head_parameters()
        self._validate_travel_arm_parameters()

        self.controller_settings = ControllerSettings(
            target_standoff_m=self.approach_standoff_m,
            max_forward_speed_mps=self.approach_max_forward_speed,
            min_forward_speed_mps=self.approach_min_forward_speed,
            max_lateral_speed_mps=self.approach_max_lateral_speed,
            max_yaw_speed_radps=self.approach_max_yaw_speed,
            forward_kp=self.approach_forward_kp,
            lateral_kp=self.approach_lateral_kp,
            yaw_kp=self.approach_yaw_kp,
            lateral_deadband_m=self.approach_lateral_deadband,
            yaw_deadband_rad=math.radians(self.approach_yaw_deadband_deg),
            lidar_slowdown_m=self.lidar_slowdown_m,
            lidar_hard_stop_m=self.lidar_hard_stop_m,
            lidar_command_margin_m=self.lidar_command_margin_m,
            max_planar_speed_mps=self.approach_max_planar_speed,
        )
        self.controller_settings.validate()

        # Live sensor subscriptions.  The raw depth stream remains available
        # when the expensive reconstructed point-cloud pipeline is disabled.
        self.color_info_sub = self.create_subscription(
            CameraInfo, self.COLOR_INFO_TOPIC,
            self._color_info_callback, qos_profile_sensor_data)
        self.depth_info_sub = self.create_subscription(
            CameraInfo, self.DEPTH_INFO_TOPIC,
            self._depth_info_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(
            Image, self.DEPTH_TOPIC,
            self._depth_callback, qos_profile_sensor_data)
        self.front_scan_sub = self.create_subscription(
            LaserScan, self.FRONT_SCAN_TOPIC,
            self._front_scan_callback, qos_profile_sensor_data)
        self.approach_debug_pub = self.create_publisher(
            String, '/ku_sparcy/approach_debug', 10)
        self.approach_head_pub = self.create_publisher(
            JointTrajectory,
            '/head_controller/joint_trajectory',
            10,
        )

        self.travel_arm_left_pub = self.create_publisher(
            JointTrajectory,
            '/arm_left_controller/joint_trajectory',
            10,
        )

        self.travel_arm_right_pub = self.create_publisher(
            JointTrajectory,
            '/arm_right_controller/joint_trajectory',
            10,
        )

        self.travel_arm_joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._travel_arm_joint_state_callback,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)

        # Camera/depth observations.
        self.color_model: Optional[CameraModel] = None
        self.depth_model: Optional[CameraModel] = None
        self.latest_rgb_stamp_sec: Optional[float] = None
        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_depth_stamp_sec: Optional[float] = None
        self.latest_depth_encoding: Optional[str] = None
        self.depth_frames_total = 0
        self.depth_conversion_failures = 0
        self.camera_info_updates = 0

        # LiDAR observations.
        self.latest_lidar: Optional[LidarClearance] = None
        self.latest_lidar_stamp_sec: Optional[float] = None
        self.latest_lidar_frame_id: Optional[str] = None
        self.lidar_frames_total = 0
        self.lidar_transform_failures = 0
        self.minimum_lidar_clearance_m: Optional[float] = None
        self.minimum_lidar_clearance_during_approach_m: Optional[float] = None

        # Odometry/path observations.
        self.current_odom_xy: Optional[Tuple[float, float]] = None
        self.approach_start_odom_xy: Optional[Tuple[float, float]] = None
        self.approach_heading_yaw_rad: Optional[float] = None
        self.approach_distance_travelled_m = 0.0

        # Day 3 state, timing, and quantitative evidence.
        self.day3_stage_started_sim_sec: Optional[float] = None
        self.approach_started_sim_sec: Optional[float] = None
        self.approach_started_wall_monotonic: Optional[float] = None
        self.approach_completed_sim_sec: Optional[float] = None
        self.approach_completed_wall_monotonic: Optional[float] = None
        self.resume_state_after_hold = 'ESTIMATE_TARGET_GEOMETRY'
        self.hold_started_sim_sec: Optional[float] = None
        self.approach_outcome = 'not_started'
        self.start_target_geometry: Optional[TargetGeometry] = None
        self.latest_target_geometry: Optional[TargetGeometry] = None
        self.final_target_geometry: Optional[TargetGeometry] = None

        # Once the requested numbered column has been identified and ranged,
        # Day 3 locks that physical column in the odometry frame.  The numeral
        # does not need to remain visible at close range.
        self.column_lock_active = False
        self.locked_target_geometry: Optional[TargetGeometry] = None
        self.locked_target_odom_xy: Optional[Tuple[float, float]] = None
        self.locked_target_confidence: Optional[float] = None
        self.locked_target_bbox_xywh = None

        # Keep recent raw-depth frames so geometry uses the depth frame
        # temporally closest to the RGB observation that produced the
        # accepted marker bbox.  Comparing an older accepted RGB detection
        # directly with the newest depth callback can create false sync holds.
        self.depth_frame_history = []
        self.depth_history_max_frames = 24

        # Day 3 needs the exact RGB timestamp belonging to the bbox used for
        # ranging. MarkerDetection intentionally has no timestamp, so capture
        # the current accepted target and the current image stamp together
        # from MissionStart._publish_debug().
        self.day3_latest_target_detection = None
        self.day3_latest_target_stamp_sec: Optional[float] = None

        # Snapshot the validated opening head pose before Day 3 moves it.
        self.opening_actual_head_tilt_rad: Optional[float] = None
        self.approach_head_commanded = False
        self.approach_head_ready = False
        self.approach_head_command_sim_sec: Optional[float] = None
        self.approach_head_camera_frame_at_command = 0

        # Collision-safe travel-arm state.
        self.travel_arm_pose_commanded = False
        self.travel_arm_pose_ready = False
        self.travel_arm_command_sim_sec: Optional[float] = None
        self.travel_arm_ready_sim_sec: Optional[float] = None
        self.travel_arm_max_error_rad: Optional[float] = None
        self.travel_arm_joint_positions: Dict[str, float] = {}
        self.travel_arm_joint_updates_after_command = 0

        self.start_lidar_clearance_m: Optional[float] = None
        self.final_lidar_clearance_m: Optional[float] = None
        self.target_stale_events = 0
        self.sensor_hold_events = 0
        self.safety_stop_events = 0
        self.final_settle_count = 0

        # Near the shelf, large edge-column corrections are completed
        # laterally before additional forward motion is allowed.
        self.near_shelf_lateral_only_active = False
        self.near_shelf_lateral_only_events = 0

        self.camera_frames_at_approach_start = 0
        self.depth_frames_at_approach_start = 0
        self.lidar_frames_at_approach_start = 0
        self.last_command = Twist()
        self.last_command_sim_sec: Optional[float] = None
        self.maximum_abs_vx = 0.0
        self.maximum_abs_vy = 0.0
        self.maximum_abs_wz = 0.0
        self.maximum_abs_heading_error_rad = 0.0
        self.last_trace_sim_sec: Optional[float] = None
        self.approach_trace: List[Dict[str, object]] = []
        self.approach_image_path: Optional[str] = None
        self.approach_image_saved = False
        self.final_approach_column_bbox = None
        self.last_geometry_failure_reason = ''
        self.last_sensor_failure_reason = ''
        self.last_sim_clock_value: Optional[float] = None
        self.last_sim_clock_advance_wall_monotonic = time.monotonic()

        self.get_logger().info(
            '[DAY3] raw_depth=%s color_info=%s depth_info=%s scan=%s '
            'standoff=%.2f m motion=%s distance_limit=%.2f m'
            % (
                self.DEPTH_TOPIC,
                self.COLOR_INFO_TOPIC,
                self.DEPTH_INFO_TOPIC,
                self.FRONT_SCAN_TOPIC,
                self.approach_standoff_m,
                self.approach_motion_enabled,
                self.approach_distance_limit_m,
            ))

    def _validate_approach_head_parameters(self) -> None:
        if not math.isfinite(self.approach_head_tilt_rad):
            raise ValueError('approach_head_tilt_rad must be finite')
        if abs(self.approach_head_tilt_rad) > 1.2:
            raise ValueError(
                'approach_head_tilt_rad is outside bounded range')
        if self.approach_head_motion_sec <= 0.0:
            raise ValueError(
                'approach_head_motion_sec must be positive')
        if not (
            0.0
            < self.approach_head_position_tolerance_rad
            <= 0.20
        ):
            raise ValueError(
                'approach_head_position_tolerance_rad is invalid')
        if self.approach_head_ready_timeout_sec <= 0.0:
            raise ValueError(
                'approach_head_ready_timeout_sec must be positive')

    def _validate_travel_arm_parameters(self) -> None:
        if self.travel_arm_motion_sec <= 0.0:
            raise ValueError(
                'travel_arm_motion_sec must be positive')

        if not (
            0.0
            < self.travel_arm_position_tolerance_rad
            <= 0.20
        ):
            raise ValueError(
                'travel_arm_position_tolerance_rad is invalid')

        if (
            self.travel_arm_ready_timeout_sec
            <= self.travel_arm_motion_sec
        ):
            raise ValueError(
                'travel_arm_ready_timeout_sec must exceed '
                'travel_arm_motion_sec')

        for positions in (
            self.TRAVEL_ARM_LEFT_TARGET_RAD,
            self.TRAVEL_ARM_RIGHT_TARGET_RAD,
        ):
            if len(positions) != 7:
                raise ValueError(
                    'travel arm pose must contain seven joints')

            if not all(math.isfinite(value) for value in positions):
                raise ValueError(
                    'travel arm pose values must be finite')

    def _read_day3_parameters(self) -> None:
        get = self.get_parameter
        self.approach_motion_enabled = bool(
            get('approach_motion_enabled').value)
        self.approach_distance_limit_m = float(
            get('approach_distance_limit_m').value)
        self.approach_result_hold_sec = float(
            get('approach_result_hold_sec').value)
        self.clock_stall_wall_timeout_sec = float(
            get('clock_stall_wall_timeout_sec').value)

        self.approach_head_tilt_rad = float(
            get('approach_head_tilt_rad').value)
        self.approach_head_motion_sec = float(
            get('approach_head_motion_sec').value)
        self.approach_head_position_tolerance_rad = float(
            get('approach_head_position_tolerance_rad').value)
        self.approach_head_ready_timeout_sec = float(
            get('approach_head_ready_timeout_sec').value)

        self.travel_arm_motion_sec = float(
            get('travel_arm_motion_sec').value)
        self.travel_arm_position_tolerance_rad = float(
            get('travel_arm_position_tolerance_rad').value)
        self.travel_arm_ready_timeout_sec = float(
            get('travel_arm_ready_timeout_sec').value)

        self.depth_min_m = float(get('depth_min_m').value)
        self.depth_max_m = float(get('depth_max_m').value)
        self.depth_min_valid_pixels = int(
            get('depth_min_valid_pixels').value)
        self.depth_min_valid_fraction = float(
            get('depth_min_valid_fraction').value)
        self.max_rgb_depth_skew = float(
            get('max_rgb_depth_stamp_skew_sec').value)
        self.depth_stale_timeout = float(
            get('depth_stale_timeout_sec').value)

        self.lidar_corridor_half_width = float(
            get('lidar_corridor_half_width_m').value)
        self.lidar_robust_percentile = float(
            get('lidar_robust_percentile').value)
        self.lidar_min_corridor_points = int(
            get('lidar_min_corridor_points').value)
        self.lidar_hard_cluster_points = int(
            get('lidar_hard_cluster_points').value)
        self.lidar_stale_timeout = float(
            get('lidar_stale_timeout_sec').value)
        self.lidar_slowdown_m = float(get('lidar_slowdown_m').value)
        self.lidar_hard_stop_m = float(get('lidar_hard_stop_m').value)
        self.lidar_hard_release_m = float(
            get('lidar_hard_release_m').value)
        self.lidar_command_margin_m = float(
            get('lidar_command_margin_m').value)

        self.approach_standoff_m = float(get('approach_standoff_m').value)
        self.approach_max_forward_speed = float(
            get('approach_max_forward_speed_mps').value)
        self.approach_min_forward_speed = float(
            get('approach_min_forward_speed_mps').value)
        self.approach_max_lateral_speed = float(
            get('approach_max_lateral_speed_mps').value)
        self.approach_max_yaw_speed = float(
            get('approach_max_yaw_speed_radps').value)
        self.approach_forward_kp = float(get('approach_forward_kp').value)
        self.approach_lateral_kp = float(get('approach_lateral_kp').value)
        self.approach_yaw_kp = float(get('approach_yaw_kp').value)
        self.approach_lateral_deadband = float(
            get('approach_lateral_deadband_m').value)
        self.approach_yaw_deadband_deg = float(
            get('approach_yaw_deadband_deg').value)
        self.approach_max_planar_speed = float(
            get('approach_max_planar_speed_mps').value)
        self.approach_linear_accel_limit = float(
            get('approach_linear_accel_limit_mps2').value)
        self.approach_yaw_accel_limit = float(
            get('approach_yaw_accel_limit_radps2').value)

        self.approach_timeout_sec = float(get('approach_timeout_sec').value)
        self.target_recovery_timeout_sec = float(
            get('target_recovery_timeout_sec').value)
        self.sensor_hold_timeout_sec = float(
            get('sensor_hold_timeout_sec').value)
        self.safety_hold_timeout_sec = float(
            get('safety_hold_timeout_sec').value)
        self.final_depth_tolerance_m = float(
            get('final_depth_tolerance_m').value)
        self.final_lidar_min_m = float(get('final_lidar_min_m').value)
        self.final_lidar_max_m = float(get('final_lidar_max_m').value)
        self.final_lateral_tolerance_m = float(
            get('final_lateral_tolerance_m').value)
        self.final_bearing_tolerance_rad = math.radians(float(
            get('final_bearing_tolerance_deg').value))
        self.final_heading_tolerance_rad = math.radians(float(
            get('final_heading_tolerance_deg').value))
        self.final_settle_required = int(get('final_settle_samples').value)

        self.trace_period_sec = float(get('trace_period_sec').value)
        self.max_trace_samples = int(get('max_trace_samples').value)

    def _validate_day3_parameters(self) -> None:
        if self.approach_distance_limit_m < 0.0:
            raise ValueError('approach_distance_limit_m cannot be negative')
        if not 0.0 <= self.approach_result_hold_sec <= 2.0:
            raise ValueError('approach_result_hold_sec must be in [0, 2]')
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError('Depth range is invalid')
        if self.depth_min_valid_pixels < 4:
            raise ValueError('depth_min_valid_pixels must be at least 4')
        if not 0.0 < self.depth_min_valid_fraction <= 1.0:
            raise ValueError('depth_min_valid_fraction must be in (0, 1]')
        if self.max_rgb_depth_skew <= 0.0:
            raise ValueError('max_rgb_depth_stamp_skew_sec must be positive')
        if self.lidar_corridor_half_width <= 0.25:
            raise ValueError('LiDAR corridor must exceed half the robot width')
        if self.lidar_min_corridor_points < 1:
            raise ValueError('lidar_min_corridor_points must be positive')
        if self.lidar_hard_cluster_points < 1:
            raise ValueError('lidar_hard_cluster_points must be positive')
        if not (
            self.lidar_hard_stop_m
            < self.lidar_hard_release_m
            < self.lidar_slowdown_m
        ):
            raise ValueError(
                'LiDAR distances must satisfy hard stop < release < slowdown')
        for name, value in (
            ('approach_timeout_sec', self.approach_timeout_sec),
            ('target_recovery_timeout_sec', self.target_recovery_timeout_sec),
            ('sensor_hold_timeout_sec', self.sensor_hold_timeout_sec),
            ('safety_hold_timeout_sec', self.safety_hold_timeout_sec),
            ('trace_period_sec', self.trace_period_sec),
            ('clock_stall_wall_timeout_sec', self.clock_stall_wall_timeout_sec),
        ):
            if value <= 0.0:
                raise ValueError(f'{name} must be positive')
        if not self.final_lidar_min_m < self.final_lidar_max_m:
            raise ValueError('Final LiDAR interval is invalid')
        if self.final_settle_required < 2:
            raise ValueError('final_settle_samples must be at least 2')
        if self.max_trace_samples < 20:
            raise ValueError('max_trace_samples must be at least 20')

    @staticmethod
    def _stamp_sec_from_header(header: Any) -> Optional[float]:
        stamp = (
            float(header.stamp.sec)
            + float(header.stamp.nanosec) * 1.0e-9
        )
        return stamp if stamp > 0.0 else None

    def _clock_callback(self, msg: Any) -> None:
        previous = self.sim_time_sec
        super()._clock_callback(msg)
        if self.sim_time_sec is None:
            return
        if previous is None or self.sim_time_sec > previous + 1.0e-9:
            self.last_sim_clock_value = self.sim_time_sec
            self.last_sim_clock_advance_wall_monotonic = time.monotonic()

    def _color_info_callback(self, msg: CameraInfo) -> None:
        try:
            self.color_model = CameraModel.from_arrays(
                msg.width, msg.height, msg.k, msg.p,
                frame_id=msg.header.frame_id,
                prefer_projection=False,
            )
            self.camera_info_updates += 1
        except ValueError as exc:
            self.get_logger().warning(
                f'[DAY3] Rejected color CameraInfo: {exc}')

    def _depth_info_callback(self, msg: CameraInfo) -> None:
        try:
            self.depth_model = CameraModel.from_arrays(
                msg.width, msg.height, msg.k, msg.p,
                frame_id=msg.header.frame_id,
                prefer_projection=True,
            )
            self.camera_info_updates += 1
        except ValueError as exc:
            self.get_logger().warning(
                f'[DAY3] Rejected depth CameraInfo: {exc}')

    def _image_callback(self, msg: Image) -> None:
        stamp = self._message_stamp_sec(msg)
        if stamp is not None:
            self.latest_rgb_stamp_sec = stamp
        super()._image_callback(msg)

    def _depth_callback(self, msg: Image) -> None:
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = normalize_depth_array(raw, msg.encoding)
        except (CvBridgeError, ValueError, TypeError) as exc:
            self.depth_conversion_failures += 1
            if self.depth_conversion_failures <= 3:
                self.get_logger().warning(
                    f'[DAY3] Depth conversion failed: {exc}')
            return
        if depth.ndim != 2 or depth.size == 0:
            return
        self.latest_depth_m = depth
        self.latest_depth_stamp_sec = (
            self._stamp_sec_from_header(msg.header) or self.sim_time_sec)
        self.latest_depth_encoding = str(msg.encoding)
        self.depth_frames_total += 1

    def _odom_callback(self, msg: Odometry) -> None:
        super()._odom_callback(msg)
        self.current_odom_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        if self.approach_start_odom_xy is not None:
            dx = self.current_odom_xy[0] - self.approach_start_odom_xy[0]
            dy = self.current_odom_xy[1] - self.approach_start_odom_xy[1]
            self.approach_distance_travelled_m = math.hypot(dx, dy)

    def _lookup_transform_components(
        self, source_frame: str,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        if not source_frame:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.BASE_FRAME,
                source_frame,
                Time(),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            (float(translation.x), float(translation.y), float(translation.z)),
            (float(rotation.x), float(rotation.y),
             float(rotation.z), float(rotation.w)),
        )

    def _front_scan_callback(self, msg: LaserScan) -> None:
        transform = self._lookup_transform_components(msg.header.frame_id)
        if transform is None:
            self.lidar_transform_failures += 1
            return
        try:
            clearance = lidar_corridor_clearance(
                msg.ranges,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                transform[0],
                transform[1],
                corridor_half_width_m=self.lidar_corridor_half_width,
                robust_percentile=self.lidar_robust_percentile,
                hard_stop_m=self.lidar_hard_stop_m,
            )
        except ValueError:
            self.lidar_transform_failures += 1
            return
        if clearance is None:
            return
        self.latest_lidar = clearance
        self.latest_lidar_stamp_sec = (
            self._stamp_sec_from_header(msg.header) or self.sim_time_sec)
        self.latest_lidar_frame_id = str(msg.header.frame_id)
        self.lidar_frames_total += 1
        value = clearance.robust_clearance_m
        if self.minimum_lidar_clearance_m is None:
            self.minimum_lidar_clearance_m = value
        else:
            self.minimum_lidar_clearance_m = min(
                self.minimum_lidar_clearance_m, value)
        if self.approach_started_sim_sec is not None:
            if self.minimum_lidar_clearance_during_approach_m is None:
                self.minimum_lidar_clearance_during_approach_m = value
            else:
                self.minimum_lidar_clearance_during_approach_m = min(
                    self.minimum_lidar_clearance_during_approach_m, value)

    def _required_transform_ready(self) -> bool:
        if self.depth_model is None or self.latest_lidar_frame_id is None:
            return False
        return (
            self._lookup_transform_components(self.depth_model.frame_id)
            is not None
            and self._lookup_transform_components(self.latest_lidar_frame_id)
            is not None
        )

    def _inputs_ready(self) -> bool:
        """Require every safety-critical sensor before any base motion starts."""
        return (
            super()._inputs_ready()
            and self.color_model is not None
            and self.depth_model is not None
            and self.latest_depth_m is not None
            and self.latest_depth_stamp_sec is not None
            and self.latest_lidar is not None
            and self.latest_lidar_stamp_sec is not None
            and self.latest_lidar.corridor_point_count
            >= self.lidar_min_corridor_points
            and self._required_transform_ready()
        )

    def _missing_startup_inputs(self) -> List[str]:
        missing: List[str] = []
        if self.sim_time_sec is None:
            missing.append('/clock')
        if self.current_yaw is None:
            missing.append('/odom')
        if self.camera_frames_total < 1:
            missing.append('RGB frame')
        if self.processed_frames < 1:
            missing.append('processed RGB frame')
        if not self._head_controller_is_ready():
            missing.append('head controller subscriber')
        if self.color_model is None:
            missing.append('RGB CameraInfo')
        if self.depth_model is None:
            missing.append('depth CameraInfo')
        if self.latest_depth_m is None:
            missing.append('raw depth frame')
        if self.latest_lidar is None:
            missing.append('front LiDAR clearance')
        elif self.latest_lidar.corridor_point_count < self.lidar_min_corridor_points:
            missing.append('enough front LiDAR corridor points')
        if (
            self.depth_model is not None
            and self._lookup_transform_components(self.depth_model.frame_id) is None
        ):
            missing.append('depth optical frame TF')
        if (
            self.latest_lidar_frame_id is not None
            and self._lookup_transform_components(self.latest_lidar_frame_id) is None
        ):
            missing.append('front LiDAR TF')
        return missing

    def _publish_velocity(self, vx: float, vy: float, wz: float) -> None:
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = float(wz)
        self.cmd_vel_pub.publish(cmd)
        self.last_command = cmd
        self.maximum_abs_vx = max(self.maximum_abs_vx, abs(float(vx)))
        self.maximum_abs_vy = max(self.maximum_abs_vy, abs(float(vy)))
        self.maximum_abs_wz = max(self.maximum_abs_wz, abs(float(wz)))

    def stop_base(self) -> None:
        for _ in range(4):
            self._publish_velocity(0.0, 0.0, 0.0)
        if hasattr(self, 'last_command_sim_sec'):
            self.last_command_sim_sec = self.sim_time_sec

    def _day3_transition(self, state: str, detail: str) -> None:
        self._transition(state, detail)
        if state in {
            'ESTIMATE_TARGET_GEOMETRY',
            'SET_TRAVEL_ARM_POSE',
            'WAIT_FOR_TRAVEL_ARM_POSE',
            'APPROACH_TARGET_COLUMN',
            'RECOVER_TARGET',
            'SENSOR_HOLD',
            'SAFETY_HOLD',
            'VERIFY_APPROACH',
        }:
            self.day3_stage_started_sim_sec = self.sim_time_sec
        self.get_logger().info(f'[DAY3] STATE -> {state}: {detail}')

    def _publish_debug(
        self,
        detections,
        new_confirmations,
        stamp_sec: float,
    ) -> None:
        # MissionStart calls this once per successfully processed RGB frame,
        # after latest_detections and the temporal filter have been updated.
        # Preserve the exact stamp belonging to the accepted target bbox.
        target = max(
            (
                item for item in detections
                if item.digit == self.shelf_column
            ),
            key=lambda item: item.confidence,
            default=None,
        )

        if target is not None:
            self.day3_latest_target_detection = target
            self.day3_latest_target_stamp_sec = float(stamp_sec)

        super()._publish_debug(
            detections,
            new_confirmations,
            stamp_sec,
        )

    def _target_fresh_for_approach(self) -> bool:
        # Retain the frozen Day 2 temporal-filter gate first.
        if not self._target_is_fresh():
            return False

        if (
            self.sim_time_sec is None
            or self.day3_latest_target_stamp_sec is None
        ):
            return False

        # Day 3 must never call a visual observation fresh for longer
        # than the depth frame that can be paired with it.
        target_age_sec = max(
            0.0,
            float(self.sim_time_sec)
            - float(self.day3_latest_target_stamp_sec),
        )

        return target_age_sec <= self.depth_stale_timeout

    def _age(self, stamp: Optional[float]) -> Optional[float]:
        if stamp is None or self.sim_time_sec is None:
            return None
        return max(0.0, self.sim_time_sec - stamp)

    def _lidar_ready_now(self) -> Tuple[bool, str]:
        if self.latest_lidar is None or self.latest_lidar_stamp_sec is None:
            return False, 'front LiDAR estimate missing'
        age = self._age(self.latest_lidar_stamp_sec)
        if age is None or age > self.lidar_stale_timeout:
            return False, 'front LiDAR estimate is stale'
        if (
            self.motion_completed_sim_sec is not None
            and self.latest_lidar_stamp_sec
            < self.motion_completed_sim_sec - 0.02
        ):
            return False, 'waiting for a front LiDAR scan after opening motion'
        if self.latest_lidar.corridor_point_count < self.lidar_min_corridor_points:
            return False, 'too few transformed points in LiDAR safety corridor'
        return True, ''

    def _current_target_geometry(
        self,
    ) -> Tuple[Optional[TargetGeometry], str]:
        if not self._target_fresh_for_approach():
            return None, 'requested marker track is stale or missing'
        if self.color_model is None or self.depth_model is None:
            return None, 'CameraInfo is missing'
        if self.latest_depth_m is None or self.latest_depth_stamp_sec is None:
            return None, 'raw depth frame is missing'
        depth_age = self._age(self.latest_depth_stamp_sec)
        if depth_age is None or depth_age > self.depth_stale_timeout:
            return None, 'raw depth frame is stale'
        if (
            self.motion_completed_sim_sec is not None
            and self.latest_depth_stamp_sec
            < self.motion_completed_sim_sec - 0.02
        ):
            return None, 'waiting for a raw depth frame after opening motion'
        confirmed = self.temporal_filter.confirmed(self.shelf_column)
        target = self.day3_latest_target_detection
        target_rgb_stamp_sec = self.day3_latest_target_stamp_sec

        if confirmed is None:
            return None, 'requested target is not temporally confirmed'

        if target is None or target_rgb_stamp_sec is None:
            return None, 'current target-marker detection is missing'

        target_rgb_stamp_sec = float(target_rgb_stamp_sec)

        # Record each new raw-depth frame observed by the controller.
        current_depth_stamp_sec = float(self.latest_depth_stamp_sec)
        if (
            not self.depth_frame_history
            or abs(
                self.depth_frame_history[-1][0]
                - current_depth_stamp_sec
            ) > 1.0e-9
        ):
            self.depth_frame_history.append(
                (
                    current_depth_stamp_sec,
                    self.latest_depth_m.copy(),
                )
            )

            if len(self.depth_frame_history) > self.depth_history_max_frames:
                del self.depth_frame_history[
                    :-self.depth_history_max_frames
                ]

        if not self.depth_frame_history:
            return None, 'no raw-depth frame is available for RGB pairing'

        # Use the depth frame closest in simulation timestamp to the RGB
        # observation that supplied this marker bbox.
        matched_depth_stamp_sec, matched_depth_m = min(
            self.depth_frame_history,
            key=lambda item: abs(item[0] - target_rgb_stamp_sec),
        )

        skew = abs(target_rgb_stamp_sec - matched_depth_stamp_sec)

        if skew > self.max_rgb_depth_skew:
            return None, (
                'No raw-depth frame is synchronized with target RGB '
                f'observation (nearest skew={skew:.3f} s)'
            )

        # Synchronization alone is insufficient: the selected frame must
        # also still satisfy the existing simulation-time freshness limit.
        if self.sim_time_sec is not None:
            matched_depth_age_sec = max(
                0.0,
                float(self.sim_time_sec) - matched_depth_stamp_sec,
            )
            if matched_depth_age_sec > self.depth_stale_timeout:
                return None, (
                    'Matched raw-depth frame is stale '
                    f'(age={matched_depth_age_sec:.3f} s)'
                )
        transform = self._lookup_transform_components(self.depth_model.frame_id)
        if transform is None:
            return None, 'depth optical frame to base TF is unavailable'
        bbox = BoundingBoxLike(
            target.bbox.x,
            target.bbox.y,
            target.bbox.width,
            target.bbox.height,
        )
        try:
            geometry = target_geometry_from_depth(
                bbox,
                self.color_model,
                self.depth_model,
                matched_depth_m,
                transform[0],
                transform[1],
                rgb_depth_stamp_skew_sec=skew,
                min_valid_pixels=self.depth_min_valid_pixels,
                min_depth_m=self.depth_min_m,
                max_depth_m=self.depth_max_m,
            )
        except ValueError as exc:
            return None, f'target geometry calculation failed: {exc}'
        if geometry is None:
            return None, 'not enough valid raw-depth pixels around marker plate'
        if geometry.depth.valid_fraction < self.depth_min_valid_fraction:
            return None, (
                'target depth valid fraction below threshold '
                f'({geometry.depth.valid_fraction:.3f})')
        if geometry.forward_m <= 0.20:
            return None, (
                'transformed target is not safely in front of robot '
                f'(x={geometry.forward_m:.3f} m)')
        self.latest_target_geometry = geometry
        return geometry, ''

    def _lock_target_column(
        self,
        geometry: TargetGeometry,
    ) -> None:
        """Lock the already-identified shelf column in the odometry frame."""
        if (
            self.current_odom_xy is None
            or self.current_yaw is None
        ):
            raise RuntimeError(
                'Cannot lock target column without odometry and yaw'
            )

        robot_x, robot_y = self.current_odom_xy
        yaw = float(self.current_yaw)

        forward = float(geometry.forward_m)
        left = float(geometry.lateral_left_m)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        target_x = (
            robot_x
            + cos_yaw * forward
            - sin_yaw * left
        )
        target_y = (
            robot_y
            + sin_yaw * forward
            + cos_yaw * left
        )

        self.locked_target_geometry = geometry
        self.locked_target_odom_xy = (
            float(target_x),
            float(target_y),
        )

        detection = self.day3_latest_target_detection
        if detection is not None:
            self.locked_target_confidence = float(
                detection.confidence)
            self.locked_target_bbox_xywh = (
                detection.bbox.as_list())
        elif self.target_confirmation is not None:
            self.locked_target_confidence = float(
                self.target_confirmation.confidence)
            self.locked_target_bbox_xywh = (
                self.target_confirmation.bbox.as_list())

        self.column_lock_active = True

        self.get_logger().info(
            '[DAY3] COLUMN LOCKED: '
            f'digit={self.shelf_column} '
            f'forward={geometry.forward_m:.3f}m '
            f'left={geometry.lateral_left_m:.3f}m '
            f'confidence={self.locked_target_confidence}'
        )

    def _locked_target_geometry_now(
        self,
    ) -> Optional[TargetGeometry]:
        """Propagate the locked shelf-column position using odometry."""
        if (
            not self.column_lock_active
            or self.locked_target_geometry is None
            or self.locked_target_odom_xy is None
            or self.current_odom_xy is None
            or self.current_yaw is None
        ):
            return None

        target_x, target_y = self.locked_target_odom_xy
        robot_x, robot_y = self.current_odom_xy

        dx = target_x - robot_x
        dy = target_y - robot_y

        yaw = float(self.current_yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # odom -> current base_footprint
        forward = cos_yaw * dx + sin_yaw * dy
        left = -sin_yaw * dx + cos_yaw * dy

        z = float(
            self.locked_target_geometry.base_point_xyz_m[2]
        )

        bearing = math.atan2(
            left,
            max(1.0e-6, forward),
        )

        geometry = replace(
            self.locked_target_geometry,
            base_point_xyz_m=(
                float(forward),
                float(left),
                z,
            ),
            base_bearing_left_rad=float(bearing),
        )

        self.latest_target_geometry = geometry
        return geometry

    def _approach_geometry_now(
        self,
    ) -> Tuple[Optional[TargetGeometry], str]:
        """Use the locked physical column throughout translation."""
        if self.column_lock_active:
            locked_geometry = self._locked_target_geometry_now()

            if locked_geometry is not None:
                return locked_geometry, 'locked_column_odometry'

            return None, 'Locked column odometry geometry unavailable'

        # Before the one-time column lock is established, live marker/depth
        # geometry remains the authoritative source.
        return self._current_target_geometry()

    def _locked_control_geometry(
        self,
        geometry: TargetGeometry,
        lidar: LidarClearance,
    ) -> TargetGeometry:
        """Use LiDAR as forward ranging after the numbered column is locked."""
        if not self.column_lock_active:
            return geometry

        # Aim at the centre of the accepted final LiDAR interval.
        lidar_target_m = 0.5 * (
            self.final_lidar_min_m
            + self.final_lidar_max_m
        )

        lidar_error_m = (
            lidar.robust_clearance_m - lidar_target_m
        )

        # compute_holonomic_command() expects a target-forward coordinate
        # relative to approach_standoff_m.  Supply an equivalent control
        # coordinate while preserving the locked column's lateral error.
        control_forward = max(
            0.21,
            self.approach_standoff_m + lidar_error_m,
        )

        left = float(geometry.lateral_left_m)
        z = float(geometry.base_point_xyz_m[2])

        return replace(
            geometry,
            base_point_xyz_m=(
                float(control_forward),
                left,
                z,
            ),
            base_bearing_left_rad=float(
                math.atan2(
                    left,
                    max(1.0e-6, control_forward),
                )
            ),
        )

    def _enter_hold(self, state: str, resume_state: str, reason: str) -> None:
        self.stop_base()
        self.resume_state_after_hold = resume_state
        self.hold_started_sim_sec = self.sim_time_sec
        if state == 'RECOVER_TARGET':
            self.target_stale_events += 1
        elif state == 'SENSOR_HOLD':
            self.sensor_hold_events += 1
        elif state == 'SAFETY_HOLD':
            self.safety_stop_events += 1
        self._day3_transition(state, reason)

    def _evidence_tick(self) -> None:
        """Preserve Day 2 evidence gates, then continue into Day 3 ranging."""
        self.stop_base()
        if self.target_confirmation is None:
            self._finish(False, 'Evidence validation reached without a confirmed target')
            return
        self._publish_column_result()
        if not self.target_annotated_image_saved:
            self._finish(False, 'Target confirmed but Day 2 annotated image was not saved')
            return

        hold_elapsed = None
        if self.sim_time_sec is not None:
            hold_elapsed = max(
                0.0,
                self.sim_time_sec
                - self.target_confirmation.confirmed_stamp_sec,
            )
        if hold_elapsed is None or hold_elapsed < self.result_hold_sec:
            return

        if self.validation_require_all_markers:
            if not self._all_marker_validation_ready():
                wait_start = (
                    self.target_wait_started_sim_sec
                    if self.target_wait_started_sim_sec is not None
                    else self.state_started_sim_sec)
                elapsed = self._simulation_elapsed(wait_start)
                if (
                    elapsed is not None
                    and elapsed > self.validation_all_markers_timeout_sec
                ):
                    missing = [
                        digit for digit in range(1, 6)
                        if digit not in self.all_confirmed_markers
                    ]
                    self._finish(
                        False,
                        'All-marker validation timeout; missing digits: '
                        + ','.join(str(item) for item in missing))
                return

        self._day3_transition(
            'ESTIMATE_TARGET_GEOMETRY',
            'Day 2 marker publication/evidence passed; estimating live bearing, '
            'raw-depth range and independent front-LiDAR clearance.')

    def _start_approach(self, geometry: TargetGeometry) -> None:
        assert self.sim_time_sec is not None
        self.start_target_geometry = geometry
        self.start_lidar_clearance_m = (
            self.latest_lidar.robust_clearance_m
            if self.latest_lidar is not None else None)
        self.minimum_lidar_clearance_during_approach_m = (
            self.start_lidar_clearance_m)
        self.approach_started_sim_sec = self.sim_time_sec
        self.approach_started_wall_monotonic = time.monotonic()
        self.approach_start_odom_xy = self.current_odom_xy
        self.approach_heading_yaw_rad = self.current_yaw
        self.camera_frames_at_approach_start = self.camera_frames_total
        self.depth_frames_at_approach_start = self.depth_frames_total
        self.lidar_frames_at_approach_start = self.lidar_frames_total
        self.last_command_sim_sec = self.sim_time_sec
        self.last_trace_sim_sec = None
        self.approach_outcome = 'running'
        self._day3_transition(
            'APPROACH_TARGET_COLUMN',
            'Bounded +x/+y/+yaw visual/depth/LiDAR approach started.')

    def _publish_approach_head_pose(self) -> None:
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            'head_1_joint',
            'head_2_joint',
        ]

        point = JointTrajectoryPoint()
        point.positions = [
            float(self.head_pan),
            float(self.approach_head_tilt_rad),
        ]

        duration = float(self.approach_head_motion_sec)
        seconds = int(duration)
        nanoseconds = int(
            round((duration - seconds) * 1.0e9)
        )

        if nanoseconds >= 1000000000:
            seconds += 1
            nanoseconds -= 1000000000

        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds
        trajectory.points = [point]

        self.approach_head_pub.publish(trajectory)

    def _approach_head_ready_tick(self) -> bool:
        if self.sim_time_sec is None:
            self._finish(
                False,
                'Simulation clock unavailable while positioning '
                'Day 3 approach head',
            )
            return False

        if not self.approach_head_commanded:
            self.opening_actual_head_tilt_rad = (
                float(self.actual_head_tilt)
                if self.actual_head_tilt is not None
                else None
            )

            self.approach_head_command_sim_sec = float(
                self.sim_time_sec)
            self.approach_head_camera_frame_at_command = (
                self.camera_frames_total)

            # Do not permit geometry from a bbox obtained before the
            # camera moved to its Day 3 shelf-viewing pose.
            self.day3_latest_target_detection = None
            self.day3_latest_target_stamp_sec = None

            self._publish_approach_head_pose()
            self.approach_head_commanded = True

            self.get_logger().info(
                '[DAY3] Approach head command published: '
                f'pan={self.head_pan:.3f} '
                f'tilt={self.approach_head_tilt_rad:.3f}'
            )
            return False

        elapsed = self._simulation_elapsed(
            self.approach_head_command_sim_sec)

        if elapsed is None:
            return False

        actual = self.actual_head_tilt

        position_ok = (
            actual is not None
            and abs(
                float(actual)
                - self.approach_head_tilt_rad
            )
            <= self.approach_head_position_tolerance_rad
        )

        new_camera_frames = (
            self.camera_frames_total
            >= self.approach_head_camera_frame_at_command + 2
        )

        target_after_command = (
            self.day3_latest_target_detection is not None
            and self.day3_latest_target_stamp_sec is not None
            and self.approach_head_command_sim_sec is not None
            and self.day3_latest_target_stamp_sec
            > self.approach_head_command_sim_sec + 0.05
        )

        if (
            position_ok
            and new_camera_frames
            and target_after_command
        ):
            if not self.approach_head_ready:
                self.approach_head_ready = True
                self.get_logger().info(
                    '[DAY3] Approach head ready: '
                    f'target={self.approach_head_tilt_rad:.3f} '
                    f'actual={float(actual):.3f} '
                    f'target_stamp='
                    f'{self.day3_latest_target_stamp_sec:.3f}'
                )

            return True

        if elapsed > self.approach_head_ready_timeout_sec:
            actual_text = (
                'None'
                if actual is None
                else f'{float(actual):.3f}'
            )
            self._finish(
                False,
                'Approach head / fresh target did not become '
                'ready before simulation-time timeout; '
                f'actual_head={actual_text}',
            )
            return False

        return False

    def _travel_arm_joint_state_callback(
        self,
        msg: JointState,
    ) -> None:
        target_names = (
            self.TRAVEL_ARM_LEFT_JOINTS
            + self.TRAVEL_ARM_RIGHT_JOINTS
        )

        saw_target = False

        for name, position in zip(msg.name, msg.position):
            if name in target_names:
                self.travel_arm_joint_positions[name] = float(position)
                saw_target = True

        if saw_target and self.travel_arm_pose_commanded:
            self.travel_arm_joint_updates_after_command += 1

    def _travel_arm_trajectory(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
    ) -> JointTrajectory:
        msg = JointTrajectory()
        msg.joint_names = list(joint_names)

        point = JointTrajectoryPoint()
        point.positions = [
            float(value)
            for value in positions
        ]

        whole_seconds = int(self.travel_arm_motion_sec)
        nanoseconds = int(round(
            (
                self.travel_arm_motion_sec
                - whole_seconds
            )
            * 1.0e9
        ))

        if nanoseconds >= 1_000_000_000:
            whole_seconds += 1
            nanoseconds -= 1_000_000_000

        point.time_from_start.sec = whole_seconds
        point.time_from_start.nanosec = nanoseconds

        msg.points = [point]
        return msg

    def _travel_arm_max_error_now(
        self,
    ) -> Optional[float]:
        targets = {}

        targets.update(dict(zip(
            self.TRAVEL_ARM_LEFT_JOINTS,
            self.TRAVEL_ARM_LEFT_TARGET_RAD,
        )))

        targets.update(dict(zip(
            self.TRAVEL_ARM_RIGHT_JOINTS,
            self.TRAVEL_ARM_RIGHT_TARGET_RAD,
        )))

        if any(
            name not in self.travel_arm_joint_positions
            for name in targets
        ):
            return None

        return max(
            abs(
                self.travel_arm_joint_positions[name]
                - target
            )
            for name, target in targets.items()
        )

    def _publish_travel_arm_pose(
        self,
        stop_base_before_publish: bool = True,
    ) -> None:
        if stop_base_before_publish:
            self.stop_base()

        left_msg = self._travel_arm_trajectory(
            self.TRAVEL_ARM_LEFT_JOINTS,
            self.TRAVEL_ARM_LEFT_TARGET_RAD,
        )

        right_msg = self._travel_arm_trajectory(
            self.TRAVEL_ARM_RIGHT_JOINTS,
            self.TRAVEL_ARM_RIGHT_TARGET_RAD,
        )

        self.travel_arm_left_pub.publish(left_msg)
        self.travel_arm_right_pub.publish(right_msg)

        self.travel_arm_pose_commanded = True
        self.travel_arm_pose_ready = False
        self.travel_arm_command_sim_sec = self.sim_time_sec
        self.travel_arm_ready_sim_sec = None
        self.travel_arm_max_error_rad = None
        self.travel_arm_joint_updates_after_command = 0

        self.get_logger().info(
            '[DAY3] TRAVEL ARM POSE COMMAND: '
            f'name={self.TRAVEL_ARM_POSE_NAME} '
            f'duration={self.travel_arm_motion_sec:.2f}s'
        )

    def _start_travel_arm_pose_early_if_possible(self) -> None:
        """Overlap safe arm compaction with the opening/perception phase."""
        if not self.approach_motion_enabled:
            return

        if self.travel_arm_pose_commanded:
            return

        if self.sim_time_sec is None:
            return

        required_joints = (
            self.TRAVEL_ARM_LEFT_JOINTS
            + self.TRAVEL_ARM_RIGHT_JOINTS
        )

        # Wait until /joint_states proves all 14 arm joints are available.
        if any(
            name not in self.travel_arm_joint_positions
            for name in required_joints
        ):
            return

        # Do NOT publish zero base velocity here.  The validated Day 1
        # clockwise turn is allowed to run concurrently with the arm tuck.
        self._publish_travel_arm_pose(
            stop_base_before_publish=False,
        )

        self.get_logger().info(
            '[DAY3] EARLY TRAVEL ARM START: '
            f'overlapping {self.travel_arm_motion_sec:.2f}s '
            'arm tuck with opening/perception.'
        )

    def _set_travel_arm_pose_tick(self) -> None:
        self.stop_base()

        if not self.travel_arm_pose_commanded:
            self._publish_travel_arm_pose()

        self._day3_transition(
            'WAIT_FOR_TRAVEL_ARM_POSE',
            'Both arms commanded to validated compact PAL home '
            'travel pose; base remains stopped.',
        )

    def _wait_for_travel_arm_pose_tick(self) -> None:
        self.stop_base()

        if not self.travel_arm_pose_commanded:
            self._finish(
                False,
                'Travel-arm wait entered before arm command',
            )
            return

        elapsed = self._simulation_elapsed(
            self.travel_arm_command_sim_sec)

        if elapsed is None:
            return

        max_error = self._travel_arm_max_error_now()

        if max_error is not None:
            self.travel_arm_max_error_rad = float(max_error)

        earliest_ready_sec = max(
            0.0,
            self.travel_arm_motion_sec - 0.20,
        )

        if (
            max_error is not None
            and max_error
            <= self.travel_arm_position_tolerance_rad
            and self.travel_arm_joint_updates_after_command >= 2
            and elapsed >= earliest_ready_sec
        ):
            self.travel_arm_pose_ready = True
            self.travel_arm_ready_sim_sec = self.sim_time_sec

            geometry = self._locked_target_geometry_now()

            if geometry is None:
                self._finish(
                    False,
                    'Travel arms reached pose but locked target '
                    'geometry is unavailable',
                )
                return

            self.get_logger().info(
                '[DAY3] TRAVEL ARM POSE READY: '
                f'max_error={max_error:.4f}rad '
                f'elapsed={elapsed:.3f}s'
            )

            self._start_approach(geometry)
            return

        if elapsed > self.travel_arm_ready_timeout_sec:
            self._finish(
                False,
                'Travel arms did not reach validated pose within '
                'simulation-time timeout',
            )

    def _estimate_geometry_tick(self) -> None:
        self.stop_base()

        if not self._approach_head_ready_tick():
            return
        if not self._target_fresh_for_approach():
            self._enter_hold(
                'RECOVER_TARGET',
                'ESTIMATE_TARGET_GEOMETRY',
                'Target became stale before ranging; holding position.')
            return
        lidar_ok, lidar_reason = self._lidar_ready_now()
        if not lidar_ok:
            self.last_sensor_failure_reason = lidar_reason
            self._enter_hold(
                'SENSOR_HOLD',
                'ESTIMATE_TARGET_GEOMETRY',
                lidar_reason)
            return
        geometry, reason = self._current_target_geometry()
        if geometry is None:
            self.last_geometry_failure_reason = reason
            self._enter_hold(
                'SENSOR_HOLD',
                'ESTIMATE_TARGET_GEOMETRY',
                reason)
            return

        if not self.column_lock_active:
            try:
                self._lock_target_column(geometry)
            except RuntimeError as exc:
                self._finish(False, str(exc))
                return

        if not self.approach_motion_enabled:
            self.start_target_geometry = geometry
            self.final_target_geometry = geometry
            self.start_lidar_clearance_m = (
                self.latest_lidar.robust_clearance_m
                if self.latest_lidar is not None else None)
            self.final_lidar_clearance_m = self.start_lidar_clearance_m
            self.approach_outcome = 'ranging_only'
            self._day3_transition(
                'VERIFY_APPROACH',
                'Motion-disabled ranging test obtained valid bearing, depth, TF '
                'projection and LiDAR clearance.')
            return

        self._day3_transition(
            'SET_TRAVEL_ARM_POSE',
            'Requested shelf column locked; compacting both arms '
            'before any shelf translation.',
        )

    def _hold_elapsed(self) -> Optional[float]:
        return self._simulation_elapsed(self.hold_started_sim_sec)

    def _recover_target_tick(self) -> None:
        self.stop_base()
        if self._target_fresh_for_approach():
            self.hold_started_sim_sec = None
            self._day3_transition(
                self.resume_state_after_hold,
                'Requested marker became fresh again; resuming from safe stop.')
            return
        elapsed = self._hold_elapsed()
        if elapsed is not None and elapsed > self.target_recovery_timeout_sec:
            self._finish(
                False,
                'Requested marker remained stale beyond target recovery timeout')

    def _sensor_hold_tick(self) -> None:
        self.stop_base()
        lidar_ok, _ = self._lidar_ready_now()
        geometry, _ = self._approach_geometry_now()
        if lidar_ok and geometry is not None:
            self.hold_started_sim_sec = None
            self._day3_transition(
                self.resume_state_after_hold,
                'Depth/LiDAR/TF data recovered; resuming from safe stop.')
            return
        elapsed = self._hold_elapsed()
        if elapsed is not None and elapsed > self.sensor_hold_timeout_sec:
            self._finish(
                False,
                'Depth, LiDAR, synchronization or TF remained unavailable '
                'beyond sensor hold timeout')

    def _hard_lidar_stop_active(self) -> bool:
        return (
            self.latest_lidar is not None
            and self.latest_lidar.minimum_clearance_m
            <= self.lidar_hard_stop_m
            and self.latest_lidar.hard_cluster_point_count
            >= self.lidar_hard_cluster_points
        )

    def _safety_hold_tick(self) -> None:
        self.stop_base()
        lidar_ok, _ = self._lidar_ready_now()

        if (
            lidar_ok
            and self.latest_lidar is not None
            and self.latest_lidar.minimum_clearance_m
            >= self.lidar_hard_release_m
        ):
            geometry, _ = self._approach_geometry_now()

            if geometry is not None:
                self.hold_started_sim_sec = None
                self._day3_transition(
                    'APPROACH_TARGET_COLUMN',
                    'LiDAR hard-stop corridor cleared; '
                    'resuming locked-column approach.',
                )
                return

        elapsed = self._hold_elapsed()

        if (
            elapsed is not None
            and elapsed > self.safety_hold_timeout_sec
        ):
            self._finish(
                False,
                'Front-LiDAR hard-stop condition persisted '
                'beyond safety timeout',
            )

    def _heading_error_rad(self) -> float:
        if self.approach_heading_yaw_rad is None or self.current_yaw is None:
            return 0.0
        return normalize_angle(
            self.approach_heading_yaw_rad - self.current_yaw)

    def _final_envelope_satisfied(
        self,
        geometry: TargetGeometry,
        lidar: LidarClearance,
    ) -> bool:
        lidar_ok = (
            self.final_lidar_min_m
            <= lidar.robust_clearance_m
            <= self.final_lidar_max_m
        )

        lateral_ok = (
            abs(geometry.lateral_left_m)
            <= self.final_lateral_tolerance_m
        )

        bearing_ok = (
            abs(geometry.base_bearing_left_rad)
            <= self.final_bearing_tolerance_rad
        )

        heading_ok = (
            abs(self._heading_error_rad())
            <= self.final_heading_tolerance_rad
        )

        if self.column_lock_active:
            # The numeral is only needed to choose the column.  Once locked,
            # final forward stand-off comes from the independent front LiDAR.
            return (
                lidar_ok
                and lateral_ok
                and bearing_ok
                and heading_ok
            )

        return (
            abs(
                geometry.forward_m
                - self.approach_standoff_m
            )
            <= self.final_depth_tolerance_m
            and lidar_ok
            and lateral_ok
            and bearing_ok
            and heading_ok
        )

    def _rate_limited_command(
        self, vx: float, vy: float, wz: float,
    ) -> Tuple[float, float, float]:
        now = self.sim_time_sec
        if now is None or self.last_command_sim_sec is None:
            dt = 0.05
        else:
            dt = max(0.001, min(0.25, now - self.last_command_sim_sec))
        limited_vx = limit_rate(
            vx, self.last_command.linear.x,
            self.approach_linear_accel_limit, dt)
        limited_vy = limit_rate(
            vy, self.last_command.linear.y,
            self.approach_linear_accel_limit, dt)
        limited_wz = limit_rate(
            wz, self.last_command.angular.z,
            self.approach_yaw_accel_limit, dt)
        self.last_command_sim_sec = now
        return limited_vx, limited_vy, limited_wz

    def _record_trace(
        self,
        geometry: Optional[TargetGeometry],
        lidar: Optional[LidarClearance],
        reason: str = '',
    ) -> None:
        if self.sim_time_sec is None:
            return
        if (
            self.last_trace_sim_sec is not None
            and self.sim_time_sec - self.last_trace_sim_sec
            < self.trace_period_sec
        ):
            return
        self.last_trace_sim_sec = self.sim_time_sec
        latest = self.temporal_filter.latest_detection(self.shelf_column)
        pixel_error = None
        confidence = None
        bbox = None
        if latest is not None and self.frame_width is not None:
            pixel_error = latest.bbox.center_x - 0.5 * self.frame_width
            confidence = latest.confidence
            bbox = latest.bbox.as_list()
        sample: Dict[str, object] = {
            'sim_stamp_sec': round(self.sim_time_sec, 6),
            'state': self.state,
            'target_pixel_error_px': (
                round(float(pixel_error), 3)
                if pixel_error is not None else None),
            'target_confidence': (
                round(float(confidence), 6)
                if confidence is not None else None),
            'target_bbox_xywh': bbox,
            'depth_age_sec': self._age(self.latest_depth_stamp_sec),
            'lidar_age_sec': self._age(self.latest_lidar_stamp_sec),
            'camera_frames_total': self.camera_frames_total,
            'depth_frames_total': self.depth_frames_total,
            'lidar_frames_total': self.lidar_frames_total,
            'odom_distance_m': round(self.approach_distance_travelled_m, 6),
            'shelf_heading_reference_deg': (
                round(math.degrees(self.approach_heading_yaw_rad), 6)
                if self.approach_heading_yaw_rad is not None else None),
            'shelf_heading_error_deg': round(
                math.degrees(self._heading_error_rad()), 6),
            'command': {
                'vx_mps': round(float(self.last_command.linear.x), 6),
                'vy_mps': round(float(self.last_command.linear.y), 6),
                'wz_radps': round(float(self.last_command.angular.z), 6),
            },
            'reason': reason,
        }
        if geometry is not None:
            sample['target_geometry'] = geometry.as_dict()
        if lidar is not None:
            sample['lidar'] = lidar.as_dict()
        self.approach_trace.append(sample)
        if len(self.approach_trace) > self.max_trace_samples:
            del self.approach_trace[0]
        self._publish_approach_debug(sample)

    def _publish_approach_debug(self, sample: Dict[str, object]) -> None:
        if self.approach_debug_pub.get_subscription_count() == 0:
            return
        message = String()
        message.data = json.dumps(sample, sort_keys=True)
        self.approach_debug_pub.publish(message)

    def _approach_tick(self) -> None:
        if not self.enable_motion or not self.approach_motion_enabled:
            self._finish(False, 'Approach state entered while motion was disabled')
            return
        elapsed = self._simulation_elapsed(self.approach_started_sim_sec)
        if elapsed is None:
            self._finish(False, 'Simulation clock unavailable during approach')
            return
        if elapsed > self.approach_timeout_sec:
            self._finish(False, 'Shelf approach exceeded simulation-time timeout')
            return

        # The numbered marker was already identified and the physical
        # shelf column was locked before translation.  Losing the numeral at
        # close range is therefore not a mission failure.

        lidar_ok, lidar_reason = self._lidar_ready_now()
        if not lidar_ok:
            self._record_trace(None, self.latest_lidar, lidar_reason)
            self._enter_hold(
                'SENSOR_HOLD',
                'APPROACH_TARGET_COLUMN',
                lidar_reason)
            return
        assert self.latest_lidar is not None

        geometry, geometry_source = self._approach_geometry_now()
        if geometry is None:
            self._record_trace(
                None,
                self.latest_lidar,
                'No live or locked target geometry available',
            )
            self._enter_hold(
                'SENSOR_HOLD',
                'APPROACH_TARGET_COLUMN',
                'No live or locked target geometry available',
            )
            return

        if self._hard_lidar_stop_active():
            self._record_trace(
                geometry, self.latest_lidar,
                'LiDAR hard-stop cluster detected')
            self._enter_hold(
                'SAFETY_HOLD',
                'APPROACH_TARGET_COLUMN',
                'Front-LiDAR hard-stop cluster detected; all base motion stopped.')
            return

        if (
            self.approach_distance_limit_m > 0.0
            and self.approach_distance_travelled_m
            >= self.approach_distance_limit_m
        ):
            self.stop_base()
            self.final_target_geometry = geometry
            self.final_lidar_clearance_m = (
                self.latest_lidar.robust_clearance_m)
            self.approach_outcome = 'partial_distance_limit'
            self._day3_transition(
                'VERIFY_APPROACH',
                'Short-distance validation limit reached without a safety event.')
            return

        if self._final_envelope_satisfied(geometry, self.latest_lidar):
            self.stop_base()
            self.final_settle_count += 1
            self._record_trace(
                geometry, self.latest_lidar,
                f'final settle {self.final_settle_count}/'
                f'{self.final_settle_required}')
            if self.final_settle_count >= self.final_settle_required:
                self.final_target_geometry = geometry
                self.final_lidar_clearance_m = (
                    self.latest_lidar.robust_clearance_m)
                self.approach_outcome = 'safe_standoff'
                self._day3_transition(
                    'VERIFY_APPROACH',
                    'Depth, LiDAR, lateral and bearing envelopes remained stable.')
            return

        self.final_settle_count = 0
        heading_error = self._heading_error_rad()
        self.maximum_abs_heading_error_rad = max(
            self.maximum_abs_heading_error_rad, abs(heading_error))
        control_geometry = self._locked_control_geometry(
            geometry,
            self.latest_lidar,
        )

        command = compute_holonomic_command(
            control_geometry,
            self.latest_lidar.robust_clearance_m,
            self.controller_settings,
            heading_error_rad=heading_error,
        )

        requested_vx = command.vx_mps

        # Near the shelf, complete significant lateral alignment before
        # allowing any additional forward progress.  The bearing envelope
        # is tighter than the raw lateral tolerance at the final stand-off,
        # so derive a conservative lateral release threshold from both.
        lateral_alignment_release_m = min(
            self.final_lateral_tolerance_m,
            0.90
            * self.approach_standoff_m
            * math.tan(self.final_bearing_tolerance_rad),
        )

        if (
            self.column_lock_active
            and not self.near_shelf_lateral_only_active
            and self.latest_lidar.robust_clearance_m
            <= self.lidar_slowdown_m
            and abs(geometry.lateral_left_m)
            > lateral_alignment_release_m
        ):
            self.near_shelf_lateral_only_active = True
            self.near_shelf_lateral_only_events += 1

            self.get_logger().info(
                '[DAY3] LATERAL-ONLY GATE ENTER: '
                f'robust_lidar='
                f'{self.latest_lidar.robust_clearance_m:.3f}m '
                f'left={geometry.lateral_left_m:.3f}m '
                f'release={lateral_alignment_release_m:.3f}m'
            )

        if self.near_shelf_lateral_only_active:
            if (
                abs(geometry.lateral_left_m)
                <= lateral_alignment_release_m
            ):
                self.near_shelf_lateral_only_active = False

                self.get_logger().info(
                    '[DAY3] LATERAL-ONLY GATE EXIT: '
                    f'left={geometry.lateral_left_m:.3f}m '
                    f'release={lateral_alignment_release_m:.3f}m'
                )
            else:
                requested_vx = 0.0

        vx, vy, wz = self._rate_limited_command(
            requested_vx,
            command.vy_mps,
            command.wz_radps,
        )
        self._publish_velocity(vx, vy, wz)
        self._record_trace(geometry, self.latest_lidar)

    def _save_approach_image(self) -> Optional[str]:
        if self.latest_frame_bgr is None or self.target_confirmation is None:
            return None
        geometry = self.final_target_geometry or self.latest_target_geometry
        lidar = self.latest_lidar
        if geometry is None or lidar is None:
            return None
        current_target = self.temporal_filter.confirmed(self.shelf_column)
        if current_target is None:
            current_target = self.target_confirmation
        current_column_bbox = self._estimate_target_column_bbox(
            current_target,
            self.latest_detections,
            self.latest_frame_bgr.shape[1],
            self.latest_frame_bgr.shape[0],
        )
        self.final_approach_column_bbox = current_column_bbox
        utc_now = datetime.now(timezone.utc)
        sim_stamp = self.sim_time_sec or 0.0
        filename = (
            f'shelf_approach_column_{self.shelf_column}_'
            f'sim_{sim_stamp:012.3f}_'
            f'utc_{utc_now.strftime("%Y%m%dT%H%M%S_%fZ")}.png')
        os.makedirs(self.image_output_dir, exist_ok=True)
        final_path = os.path.join(self.image_output_dir, filename)
        temporary_path = final_path + '.tmp.png'
        lines = [
            'KU SPARCy ERC 2026 - DAY 3 LIVE APPROACH EVIDENCE',
            f'target marker: {self.shelf_column} | outcome: {self.approach_outcome}',
            f'depth/base forward: {geometry.forward_m:.3f} m | '
            f'standoff target: {self.approach_standoff_m:.3f} m',
            f'lateral left: {geometry.lateral_left_m:+.3f} m | '
            f'target bearing: {math.degrees(geometry.base_bearing_left_rad):+.2f} deg',
            f'shelf-heading error: {math.degrees(self._heading_error_rad()):+.2f} deg',
            f'LiDAR robust/min: {lidar.robust_clearance_m:.3f}/'
            f'{lidar.minimum_clearance_m:.3f} m',
            f'odom travel: {self.approach_distance_travelled_m:.3f} m | '
            f'safety stops: {self.safety_stop_events}',
            f'sim timestamp: {sim_stamp:.6f} s',
            f'UTC timestamp: {utc_now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}',
        ]
        annotated = draw_detection_overlay(
            self.latest_frame_bgr,
            detections=self.latest_detections,
            target_digit=self.shelf_column,
            confirmed_target=current_target,
            target_column_bbox=current_column_bbox,
            lines=lines,
        )
        if not cv2.imwrite(temporary_path, annotated):
            raise OSError(f'cv2.imwrite returned false for {temporary_path}')
        os.replace(temporary_path, final_path)
        return final_path

    def _verify_approach_tick(self) -> None:
        self.stop_base()
        if self.approach_completed_sim_sec is None:
            self.approach_completed_sim_sec = self.sim_time_sec
            self.approach_completed_wall_monotonic = time.monotonic()
            if self.final_target_geometry is None:
                geometry, _ = self._current_target_geometry()
                self.final_target_geometry = geometry
            if self.final_lidar_clearance_m is None and self.latest_lidar is not None:
                self.final_lidar_clearance_m = (
                    self.latest_lidar.robust_clearance_m)
            try:
                path = self._save_approach_image()
                if path is None:
                    raise OSError('live RGB/target/range evidence unavailable')
                self.approach_image_path = path
                self.approach_image_saved = True
                self.get_logger().info(
                    '[DAY3] Final approach image saved: %s' % path)
            except (OSError, cv2.error) as exc:
                self._finish(
                    False,
                    f'Could not save final live approach image: {exc}')
                return

        elapsed = self._simulation_elapsed(self.approach_completed_sim_sec)
        if elapsed is None or elapsed < self.approach_result_hold_sec:
            return
        if not self.approach_image_saved:
            self._finish(False, 'Final approach image was not saved')
            return
        if self.approach_outcome == 'safe_standoff':
            self._finish(
                True,
                'Requested shelf column reached at controlled depth/LiDAR '
                'stand-off after one-time column identification with '
                'odometry/LiDAR approach')
        elif self.approach_outcome == 'partial_distance_limit':
            self._finish(
                True,
                'Short-distance approach validation completed safely')
        elif self.approach_outcome == 'ranging_only':
            self._finish(
                True,
                'Motion-disabled bearing/depth/LiDAR ranging validation completed')
        else:
            self._finish(False, f'Unknown Day 3 approach outcome: {self.approach_outcome}')

    def _control_tick(self) -> None:
        if self.done:
            return
        if (
            self.sim_time_sec is not None
            and time.monotonic() - self.last_sim_clock_advance_wall_monotonic
            > self.clock_stall_wall_timeout_sec
        ):
            self._finish(
                False,
                'Gazebo /clock stopped advancing beyond wall-time watchdog')
            return
        self._publish_column_result()

        if self.state == 'WAITING_FOR_INPUTS':
            if self._inputs_ready():
                self._start_motion()
                return
            wall_elapsed = time.monotonic() - self.node_started_wall_monotonic
            if wall_elapsed > self.ready_timeout_sec:
                self._finish(
                    False,
                    'Readiness timeout; missing: '
                    + ', '.join(self._missing_startup_inputs()))
            return

        # Start the collision-safe travel-arm pose as early as possible.
        # This is deliberately independent of the frozen Day 2 state machine:
        # opening rotation and perception continue normally in parallel.
        self._start_travel_arm_pose_early_if_possible()

        # Frozen Day 2 motion/search methods are reused directly.
        if self.state == 'ROTATING_FAST':
            self._fast_rotation_tick()
        elif self.state == 'WAITING_FOR_TARGET':
            self._waiting_for_target_tick()
        elif self.state == 'SEARCH_FOR_SHELF':
            self._general_search_tick()
        elif self.state == 'ALIGN_TO_TARGET':
            self._general_align_tick()
        elif self.state == 'VERIFYING_EVIDENCE':
            self._evidence_tick()
        elif self.state == 'ESTIMATE_TARGET_GEOMETRY':
            self._estimate_geometry_tick()
        elif self.state == 'SET_TRAVEL_ARM_POSE':
            self._set_travel_arm_pose_tick()
        elif self.state == 'WAIT_FOR_TRAVEL_ARM_POSE':
            self._wait_for_travel_arm_pose_tick()
        elif self.state == 'APPROACH_TARGET_COLUMN':
            self._approach_tick()
        elif self.state == 'RECOVER_TARGET':
            self._recover_target_tick()
        elif self.state == 'SENSOR_HOLD':
            self._sensor_hold_tick()
        elif self.state == 'SAFETY_HOLD':
            self._safety_hold_tick()
        elif self.state == 'VERIFY_APPROACH':
            self._verify_approach_tick()
        else:
            self._finish(False, f'Unknown Day 3 state: {self.state}')

    def _build_result(self, passed: bool, reason: str) -> Dict[str, Any]:
        approach_head_actual_tilt_rad = (
            float(self.actual_head_tilt)
            if self.actual_head_tilt is not None
            else None
        )

        current_actual_head_tilt = self.actual_head_tilt

        if self.opening_actual_head_tilt_rad is not None:
            self.actual_head_tilt = (
                self.opening_actual_head_tilt_rad)

        try:
            result = super()._build_result(passed, reason)
        finally:
            self.actual_head_tilt = current_actual_head_tilt

        approach_head_position_error_rad = (
            self.approach_head_tilt_rad
            - approach_head_actual_tilt_rad
            if approach_head_actual_tilt_rad is not None
            else None
        )

        now_wall = time.monotonic()
        approach_sim_duration = None
        if self.approach_started_sim_sec is not None:
            end = self.approach_completed_sim_sec or self.sim_time_sec
            if end is not None:
                approach_sim_duration = max(
                    0.0, end - self.approach_started_sim_sec)
        approach_wall_duration = None
        if self.approach_started_wall_monotonic is not None:
            end_wall = self.approach_completed_wall_monotonic or now_wall
            approach_wall_duration = max(
                0.0, end_wall - self.approach_started_wall_monotonic)
        mission_sim_duration = None
        if self.node_started_sim_sec is not None and self.sim_time_sec is not None:
            mission_sim_duration = max(
                0.0, self.sim_time_sec - self.node_started_sim_sec)

        camera_during = max(
            0, self.camera_frames_total - self.camera_frames_at_approach_start)
        depth_during = max(
            0, self.depth_frames_total - self.depth_frames_at_approach_start)
        lidar_during = max(
            0, self.lidar_frames_total - self.lidar_frames_at_approach_start)
        rate_denominator = (
            approach_sim_duration
            if approach_sim_duration is not None and approach_sim_duration > 1.0e-6
            else None)
        camera_hz = (
            camera_during / rate_denominator
            if rate_denominator is not None else None)
        depth_hz = (
            depth_during / rate_denominator
            if rate_denominator is not None else None)
        lidar_hz = (
            lidar_during / rate_denominator
            if rate_denominator is not None else None)
        final_detection = self.temporal_filter.latest_detection(self.shelf_column)
        final_confidence = (
            float(final_detection.confidence)
            if final_detection is not None
            else self.locked_target_confidence
        )
        final_bbox = (
            final_detection.bbox.as_list()
            if final_detection is not None
            else self.locked_target_bbox_xywh
        )

        result.update({
            'day': 3,
            'day2_commit_baseline': (
                '8c921d7ec228297efe683f7eb973ecb55973eaf0'),
            'day2_detector_reused': True,
            'raw_depth_topic': self.DEPTH_TOPIC,
            'color_camera_info_topic': self.COLOR_INFO_TOPIC,
            'depth_camera_info_topic': self.DEPTH_INFO_TOPIC,
            'front_lidar_topic': self.FRONT_SCAN_TOPIC,
            'base_frame': self.BASE_FRAME,
            'approach_motion_enabled': self.approach_motion_enabled,
            'approach_distance_limit_m': self.approach_distance_limit_m,
            'approach_outcome': self.approach_outcome,
            'travel_arm_pose_name': self.TRAVEL_ARM_POSE_NAME,
            'travel_arm_pose_required': bool(
                self.approach_motion_enabled),
            'travel_arm_pose_commanded': (
                self.travel_arm_pose_commanded),
            'travel_arm_pose_ready': self.travel_arm_pose_ready,
            'travel_arm_motion_sec': self.travel_arm_motion_sec,
            'travel_arm_position_tolerance_rad': (
                self.travel_arm_position_tolerance_rad),
            'travel_arm_ready_timeout_sec': (
                self.travel_arm_ready_timeout_sec),
            'travel_arm_command_sim_sec': (
                self.travel_arm_command_sim_sec),
            'travel_arm_ready_sim_sec': (
                self.travel_arm_ready_sim_sec),
            'travel_arm_motion_actual_sim_sec': (
                self.travel_arm_ready_sim_sec
                - self.travel_arm_command_sim_sec
                if (
                    self.travel_arm_ready_sim_sec is not None
                    and self.travel_arm_command_sim_sec is not None
                )
                else None
            ),
            'travel_arm_max_error_rad': (
                self.travel_arm_max_error_rad),
            'travel_arm_joint_updates_after_command': (
                self.travel_arm_joint_updates_after_command),
            'travel_arm_left_target_rad': list(
                self.TRAVEL_ARM_LEFT_TARGET_RAD),
            'travel_arm_right_target_rad': list(
                self.TRAVEL_ARM_RIGHT_TARGET_RAD),
            'travel_arm_actual_positions_rad': {
                name: self.travel_arm_joint_positions.get(name)
                for name in (
                    self.TRAVEL_ARM_LEFT_JOINTS
                    + self.TRAVEL_ARM_RIGHT_JOINTS
                )
            },
            'column_lock_active': self.column_lock_active,
            'column_lock_mode': (
                'identify_once_then_odom_lidar'
                if self.column_lock_active
                else 'not_locked'
            ),
            'locked_target_odom_xy': (
                list(self.locked_target_odom_xy)
                if self.locked_target_odom_xy is not None
                else None
            ),
            'locked_target_confidence': (
                self.locked_target_confidence),
            'locked_target_bbox_xywh': (
                self.locked_target_bbox_xywh),
            'final_geometry_source': (
                'locked_column_odometry'
                if self.column_lock_active
                else 'live_marker_depth'
            ),
            'final_forward_standoff_source': (
                'front_lidar'
                if self.column_lock_active
                else 'marker_depth_and_front_lidar'
            ),
            'opening_actual_head_tilt_snapshot_rad': (
                self.opening_actual_head_tilt_rad),
            'approach_head_tilt_target_rad': (
                self.approach_head_tilt_rad),
            'approach_head_actual_tilt_rad': (
                approach_head_actual_tilt_rad),
            'approach_head_position_error_rad': (
                approach_head_position_error_rad),
            'approach_head_commanded': (
                self.approach_head_commanded),
            'approach_head_ready': (
                self.approach_head_ready),
            'mission_duration_sim_sec': mission_sim_duration,
            'mission_duration_wall_sec': (
                now_wall - self.node_started_wall_monotonic),
            'approach_duration_sim_sec': approach_sim_duration,
            'approach_duration_wall_sec': approach_wall_duration,
            'start_target_geometry': (
                self.start_target_geometry.as_dict()
                if self.start_target_geometry is not None else None),
            'final_target_geometry': (
                self.final_target_geometry.as_dict()
                if self.final_target_geometry is not None else None),
            'start_depth_forward_m': (
                self.start_target_geometry.forward_m
                if self.start_target_geometry is not None else None),
            'final_depth_forward_m': (
                self.final_target_geometry.forward_m
                if self.final_target_geometry is not None else None),
            'start_lidar_clearance_m': self.start_lidar_clearance_m,
            'final_lidar_clearance_m': self.final_lidar_clearance_m,
            'minimum_lidar_clearance_m': self.minimum_lidar_clearance_m,
            'minimum_lidar_clearance_during_approach_m': (
                self.minimum_lidar_clearance_during_approach_m),
            'approach_distance_travelled_m': (
                self.approach_distance_travelled_m),
            'approach_heading_reference_deg': (
                math.degrees(self.approach_heading_yaw_rad)
                if self.approach_heading_yaw_rad is not None else None),
            'final_heading_error_deg': math.degrees(
                self._heading_error_rad()),
            'maximum_abs_heading_error_deg': math.degrees(
                self.maximum_abs_heading_error_rad),
            'camera_frames_during_approach': camera_during,
            'depth_frames_total': self.depth_frames_total,
            'depth_frames_during_approach': depth_during,
            'lidar_frames_total': self.lidar_frames_total,
            'lidar_frames_during_approach': lidar_during,
            'camera_effective_hz_during_approach': camera_hz,
            'depth_effective_hz_during_approach': depth_hz,
            'lidar_effective_hz_during_approach': lidar_hz,
            'depth_encoding': self.latest_depth_encoding,
            'depth_conversion_failures': self.depth_conversion_failures,
            'lidar_transform_failures': self.lidar_transform_failures,
            'camera_info_updates': self.camera_info_updates,
            'target_stale_events': self.target_stale_events,
            'sensor_hold_events': self.sensor_hold_events,
            'safety_stop_events': self.safety_stop_events,
            'final_settle_samples': self.final_settle_count,
            'final_target_confidence': final_confidence,
            'final_target_bbox_xywh': final_bbox,
            'final_target_column_bbox_xywh': (
                self.final_approach_column_bbox.as_list()
                if self.final_approach_column_bbox is not None else None),
            'maximum_command_abs_vx_mps': self.maximum_abs_vx,
            'maximum_command_abs_vy_mps': self.maximum_abs_vy,
            'maximum_command_abs_wz_radps': self.maximum_abs_wz,
            'controller_configuration': {
                'coordinate_convention': (
                    '+x forward, +y left, +yaw counter-clockwise'),
                'approach_standoff_m': self.approach_standoff_m,
                'max_forward_speed_mps': self.approach_max_forward_speed,
                'min_forward_speed_mps': self.approach_min_forward_speed,
                'max_lateral_speed_mps': self.approach_max_lateral_speed,
                'max_yaw_speed_radps': self.approach_max_yaw_speed,
                'forward_kp': self.approach_forward_kp,
                'lateral_kp': self.approach_lateral_kp,
                'yaw_kp': self.approach_yaw_kp,
                'lidar_corridor_half_width_m': (
                    self.lidar_corridor_half_width),
                'lidar_slowdown_m': self.lidar_slowdown_m,
                'lidar_hard_stop_m': self.lidar_hard_stop_m,
                'lidar_hard_release_m': self.lidar_hard_release_m,
                'final_depth_tolerance_m': self.final_depth_tolerance_m,
                'final_lidar_interval_m': [
                    self.final_lidar_min_m, self.final_lidar_max_m],
                'final_lateral_tolerance_m': (
                    self.final_lateral_tolerance_m),
                'final_bearing_tolerance_deg': math.degrees(
                    self.final_bearing_tolerance_rad),
                'final_heading_tolerance_deg': math.degrees(
                    self.final_heading_tolerance_rad),
                'yaw_control_mode': 'hold post-opening shelf-facing heading',
            },
            'approach_annotated_image_saved': self.approach_image_saved,
            'approach_annotated_image_path': self.approach_image_path,
            'approach_image_contains_live_range': self.approach_image_saved,
            'approach_image_contains_sim_timestamp': self.approach_image_saved,
            'approach_image_contains_utc_timestamp': self.approach_image_saved,
            'last_geometry_failure_reason': self.last_geometry_failure_reason,
            'last_sensor_failure_reason': self.last_sensor_failure_reason,
            'approach_trace_samples': len(self.approach_trace),
            'approach_trace': self.approach_trace,
        })
        return result

    def _finish(self, passed: bool, reason: str) -> None:
        if self.done:
            return
        self.stop_base()
        if self.target_confirmation is not None:
            self._publish_column_result(force=True)
        result = self._build_result(passed, reason)
        try:
            self._write_result(result)
        except OSError as exc:
            passed = False
            reason = f'Could not write Day 3 result file: {exc}'
            result['passed'] = False
            result['state'] = 'FAILED'
            result['reason'] = reason
        self.passed = passed
        self.failure_reason = '' if passed else reason
        self.state = 'PASSED' if passed else 'FAILED'
        self.done = True
        self._publish_status(self.state, reason)
        level = self.get_logger().info if passed else self.get_logger().error
        level('[DAY3][%s] %s | result=%s'
              % (self.state, reason, self.result_path))


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[Day3Mission] = None
    exit_code = 1
    try:
        node = Day3Mission()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.10)
        if node.done and node.passed:
            exit_code = 0
    except KeyboardInterrupt:
        if node is not None:
            node.stop_base()
            node.get_logger().warning(
                '[DAY3] Interrupted; repeated zero-velocity commands sent')
        exit_code = 130
    except Exception as exc:
        if node is not None:
            node.stop_base()
            node.get_logger().error(
                '[DAY3][FAILED] Unhandled exception: %s' % exc)
        else:
            print('[DAY3][FAILED] Could not initialize day3_mission: %s' % exc)
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
