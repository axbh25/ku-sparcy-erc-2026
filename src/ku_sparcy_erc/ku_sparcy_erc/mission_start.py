#!/usr/bin/env python3
"""Day 2 integrated opening motion and shelf-marker perception.

This node extends the validated Day 1 opening sequence without changing the
Day 1 implementation.  It starts processing RGB frames immediately, detects
official shelf marker textures 1-5 using deterministic OpenCV template-shape
matching, temporally confirms the requested marker, publishes the required ERC
Int32 result, saves live annotated evidence, and supports two search modes:

* phase1_fast_start=true: preserve the Day 1 closed-loop clockwise 90 degree
  opening turn and continue perception throughout the turn.
* phase1_fast_start=false: orientation-independent clockwise visual search,
  followed by pixel-space alignment to the confirmed requested marker.

Motion deadlines use Gazebo /clock simulation time.  Readiness deadlines use
wall time so a dead simulation still fails rather than waiting forever.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ku_sparcy_erc.marker_detector import (
    BoundingBox,
    ConfirmedMarker,
    DetectorSettings,
    MarkerDetection,
    ShelfMarkerDetector,
    TemporalMarkerFilter,
    TemporalSettings,
    draw_detection_overlay,
)


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract planar yaw from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class MissionStart(Node):
    """Opening motion plus temporally filtered target-marker perception."""

    VALID_COLOURS = {'red', 'green', 'yellow', 'blue'}
    CAMERA_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
    COLUMN_TOPIC = '/erc/shelf_column_identification'

    def __init__(self) -> None:
        super().__init__('mission_start')

        # Required competition inputs.
        self.declare_parameter('shelf_column_number', 1)
        self.declare_parameter('book_colour', 'red')
        self.declare_parameter('phase1_fast_start', True)
        self.declare_parameter('enable_motion', True)

        # Day 1 controller baseline.  Keep these defaults synchronized with
        # config/opening_sequence.yaml unless a later measured retune is made.
        self.declare_parameter('clockwise_rotation_deg', 90.0)
        self.declare_parameter('max_angular_speed', 0.60)
        self.declare_parameter('min_angular_speed', 0.10)
        self.declare_parameter('yaw_kp', 1.40)
        self.declare_parameter('yaw_tolerance_deg', 1.50)
        self.declare_parameter('settle_samples', 5)
        self.declare_parameter('ready_timeout_sec', 25.0)
        self.declare_parameter('motion_timeout_sec', 10.0)
        self.declare_parameter('head_pan_rad', 0.0)
        self.declare_parameter('head_tilt_rad', 0.25)
        self.declare_parameter('head_motion_sec', 1.5)
        self.declare_parameter('head_tilt_tolerance_rad', 0.05)

        # Perception parameters.
        self.declare_parameter('perception_enabled', True)
        self.declare_parameter('process_rate_hz', 12.0)
        self.declare_parameter('roi_top_fraction', 0.04)
        self.declare_parameter('roi_bottom_fraction', 0.68)
        self.declare_parameter('dark_threshold', 115)
        self.declare_parameter('min_bright_ring_fraction', 0.48)
        self.declare_parameter('min_neutral_dark_fraction', 0.65)
        self.declare_parameter('min_classifier_confidence', 0.58)
        self.declare_parameter('min_classifier_margin', 0.015)
        self.declare_parameter('confirmation_frames', 3)
        self.declare_parameter('confirmation_window_sec', 0.75)
        self.declare_parameter('confirmation_min_span_sec', 0.04)
        self.declare_parameter('confirmation_average_confidence', 0.62)
        self.declare_parameter('target_stale_timeout_sec', 1.0)
        self.declare_parameter('post_turn_detection_timeout_sec', 6.0)
        self.declare_parameter('publish_period_sec', 0.20)
        self.declare_parameter('result_hold_sec', 0.30)

        # General orientation-independent visual search and alignment.
        self.declare_parameter('general_search_angular_speed', 0.40)
        self.declare_parameter('general_search_timeout_sec', 25.0)
        self.declare_parameter('general_search_max_sweep_deg', 390.0)
        self.declare_parameter('general_align_kp', 0.0025)
        self.declare_parameter('general_align_max_speed', 0.25)
        self.declare_parameter('general_align_min_speed', 0.07)
        self.declare_parameter('general_align_tolerance_px', 30.0)
        self.declare_parameter('general_align_settle_samples', 4)

        # Validation/evidence outputs.  The package source directory is part of
        # the official Docker bind mount, so files persist on the host repo.
        self.declare_parameter(
            'result_path', '/opt/erc_ws/src/ku_sparcy_erc/day2_result.json')
        self.declare_parameter(
            'image_output_dir',
            '/opt/erc_ws/src/ku_sparcy_erc/erc_images')
        self.declare_parameter('validation_require_all_markers', False)
        self.declare_parameter('validation_all_markers_timeout_sec', 6.0)

        self._read_parameters()
        self._validate_parameters()

        # ROS interfaces.
        result_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.column_pub = self.create_publisher(
            Int32, self.COLUMN_TOPIC, result_qos)
        self.status_pub = self.create_publisher(
            String, '/ku_sparcy/mission_status', status_qos)
        self.debug_pub = self.create_publisher(
            String, '/ku_sparcy/marker_debug', 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(
            Image, self.CAMERA_TOPIC, self._image_callback,
            qos_profile_sensor_data)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback,
            qos_profile_sensor_data)
        self.clock_sub = self.create_subscription(
            Clock, '/clock', self._clock_callback, qos_profile_sensor_data)

        # Perception objects are initialized before the first spin, so the first
        # accepted image callback can be processed immediately.
        marker_directory = os.path.join(
            get_package_share_directory('erc_description'),
            'models', 'number_marker', 'textures')
        detector_settings = DetectorSettings(
            roi_top_fraction=self.roi_top_fraction,
            roi_bottom_fraction=self.roi_bottom_fraction,
            dark_threshold=self.dark_threshold,
            min_bright_ring_fraction=self.min_bright_ring_fraction,
            min_neutral_dark_fraction=self.min_neutral_dark_fraction,
            min_classifier_confidence=self.min_classifier_confidence,
            min_classifier_margin=self.min_classifier_margin,
        )
        temporal_settings = TemporalSettings(
            required_frames=self.confirmation_frames,
            window_sec=self.confirmation_window_sec,
            min_span_sec=self.confirmation_min_span_sec,
            min_average_confidence=self.confirmation_average_confidence,
            fresh_track_timeout_sec=self.target_stale_timeout_sec,
        )
        self.bridge = CvBridge()
        self.detector = ShelfMarkerDetector(
            marker_directory, detector_settings)
        self.temporal_filter = TemporalMarkerFilter(temporal_settings)
        self.marker_template_directory = marker_directory

        # Timing and state.
        self.node_started_wall_monotonic = time.monotonic()
        self.node_started_unix_time = time.time()
        self.node_started_sim_sec: Optional[float] = None
        self.sim_time_sec: Optional[float] = None
        self.state_started_sim_sec: Optional[float] = None
        self.motion_started_wall_monotonic: Optional[float] = None
        self.motion_started_sim_sec: Optional[float] = None
        self.motion_completed_wall_monotonic: Optional[float] = None
        self.motion_completed_sim_sec: Optional[float] = None
        self.target_wait_started_sim_sec: Optional[float] = None

        self.state = 'WAITING_FOR_INPUTS'
        self.done = False
        self.passed = False
        self.failure_reason = ''

        # Robot observations.
        self.current_yaw: Optional[float] = None
        self.previous_yaw: Optional[float] = None
        self.unwrapped_yaw = 0.0
        self.search_start_unwrapped_yaw: Optional[float] = None
        self.motion_start_unwrapped_yaw: Optional[float] = None
        self.general_search_signed_rotation_rad = 0.0
        self.initial_yaw: Optional[float] = None
        self.target_yaw: Optional[float] = None
        self.actual_head_tilt: Optional[float] = None
        self.head_command_sent = False
        self.rotation_settle_samples = 0
        self.align_settle_samples = 0
        self.rotation_completed = False

        # Camera and perception observations.
        self.camera_frames_total = 0
        self.camera_frames_during_motion = 0
        self.frames_at_motion_start = 0
        self.processed_frames = 0
        self.last_processed_stamp_sec: Optional[float] = None
        self.last_image_stamp_ns: Optional[int] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.latest_frame_bgr: Optional[np.ndarray] = None
        self.latest_detections: List[MarkerDetection] = []
        self.latest_detector_diagnostics: Dict[str, object] = {}
        self.detector_processing_wall_sec_total = 0.0
        self.detector_processing_wall_sec_max = 0.0
        self.detector_processing_samples = 0
        self.all_confirmed_markers: Dict[int, ConfirmedMarker] = {}

        self.target_first_detection_sim_sec: Optional[float] = None
        self.target_first_detection_wall_sec: Optional[float] = None
        self.target_first_detection_bbox: Optional[List[int]] = None
        self.target_confirmation: Optional[ConfirmedMarker] = None
        self.target_confirmation_yaw: Optional[float] = None
        self.target_confirmation_wall_sec: Optional[float] = None
        self.target_confirmation_state: Optional[str] = None
        self.target_confirmed_during_motion = False
        self.target_annotated_image_path: Optional[str] = None
        self.target_annotated_image_saved = False
        self.target_column_bbox: Optional[BoundingBox] = None
        self.target_publication_count = 0
        self.last_publish_sim_sec: Optional[float] = None
        self.last_publish_wall_monotonic: Optional[float] = None
        self.state_history: List[Dict[str, object]] = [{
            'state': self.state,
            'sim_stamp_sec': None,
            'wall_elapsed_sec': 0.0,
        }]

        self.control_timer = self.create_timer(0.05, self._control_tick)
        self._publish_status(
            'WAITING_FOR_INPUTS',
            'Waiting for /clock, odometry, RGB camera and head controller.')
        self.get_logger().info(
            '[DAY2] target=%d colour=%s fast_start=%s motion=%s templates=%s'
            % (
                self.shelf_column,
                self.book_colour,
                self.phase1_fast_start,
                self.enable_motion,
                self.marker_template_directory,
            )
        )

    def _read_parameters(self) -> None:
        get = self.get_parameter
        self.shelf_column = int(get('shelf_column_number').value)
        self.book_colour = str(get('book_colour').value).strip().lower()
        self.phase1_fast_start = bool(get('phase1_fast_start').value)
        self.enable_motion = bool(get('enable_motion').value)

        self.rotation_deg = float(get('clockwise_rotation_deg').value)
        self.max_speed = float(get('max_angular_speed').value)
        self.min_speed = float(get('min_angular_speed').value)
        self.yaw_kp = float(get('yaw_kp').value)
        self.yaw_tolerance_rad = math.radians(
            float(get('yaw_tolerance_deg').value))
        self.rotation_settle_required = int(get('settle_samples').value)
        self.ready_timeout_sec = float(get('ready_timeout_sec').value)
        self.motion_timeout_sec = float(get('motion_timeout_sec').value)
        self.head_pan = float(get('head_pan_rad').value)
        self.head_tilt = float(get('head_tilt_rad').value)
        self.head_motion_sec = float(get('head_motion_sec').value)
        self.head_tilt_tolerance = float(
            get('head_tilt_tolerance_rad').value)

        self.perception_enabled = bool(get('perception_enabled').value)
        self.process_rate_hz = float(get('process_rate_hz').value)
        self.roi_top_fraction = float(get('roi_top_fraction').value)
        self.roi_bottom_fraction = float(get('roi_bottom_fraction').value)
        self.dark_threshold = int(get('dark_threshold').value)
        self.min_bright_ring_fraction = float(
            get('min_bright_ring_fraction').value)
        self.min_neutral_dark_fraction = float(
            get('min_neutral_dark_fraction').value)
        self.min_classifier_confidence = float(
            get('min_classifier_confidence').value)
        self.min_classifier_margin = float(
            get('min_classifier_margin').value)
        self.confirmation_frames = int(get('confirmation_frames').value)
        self.confirmation_window_sec = float(
            get('confirmation_window_sec').value)
        self.confirmation_min_span_sec = float(
            get('confirmation_min_span_sec').value)
        self.confirmation_average_confidence = float(
            get('confirmation_average_confidence').value)
        self.target_stale_timeout_sec = float(
            get('target_stale_timeout_sec').value)
        self.post_turn_detection_timeout_sec = float(
            get('post_turn_detection_timeout_sec').value)
        self.publish_period_sec = float(get('publish_period_sec').value)
        self.result_hold_sec = float(get('result_hold_sec').value)

        self.general_search_speed = float(
            get('general_search_angular_speed').value)
        self.general_search_timeout_sec = float(
            get('general_search_timeout_sec').value)
        self.general_search_max_sweep_rad = math.radians(
            float(get('general_search_max_sweep_deg').value))
        self.general_align_kp = float(get('general_align_kp').value)
        self.general_align_max_speed = float(
            get('general_align_max_speed').value)
        self.general_align_min_speed = float(
            get('general_align_min_speed').value)
        self.general_align_tolerance_px = float(
            get('general_align_tolerance_px').value)
        self.general_align_settle_required = int(
            get('general_align_settle_samples').value)

        self.result_path = str(get('result_path').value)
        self.image_output_dir = str(get('image_output_dir').value)
        self.validation_require_all_markers = bool(
            get('validation_require_all_markers').value)
        self.validation_all_markers_timeout_sec = float(
            get('validation_all_markers_timeout_sec').value)

    def _validate_parameters(self) -> None:
        if not 1 <= self.shelf_column <= 5:
            raise ValueError('shelf_column_number must be an integer from 1 to 5')
        if self.book_colour not in self.VALID_COLOURS:
            raise ValueError(
                'book_colour must be one of: blue, green, red, yellow')
        if self.rotation_deg <= 0.0 or self.rotation_deg > 180.0:
            raise ValueError('clockwise_rotation_deg must be in (0, 180]')
        if not 0.0 < self.min_speed <= self.max_speed:
            raise ValueError(
                'Day 1 speeds must satisfy 0 < min <= max')
        if self.yaw_kp <= 0.0 or self.yaw_tolerance_rad <= 0.0:
            raise ValueError('Day 1 yaw controller values must be positive')
        if self.rotation_settle_required < 1:
            raise ValueError('settle_samples must be at least 1')
        if self.ready_timeout_sec <= 0.0 or self.motion_timeout_sec <= 0.0:
            raise ValueError('timeouts must be positive')
        if self.head_tilt_tolerance <= 0.0:
            raise ValueError('head_tilt_tolerance_rad must be positive')
        if not self.perception_enabled:
            raise ValueError('Day 2 requires perception_enabled:=true')
        if not 1.0 <= self.process_rate_hz <= 30.0:
            raise ValueError('process_rate_hz must be in [1, 30]')
        if not 0.0 < self.publish_period_sec <= 2.0:
            raise ValueError('publish_period_sec must be in (0, 2]')
        if not 0.0 <= self.result_hold_sec <= 2.0:
            raise ValueError('result_hold_sec must be in [0, 2]')
        if not 0.0 < self.general_search_speed <= self.max_speed:
            raise ValueError(
                'general_search_angular_speed must be positive and no larger '
                'than the validated Day 1 max speed')
        if self.general_search_timeout_sec <= 0.0:
            raise ValueError('general_search_timeout_sec must be positive')
        if self.general_search_max_sweep_rad < math.radians(360.0):
            raise ValueError('general search sweep must be at least 360 degrees')
        if not 0.0 < self.general_align_min_speed <= self.general_align_max_speed:
            raise ValueError('general alignment speeds are invalid')
        if self.general_align_settle_required < 1:
            raise ValueError('general_align_settle_samples must be at least 1')
        if not self.result_path:
            raise ValueError('result_path cannot be empty')
        if not self.image_output_dir:
            raise ValueError('image_output_dir cannot be empty')

    def _clock_callback(self, msg: Clock) -> None:
        stamp = float(msg.clock.sec) + float(msg.clock.nanosec) * 1.0e-9
        self.sim_time_sec = stamp
        if self.node_started_sim_sec is None:
            self.node_started_sim_sec = stamp

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        if self.previous_yaw is not None:
            self.unwrapped_yaw += normalize_angle(yaw - self.previous_yaw)
        self.previous_yaw = yaw
        self.current_yaw = yaw

    def _joint_state_callback(self, msg: JointState) -> None:
        try:
            index = msg.name.index('head_2_joint')
        except ValueError:
            return
        if index < len(msg.position):
            self.actual_head_tilt = float(msg.position[index])

    def _message_stamp_sec(self, msg: Image) -> Optional[float]:
        stamp = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1.0e-9
        )
        if stamp > 0.0:
            return stamp
        return self.sim_time_sec

    def _image_callback(self, msg: Image) -> None:
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return
        self.camera_frames_total += 1
        if self.motion_started_sim_sec is not None and not self.rotation_completed:
            self.camera_frames_during_motion = max(
                0, self.camera_frames_total - self.frames_at_motion_start)
        self.last_image_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )
        self.frame_width = int(msg.width)
        self.frame_height = int(msg.height)

        stamp_sec = self._message_stamp_sec(msg)
        if stamp_sec is None:
            return
        minimum_interval = 1.0 / self.process_rate_hz
        if (
            self.last_processed_stamp_sec is not None
            and stamp_sec - self.last_processed_stamp_sec < minimum_interval
        ):
            return

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().warning(
                f'[DAY2] Could not convert RGB frame: {exc}')
            return
        if frame_bgr is None or frame_bgr.size == 0:
            return

        self.last_processed_stamp_sec = stamp_sec
        self.processed_frames += 1
        self.latest_frame_bgr = frame_bgr

        detector_wall_start = time.perf_counter()
        try:
            detections, diagnostics = self.detector.detect(frame_bgr)
        except (ValueError, cv2.error) as exc:
            elapsed_wall = time.perf_counter() - detector_wall_start
            self.detector_processing_wall_sec_total += elapsed_wall
            self.detector_processing_wall_sec_max = max(
                self.detector_processing_wall_sec_max, elapsed_wall)
            self.detector_processing_samples += 1
            self.get_logger().warning(
                f'[DAY2] Detector rejected frame {self.processed_frames}: {exc}')
            return
        elapsed_wall = time.perf_counter() - detector_wall_start
        self.detector_processing_wall_sec_total += elapsed_wall
        self.detector_processing_wall_sec_max = max(
            self.detector_processing_wall_sec_max, elapsed_wall)
        self.detector_processing_samples += 1

        self.latest_detections = detections
        self.latest_detector_diagnostics = diagnostics
        new_confirmations = self.temporal_filter.update(
            detections,
            stamp_sec=stamp_sec,
            frame_index=self.processed_frames,
        )
        self.all_confirmed_markers = self.temporal_filter.all_confirmed()

        target_single_frame = max(
            (item for item in detections if item.digit == self.shelf_column),
            key=lambda item: item.confidence,
            default=None,
        )
        if (
            target_single_frame is not None
            and self.target_first_detection_sim_sec is None
        ):
            self.target_first_detection_sim_sec = stamp_sec
            self.target_first_detection_wall_sec = (
                time.monotonic() - self.node_started_wall_monotonic)
            self.target_first_detection_bbox = (
                target_single_frame.bbox.as_list())
            self.get_logger().info(
                '[DAY2] First accepted target observation: digit=%d '
                'confidence=%.3f bbox=%s sim=%.3f'
                % (
                    self.shelf_column,
                    target_single_frame.confidence,
                    target_single_frame.bbox.as_list(),
                    stamp_sec,
                )
            )

        for marker in new_confirmations:
            self.get_logger().info(
                '[DAY2] Temporal confirmation: digit=%d confidence=%.3f '
                'frames=%d bbox=%s sim=%.3f'
                % (
                    marker.digit,
                    marker.confidence,
                    marker.confirming_frames,
                    marker.bbox.as_list(),
                    marker.confirmed_stamp_sec,
                )
            )

        confirmed_target = self.temporal_filter.confirmed(self.shelf_column)
        if confirmed_target is not None and self.target_confirmation is None:
            self._accept_target_confirmation(
                confirmed_target, frame_bgr, detections)

        self._publish_debug(detections, new_confirmations, stamp_sec)

    def _accept_target_confirmation(
        self,
        marker: ConfirmedMarker,
        frame_bgr: np.ndarray,
        detections: Sequence[MarkerDetection],
    ) -> None:
        self.target_confirmation = marker
        self.target_confirmation_yaw = self.current_yaw
        self.target_confirmation_wall_sec = (
            time.monotonic() - self.node_started_wall_monotonic)
        self.target_confirmation_state = self.state
        self.target_confirmed_during_motion = (
            self.motion_started_sim_sec is not None
            and not self.rotation_completed
        )
        self.target_column_bbox = self._estimate_target_column_bbox(
            marker, detections, frame_bgr.shape[1], frame_bgr.shape[0])
        self._publish_column_result(force=True)
        try:
            path = self._save_annotated_target_image(
                frame_bgr, detections, marker)
            self.target_annotated_image_path = path
            self.target_annotated_image_saved = True
            self.get_logger().info(
                '[DAY2] Annotated target image saved: %s' % path)
        except (OSError, cv2.error) as exc:
            self.target_annotated_image_saved = False
            self.get_logger().error(
                '[DAY2] Could not save annotated target image: %s' % exc)

        self._publish_status(
            'TARGET_MARKER_FOUND',
            'Requested marker was temporally confirmed and published.')
        self.get_logger().info(
            '[DAY2] TARGET CONFIRMED digit=%d confidence=%.3f frames=%d '
            'during_motion=%s'
            % (
                marker.digit,
                marker.confidence,
                marker.confirming_frames,
                self.target_confirmed_during_motion,
            )
        )

    @staticmethod
    def _estimate_target_column_bbox(
        marker: ConfirmedMarker,
        detections: Sequence[MarkerDetection],
        frame_width: int,
        frame_height: int,
    ) -> BoundingBox:
        """Estimate a visible shelf-column region from live marker geometry.

        The scoring guide asks for a box around the target shelf column, not
        merely the digit glyph.  When multiple markers are visible, their
        centre spacing gives the most useful projected column width.  During
        an early partial view, the known square marker plate and 1 m shelf
        column geometry provide a conservative fallback based on glyph height.
        This is annotation geometry only; it is never used to identify the
        requested digit or to read hidden simulator state.
        """
        centres = sorted(
            detection.bbox.center_x for detection in detections
            if detection.bbox.height > 0
        )
        spacings = [
            centres[index + 1] - centres[index]
            for index in range(len(centres) - 1)
            if centres[index + 1] - centres[index] >= 18.0
        ]
        if spacings:
            projected_width = 0.88 * float(np.median(spacings))
        else:
            projected_width = 5.2 * float(marker.bbox.height)
        projected_width = clamp(
            projected_width, 60.0, 0.32 * float(frame_width))

        x1 = int(round(marker.bbox.center_x - 0.5 * projected_width))
        x2 = int(round(marker.bbox.center_x + 0.5 * projected_width))
        y1 = int(round(marker.bbox.y - 0.65 * marker.bbox.height))
        y2 = max(
            int(round(0.92 * frame_height)),
            int(round(marker.bbox.y2 + 5.0 * marker.bbox.height)),
        )
        return BoundingBox(
            x=x1,
            y=y1,
            width=max(1, x2 - x1),
            height=max(1, y2 - y1),
        ).clipped(frame_width, frame_height)

    def _save_annotated_target_image(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[MarkerDetection],
        marker: ConfirmedMarker,
    ) -> str:
        os.makedirs(self.image_output_dir, exist_ok=True)
        sim_stamp = marker.confirmed_stamp_sec
        utc_now = datetime.now(timezone.utc)
        utc_text = utc_now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        filename = (
            f'shelf_column_{self.shelf_column}_'
            f'sim_{sim_stamp:012.3f}_'
            f'utc_{utc_now.strftime("%Y%m%dT%H%M%S_%fZ")}.png'
        )
        final_path = os.path.join(self.image_output_dir, filename)
        temporary_path = final_path + '.tmp.png'
        yaw_text = (
            f'{math.degrees(self.current_yaw):.2f}'
            if self.current_yaw is not None else 'unknown')
        lines = [
            'KU SPARCy ERC 2026 - LIVE CAMERA EVIDENCE',
            f'target shelf marker: {self.shelf_column}',
            f'confidence: {marker.confidence:.3f} | '
            f'confirming frames: {marker.confirming_frames}',
            f'sim timestamp: {sim_stamp:.6f} s',
            f'UTC timestamp: {utc_text}',
            f'robot yaw at confirmation: {yaw_text} deg',
            f'target column box: {self.target_column_bbox.as_list() if self.target_column_bbox else None}',
            f'state: {self.state}',
        ]
        annotated = draw_detection_overlay(
            frame_bgr,
            detections=detections,
            target_digit=self.shelf_column,
            confirmed_target=marker,
            target_column_bbox=self.target_column_bbox,
            lines=lines,
        )
        if not cv2.imwrite(temporary_path, annotated):
            raise OSError(f'cv2.imwrite returned false for {temporary_path}')
        os.replace(temporary_path, final_path)
        return final_path

    def _head_controller_is_ready(self) -> bool:
        return self.head_pub.get_subscription_count() > 0

    def _send_head_command(self) -> None:
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [self.head_pan, self.head_tilt]
        point.time_from_start.sec = int(self.head_motion_sec)
        point.time_from_start.nanosec = int(
            (self.head_motion_sec - int(self.head_motion_sec)) * 1.0e9)
        msg.points = [point]
        self.head_pub.publish(msg)
        self.head_command_sent = True
        self.get_logger().info(
            '[DAY2] Head command published: pan=%.3f tilt=%.3f'
            % (self.head_pan, self.head_tilt))

    def _publish_base_speed(self, angular_z: float) -> None:
        cmd = Twist()
        cmd.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd)

    def stop_base(self) -> None:
        for _ in range(4):
            self._publish_base_speed(0.0)

    def _publish_column_result(self, force: bool = False) -> None:
        if self.target_confirmation is None:
            return
        now_wall = time.monotonic()
        if not force:
            if self.sim_time_sec is not None and self.last_publish_sim_sec is not None:
                if self.sim_time_sec - self.last_publish_sim_sec < self.publish_period_sec:
                    return
            elif self.last_publish_wall_monotonic is not None:
                if now_wall - self.last_publish_wall_monotonic < self.publish_period_sec:
                    return
        msg = Int32()
        msg.data = int(self.shelf_column)
        self.column_pub.publish(msg)
        self.target_publication_count += 1
        self.last_publish_sim_sec = self.sim_time_sec
        self.last_publish_wall_monotonic = now_wall
        if force or self.target_publication_count == 1:
            self.get_logger().info(
                '[DAY2] Published %d on %s'
                % (self.shelf_column, self.COLUMN_TOPIC))

    def _publish_status(self, state: str, detail: str) -> None:
        msg = String()
        msg.data = json.dumps({
            'state': state,
            'detail': detail,
            'shelf_column_number': self.shelf_column,
            'book_colour': self.book_colour,
            'phase1_fast_start': self.phase1_fast_start,
        }, sort_keys=True)
        self.status_pub.publish(msg)

    def _publish_debug(
        self,
        detections: Sequence[MarkerDetection],
        new_confirmations: Sequence[ConfirmedMarker],
        stamp_sec: float,
    ) -> None:
        if self.debug_pub.get_subscription_count() == 0:
            return
        msg = String()
        msg.data = json.dumps({
            'sim_stamp_sec': round(stamp_sec, 6),
            'processed_frame': self.processed_frames,
            'state': self.state,
            'detections': [item.as_dict() for item in detections],
            'new_confirmations': [
                item.as_dict() for item in new_confirmations],
            'confirmed_digits': sorted(self.all_confirmed_markers),
        }, sort_keys=True)
        self.debug_pub.publish(msg)

    def _transition(self, state: str, detail: str) -> None:
        self.state = state
        self.state_started_sim_sec = self.sim_time_sec
        self.state_history.append({
            'state': state,
            'sim_stamp_sec': self.sim_time_sec,
            'wall_elapsed_sec': (
                time.monotonic() - self.node_started_wall_monotonic),
        })
        self._publish_status(state, detail)
        self.get_logger().info(f'[DAY2] STATE -> {state}: {detail}')

    def _inputs_ready(self) -> bool:
        return (
            self.sim_time_sec is not None
            and self.current_yaw is not None
            and self.camera_frames_total >= 1
            and self.processed_frames >= 1
            and self._head_controller_is_ready()
        )

    def _start_motion(self) -> None:
        assert self.current_yaw is not None
        assert self.sim_time_sec is not None
        self.initial_yaw = self.current_yaw
        self.motion_start_unwrapped_yaw = self.unwrapped_yaw
        self.frames_at_motion_start = self.camera_frames_total
        self.motion_started_wall_monotonic = time.monotonic()
        self.motion_started_sim_sec = self.sim_time_sec
        self._send_head_command()

        if self.phase1_fast_start:
            self.target_yaw = normalize_angle(
                self.initial_yaw - math.radians(self.rotation_deg))
            self._transition(
                'ROTATING_FAST',
                'Day 1 closed-loop clockwise 90 degree turn; vision remains active.')
            self.get_logger().info(
                '[DAY2] Fast turn initial=%.2f deg target=%.2f deg'
                % (
                    math.degrees(self.initial_yaw),
                    math.degrees(self.target_yaw),
                ))
        else:
            self.search_start_unwrapped_yaw = self.unwrapped_yaw
            self._transition(
                'SEARCH_FOR_SHELF',
                'Orientation-independent clockwise visual sweep started.')

    def _simulation_elapsed(self, start: Optional[float]) -> Optional[float]:
        if start is None or self.sim_time_sec is None:
            return None
        return max(0.0, self.sim_time_sec - start)

    def _head_reached(self) -> bool:
        return (
            self.actual_head_tilt is not None
            and abs(self.actual_head_tilt - self.head_tilt)
            <= self.head_tilt_tolerance
        )

    def _complete_fast_turn(self) -> None:
        self.stop_base()
        self.rotation_completed = True
        self.motion_completed_sim_sec = self.sim_time_sec
        self.motion_completed_wall_monotonic = time.monotonic()
        self.target_wait_started_sim_sec = self.sim_time_sec
        if self.target_confirmation is not None:
            self._transition(
                'VERIFYING_EVIDENCE',
                'Turn complete and target already confirmed; validating outputs.')
        else:
            self._transition(
                'WAITING_FOR_TARGET',
                'Turn complete; waiting briefly for temporal target confirmation.')

    def _fast_rotation_tick(self) -> None:
        if not self.enable_motion:
            self._finish(False, 'enable_motion was false; no opening motion executed')
            return
        if self.current_yaw is None or self.target_yaw is None:
            self._finish(False, 'Odometry became unavailable during fast turn')
            return
        elapsed = self._simulation_elapsed(self.motion_started_sim_sec)
        if elapsed is None:
            self._finish(False, 'Simulation clock became unavailable during fast turn')
            return
        if elapsed > self.motion_timeout_sec:
            self._finish(False, 'Fast opening turn exceeded simulated motion_timeout_sec')
            return

        error = normalize_angle(self.target_yaw - self.current_yaw)
        if abs(error) <= self.yaw_tolerance_rad:
            self.stop_base()
            self.rotation_settle_samples += 1
            if (
                self.rotation_settle_samples >= self.rotation_settle_required
                and self._head_reached()
            ):
                self._complete_fast_turn()
            return

        self.rotation_settle_samples = 0
        speed = clamp(
            abs(self.yaw_kp * error), self.min_speed, self.max_speed)
        self._publish_base_speed(math.copysign(speed, error))

    def _target_is_fresh(self) -> bool:
        if self.target_confirmation is None or self.sim_time_sec is None:
            return False
        current = self.temporal_filter.confirmed(self.shelf_column)
        if current is None:
            return False
        return (
            self.sim_time_sec - current.last_seen_stamp_sec
            <= self.target_stale_timeout_sec
        )

    def _general_search_tick(self) -> None:
        if not self.enable_motion:
            self._finish(False, 'enable_motion was false; no general search executed')
            return
        elapsed = self._simulation_elapsed(self.motion_started_sim_sec)
        if elapsed is None:
            self._finish(False, 'Simulation clock unavailable during general search')
            return
        if elapsed > self.general_search_timeout_sec:
            self._finish(False, 'General shelf search exceeded simulated timeout')
            return
        if self.search_start_unwrapped_yaw is None:
            self._finish(False, 'General search sweep origin was not recorded')
            return
        swept = abs(self.unwrapped_yaw - self.search_start_unwrapped_yaw)
        if swept > self.general_search_max_sweep_rad:
            self._finish(False, 'General shelf search exceeded maximum angular sweep')
            return

        if self.target_confirmation is not None:
            # Record only the actual SEARCH_FOR_SHELF sweep.  Alignment may
            # subsequently rotate in either direction and must not alter the
            # search-direction validation measurement.
            self.general_search_signed_rotation_rad += (
                self.unwrapped_yaw - self.search_start_unwrapped_yaw
            )
            self.stop_base()
            self.align_settle_samples = 0
            self._transition(
                'ALIGN_TO_TARGET',
                'Target confirmed; aligning camera with requested shelf marker.')
            return
        self._publish_base_speed(-abs(self.general_search_speed))

    def _general_align_tick(self) -> None:
        if not self.enable_motion:
            self._finish(False, 'enable_motion was false during general alignment')
            return
        if self.frame_width is None:
            self._finish(False, 'Camera width unavailable during general alignment')
            return
        latest = self.temporal_filter.latest_detection(self.shelf_column)
        if latest is None or not self._target_is_fresh():
            self.align_settle_samples = 0
            self.search_start_unwrapped_yaw = self.unwrapped_yaw
            self._transition(
                'SEARCH_FOR_SHELF',
                'Target became stale during alignment; resuming visual search.')
            return

        pixel_error = latest.bbox.center_x - 0.5 * self.frame_width
        if abs(pixel_error) <= self.general_align_tolerance_px:
            self.stop_base()
            self.align_settle_samples += 1
            if (
                self.align_settle_samples >= self.general_align_settle_required
                and self._head_reached()
            ):
                self.rotation_completed = True
                self.motion_completed_sim_sec = self.sim_time_sec
                self.motion_completed_wall_monotonic = time.monotonic()
                self._transition(
                    'VERIFYING_EVIDENCE',
                    'General search found and aligned to the requested marker.')
            return

        self.align_settle_samples = 0
        speed = clamp(
            self.general_align_kp * abs(pixel_error),
            self.general_align_min_speed,
            self.general_align_max_speed,
        )
        # A target to the right requires clockwise rotation (negative z).
        angular_z = -math.copysign(speed, pixel_error)
        self._publish_base_speed(angular_z)

    def _all_marker_validation_ready(self) -> bool:
        return all(digit in self.all_confirmed_markers for digit in range(1, 6))

    def _evidence_tick(self) -> None:
        self.stop_base()
        if self.target_confirmation is None:
            self._finish(False, 'Evidence validation reached without a confirmed target')
            return
        self._publish_column_result()
        if not self.target_annotated_image_saved:
            self._finish(False, 'Target confirmed but annotated image was not saved')
            return

        hold_elapsed = None
        if self.sim_time_sec is not None:
            hold_elapsed = max(
                0.0,
                self.sim_time_sec - self.target_confirmation.confirmed_stamp_sec,
            )
        if hold_elapsed is None or hold_elapsed < self.result_hold_sec:
            return

        if self.validation_require_all_markers:
            if self._all_marker_validation_ready():
                self._finish(
                    True,
                    'Target confirmed, official result published, annotated image '
                    'saved, and all five markers confirmed for validation')
                return
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
                    if digit not in self.all_confirmed_markers]
                self._finish(
                    False,
                    'All-marker validation timeout; missing digits: '
                    + ','.join(str(item) for item in missing))
            return

        self._finish(
            True,
            'Target marker confirmed, published and saved with live annotation')

    def _waiting_for_target_tick(self) -> None:
        self.stop_base()
        if self.target_confirmation is not None:
            self._transition(
                'VERIFYING_EVIDENCE',
                'Target confirmed after opening turn; validating evidence.')
            return
        elapsed = self._simulation_elapsed(self.target_wait_started_sim_sec)
        if elapsed is not None and elapsed > self.post_turn_detection_timeout_sec:
            self._finish(False, 'Target marker was not confirmed after opening turn')

    def _control_tick(self) -> None:
        if self.done:
            return
        self._publish_column_result()

        if self.state == 'WAITING_FOR_INPUTS':
            if self._inputs_ready():
                self._start_motion()
                return
            wall_elapsed = time.monotonic() - self.node_started_wall_monotonic
            if wall_elapsed > self.ready_timeout_sec:
                missing = []
                if self.sim_time_sec is None:
                    missing.append('/clock')
                if self.current_yaw is None:
                    missing.append('/odom')
                if self.camera_frames_total < 1:
                    missing.append('RGB camera frame')
                if self.processed_frames < 1:
                    missing.append('processed RGB frame')
                if not self._head_controller_is_ready():
                    missing.append('head controller subscriber')
                self._finish(
                    False,
                    'Readiness timeout; missing: ' + ', '.join(missing))
            return

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
        else:
            self._finish(False, f'Unknown Day 2 state: {self.state}')

    def _observed_left_to_right(self) -> List[int]:
        confirmed_digits = set(self.all_confirmed_markers)

        # Validation-only coherent-frame layout reconstruction.
        #
        # ConfirmedMarker bboxes may have been last observed at different
        # robot yaws while the base was rotating.  Sorting those historical
        # pixel positions can therefore produce a false left-to-right order.
        #
        # When validation has confirmed all official digits 1..5 and the
        # current detector frame contains exactly five physical candidates,
        # reconstruct the order from that one coherent frame.  Accepted live
        # detections label their matching candidate directly.  If exactly one
        # candidate was rejected by the normal classifier thresholds, assign
        # it the one remaining confirmed digit.  This uses only live image
        # geometry plus the official one-each-of-1..5 rule; it does not use
        # the random seed or hidden simulator state.
        if (
            self.validation_require_all_markers
            and confirmed_digits == set(range(1, 6))
        ):
            diagnostics = self.latest_detector_diagnostics or {}
            top_candidates = diagnostics.get('top_candidates', [])

            if (
                diagnostics.get('classified_candidate_count') == 5
                and isinstance(top_candidates, list)
                and len(top_candidates) == 5
            ):
                slots = {}
                valid_slots = True

                for candidate in top_candidates:
                    if not isinstance(candidate, dict):
                        valid_slots = False
                        break

                    bbox = candidate.get('bbox_xywh')
                    if not (
                        isinstance(bbox, (list, tuple))
                        and len(bbox) == 4
                    ):
                        valid_slots = False
                        break

                    key = tuple(int(value) for value in bbox)

                    if key in slots:
                        valid_slots = False
                        break

                    x, _, width, _ = key
                    if width <= 0:
                        valid_slots = False
                        break

                    slots[key] = {
                        'center_x': x + 0.5 * width,
                        'digit': None,
                    }

                if valid_slots and len(slots) == 5:
                    used_digits = set()
                    mapping_ok = True

                    for detection in self.latest_detections:
                        key = tuple(
                            int(value)
                            for value in detection.bbox.as_list()
                        )
                        digit = int(detection.digit)

                        if key not in slots:
                            mapping_ok = False
                            break

                        if (
                            slots[key]['digit'] is not None
                            or digit in used_digits
                        ):
                            mapping_ok = False
                            break

                        slots[key]['digit'] = digit
                        used_digits.add(digit)

                    missing_digits = sorted(
                        confirmed_digits - used_digits
                    )
                    unmatched = [
                        key
                        for key, slot in slots.items()
                        if slot['digit'] is None
                    ]

                    # Accept either:
                    #   - five accepted current detections, or
                    #   - four accepted detections plus one threshold-rejected
                    #     candidate whose digit is uniquely determined.
                    if (
                        mapping_ok
                        and len(used_digits) >= 4
                        and len(missing_digits) == len(unmatched)
                        and len(missing_digits) <= 1
                    ):
                        if len(missing_digits) == 1:
                            slots[unmatched[0]]['digit'] = (
                                missing_digits[0]
                            )

                        if all(
                            slot['digit'] is not None
                            for slot in slots.values()
                        ):
                            ordered = sorted(
                                slots.values(),
                                key=lambda slot: slot['center_x'],
                            )
                            return [
                                int(slot['digit'])
                                for slot in ordered
                            ]

        # Normal/default fallback: retain the original behavior.
        markers = [
            self.all_confirmed_markers[digit]
            for digit in sorted(self.all_confirmed_markers)
        ]
        markers.sort(key=lambda item: item.bbox.center_x)
        return [marker.digit for marker in markers]

    def _build_result(self, passed: bool, reason: str) -> Dict[str, Any]:
        now_wall = time.monotonic()
        wall_motion_duration = None
        if self.motion_started_wall_monotonic is not None:
            end = self.motion_completed_wall_monotonic or now_wall
            wall_motion_duration = max(
                0.0, end - self.motion_started_wall_monotonic)

        sim_motion_duration = None
        if self.motion_started_sim_sec is not None:
            end_sim = self.motion_completed_sim_sec or self.sim_time_sec
            if end_sim is not None:
                sim_motion_duration = max(
                    0.0, end_sim - self.motion_started_sim_sec)

        signed_rotation_deg = None
        general_search_signed_rotation_deg = None
        final_error_deg = None
        if self.initial_yaw is not None and self.current_yaw is not None:
            if self.phase1_fast_start:
                signed_rotation_deg = math.degrees(
                    normalize_angle(self.current_yaw - self.initial_yaw))
            else:
                signed_rotation_deg = math.degrees(
                    self.unwrapped_yaw - (
                        self.motion_start_unwrapped_yaw
                        if self.motion_start_unwrapped_yaw is not None
                        else self.unwrapped_yaw))
        if not self.phase1_fast_start:
            general_search_signed_rotation_deg = math.degrees(
                self.general_search_signed_rotation_rad
            )

        if (
            self.phase1_fast_start
            and self.target_yaw is not None
            and self.current_yaw is not None
        ):
            final_error_deg = math.degrees(
                normalize_angle(self.target_yaw - self.current_yaw))

        confirmation_rotation_progress = None
        if (
            self.initial_yaw is not None
            and self.target_confirmation_yaw is not None
        ):
            confirmation_rotation_progress = abs(math.degrees(
                normalize_angle(
                    self.target_confirmation_yaw - self.initial_yaw)))

        time_to_first_sim = None
        if (
            self.target_first_detection_sim_sec is not None
            and self.node_started_sim_sec is not None
        ):
            time_to_first_sim = max(
                0.0,
                self.target_first_detection_sim_sec - self.node_started_sim_sec)
        time_to_confirmed_sim = None
        if (
            self.target_confirmation is not None
            and self.node_started_sim_sec is not None
        ):
            time_to_confirmed_sim = max(
                0.0,
                self.target_confirmation.confirmed_stamp_sec
                - self.node_started_sim_sec)

        latest_target = self.temporal_filter.latest_detection(
            self.shelf_column)
        final_target_pixel_error = None
        if latest_target is not None and self.frame_width is not None:
            final_target_pixel_error = (
                latest_target.bbox.center_x - 0.5 * self.frame_width)

        target = (
            self.target_confirmation.as_dict()
            if self.target_confirmation is not None else None)
        confirmed = {
            str(digit): marker.as_dict()
            for digit, marker in sorted(self.all_confirmed_markers.items())
        }
        observed_order = self._observed_left_to_right()
        target_column_index = (
            observed_order.index(self.shelf_column) + 1
            if len(observed_order) == 5 and self.shelf_column in observed_order
            else None
        )

        # Day 1-compatible fields are deliberately retained so its validator
        # can regression-check a fast-start Day 2 result.
        return {
            'passed': bool(passed),
            'reason': reason,
            'state': 'PASSED' if passed else 'FAILED',
            'day': 2,
            'shelf_column_number': self.shelf_column,
            'book_colour': self.book_colour,
            'phase1_fast_start': self.phase1_fast_start,
            'search_mode': (
                'phase1_fast' if self.phase1_fast_start
                else 'orientation_independent'),
            'enable_motion': self.enable_motion,
            'perception_enabled': self.perception_enabled,
            'commanded_clockwise_rotation_deg': self.rotation_deg,
            'signed_rotation_deg': signed_rotation_deg,
            'general_search_signed_rotation_deg': (
                general_search_signed_rotation_deg
            ),
            'absolute_rotation_deg': (
                abs(signed_rotation_deg)
                if signed_rotation_deg is not None else None),
            'final_error_deg': final_error_deg,
            'duration_sec': sim_motion_duration,
            'wall_duration_sec': wall_motion_duration,
            'duration_clock': 'simulation',
            'initial_yaw_deg': (
                math.degrees(self.initial_yaw)
                if self.initial_yaw is not None else None),
            'target_yaw_deg': (
                math.degrees(self.target_yaw)
                if self.target_yaw is not None else None),
            'final_yaw_deg': (
                math.degrees(self.current_yaw)
                if self.current_yaw is not None else None),
            'head_command_sent': self.head_command_sent,
            'head_pan_rad': self.head_pan,
            'head_tilt_rad': self.head_tilt,
            'actual_head_tilt_rad': self.actual_head_tilt,
            'head_tilt_error_rad': (
                self.head_tilt - self.actual_head_tilt
                if self.actual_head_tilt is not None else None),
            'head_tilt_tolerance_rad': self.head_tilt_tolerance,
            'camera_topic': self.CAMERA_TOPIC,
            'camera_frames_total': self.camera_frames_total,
            'camera_frames_during_motion': self.camera_frames_during_motion,
            'processed_frames': self.processed_frames,
            'last_image_stamp_ns': self.last_image_stamp_ns,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'marker_template_directory': self.marker_template_directory,
            'detector_configuration': {
                'process_rate_hz': self.process_rate_hz,
                'roi_top_fraction': self.roi_top_fraction,
                'roi_bottom_fraction': self.roi_bottom_fraction,
                'dark_threshold': self.dark_threshold,
                'min_bright_ring_fraction': self.min_bright_ring_fraction,
                'min_neutral_dark_fraction': self.min_neutral_dark_fraction,
                'min_classifier_confidence': self.min_classifier_confidence,
                'min_classifier_margin': self.min_classifier_margin,
                'confirmation_frames': self.confirmation_frames,
                'confirmation_window_sec': self.confirmation_window_sec,
                'confirmation_min_span_sec': self.confirmation_min_span_sec,
                'confirmation_average_confidence': (
                    self.confirmation_average_confidence),
                'result_hold_sec': self.result_hold_sec,
            },
            'target_marker': target,
            'target_first_detection_bbox_xywh': (
                self.target_first_detection_bbox),
            'time_to_first_detection_sim_sec': time_to_first_sim,
            'time_to_first_detection_wall_sec': (
                self.target_first_detection_wall_sec),
            'time_to_confirmed_detection_sim_sec': time_to_confirmed_sim,
            'time_to_confirmed_detection_wall_sec': (
                self.target_confirmation_wall_sec),
            'target_confirmation_state': self.target_confirmation_state,
            'target_confirmation_yaw_deg': (
                math.degrees(self.target_confirmation_yaw)
                if self.target_confirmation_yaw is not None else None),
            'rotation_progress_deg_at_confirmation': (
                confirmation_rotation_progress),
            'target_confirmed_during_motion': (
                self.target_confirmed_during_motion),
            'final_target_pixel_error_px': final_target_pixel_error,
            'general_align_tolerance_px': self.general_align_tolerance_px,
            'column_result_topic': self.COLUMN_TOPIC,
            'column_result_message_type': 'std_msgs/msg/Int32',
            'published_column_value': (
                self.shelf_column if self.target_publication_count > 0 else None),
            'column_publication_count': self.target_publication_count,
            'annotated_image_saved': self.target_annotated_image_saved,
            'annotated_image_path': self.target_annotated_image_path,
            'annotated_image_contains_sim_timestamp': True,
            'annotated_image_contains_wall_timestamp': True,
            'annotated_image_contains_target_column_box': (
                self.target_column_bbox is not None),
            'target_column_bbox_xywh': (
                self.target_column_bbox.as_list()
                if self.target_column_bbox is not None else None),
            'confirmed_markers': confirmed,
            'confirmed_marker_count': len(confirmed),
            'observed_left_to_right': observed_order,
            'target_column_index_left_to_right': target_column_index,
            'validation_require_all_markers': (
                self.validation_require_all_markers),
            'latest_detector_diagnostics': (
                self.latest_detector_diagnostics),
            'detector_processing_ms_mean': (
                1000.0 * self.detector_processing_wall_sec_total
                / self.detector_processing_samples
                if self.detector_processing_samples > 0 else None),
            'detector_processing_ms_max': (
                1000.0 * self.detector_processing_wall_sec_max
                if self.detector_processing_samples > 0 else None),
            'detector_processing_samples': self.detector_processing_samples,
            'result_path': self.result_path,
            'node_wall_elapsed_sec': (
                now_wall - self.node_started_wall_monotonic),
            'node_started_unix_time': self.node_started_unix_time,
            'state_history': self.state_history,
            'generated_unix_time': time.time(),
        }

    def _write_result(self, result: Dict[str, Any]) -> None:
        result_dir = os.path.dirname(self.result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        temporary = self.result_path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, self.result_path)

    def _finish(self, passed: bool, reason: str) -> None:
        if self.done:
            return
        self.stop_base()
        # Send one last result sample before serializing the publication count.
        if self.target_confirmation is not None:
            self._publish_column_result(force=True)
        result = self._build_result(passed, reason)
        try:
            self._write_result(result)
        except OSError as exc:
            passed = False
            reason = f'Could not write Day 2 result file: {exc}'
            result['passed'] = False
            result['state'] = 'FAILED'
            result['reason'] = reason

        self.passed = passed
        self.failure_reason = '' if passed else reason
        self.state = 'PASSED' if passed else 'FAILED'
        self.done = True
        self._publish_status(self.state, reason)
        level = self.get_logger().info if passed else self.get_logger().error
        level('[DAY2][%s] %s | result=%s'
              % (self.state, reason, self.result_path))


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[MissionStart] = None
    exit_code = 1
    try:
        node = MissionStart()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.10)
        if node.done and node.passed:
            exit_code = 0
    except KeyboardInterrupt:
        if node is not None:
            node.stop_base()
            node.get_logger().warning(
                '[DAY2] Interrupted; repeated base stop commands sent')
        exit_code = 130
    except Exception as exc:
        if node is not None:
            node.stop_base()
            node.get_logger().error(
                '[DAY2][FAILED] Unhandled exception: %s' % exc)
        else:
            print('[DAY2][FAILED] Could not initialize mission_start: %s' % exc)
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
