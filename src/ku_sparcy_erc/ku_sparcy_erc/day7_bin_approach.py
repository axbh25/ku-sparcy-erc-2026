#!/usr/bin/env python3
"""Day 7: identify the red collection bin and approach it with the book held."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros

try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:  # pragma: no cover
    Contacts = None  # type: ignore

from ku_sparcy_erc.bin_perception import (
    BinDetection,
    BinDetectorSettings,
    consistent_detection,
    detect_red_bin,
)
from ku_sparcy_erc.day567_common import (
    atomic_write_json,
    book_half_diagonal_m,
    clamp,
    contact_collision_names,
    limit_planar,
    load_passed_json,
    meaningful_robot_collision,
    normalize_angle,
    selected_fingertip_tokens,
    vector_base_to_odom,
    vector_odom_to_base,
    yaw_from_odom,
)
from ku_sparcy_erc.grasp_planner import (
    deterministic_seeds,
    interpolate_joint_path,
    screen_joint_path,
    solve_with_seeds,
)
from ku_sparcy_erc.urdf_kinematics import (
    URDFKinematicModel,
    rpy_matrix,
)
from ku_sparcy_erc.range_fusion import (
    BoundingBoxLike,
    CameraModel,
    LidarClearance,
    lidar_corridor_clearance,
    normalize_depth_array,
    target_geometry_from_depth,
)


class Day7BinApproach(Node):
    RGB_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
    COLOR_INFO_TOPIC = '/head_front_camera/head_front_camera/color/camera_info'
    DEPTH_TOPIC = '/head_front_camera/head_front_camera/depth/image_rect_raw'
    DEPTH_INFO_TOPIC = '/head_front_camera/head_front_camera/depth/camera_info'
    FRONT_SCAN_TOPIC = '/scan_front_raw'

    def __init__(self) -> None:
        super().__init__('ku_sparcy_day7_bin_approach')
        self.declare_parameter(
            'day5_result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day5_result.json')
        self.declare_parameter(
            'day6_result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day6_result.json')
        self.declare_parameter(
            'result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day7_result.json')
        self.declare_parameter(
            'image_output_dir',
            '/opt/erc_ws/src/ku_sparcy_erc/erc_images')
        self.declare_parameter('ready_wall_timeout_sec', 45.0)
        self.declare_parameter('clock_stall_wall_timeout_sec', 20.0)
        self.declare_parameter(
            'head_search_poses_pan_tilt_rad',
            [0.0, -0.15, 0.70, -0.20, -0.70, -0.20, 0.0, -0.35])
        self.declare_parameter('head_motion_sec', 2.5)
        self.declare_parameter('head_settle_tolerance_rad', 0.05)
        self.declare_parameter('head_settle_samples', 4)
        self.declare_parameter('head_pose_timeout_sec', 7.0)
        self.declare_parameter('head_dwell_sec', 1.0)
        self.declare_parameter('bin_confirm_frames', 3)
        self.declare_parameter('bin_confirm_window_sec', 0.90)
        self.declare_parameter('bin_center_tolerance_px', 42.0)
        self.declare_parameter('bin_min_saturation', 110)
        self.declare_parameter('bin_min_value', 55)
        self.declare_parameter('bin_min_area_px', 450)
        self.declare_parameter('bin_min_width_px', 34)
        self.declare_parameter('bin_min_height_px', 18)
        self.declare_parameter('bin_min_aspect_ratio', 1.10)
        self.declare_parameter('bin_min_fill_fraction', 0.30)
        self.declare_parameter('max_rgb_depth_stamp_skew_sec', 0.20)
        self.declare_parameter('depth_stale_timeout_sec', 0.55)
        self.declare_parameter('depth_min_valid_pixels', 18)
        self.declare_parameter('depth_min_valid_fraction', 0.10)
        self.declare_parameter('bin_min_standoff_m', 0.85)
        self.declare_parameter('carried_book_clearance_margin_m', 0.18)
        self.declare_parameter('bin_max_standoff_m', 1.25)
        self.declare_parameter('final_distance_tolerance_m', 0.12)
        self.declare_parameter('final_lateral_tolerance_m', 0.12)
        self.declare_parameter('final_bearing_tolerance_deg', 5.0)
        self.declare_parameter('final_settle_samples', 5)
        self.declare_parameter('max_forward_speed_mps', 0.18)
        self.declare_parameter('max_lateral_speed_mps', 0.10)
        self.declare_parameter('max_yaw_speed_radps', 0.22)
        self.declare_parameter('forward_kp', 0.45)
        self.declare_parameter('lateral_kp', 0.65)
        self.declare_parameter('yaw_kp', 0.85)
        self.declare_parameter('approach_timeout_sec', 18.0)
        self.declare_parameter('front_hard_stop_m', 0.55)
        self.declare_parameter('lidar_corridor_half_width_m', 0.42)
        self.declare_parameter('lidar_robust_percentile', 10.0)
        self.declare_parameter('lidar_hard_cluster_points', 3)
        self.declare_parameter('scan_stale_timeout_sec', 0.60)
        self.declare_parameter('gripper_hold_position_m', 0.010)
        self.declare_parameter('gripper_hold_period_sec', 0.20)
        self.declare_parameter('retention_contact_stale_sec', 0.80)

        # Physically validated Day 7 high/compact carry pose.
        self.declare_parameter('carry_raise_distance_m', 0.16)
        self.declare_parameter('carry_retract_distance_m', 0.12)
        self.declare_parameter('carry_raise_motion_sec', 4.0)
        self.declare_parameter('carry_raise_joint_tolerance_rad', 0.045)
        self.declare_parameter('carry_raise_settle_samples', 3)
        self.declare_parameter('carry_raise_timeout_sec', 8.0)
        self.declare_parameter('carry_raise_max_joint_delta_rad', 0.45)

        # Physically validated rotation-in-place strategy at Start/End Zone.
        self.declare_parameter(
            'preapproach_align_max_yaw_speed_radps', 0.10)
        self.declare_parameter(
            'preapproach_align_tolerance_deg', 2.0)
        self.declare_parameter(
            'preapproach_align_settle_samples', 5)
        self.declare_parameter(
            'preapproach_align_timeout_sec', 15.0)

        # Once aligned, regulate distance primarily along the base X axis.
        self.declare_parameter('max_reverse_speed_mps', 0.10)
        self.declare_parameter(
            'aligned_approach_max_yaw_speed_radps', 0.10)

        get = self.get_parameter
        self.day5_path = str(get('day5_result_path').value)
        self.day6_path = str(get('day6_result_path').value)
        self.result_path = str(get('result_path').value)
        self.image_output_dir = str(get('image_output_dir').value)
        self.ready_wall_timeout = float(get('ready_wall_timeout_sec').value)
        self.clock_stall_wall_timeout = float(
            get('clock_stall_wall_timeout_sec').value)
        flat_poses = [float(v) for v in get(
            'head_search_poses_pan_tilt_rad').value]
        if len(flat_poses) % 2 != 0:
            raise ValueError('head search pose list must contain pan/tilt pairs')
        self.head_poses = [
            (flat_poses[index], flat_poses[index + 1])
            for index in range(0, len(flat_poses), 2)]
        self.head_motion_sec = float(get('head_motion_sec').value)
        self.head_tolerance = float(get('head_settle_tolerance_rad').value)
        self.head_settle_required = int(get('head_settle_samples').value)
        self.head_pose_timeout = float(get('head_pose_timeout_sec').value)
        self.head_dwell_sec = float(get('head_dwell_sec').value)
        self.bin_confirm_frames = int(get('bin_confirm_frames').value)
        self.bin_confirm_window = float(get('bin_confirm_window_sec').value)
        self.bin_center_tolerance = float(
            get('bin_center_tolerance_px').value)
        self.detector_settings = BinDetectorSettings(
            min_saturation=int(get('bin_min_saturation').value),
            min_value=int(get('bin_min_value').value),
            min_area_px=int(get('bin_min_area_px').value),
            min_width_px=int(get('bin_min_width_px').value),
            min_height_px=int(get('bin_min_height_px').value),
            min_aspect_ratio=float(get('bin_min_aspect_ratio').value),
            min_fill_fraction=float(get('bin_min_fill_fraction').value),
        )
        self.max_rgb_depth_skew = float(
            get('max_rgb_depth_stamp_skew_sec').value)
        self.depth_stale_timeout = float(get('depth_stale_timeout_sec').value)
        self.depth_min_valid_pixels = int(
            get('depth_min_valid_pixels').value)
        self.depth_min_valid_fraction = float(
            get('depth_min_valid_fraction').value)
        self.bin_min_standoff = float(get('bin_min_standoff_m').value)
        self.book_clearance_margin = float(
            get('carried_book_clearance_margin_m').value)
        self.bin_max_standoff = float(get('bin_max_standoff_m').value)
        self.final_distance_tolerance = float(
            get('final_distance_tolerance_m').value)
        self.final_lateral_tolerance = float(
            get('final_lateral_tolerance_m').value)
        self.final_bearing_tolerance = math.radians(float(
            get('final_bearing_tolerance_deg').value))
        self.final_settle_required = int(get('final_settle_samples').value)
        self.max_forward_speed = float(get('max_forward_speed_mps').value)
        self.max_lateral_speed = float(get('max_lateral_speed_mps').value)
        self.max_yaw_speed = float(get('max_yaw_speed_radps').value)
        self.forward_kp = float(get('forward_kp').value)
        self.lateral_kp = float(get('lateral_kp').value)
        self.yaw_kp = float(get('yaw_kp').value)
        self.approach_timeout = float(get('approach_timeout_sec').value)
        self.front_hard_stop = float(get('front_hard_stop_m').value)
        self.corridor_half_width = float(
            get('lidar_corridor_half_width_m').value)
        self.lidar_percentile = float(get('lidar_robust_percentile').value)
        self.hard_cluster_points = int(
            get('lidar_hard_cluster_points').value)
        self.scan_stale_timeout = float(get('scan_stale_timeout_sec').value)
        self.gripper_hold_position = float(
            get('gripper_hold_position_m').value)
        self.gripper_hold_period = float(get('gripper_hold_period_sec').value)
        self.retention_stale = float(
            get('retention_contact_stale_sec').value)

        self.carry_raise_distance = float(
            get('carry_raise_distance_m').value)
        self.carry_retract_distance = float(
            get('carry_retract_distance_m').value)
        self.carry_raise_motion_sec = float(
            get('carry_raise_motion_sec').value)
        self.carry_raise_joint_tolerance = float(
            get('carry_raise_joint_tolerance_rad').value)
        self.carry_raise_settle_required = int(
            get('carry_raise_settle_samples').value)
        self.carry_raise_timeout = float(
            get('carry_raise_timeout_sec').value)
        self.carry_raise_max_joint_delta_limit = float(
            get('carry_raise_max_joint_delta_rad').value)

        self.preapproach_align_max_yaw_speed = float(
            get('preapproach_align_max_yaw_speed_radps').value)
        self.preapproach_align_tolerance = math.radians(float(
            get('preapproach_align_tolerance_deg').value))
        self.preapproach_align_settle_required = int(
            get('preapproach_align_settle_samples').value)
        self.preapproach_align_timeout = float(
            get('preapproach_align_timeout_sec').value)

        self.max_reverse_speed = float(
            get('max_reverse_speed_mps').value)
        self.aligned_approach_max_yaw_speed = float(
            get('aligned_approach_max_yaw_speed_radps').value)

        self._validate_parameters()

        self.day5 = load_passed_json(self.day5_path, label='Day 5 result')
        self.day6 = load_passed_json(self.day6_path, label='Day 6 result')
        if self.day5.get('retention_verified') is not True:
            raise ValueError('Day 5 result did not verify retained book')
        self.selected_arm = str(self.day5.get('selected_arm') or '').lower()
        if self.selected_arm not in {'left', 'right'}:
            raise ValueError('invalid selected arm')

        grasp_plan = self.day5.get('grasp_plan')
        if not isinstance(grasp_plan, dict):
            raise ValueError('Day 5 result has no validated grasp plan')

        candidate_plans = grasp_plan.get('candidate_plans')
        if not isinstance(candidate_plans, dict):
            raise ValueError('Day 5 grasp plan has no candidate plans')

        selected_candidate = candidate_plans.get(self.selected_arm)
        if not isinstance(selected_candidate, dict):
            raise ValueError('Day 5 grasp plan has no selected-arm candidate')

        lift_solution = selected_candidate.get('lift')
        if not isinstance(lift_solution, dict) or not lift_solution.get('success'):
            raise ValueError('Day 5 selected-arm lift solution was not valid')

        joint_names = lift_solution.get('joint_names')
        if not isinstance(joint_names, list) or len(joint_names) != 7:
            raise ValueError('Day 5 lift solution has invalid joint names')

        lift_xyz = grasp_plan.get('lift_xyz_m')
        if not isinstance(lift_xyz, list) or len(lift_xyz) != 3:
            raise ValueError('Day 5 grasp plan has invalid lift XYZ')

        target_rpy = selected_candidate.get('target_rpy_base')
        if not isinstance(target_rpy, list) or len(target_rpy) != 3:
            raise ValueError('Day 5 grasp plan has invalid target orientation')

        self.selected_joint_names = tuple(str(v) for v in joint_names)
        self.day5_lift_xyz = np.asarray(lift_xyz, dtype=np.float64)
        self.carry_target_rotation = rpy_matrix(target_rpy)

        self.carried_forward_extent = float(
            self.day6.get('carried_book_forward_extent_m'))
        self.carried_swept_radius = float(
            self.day6.get('carried_book_swept_radius_m'))

        # The validated compact pose translates the gripper/book 12 cm
        # toward the torso while preserving the grasp orientation.
        self.compact_forward_extent = max(
            0.0,
            self.carried_forward_extent - self.carry_retract_distance,
        )

        self.final_standoff = clamp(
            max(
                self.bin_min_standoff,
                self.compact_forward_extent + self.book_clearance_margin),
            self.bin_min_standoff,
            self.bin_max_standoff)

        self.bridge = CvBridge()
        self.started_wall = time.monotonic()
        self.last_clock_wall = time.monotonic()
        self.sim_time_sec: Optional[float] = None
        self.state = 'WAIT_INPUTS'
        self.state_started_sim: Optional[float] = None
        self.done = False
        self.passed = False
        self.reason = ''
        self.current_xy: Optional[Tuple[float, float]] = None
        self.current_yaw: Optional[float] = None
        self.head_positions: Dict[str, float] = {}
        self.joint_positions: Dict[str, float] = {}

        self.carry_raise_target_xyz: Optional[Tuple[float, float, float]] = None
        self.carry_raise_target_positions: Optional[Tuple[float, ...]] = None
        self.carry_raise_command_sim: Optional[float] = None
        self.carry_raise_settle_count = 0
        self.carry_raise_completed = False
        self.carry_raise_max_joint_delta: Optional[float] = None
        self.carry_raise_screen = None
        self.carry_raise_ik_position_error_m: Optional[float] = None
        self.carry_raise_ik_orientation_error_rad: Optional[float] = None

        self.preapproach_align_started_sim: Optional[float] = None
        self.preapproach_align_settle_count = 0
        self.preapproach_initial_bearing_rad: Optional[float] = None
        self.rotation_alignment_completed = False

        self.current_pose_index = 0
        self.head_command_sim: Optional[float] = None
        self.head_settle_count = 0
        self.head_dwell_started_sim: Optional[float] = None
        self.rgb_frames_at_head_command = 0
        self.color_model: Optional[CameraModel] = None
        self.depth_model: Optional[CameraModel] = None
        self.depth_history: List[Tuple[float, np.ndarray]] = []
        self.depth_encoding: Optional[str] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_rgb_stamp: Optional[float] = None
        self.rgb_frames = 0
        self.depth_frames = 0
        self.detection_history: List[Tuple[float, BinDetection]] = []
        self.confirmed_detection: Optional[BinDetection] = None
        self.confirmed_geometry = None
        self.locked_bin_odom_xy: Optional[Tuple[float, float]] = None
        self.lock_sim_time: Optional[float] = None
        self.front_scan: Optional[LidarClearance] = None
        self.front_scan_stamp: Optional[float] = None
        self.minimum_front_clearance: Optional[float] = None
        self.approach_started_sim: Optional[float] = None
        self.final_settle_count = 0
        self.last_gripper_hold_sim: Optional[float] = None
        self.last_fingertip_contact_sim: Optional[float] = None
        self.fingertip_contact_messages = 0
        self.unintended_robot_contacts = 0
        self.bin_contact_messages = 0
        self.evidence_image_path: Optional[str] = None
        self.path_trace: List[Dict[str, object]] = []
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)
        self.clock_sub = self.create_subscription(
            Clock, '/clock', self._clock_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile_sensor_data)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 20)
        self.rgb_sub = self.create_subscription(
            Image, self.RGB_TOPIC, self._rgb_callback, qos_profile_sensor_data)
        self.color_info_sub = self.create_subscription(
            CameraInfo, self.COLOR_INFO_TOPIC,
            self._color_info_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(
            Image, self.DEPTH_TOPIC,
            self._depth_callback, qos_profile_sensor_data)
        self.depth_info_sub = self.create_subscription(
            CameraInfo, self.DEPTH_INFO_TOPIC,
            self._depth_info_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(
            LaserScan, self.FRONT_SCAN_TOPIC,
            self._scan_callback, qos_profile_sensor_data)
        if Contacts is None:
            raise RuntimeError('ros_gz_interfaces/Contacts is required')
        self.contact_sub = self.create_subscription(
            Contacts, '/contacts', self._contact_callback,
            qos_profile_sensor_data)
        self.bin_contact_sub = self.create_subscription(
            Contacts, '/bin_contacts', self._bin_contact_callback,
            qos_profile_sensor_data)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            f'/gripper_{self.selected_arm}_controller/joint_trajectory', 10)

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            f'/arm_{self.selected_arm}_controller/joint_trajectory',
            10)

        self.timer = self.create_timer(0.05, self._tick)
        self.get_logger().info(
            '[DAY7] bin search and safe approach ready; arm=%s '
            'held-book-aware standoff=%.3f m' % (
                self.selected_arm, self.final_standoff))

    def _validate_parameters(self) -> None:
        if not self.head_poses:
            raise ValueError('at least one head search pose is required')
        for pan, tilt in self.head_poses:
            if not -1.20 <= pan <= 1.20:
                raise ValueError('head pan outside conservative limit')
            if not -0.90 <= tilt <= 0.30:
                raise ValueError('head tilt outside conservative hard-limit margin')
        if not 0.0 <= self.gripper_hold_position <= 0.069:
            raise ValueError('gripper hold position outside public clamp')

        if not 0.05 <= self.carry_raise_distance <= 0.18:
            raise ValueError('carry_raise_distance_m outside validated bound')
        if not 1.0 <= self.carry_raise_motion_sec <= 6.0:
            raise ValueError('carry_raise_motion_sec outside bounded interval')
        if not 0.01 <= self.carry_raise_joint_tolerance <= 0.08:
            raise ValueError(
                'carry_raise_joint_tolerance_rad outside bounded interval')
        if self.carry_raise_settle_required < 2:
            raise ValueError('carry_raise_settle_samples must be at least 2')
        if self.carry_raise_timeout <= self.carry_raise_motion_sec:
            raise ValueError(
                'carry_raise_timeout_sec must exceed carry motion duration')
        if not 0.20 <= self.carry_raise_max_joint_delta_limit <= 0.50:
            raise ValueError(
                'carry_raise_max_joint_delta_rad outside conservative bound')

        if not 0.05 <= self.carry_retract_distance <= 0.18:
            raise ValueError(
                'carry_retract_distance_m outside validated bound')
        if not 0.05 <= self.preapproach_align_max_yaw_speed <= 0.12:
            raise ValueError(
                'preapproach alignment yaw speed outside validated bound')
        if not math.radians(1.0) <= self.preapproach_align_tolerance <= math.radians(5.0):
            raise ValueError(
                'preapproach alignment tolerance outside bounded interval')
        if self.preapproach_align_settle_required < 3:
            raise ValueError(
                'preapproach alignment settle samples too small')
        if not 8.0 <= self.preapproach_align_timeout <= 25.0:
            raise ValueError(
                'preapproach alignment timeout outside bounded interval')
        if not 0.03 <= self.max_reverse_speed <= 0.12:
            raise ValueError(
                'max_reverse_speed_mps outside bounded interval')
        if not 0.05 <= self.aligned_approach_max_yaw_speed <= 0.12:
            raise ValueError(
                'aligned approach yaw speed outside bounded interval')

        self.detector_settings.validate()

    def _clock_callback(self, msg: Clock) -> None:
        value = float(msg.clock.sec) + float(msg.clock.nanosec) * 1.0e-9
        if self.sim_time_sec is None or value > self.sim_time_sec + 1.0e-9:
            self.last_clock_wall = time.monotonic()
        self.sim_time_sec = value

    def _odom_callback(self, msg: Odometry) -> None:
        self.current_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y))
        self.current_yaw = float(yaw_from_odom(msg))

    def _joint_callback(self, msg: JointState) -> None:
        for index, name in enumerate(msg.name):
            if index >= len(msg.position):
                continue

            value = float(msg.position[index])
            self.joint_positions[name] = value

            if name in {'head_1_joint', 'head_2_joint'}:
                self.head_positions[name] = value

    def _color_info_callback(self, msg: CameraInfo) -> None:
        try:
            self.color_model = CameraModel.from_arrays(
                msg.width, msg.height, msg.k, msg.p,
                frame_id=msg.header.frame_id,
                prefer_projection=False)
        except ValueError:
            return

    def _depth_info_callback(self, msg: CameraInfo) -> None:
        try:
            self.depth_model = CameraModel.from_arrays(
                msg.width, msg.height, msg.k, msg.p,
                frame_id=msg.header.frame_id,
                prefer_projection=False)
        except ValueError:
            return

    @staticmethod
    def _stamp(msg: Image) -> Optional[float]:
        value = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9
        return value if value > 0.0 else None

    def _depth_callback(self, msg: Image) -> None:
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = normalize_depth_array(raw, msg.encoding)
        except (CvBridgeError, ValueError):
            return
        stamp = self._stamp(msg) or self.sim_time_sec
        if stamp is None:
            return
        self.depth_history.append((float(stamp), depth.copy()))
        if len(self.depth_history) > 24:
            del self.depth_history[:-24]
        self.depth_encoding = str(msg.encoding)
        self.depth_frames += 1

    def _lookup_transform_components(
        self,
        target_frame: str,
        source_frame: str,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time())
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None
        t = transform.transform.translation
        q = transform.transform.rotation
        return (
            (float(t.x), float(t.y), float(t.z)),
            (float(q.x), float(q.y), float(q.z), float(q.w)),
        )

    def _gripper_pixel(self) -> Optional[Tuple[float, float]]:
        if self.color_model is None:
            return None
        frame = self.color_model.frame_id
        transform = self._lookup_transform_components(
            frame,
            f'gripper_{self.selected_arm}_grasping_link')
        if transform is None:
            return None
        x, y, z = transform[0]
        if z <= 0.05:
            return None
        return (
            self.color_model.fx * x / z + self.color_model.cx,
            self.color_model.fy * y / z + self.color_model.cy,
        )

    def _rgb_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError:
            return
        if frame is None or frame.size == 0:
            return
        stamp = self._stamp(msg) or self.sim_time_sec
        if stamp is None:
            return
        self.latest_frame = frame
        self.latest_rgb_stamp = float(stamp)
        self.rgb_frames += 1
        if self.state != 'DWELL_HEAD_POSE':
            return
        candidates = detect_red_bin(
            frame,
            self.detector_settings,
            excluded_center_px=self._gripper_pixel(),
            excluded_radius_px=85.0)
        if not candidates:
            return
        best = candidates[0]
        self.detection_history.append((float(stamp), best))
        if len(self.detection_history) > 40:
            del self.detection_history[:-40]
        confirmed = consistent_detection(
            self.detection_history,
            required_frames=self.bin_confirm_frames,
            window_sec=self.bin_confirm_window,
            center_tolerance_px=self.bin_center_tolerance)
        if confirmed is None:
            return
        geometry = self._geometry_for_detection(confirmed, float(stamp))
        if geometry is None:
            return
        self.confirmed_detection = confirmed
        self.confirmed_geometry = geometry
        if self.current_xy is None or self.current_yaw is None:
            return
        self.locked_bin_odom_xy = vector_base_to_odom(
            geometry.forward_m,
            geometry.lateral_left_m,
            self.current_xy,
            self.current_yaw)
        self.lock_sim_time = self.sim_time_sec
        self._save_evidence_image(frame, confirmed, geometry)
        self._transition(
            'RESTORE_HEAD',
            'large red bin temporally confirmed with live depth and TF; '
            'locking bin position in odometry')

    def _geometry_for_detection(
        self,
        detection: BinDetection,
        rgb_stamp: float,
    ):
        if (
            self.color_model is None
            or self.depth_model is None
            or not self.depth_history
            or self.sim_time_sec is None
        ):
            return None
        depth_stamp, depth = min(
            self.depth_history,
            key=lambda item: abs(item[0] - rgb_stamp))
        skew = abs(float(depth_stamp) - float(rgb_stamp))
        if skew > self.max_rgb_depth_skew:
            return None
        if self.sim_time_sec - float(depth_stamp) > self.depth_stale_timeout:
            return None
        transform = self._lookup_transform_components(
            'base_footprint', self.depth_model.frame_id)
        if transform is None:
            return None
        x, y, width, height = detection.bbox_xywh
        bbox = BoundingBoxLike(x, y, width, height)
        try:
            geometry = target_geometry_from_depth(
                bbox,
                self.color_model,
                self.depth_model,
                depth,
                transform[0],
                transform[1],
                rgb_depth_stamp_skew_sec=skew,
                min_valid_pixels=self.depth_min_valid_pixels,
                min_depth_m=0.20,
                max_depth_m=5.0)
        except ValueError:
            return None
        if geometry is None:
            return None
        if geometry.depth.valid_fraction < self.depth_min_valid_fraction:
            return None
        if geometry.forward_m <= 0.20:
            return None
        return geometry

    def _scan_callback(self, msg: LaserScan) -> None:
        transform = self._lookup_transform_components(
            'base_footprint', msg.header.frame_id)
        if transform is None:
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
                corridor_half_width_m=self.corridor_half_width,
                robust_percentile=self.lidar_percentile,
                hard_stop_m=self.front_hard_stop)
        except ValueError:
            return
        if clearance is None:
            return
        self.front_scan = clearance
        self.front_scan_stamp = self.sim_time_sec
        value = clearance.minimum_clearance_m
        self.minimum_front_clearance = (
            value if self.minimum_front_clearance is None
            else min(self.minimum_front_clearance, value))

    def _contact_callback(self, msg) -> None:
        fingertip_tokens = selected_fingertip_tokens(self.selected_arm)
        for contact in getattr(msg, 'contacts', []):
            names = contact_collision_names(contact)
            selected_tip = any(
                any(token in name for token in fingertip_tokens)
                for name in names)
            meaningful = [name for name in names if meaningful_robot_collision(name)]
            if selected_tip:
                self.fingertip_contact_messages += 1
                if self.sim_time_sec is not None:
                    self.last_fingertip_contact_sim = self.sim_time_sec
            unexpected = [
                name for name in meaningful
                if not any(token in name for token in fingertip_tokens)
            ]
            if unexpected:
                self.unintended_robot_contacts += 1
                self._finish(
                    False,
                    'unintended robot contact during bin approach: '
                    + ', '.join(unexpected[:4]))
                return

    def _bin_contact_callback(self, msg) -> None:
        """Reject only premature robot/book contact with the collection bin.

        The official bin rests on the table, so /bin_contacts continuously
        contains a legitimate bin-table support contact.  That static arena
        contact must not be treated as a Day 7 approach collision.
        """
        for contact in getattr(msg, 'contacts', []):
            names = contact_collision_names(contact)

            bin_names = [
                name for name in names
                if 'erc_collection_bin::' in name
            ]
            if not bin_names:
                continue

            other_names = [
                name for name in names
                if 'erc_collection_bin::' not in name
            ]

            premature = [
                name for name in other_names
                if (
                    meaningful_robot_collision(name)
                    or name.startswith('book_col_')
                )
            ]

            if not premature:
                continue

            self.bin_contact_messages += 1
            if not self.done:
                self._finish(
                    False,
                    'bin contact occurred during Day 7 approach before '
                    'the Day 8 placement stage: '
                    + ', '.join(premature[:4]))
            return

    def _publish_head(self, pan: float, tilt: float) -> None:
        message = JointTrajectory()
        message.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(pan), float(tilt)]
        whole = int(self.head_motion_sec)
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int(
            round((self.head_motion_sec - whole) * 1.0e9))
        message.points = [point]
        self.head_pub.publish(message)
        self.head_command_sim = self.sim_time_sec
        self.rgb_frames_at_head_command = self.rgb_frames
        self.head_settle_count = 0

    def _head_at(self, pan: float, tilt: float) -> bool:
        actual_pan = self.head_positions.get('head_1_joint')
        actual_tilt = self.head_positions.get('head_2_joint')
        return (
            actual_pan is not None
            and actual_tilt is not None
            and abs(actual_pan - pan) <= self.head_tolerance
            and abs(actual_tilt - tilt) <= self.head_tolerance)

    def _gripper_message(self) -> JointTrajectory:
        message = JointTrajectory()
        message.joint_names = [
            f'gripper_{self.selected_arm}_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [self.gripper_hold_position]
        point.time_from_start.nanosec = 200_000_000
        message.points = [point]
        return message

    def _maintain_gripper(self) -> None:
        if self.sim_time_sec is None:
            return
        if (
            self.last_gripper_hold_sim is None
            or self.sim_time_sec - self.last_gripper_hold_sim
            >= self.gripper_hold_period
        ):
            self.gripper_pub.publish(self._gripper_message())
            self.last_gripper_hold_sim = self.sim_time_sec

    def _retention_ok(self) -> bool:
        return (
            self.last_fingertip_contact_sim is not None
            and self.sim_time_sec is not None
            and self.sim_time_sec - self.last_fingertip_contact_sim
            <= self.retention_stale
            and self.unintended_robot_contacts == 0)

    def _scan_fresh(self) -> bool:
        return (
            self.front_scan_stamp is not None
            and self.sim_time_sec is not None
            and self.sim_time_sec - self.front_scan_stamp
            <= self.scan_stale_timeout)

    def _inputs_ready(self) -> bool:
        return (
            self.sim_time_sec is not None
            and self.current_xy is not None
            and self.current_yaw is not None
            and self.color_model is not None
            and self.depth_model is not None
            and bool(self.depth_history)
            and self.latest_frame is not None
            and self.front_scan is not None
            and self._scan_fresh()
            and all(
                name in self.joint_positions
                for name in self.selected_joint_names)
            and self.head_pub.get_subscription_count() > 0
            and self.gripper_pub.get_subscription_count() > 0
            and self.arm_pub.get_subscription_count() > 0)

    def _plan_carry_raise(self) -> Tuple[bool, str]:
        """Plan the validated high/compact held-book carry pose."""
        try:
            share = get_package_share_directory('erc_description')
            urdf_path = os.path.join(share, 'urdf', 'tiago_pro.urdf')
            model = URDFKinematicModel.from_file(urdf_path)

            tip = f'gripper_{self.selected_arm}_grasping_link'
            chain = model.chain('base_footprint', tip)

            if not all(
                name in self.joint_positions
                for name in self.selected_joint_names
            ):
                return False, 'selected-arm joint state unavailable'

            current = np.asarray(
                [
                    self.joint_positions[name]
                    for name in self.selected_joint_names
                ],
                dtype=np.float64,
            )

            target_xyz = (
                self.day5_lift_xyz
                + np.asarray(
                    [
                        -self.carry_retract_distance,
                        0.0,
                        self.carry_raise_distance,
                    ],
                    dtype=np.float64,
                )
            )

            lower, upper = model.limits(
                self.selected_joint_names,
                margin=0.025,
            )

            seeds = deterministic_seeds(
                current,
                lower,
                upper,
            )

            solution = solve_with_seeds(
                model,
                chain,
                self.selected_joint_names,
                self.joint_positions,
                target_xyz,
                self.carry_target_rotation,
                seeds,
            )

            if not solution.success:
                return (
                    False,
                    'carry-raise IK did not converge: '
                    + str(solution.reason),
                )

            path = interpolate_joint_path(
                current,
                solution.positions,
                samples=40,
            )

            # The robot is no longer beside the shelf on Day 7.
            # A remote shelf plane disables shelf-specific proximity while
            # retaining the planner's bounded workspace/body path checks.
            screen = screen_joint_path(
                model,
                chain,
                self.selected_joint_names,
                self.joint_positions,
                path,
                shelf_plane_x_m=10.0,
                book_surface_xyz_m=self.day5_lift_xyz,
                stage='high_carry',
            )

            if not screen.passed:
                return (
                    False,
                    'carry-raise path screen rejected motion: '
                    + '; '.join(screen.violations[:4]),
                )

            delta = np.asarray(solution.positions) - current
            maximum_delta = float(np.max(np.abs(delta)))

            if maximum_delta > self.carry_raise_max_joint_delta_limit:
                return (
                    False,
                    'carry-raise joint displacement exceeds bounded limit: '
                    f'{maximum_delta:.3f} rad',
                )

            self.carry_raise_target_xyz = tuple(
                float(v) for v in target_xyz)
            self.carry_raise_target_positions = tuple(
                float(v) for v in solution.positions)
            self.carry_raise_max_joint_delta = maximum_delta
            self.carry_raise_screen = screen.as_dict()
            self.carry_raise_ik_position_error_m = float(
                solution.position_error_m)
            self.carry_raise_ik_orientation_error_rad = float(
                solution.orientation_error_rad)

            return True, (
                'validated high/compact carry planned: '
                'retract=%.3f m raise=%.3f m; '
                'IK position error=%.4f m max joint delta=%.3f rad'
                % (
                    self.carry_retract_distance,
                    self.carry_raise_distance,
                    self.carry_raise_ik_position_error_m,
                    maximum_delta,
                )
            )

        except Exception as exc:
            return False, f'carry-raise planning exception: {exc}'


    def _publish_carry_raise(self) -> None:
        if self.carry_raise_target_positions is None:
            raise RuntimeError('carry raise target is unavailable')

        message = JointTrajectory()
        message.joint_names = list(self.selected_joint_names)

        point = JointTrajectoryPoint()
        point.positions = list(self.carry_raise_target_positions)

        whole = int(self.carry_raise_motion_sec)
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int(
            round(
                (self.carry_raise_motion_sec - whole)
                * 1.0e9
            )
        )

        message.points = [point]
        self.arm_pub.publish(message)

        self.carry_raise_command_sim = self.sim_time_sec
        self.carry_raise_settle_count = 0


    def _carry_raise_error(self) -> Optional[float]:
        if self.carry_raise_target_positions is None:
            return None

        if not all(
            name in self.joint_positions
            for name in self.selected_joint_names
        ):
            return None

        return max(
            abs(
                self.joint_positions[name]
                - target
            )
            for name, target in zip(
                self.selected_joint_names,
                self.carry_raise_target_positions,
            )
        )


    def _publish_velocity(self, vx: float, vy: float, wz: float) -> None:
        message = Twist()
        message.linear.x = float(vx)
        message.linear.y = float(vy)
        message.angular.z = float(wz)
        self.cmd_pub.publish(message)

    def _stop_base(self) -> None:
        for _ in range(3):
            self._publish_velocity(0.0, 0.0, 0.0)

    def _transition(self, state: str, reason: str) -> None:
        self.state = state
        self.state_started_sim = self.sim_time_sec
        self.get_logger().info(f'[DAY7] STATE -> {state}: {reason}')
        self._write_result(False, f'running: {reason}')

    def _elapsed(self) -> Optional[float]:
        if self.sim_time_sec is None or self.state_started_sim is None:
            return None
        return max(0.0, self.sim_time_sec - self.state_started_sim)

    def _save_evidence_image(self, frame, detection, geometry) -> None:
        image = frame.copy()
        x, y, width, height = detection.bbox_xywh
        cv2.rectangle(
            image, (x, y), (x + width, y + height), (0, 255, 255), 3)
        lines = [
            'KU SPARCy ERC 2026 - RED COLLECTION BIN',
            f'confidence: {detection.confidence:.3f}',
            f'base forward/left: {geometry.forward_m:.3f}/{geometry.lateral_left_m:.3f} m',
            f'held-book-aware standoff: {self.final_standoff:.3f} m',
            f'sim time: {self.sim_time_sec}',
            f'UTC: {datetime.now(timezone.utc).isoformat()}',
        ]
        y_text = 22
        for line in lines:
            cv2.putText(
                image, line, (8, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y_text += 18
        os.makedirs(self.image_output_dir, exist_ok=True)
        filename = (
            'day7_bin_'
            f'sim_{(self.sim_time_sec or 0.0):012.3f}_'
            f'utc_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}.png')
        path = os.path.join(self.image_output_dir, filename)
        temporary = path + '.tmp.png'
        if cv2.imwrite(temporary, image):
            os.replace(temporary, path)
            self.evidence_image_path = path

    def _hard_stop(self) -> bool:
        return (
            self.front_scan is not None
            and self.front_scan.minimum_clearance_m <= self.front_hard_stop
            and self.front_scan.hard_cluster_point_count
            >= self.hard_cluster_points)

    def _record_path(self) -> None:
        if self.current_xy is None or self.current_yaw is None:
            return
        if self.path_trace and self.sim_time_sec is not None:
            previous = self.path_trace[-1].get('sim_time_sec')
            if isinstance(previous, (int, float)) and self.sim_time_sec - previous < 0.25:
                return
        forward = left = None
        if self.locked_bin_odom_xy is not None:
            forward, left = vector_odom_to_base(
                self.locked_bin_odom_xy,
                self.current_xy,
                self.current_yaw)
        self.path_trace.append({
            'sim_time_sec': self.sim_time_sec,
            'state': self.state,
            'odom_xy': list(self.current_xy),
            'yaw_rad': self.current_yaw,
            'bin_forward_m': forward,
            'bin_left_m': left,
            'front_lidar_min_m': (
                self.front_scan.minimum_clearance_m
                if self.front_scan is not None else None),
            'front_lidar_robust_m': (
                self.front_scan.robust_clearance_m
                if self.front_scan is not None else None),
        })
        if len(self.path_trace) > 400:
            del self.path_trace[:-400]

    def _tick(self) -> None:
        if self.done:
            return
        self._maintain_gripper()
        self._record_path()
        if time.monotonic() - self.last_clock_wall > self.clock_stall_wall_timeout:
            self._finish(False, 'Gazebo /clock stopped advancing')
            return
        if self.state != 'WAIT_INPUTS' and not self._retention_ok():
            self._stop_base()
            self._finish(False, 'held-book fingertip contact became stale')
            return
        if self.state == 'WAIT_INPUTS':
            self._stop_base()
            if self._inputs_ready():
                self.current_pose_index = 0
                pan, tilt = self.head_poses[0]
                self._publish_head(pan, tilt)
                self._transition(
                    'WAIT_HEAD_POSE',
                    f'commanded bin-search head pose 0 pan={pan:.3f} '
                    f'tilt={tilt:.3f}')
            elif time.monotonic() - self.started_wall > self.ready_wall_timeout:
                self._finish(False, 'Day 7 readiness timed out')
            return

        if self.state == 'WAIT_HEAD_POSE':
            self._stop_base()
            pan, tilt = self.head_poses[self.current_pose_index]
            elapsed = (
                self.sim_time_sec - self.head_command_sim
                if self.sim_time_sec is not None
                and self.head_command_sim is not None else None)
            if self._head_at(pan, tilt):
                self.head_settle_count += 1
            else:
                self.head_settle_count = 0
            if (
                self.head_settle_count >= self.head_settle_required
                and self.rgb_frames >= self.rgb_frames_at_head_command + 2
            ):
                self.head_dwell_started_sim = self.sim_time_sec
                self.detection_history.clear()
                self._transition(
                    'DWELL_HEAD_POSE',
                    f'head pose {self.current_pose_index} settled; '
                    'collecting live red-bin observations')
                return
            if elapsed is not None and elapsed > self.head_pose_timeout:
                self._finish(False, 'bin-search head pose failed to settle')
            return

        if self.state == 'DWELL_HEAD_POSE':
            self._stop_base()
            if self.confirmed_detection is not None:
                return
            elapsed = (
                self.sim_time_sec - self.head_dwell_started_sim
                if self.sim_time_sec is not None
                and self.head_dwell_started_sim is not None else None)
            if elapsed is not None and elapsed >= self.head_dwell_sec:
                self.current_pose_index += 1
                if self.current_pose_index >= len(self.head_poses):
                    self._finish(
                        False,
                        'all safe head poses exhausted without a temporally '
                        'confirmed large red collection bin')
                    return
                pan, tilt = self.head_poses[self.current_pose_index]
                self._publish_head(pan, tilt)
                self._transition(
                    'WAIT_HEAD_POSE',
                    f'commanded bin-search head pose {self.current_pose_index} '
                    f'pan={pan:.3f} tilt={tilt:.3f}')
            return

        if self.state == 'RESTORE_HEAD':
            self._stop_base()
            self._publish_head(0.0, -0.15)
            self._transition(
                'WAIT_RESTORE_HEAD',
                'restoring centered safe bin-approach head pose')
            return

        if self.state == 'WAIT_RESTORE_HEAD':
            self._stop_base()
            elapsed = (
                self.sim_time_sec - self.head_command_sim
                if self.sim_time_sec is not None
                and self.head_command_sim is not None else None)
            if self._head_at(0.0, -0.15):
                self.head_settle_count += 1
            else:
                self.head_settle_count = 0
            if self.head_settle_count >= self.head_settle_required:
                self._transition(
                    'PLAN_CARRY_RAISE',
                    'bin-approach head restored; base remains stopped while '
                    'planning the validated held-book vertical clearance raise')
                return
            if elapsed is not None and elapsed > self.head_pose_timeout:
                self._finish(False, 'bin-approach head restore timed out')
            return

        if self.state == 'PLAN_CARRY_RAISE':
            self._stop_base()

            ok, reason = self._plan_carry_raise()
            if not ok:
                self._finish(False, reason)
                return

            self._publish_carry_raise()
            self._transition(
                'WAIT_CARRY_RAISE',
                reason
                + '; selected arm trajectory published while base remains '
                  'stationary and gripper position hold remains active')
            return

        if self.state == 'WAIT_CARRY_RAISE':
            self._stop_base()

            if (
                self.sim_time_sec is None
                or self.carry_raise_command_sim is None
            ):
                return

            error = self._carry_raise_error()

            if (
                error is not None
                and error <= self.carry_raise_joint_tolerance
            ):
                self.carry_raise_settle_count += 1
            else:
                self.carry_raise_settle_count = 0

            if (
                self.carry_raise_settle_count
                >= self.carry_raise_settle_required
            ):
                self.carry_raise_completed = True

                if (
                    self.current_xy is None
                    or self.current_yaw is None
                    or self.locked_bin_odom_xy is None
                ):
                    self._finish(
                        False,
                        'pose unavailable before live-bin alignment')
                    return

                forward, left = vector_odom_to_base(
                    self.locked_bin_odom_xy,
                    self.current_xy,
                    self.current_yaw)

                self.preapproach_initial_bearing_rad = math.atan2(
                    left,
                    forward,
                )
                self.preapproach_align_started_sim = self.sim_time_sec
                self.preapproach_align_settle_count = 0

                self._transition(
                    'ALIGN_BIN_FROM_START',
                    'validated high/compact carry settled; rotating base '
                    'in place toward live locked collection-bin bearing '
                    'before any translational approach')
                return

            elapsed = (
                self.sim_time_sec
                - self.carry_raise_command_sim
            )

            if elapsed > self.carry_raise_timeout:
                self._finish(
                    False,
                    'held-book carry raise failed to settle before timeout')
            return

        if self.state == 'ALIGN_BIN_FROM_START':
            self._stop_base()

            if (
                self.sim_time_sec is None
                or self.preapproach_align_started_sim is None
                or self.current_xy is None
                or self.current_yaw is None
                or self.locked_bin_odom_xy is None
            ):
                return

            elapsed = (
                self.sim_time_sec
                - self.preapproach_align_started_sim
            )

            if elapsed > self.preapproach_align_timeout:
                self._finish(
                    False,
                    'live-bin rotation-in-place alignment exceeded '
                    'simulation-time timeout')
                return

            forward, left = vector_odom_to_base(
                self.locked_bin_odom_xy,
                self.current_xy,
                self.current_yaw)

            bearing = math.atan2(
                left,
                forward,
            )

            if abs(bearing) <= self.preapproach_align_tolerance:
                self._stop_base()
                self.preapproach_align_settle_count += 1

                if (
                    self.preapproach_align_settle_count
                    >= self.preapproach_align_settle_required
                ):
                    self.rotation_alignment_completed = True
                    self.approach_started_sim = self.sim_time_sec
                    self.final_settle_count = 0

                    self._transition(
                        'APPROACH_BIN',
                        'live collection-bin bearing aligned using '
                        'rotation-only motion with validated high/compact '
                        'held-book pose')
                return

            self.preapproach_align_settle_count = 0

            wz = clamp(
                self.yaw_kp * bearing,
                -self.preapproach_align_max_yaw_speed,
                self.preapproach_align_max_yaw_speed,
            )

            # Physically validated condition:
            # absolutely no X/Y translation during alignment.
            self._publish_velocity(
                0.0,
                0.0,
                wz,
            )
            return

        if self.state == 'APPROACH_BIN':
            if (
                self.sim_time_sec is None
                or self.approach_started_sim is None
                or self.current_xy is None
                or self.current_yaw is None
                or self.locked_bin_odom_xy is None
            ):
                self._stop_base()
                return
            if self.sim_time_sec - self.approach_started_sim > self.approach_timeout:
                self._finish(False, 'bin approach exceeded simulation-time timeout')
                return
            if not self._scan_fresh():
                self._stop_base()
                return

            forward, left = vector_odom_to_base(
                self.locked_bin_odom_xy,
                self.current_xy,
                self.current_yaw)

            distance_error = forward - self.final_standoff
            bearing = math.atan2(left, forward)

            if (
                abs(distance_error) <= self.final_distance_tolerance
                and abs(left) <= self.final_lateral_tolerance
                and abs(bearing) <= self.final_bearing_tolerance
            ):
                self._stop_base()
                self.final_settle_count += 1

                if self.final_settle_count >= self.final_settle_required:
                    self._transition(
                        'VERIFY_BIN_APPROACH',
                        'high/compact held-book bin standoff remained stable')
                return

            self.final_settle_count = 0

            # A front hard stop is only fatal while trying to move TOWARD
            # the bin.  If we start too close, controlled reverse motion
            # increases clearance and must remain available.
            if (
                distance_error > self.final_distance_tolerance
                and self._hard_stop()
            ):
                self._stop_base()
                self._finish(
                    False,
                    'front LiDAR hard stop during forward bin approach')
                return

            # If heading error grows, stop translation and realign first.
            if abs(bearing) > 2.0 * self.final_bearing_tolerance:
                vx = 0.0
            else:
                vx = clamp(
                    self.forward_kp * distance_error,
                    -self.max_reverse_speed,
                    self.max_forward_speed,
                )

            wz = clamp(
                self.yaw_kp * bearing,
                -self.aligned_approach_max_yaw_speed,
                self.aligned_approach_max_yaw_speed,
            )

            # No lateral base translation with the held book.
            self._publish_velocity(
                vx,
                0.0,
                wz,
            )
            return

        if self.state == 'VERIFY_BIN_APPROACH':
            self._stop_base()
            if self.bin_contact_messages != 0:
                self._finish(False, 'premature bin contact occurred')
                return
            if not self._retention_ok():
                self._finish(False, 'book was not retained at bin approach pose')
                return
            self._finish(
                True,
                'red collection bin identified from live RGB-D and reached at '
                'a held-book-aware, collision-free placement-ready standoff')
            return

        self._finish(False, f'unknown Day 7 state: {self.state}')

    def _write_result(self, passed: bool, reason: str) -> None:
        final_forward = final_left = None
        if (
            self.locked_bin_odom_xy is not None
            and self.current_xy is not None
            and self.current_yaw is not None
        ):
            final_forward, final_left = vector_odom_to_base(
                self.locked_bin_odom_xy,
                self.current_xy,
                self.current_yaw)
        result = {
            'day': 7,
            'passed': bool(passed),
            'reason': reason,
            'state': self.state,
            'selected_arm': self.selected_arm,
            'day5_result_path': self.day5_path,
            'day6_result_path': self.day6_path,
            'bin_detection': (
                self.confirmed_detection.as_dict()
                if self.confirmed_detection is not None else None),
            'bin_geometry': (
                self.confirmed_geometry.as_dict()
                if self.confirmed_geometry is not None else None),
            'locked_bin_odom_xy': (
                list(self.locked_bin_odom_xy)
                if self.locked_bin_odom_xy is not None else None),
            'bin_lock_sim_time_sec': self.lock_sim_time,
            'bin_evidence_image_path': self.evidence_image_path,
            'final_held_book_aware_standoff_m': self.final_standoff,
            'carried_book_forward_extent_m': self.carried_forward_extent,
            'carried_book_clearance_margin_m': self.book_clearance_margin,
            'carry_raise_distance_m': self.carry_raise_distance,
            'carry_retract_distance_m': self.carry_retract_distance,
            'carry_raise_completed': self.carry_raise_completed,
            'carry_raise_target_xyz_m': (
                list(self.carry_raise_target_xyz)
                if self.carry_raise_target_xyz is not None else None),
            'carry_raise_target_positions_rad': (
                list(self.carry_raise_target_positions)
                if self.carry_raise_target_positions is not None else None),
            'carry_raise_max_joint_delta_rad': (
                self.carry_raise_max_joint_delta),
            'carry_raise_ik_position_error_m': (
                self.carry_raise_ik_position_error_m),
            'carry_raise_ik_orientation_error_rad': (
                self.carry_raise_ik_orientation_error_rad),
            'carry_raise_collision_screen': self.carry_raise_screen,
            'carried_book_swept_radius_m': self.carried_swept_radius,
            'compact_book_forward_extent_m': self.compact_forward_extent,
            'preapproach_initial_bin_bearing_rad': (
                self.preapproach_initial_bearing_rad),
            'rotation_alignment_completed': (
                self.rotation_alignment_completed),
            'final_bin_forward_m': final_forward,
            'final_bin_left_m': final_left,
            'final_odom_xy': (
                list(self.current_xy) if self.current_xy is not None else None),
            'final_yaw_rad': self.current_yaw,
            'minimum_front_clearance_m': self.minimum_front_clearance,
            'fingertip_contact_messages': self.fingertip_contact_messages,
            'last_fingertip_contact_sim_sec': self.last_fingertip_contact_sim,
            'unintended_robot_contacts': self.unintended_robot_contacts,
            'bin_contact_messages': self.bin_contact_messages,
            'gripper_public_topic': (
                f'/gripper_{self.selected_arm}_controller/joint_trajectory'),
            'gripper_command_mode': 'position',
            'effort_commanded': False,
            'torso_commanded': False,
            'bin_detection_uses_live_rgb_depth_tf': True,
            'hidden_simulator_oracle_used': False,
            'validation_layout_oracle_used': False,
            'path_trace': list(self.path_trace),
            'simulation_time_sec': self.sim_time_sec,
            'wall_duration_sec': time.monotonic() - self.started_wall,
            'completed_utc': datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.result_path, result)

    def _finish(self, passed: bool, reason: str) -> None:
        if self.done:
            return
        self._stop_base()
        self.done = True
        self.passed = bool(passed)
        self.reason = str(reason)
        self.state = 'DONE' if passed else 'FAILED'
        self._write_result(passed, reason)
        label = 'PASSED' if passed else 'FAILED'
        self.get_logger().info(f'[DAY7][{label}] {reason}')


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[Day7BinApproach] = None
    code = 1
    try:
        node = Day7BinApproach()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.10)
        code = 0 if node.done and node.passed else 1
    except KeyboardInterrupt:
        if node is not None:
            node._finish(False, 'interrupted by operator')
        code = 130
    except Exception as exc:
        if node is not None:
            node._finish(False, f'unhandled exception: {exc}')
        else:
            print(f'[DAY7][FAILED] {exc}')
        code = 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
