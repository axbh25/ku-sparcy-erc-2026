#!/usr/bin/env python3
"""Guarded Day 4 one-arm grasp laboratory for the merged official simulator.

The normal competition entry point still stops after perception, row
publication, evidence, synchronized book geometry, and a non-executing
pre-grasp screen.  This separate node is launched only after those gates pass.
It parses the official URDF, creates a bounded numerical IK sequence, and then
executes one arm through explicit development checkpoints.

No effort command interface is created or commanded.  The selected PAL Pro
gripper receives only position trajectories through the public clamped topic.
Effort is recorded from /joint_states as diagnostic state only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import time
import xml.etree.ElementTree as ET
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
from sensor_msgs.msg import Image, JointState, LaserScan
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros

try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:  # pragma: no cover - live container dependency probe catches this
    Contacts = None  # type: ignore

from ku_sparcy_erc.grasp_planner import GraspPlan, build_grasp_plan
from ku_sparcy_erc.range_fusion import LidarClearance, lidar_corridor_clearance
from ku_sparcy_erc.urdf_kinematics import URDFKinematicModel


STAGE_ORDER = (
    'plan', 'pregrasp', 'open', 'no_contact',
    'contact_pose', 'close', 'extract', 'lift')
ARM_STAGE_KEYS = {
    'pregrasp': 'pregrasp',
    'no_contact': 'no_contact',
    'contact_pose': 'grasp',
    'extract': 'extract',
    'lift': 'lift',
}


class StagedGraspExperiment(Node):
    """Execute a collision-screened one-arm grasp with manual development gates."""

    RGB_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
    SCAN_TOPIC = '/scan_front_raw'
    CONTACT_TOPIC = '/contacts'

    def __init__(self) -> None:
        super().__init__('ku_sparcy_day4_grasp_experiment')
        self.declare_parameter(
            'source_result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day4_result_full_gui.json')
        self.declare_parameter(
            'grasp_result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day4_grasp_result.json')
        self.declare_parameter(
            'image_output_dir',
            '/opt/erc_ws/src/ku_sparcy_erc/erc_images')
        self.declare_parameter('target_stage', 'lift')
        self.declare_parameter('manual_approval_required', True)
        self.declare_parameter('manual_approval_wall_timeout_sec', 600.0)
        self.declare_parameter('clock_stall_wall_timeout_sec', 20.0)
        self.declare_parameter('ready_wall_timeout_sec', 40.0)
        self.declare_parameter('joint_position_tolerance_rad', 0.045)
        self.declare_parameter('joint_settle_samples', 3)
        self.declare_parameter('pregrasp_motion_sec', 4.0)
        self.declare_parameter('no_contact_motion_sec', 2.0)
        self.declare_parameter('contact_motion_sec', 1.5)
        self.declare_parameter('extract_motion_sec', 2.0)
        self.declare_parameter('lift_motion_sec', 2.0)
        self.declare_parameter('stage_timeout_margin_sec', 2.5)
        self.declare_parameter('gripper_open_position_m', 0.040)
        self.declare_parameter('gripper_close_positions_m', [0.018, 0.014, 0.010])
        self.declare_parameter('gripper_motion_sec', 1.05)
        self.declare_parameter('gripper_hold_sec', 1.0)
        self.declare_parameter('front_clearance_abort_m', 0.70)
        self.declare_parameter('lidar_corridor_half_width_m', 0.38)
        self.declare_parameter('lidar_robust_percentile', 10.0)
        self.declare_parameter('lidar_hard_cluster_points', 3)
        self.declare_parameter('source_pose_tolerance_m', 0.035)
        self.declare_parameter('source_yaw_tolerance_deg', 2.0)
        self.declare_parameter('pregrasp_offset_m', 0.18)
        self.declare_parameter('no_contact_offset_m', 0.070)
        self.declare_parameter('grasp_insertion_depth_m', 0.020)
        self.declare_parameter('extract_distance_m', 0.10)
        self.declare_parameter('lift_distance_m', 0.08)

        get = self.get_parameter
        self.source_result_path = str(get('source_result_path').value)
        self.grasp_result_path = str(get('grasp_result_path').value)
        self.image_output_dir = str(get('image_output_dir').value)
        self.target_stage = str(get('target_stage').value).strip().lower()
        self.manual_approval_required = bool(get('manual_approval_required').value)
        self.manual_approval_wall_timeout = float(
            get('manual_approval_wall_timeout_sec').value)
        self.clock_stall_wall_timeout = float(
            get('clock_stall_wall_timeout_sec').value)
        self.ready_wall_timeout = float(get('ready_wall_timeout_sec').value)
        self.joint_tolerance = float(get('joint_position_tolerance_rad').value)
        self.joint_settle_required = int(get('joint_settle_samples').value)
        self.motion_durations = {
            'pregrasp': float(get('pregrasp_motion_sec').value),
            'no_contact': float(get('no_contact_motion_sec').value),
            'contact_pose': float(get('contact_motion_sec').value),
            'extract': float(get('extract_motion_sec').value),
            'lift': float(get('lift_motion_sec').value),
        }
        self.stage_timeout_margin = float(get('stage_timeout_margin_sec').value)
        self.gripper_open_position = float(get('gripper_open_position_m').value)
        self.gripper_close_positions = [
            float(value) for value in get('gripper_close_positions_m').value]
        self.gripper_motion_sec = float(get('gripper_motion_sec').value)
        self.gripper_hold_sec = float(get('gripper_hold_sec').value)
        self.front_clearance_abort = float(get('front_clearance_abort_m').value)
        self.lidar_corridor_half_width = float(
            get('lidar_corridor_half_width_m').value)
        self.lidar_robust_percentile = float(
            get('lidar_robust_percentile').value)
        self.lidar_hard_cluster_points = int(
            get('lidar_hard_cluster_points').value)
        self.source_pose_tolerance = float(get('source_pose_tolerance_m').value)
        self.source_yaw_tolerance = math.radians(float(
            get('source_yaw_tolerance_deg').value))
        self.pregrasp_offset = float(get('pregrasp_offset_m').value)
        self.no_contact_offset = float(get('no_contact_offset_m').value)
        self.insertion_depth = float(get('grasp_insertion_depth_m').value)
        self.extract_distance = float(get('extract_distance_m').value)
        self.lift_distance = float(get('lift_distance_m').value)
        self._validate_parameters()

        self.source_result = self._load_source_result()
        self.bridge = CvBridge()
        self.sim_time_sec: Optional[float] = None
        self.last_clock_wall = time.monotonic()
        self.started_wall = time.monotonic()
        self.started_sim: Optional[float] = None
        self.state = 'WAIT_INPUTS'
        self.done = False
        self.passed = False
        self.reason = ''
        self.plan: Optional[GraspPlan] = None
        self.selected_arm: Optional[str] = None
        self.selected_joint_names: Tuple[str, ...] = ()
        self.joint_positions: Dict[str, float] = {}
        self.joint_efforts: Dict[str, float] = {}
        self.latest_scan_front_min: Optional[float] = None
        self.latest_lidar: Optional[LidarClearance] = None
        self.latest_lidar_robust: Optional[float] = None
        self.lidar_transform_failures = 0
        self.current_odom_xy: Optional[Tuple[float, float]] = None
        self.current_yaw_rad: Optional[float] = None
        self.source_pose_error_m: Optional[float] = None
        self.source_yaw_error_rad: Optional[float] = None
        self.latest_frame_bgr: Optional[np.ndarray] = None
        self.latest_frame_stamp: Optional[float] = None
        self.approval_received = False
        self.approval_wait_started_wall: Optional[float] = None
        self.pending_stage: Optional[str] = None
        self.active_stage: Optional[str] = None
        self.stage_started_sim: Optional[float] = None
        self.stage_target_positions: Optional[Tuple[float, ...]] = None
        self.stage_settle_count = 0
        self.completed_stages: List[str] = []
        self.stage_records: List[Dict[str, object]] = []
        self.evidence_paths: Dict[str, str] = {}
        self.gripper_actual_position: Optional[float] = None
        self.gripper_actual_effort: Optional[float] = None
        self.gripper_effort_samples: List[float] = []
        self.gripper_commanded = False
        self.arm_commanded = False
        self.contact_messages_total = 0
        self.selected_fingertip_contact_messages = 0
        self.premature_selected_fingertip_contacts = 0
        self.unintended_selected_arm_contacts = 0
        self.contact_pairs: List[Dict[str, object]] = []
        self.last_hold_publish_sim: Optional[float] = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)

        self.clock_sub = self.create_subscription(
            Clock, '/clock', self._clock_callback, qos_profile_sensor_data)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 20)
        self.scan_sub = self.create_subscription(
            LaserScan, self.SCAN_TOPIC, self._scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile_sensor_data)
        self.rgb_sub = self.create_subscription(
            Image, self.RGB_TOPIC, self._image_callback, qos_profile_sensor_data)
        if Contacts is not None:
            self.contact_sub = self.create_subscription(
                Contacts, self.CONTACT_TOPIC, self._contact_callback,
                qos_profile_sensor_data)
        self.approval_sub = self.create_subscription(
            Bool, '/ku_sparcy/day4_continue', self._approval_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_publishers = {
            arm: self.create_publisher(
                JointTrajectory, f'/arm_{arm}_controller/joint_trajectory', 10)
            for arm in ('left', 'right')
        }
        self.gripper_publishers = {
            arm: self.create_publisher(
                JointTrajectory, f'/gripper_{arm}_controller/joint_trajectory', 10)
            for arm in ('left', 'right')
        }
        self.status_pub = self.create_publisher(
            String, '/ku_sparcy/day4_grasp_status', 10)
        self.timer = self.create_timer(0.05, self._tick)
        if Contacts is None:
            raise RuntimeError(
                'ros_gz_interfaces/Contacts is unavailable; staged grasp '
                'contact monitoring is mandatory')

        self.get_logger().info(
            '[DAY4-GRASP] guarded experiment ready; '
            f'target_stage={self.target_stage} manual={self.manual_approval_required}')

    def _validate_parameters(self) -> None:
        if self.target_stage not in STAGE_ORDER:
            raise ValueError(f'target_stage must be one of {STAGE_ORDER}')
        if not 0.0 < self.joint_tolerance <= 0.15:
            raise ValueError('joint_position_tolerance_rad is invalid')
        if self.joint_settle_required < 2:
            raise ValueError('joint_settle_samples must be at least 2')
        if not 0.0 <= self.gripper_open_position <= 0.069:
            raise ValueError('gripper open target must stay in [0, 0.069]')
        if not self.gripper_close_positions:
            raise ValueError('gripper close sequence cannot be empty')
        if any(not 0.0 <= value <= 0.069 for value in self.gripper_close_positions):
            raise ValueError('all gripper close targets must stay in [0, 0.069]')
        if any(value <= 0.0 for value in self.motion_durations.values()):
            raise ValueError('all arm motion durations must be positive')
        if self.front_clearance_abort < 0.60:
            raise ValueError('front clearance abort threshold is too small')
        if self.lidar_corridor_half_width <= 0.25:
            raise ValueError('LiDAR corridor must exceed half the robot width')
        if not 0.0 < self.lidar_robust_percentile < 50.0:
            raise ValueError('lidar_robust_percentile is invalid')
        if self.lidar_hard_cluster_points < 2:
            raise ValueError('lidar_hard_cluster_points must be at least 2')
        if not 0.0 < self.source_pose_tolerance <= 0.10:
            raise ValueError('source_pose_tolerance_m is invalid')
        if not 0.0 < self.source_yaw_tolerance <= math.radians(10.0):
            raise ValueError('source_yaw_tolerance_deg is invalid')

    def _load_source_result(self) -> Dict[str, object]:
        with open(self.source_result_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        if not data.get('passed'):
            raise ValueError('source Day 4 result did not pass')
        if data.get('official_upstream_required_commit') != (
            '93554d4f9335b2ee3acb49c6b332611f6ad2a964'
        ):
            raise ValueError(
                'source result was not generated against the reviewed official '
                'gripper-update baseline')
        geometry = data.get('target_book_geometry')
        if not isinstance(geometry, dict):
            raise ValueError('source result has no target_book_geometry')
        point = geometry.get('base_point_xyz_m')
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError('source result has invalid book base_point_xyz_m')
        clearance = data.get('final_lidar_clearance_m')
        if clearance is None or not math.isfinite(float(clearance)):
            raise ValueError('source result has no final LiDAR clearance')
        if not 0.82 <= float(clearance) <= 1.18:
            raise ValueError(
                'source result is outside the validated Day 3 LiDAR stand-off '
                f'interval: {float(clearance):.3f} m')
        if not data.get('column_lock_active'):
            raise ValueError('source result did not preserve physical column lock')
        if data.get('day4_outcome') != 'pregrasp_geometry_ready':
            raise ValueError('source result did not complete the safe pre-grasp geometry gate')
        if data.get('day4_new_arm_trajectory_commanded') is not False:
            raise ValueError('source result unexpectedly contains Day 4 arm motion')
        if data.get('day4_gripper_commanded') is not False:
            raise ValueError('source result unexpectedly contains a gripper command')
        if data.get('selected_manipulation_arm') not in {'left', 'right'}:
            raise ValueError('source result has no valid selected arm')
        pose = data.get('day4_final_base_odom_xy')
        if not isinstance(pose, list) or len(pose) != 2:
            raise ValueError('source result has no final base odometry snapshot')
        yaw = data.get('day4_final_base_yaw_rad')
        if yaw is None or not math.isfinite(float(yaw)):
            raise ValueError('source result has no final base yaw snapshot')
        return data

    def _clock_callback(self, msg: Clock) -> None:
        value = float(msg.clock.sec) + float(msg.clock.nanosec) * 1.0e-9
        if self.sim_time_sec is None or value > self.sim_time_sec + 1.0e-9:
            self.last_clock_wall = time.monotonic()
        self.sim_time_sec = value
        if self.started_sim is None and value > 0.0:
            self.started_sim = value

    def _joint_callback(self, msg: JointState) -> None:
        for index, name in enumerate(msg.name):
            if index < len(msg.position):
                self.joint_positions[name] = float(msg.position[index])
            if index < len(msg.effort):
                effort = float(msg.effort[index])
                self.joint_efforts[name] = effort
        if self.selected_arm is not None:
            gripper_name = f'gripper_{self.selected_arm}_finger_joint'
            self.gripper_actual_position = self.joint_positions.get(gripper_name)
            self.gripper_actual_effort = self.joint_efforts.get(gripper_name)
            if self.active_stage in {'close', 'extract', 'lift'} and self.gripper_actual_effort is not None:
                self.gripper_effort_samples.append(self.gripper_actual_effort)
                if len(self.gripper_effort_samples) > 1000:
                    del self.gripper_effort_samples[:-1000]

    @staticmethod
    def _yaw_from_odom(msg: Odometry) -> float:
        q = msg.pose.pose.orientation
        siny = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
        cosy = 1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2)
        return math.atan2(siny, cosy)

    @staticmethod
    def _angle_error(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _odom_callback(self, msg: Odometry) -> None:
        self.current_odom_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        self.current_yaw_rad = self._yaw_from_odom(msg)

    def _lookup_transform_components(
        self, source_frame: str,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        if not source_frame:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint', source_frame, Time())
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

    def _scan_callback(self, msg: LaserScan) -> None:
        # Preserve Day 3's transformed swept-corridor semantics; raw scan
        # angle zero is not assumed to be robot-forward.
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
                hard_stop_m=self.front_clearance_abort,
            )
        except ValueError:
            self.lidar_transform_failures += 1
            return
        if clearance is None:
            return
        self.latest_lidar = clearance
        self.latest_scan_front_min = float(clearance.minimum_clearance_m)
        self.latest_lidar_robust = float(clearance.robust_clearance_m)

    def _image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError:
            return
        if frame is None or frame.size == 0:
            return
        self.latest_frame_bgr = frame
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9
        self.latest_frame_stamp = stamp if stamp > 0.0 else self.sim_time_sec

    def _contact_callback(self, msg) -> None:
        self.contact_messages_total += 1
        if self.selected_arm is None:
            return
        fingertip_tokens = (
            f'gripper_{self.selected_arm}_fingertip_left_link',
            f'gripper_{self.selected_arm}_fingertip_right_link',
        )
        selected_prefixes = (
            f'arm_{self.selected_arm}_',
            f'gripper_{self.selected_arm}_',
        )
        for contact in getattr(msg, 'contacts', []):
            first = str(getattr(getattr(contact, 'collision1', None), 'name', ''))
            second = str(getattr(getattr(contact, 'collision2', None), 'name', ''))
            names = (first, second)
            depth_values = [float(value) for value in getattr(contact, 'depths', [])]
            pair = {
                'collision1': first,
                'collision2': second,
                'maximum_depth_m': max(depth_values) if depth_values else None,
                'stage': self.active_stage,
                'sim_time_sec': self.sim_time_sec,
            }
            self.contact_pairs.append(pair)
            if len(self.contact_pairs) > 250:
                del self.contact_pairs[:-250]

            selected_names = [
                name for name in names
                if any(token in name for token in selected_prefixes)
            ]
            if not selected_names:
                continue
            fingertip_contact = any(
                any(token in name for token in fingertip_tokens)
                for name in selected_names
            )
            non_fingertip_selected_contact = any(
                not any(token in name for token in fingertip_tokens)
                for name in selected_names
            )
            if non_fingertip_selected_contact:
                self.unintended_selected_arm_contacts += 1
                self._finish(
                    False,
                    'selected arm or non-fingertip gripper link contacted an '
                    'object; inspect contact_pairs before continuing',
                )
                return
            if fingertip_contact:
                self.selected_fingertip_contact_messages += 1
                if self.active_stage not in {
                    'contact_pose', 'close', 'extract', 'lift'
                }:
                    self.premature_selected_fingertip_contacts += 1
                    self._finish(
                        False,
                        'selected gripper fingertip contacted an object before '
                        'the approved contact stage',
                    )
                    return

    def _approval_callback(self, msg: Bool) -> None:
        if bool(msg.data):
            self.approval_received = True
            self.get_logger().info('[DAY4-GRASP] manual CONTINUE received')

    def _stop_base(self) -> None:
        message = Twist()
        for _ in range(3):
            self.cmd_vel_pub.publish(message)

    def _transition(self, state: str, detail: str) -> None:
        self.state = state
        message = String()
        message.data = json.dumps({
            'state': state,
            'detail': detail,
            'sim_time_sec': self.sim_time_sec,
        }, sort_keys=True)
        self.status_pub.publish(message)
        self.get_logger().info(f'[DAY4-GRASP] STATE -> {state}: {detail}')
        self._write_result(False, f'running: {detail}')

    def _required_joint_names(self) -> List[str]:
        names = []
        for arm in ('left', 'right'):
            names.extend(f'arm_{arm}_{index}_joint' for index in range(1, 8))
            names.append(f'gripper_{arm}_finger_joint')
        names.append('torso_lift_joint')
        return names

    def _inputs_ready(self) -> bool:
        return (
            self.sim_time_sec is not None
            and self.latest_scan_front_min is not None
            and self.latest_lidar is not None
            and self.latest_frame_bgr is not None
            and self.current_odom_xy is not None
            and self.current_yaw_rad is not None
            and all(name in self.joint_positions for name in self._required_joint_names())
            and all(
                publisher.get_subscription_count() > 0
                for publisher in self.arm_publishers.values()
            )
            and all(
                publisher.get_subscription_count() > 0
                for publisher in self.gripper_publishers.values()
            )
        )

    @staticmethod
    def _official_assets_are_current(share: str) -> Tuple[bool, str]:
        book_path = os.path.join(
            share, 'models', 'book', 'sdf', 'erc_book.sdf')
        urdf_path = os.path.join(share, 'urdf', 'tiago_pro.urdf')
        try:
            book_root = ET.parse(book_path).getroot()
            sizes = [
                [float(value) for value in element.text.split()]
                for element in book_root.findall('.//geometry/box/size')
                if element.text
            ]
            if not sizes or any(
                len(size) != 3
                or abs(size[0] - 0.25) > 1.0e-9
                or abs(size[1] - 0.02) > 1.0e-9
                or abs(size[2] - 0.16) > 1.0e-9
                for size in sizes
            ):
                return False, f'installed book dimensions are not 0.25 x 0.02 x 0.16 m: {sizes}'
            urdf_root = ET.parse(urdf_path).getroot()
            mimic_efforts = {}
            for joint in urdf_root.findall('joint'):
                name = joint.attrib.get('name', '')
                if (
                    name.startswith('gripper_')
                    and joint.find('mimic') is not None
                ):
                    limit = joint.find('limit')
                    if limit is not None and 'effort' in limit.attrib:
                        mimic_efforts[name] = float(limit.attrib['effort'])

            expected_10_names = {
                'gripper_left_finger_right_joint',
                'gripper_right_finger_right_joint',
            }
            effort_10_names = {
                name for name, value in mimic_efforts.items()
                if abs(value - 10.0) <= 1.0e-9
            }
            effort_40_names = {
                name for name, value in mimic_efforts.items()
                if abs(value - 40.0) <= 1.0e-9
            }

            if (
                len(mimic_efforts) != 14
                or effort_10_names != expected_10_names
                or len(effort_40_names) != 12
                or effort_10_names | effort_40_names != set(mimic_efforts)
            ):
                return False, (
                    'installed URDF mimic-joint effort limits do not match '
                    'official structure: 12 at 40.0 and two '
                    '*_finger_right_joint joints at 10.0'
                )
            for side in ('left', 'right'):
                name = f'gripper_{side}_finger_joint'
                element = next((
                    joint for joint in urdf_root.findall('.//ros2_control/joint')
                    if joint.attrib.get('name') == name
                ), None)
                if element is None:
                    return False, f'{name} missing from ros2_control block'
                commands = [
                    item.attrib.get('name')
                    for item in element.findall('command_interface')
                ]
                if commands != ['position']:
                    return False, f'{name} command interfaces changed: {commands}'
            return True, ''
        except (OSError, ET.ParseError, ValueError) as exc:
            return False, f'could not verify installed official assets: {exc}'

    def _make_plan(self) -> None:
        assert self.current_odom_xy is not None
        assert self.current_yaw_rad is not None
        source_xy = self.source_result['day4_final_base_odom_xy']
        source_yaw = float(self.source_result['day4_final_base_yaw_rad'])
        self.source_pose_error_m = math.hypot(
            self.current_odom_xy[0] - float(source_xy[0]),
            self.current_odom_xy[1] - float(source_xy[1]),
        )
        self.source_yaw_error_rad = abs(self._angle_error(
            self.current_yaw_rad, source_yaw))
        if self.source_pose_error_m > self.source_pose_tolerance:
            self._finish(
                False,
                'base moved since the source Day 4 geometry result '
                f'(error={self.source_pose_error_m:.4f} m)')
            return
        if self.source_yaw_error_rad > self.source_yaw_tolerance:
            self._finish(
                False,
                'base yaw changed since the source Day 4 geometry result '
                f'(error={math.degrees(self.source_yaw_error_rad):.3f} deg)')
            return

        share = get_package_share_directory('erc_description')
        assets_ok, assets_reason = self._official_assets_are_current(share)
        if not assets_ok:
            self._finish(False, assets_reason)
            return
        urdf_path = os.path.join(share, 'urdf', 'tiago_pro.urdf')
        model = URDFKinematicModel.from_file(urdf_path)
        geometry = self.source_result['target_book_geometry']
        surface = geometry['base_point_xyz_m']
        clearance = float(self.source_result['final_lidar_clearance_m'])
        preferred = self.source_result.get('selected_manipulation_arm')
        self.plan = build_grasp_plan(
            model,
            self.joint_positions,
            surface,
            clearance,
            preferred_arm=str(preferred) if preferred else None,
            pregrasp_offset_m=self.pregrasp_offset,
            no_contact_offset_m=self.no_contact_offset,
            insertion_depth_m=self.insertion_depth,
            extract_distance_m=self.extract_distance,
            lift_distance_m=self.lift_distance,
        )
        self.selected_arm = self.plan.selected_arm
        selected = self.plan.candidate_plans[self.selected_arm]
        self.selected_joint_names = selected.joint_names
        if not self.plan.passed:
            self._finish(False, f'grasp plan rejected: {self.plan.reason}')
            return
        self._save_stage_image('plan')
        self.completed_stages.append('plan')
        if self.target_stage == 'plan':
            self._finish(True, 'collision-screened IK plan generated; no robot manipulation motion executed')
            return
        self._request_approval('pregrasp')

    def _request_approval(self, stage: str) -> None:
        self.pending_stage = stage
        self.approval_received = not self.manual_approval_required
        self.approval_wait_started_wall = time.monotonic()
        self._transition(
            'WAIT_APPROVAL',
            f'approval required before {stage}; publish Bool true on /ku_sparcy/day4_continue')

    def _approval_tick(self) -> None:
        self._stop_base()
        if self.approval_received and self.pending_stage is not None:
            stage = self.pending_stage
            self.pending_stage = None
            self.approval_received = False
            self._start_stage(stage)
            return
        if (
            self.approval_wait_started_wall is not None
            and time.monotonic() - self.approval_wait_started_wall
            > self.manual_approval_wall_timeout
        ):
            self._finish(False, 'manual approval timed out in development mode')

    def _arm_solution_for(self, stage: str):
        assert self.plan is not None and self.selected_arm is not None
        selected = self.plan.candidate_plans[self.selected_arm]
        key = ARM_STAGE_KEYS[stage]
        return getattr(selected, key)

    def _arm_trajectory(self, positions: Sequence[float], duration: float) -> JointTrajectory:
        message = JointTrajectory()
        message.joint_names = list(self.selected_joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        whole = int(duration)
        nano = int(round((duration - whole) * 1.0e9))
        if nano >= 1_000_000_000:
            whole += 1
            nano -= 1_000_000_000
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = nano
        message.points = [point]
        return message

    def _gripper_trajectory(self, positions: Sequence[float], total_duration: float) -> JointTrajectory:
        assert self.selected_arm is not None
        message = JointTrajectory()
        message.joint_names = [f'gripper_{self.selected_arm}_finger_joint']
        for index, value in enumerate(positions, start=1):
            point = JointTrajectoryPoint()
            point.positions = [float(value)]
            elapsed = float(total_duration) * index / len(positions)
            whole = int(elapsed)
            nano = int(round((elapsed - whole) * 1.0e9))
            point.time_from_start.sec = whole
            point.time_from_start.nanosec = min(nano, 999_999_999)
            message.points.append(point)
        return message

    def _start_stage(self, stage: str) -> None:
        assert self.selected_arm is not None
        self.active_stage = stage
        self.stage_started_sim = self.sim_time_sec
        self.stage_settle_count = 0
        self.last_hold_publish_sim = None
        if stage in {
            'pregrasp', 'no_contact', 'contact_pose', 'extract', 'lift'
        }:
            solution = self._arm_solution_for(stage)
            self.stage_target_positions = tuple(solution.positions)
            self.arm_publishers[self.selected_arm].publish(
                self._arm_trajectory(solution.positions, self.motion_durations[stage]))
            self.arm_commanded = True
            self._transition(f'MOVE_{stage.upper()}', f'{self.selected_arm} arm trajectory published')
        elif stage == 'open':
            self.stage_target_positions = None
            self.gripper_publishers[self.selected_arm].publish(
                self._gripper_trajectory([self.gripper_open_position], self.gripper_motion_sec))
            self.gripper_commanded = True
            self._transition('OPEN_GRIPPER', 'public position command published to open gripper')
        elif stage == 'close':
            # The separate contact_pose stage has already placed the open
            # fingers around the bounded insertion target. This stage changes
            # only the public gripper position command.
            self.stage_target_positions = None
            self.gripper_publishers[self.selected_arm].publish(
                self._gripper_trajectory(
                    self.gripper_close_positions,
                    self.gripper_motion_sec,
                ))
            self.gripper_commanded = True
            self._transition(
                'CLOSE_GRIPPER',
                'position-only stepped closure published after contact-pose approval',
            )
        else:
            self._finish(False, f'unknown stage: {stage}')

    def _joint_error(self) -> Optional[float]:
        if self.stage_target_positions is None:
            return None
        if any(name not in self.joint_positions for name in self.selected_joint_names):
            return None
        return max(
            abs(self.joint_positions[name] - target)
            for name, target in zip(self.selected_joint_names, self.stage_target_positions)
        )

    def _elapsed(self) -> Optional[float]:
        if self.sim_time_sec is None or self.stage_started_sim is None:
            return None
        return max(0.0, self.sim_time_sec - self.stage_started_sim)

    def _maintain_gripper_hold(self) -> None:
        if (
            self.selected_arm is None
            or not self.gripper_commanded
            or not self.gripper_close_positions
            or self.sim_time_sec is None
        ):
            return
        if (
            self.last_hold_publish_sim is None
            or self.sim_time_sec - self.last_hold_publish_sim >= 0.20
        ):
            self.gripper_publishers[self.selected_arm].publish(
                self._gripper_trajectory(
                    [self.gripper_close_positions[-1]], 0.20))
            self.last_hold_publish_sim = self.sim_time_sec

    def _arm_motion_tick(self, stage: str) -> None:
        self._stop_base()
        if stage in {'extract', 'lift'}:
            self._maintain_gripper_hold()
        elapsed = self._elapsed()
        if elapsed is None:
            return
        duration = self.motion_durations[stage]
        error = self._joint_error()
        if error is not None and error <= self.joint_tolerance and elapsed >= max(0.0, duration - 0.20):
            self.stage_settle_count += 1
        else:
            self.stage_settle_count = 0
        if self.stage_settle_count >= self.joint_settle_required:
            self._complete_stage(stage, error)
            return
        if elapsed > duration + self.stage_timeout_margin:
            self._finish(False, f'{stage} arm trajectory did not settle; max error={error}')

    def _open_tick(self) -> None:
        self._stop_base()
        elapsed = self._elapsed()
        if elapsed is None:
            return
        position = self.gripper_actual_position
        if (
            position is not None
            and abs(position - self.gripper_open_position) <= 0.012
            and elapsed >= self.gripper_motion_sec - 0.10
        ):
            self._complete_stage('open', abs(position - self.gripper_open_position))
            return
        if elapsed > self.gripper_motion_sec + self.stage_timeout_margin:
            self._finish(False, f'gripper did not reach open target; actual={position}')

    def _close_tick(self) -> None:
        self._stop_base()
        elapsed = self._elapsed()
        if elapsed is None:
            return
        final_target = self.gripper_close_positions[-1]
        if elapsed >= self.gripper_motion_sec:
            if (
                self.last_hold_publish_sim is None
                or self.sim_time_sec - self.last_hold_publish_sim >= 0.20
            ):
                self.gripper_publishers[self.selected_arm].publish(
                    self._gripper_trajectory([final_target], 0.20))
                self.last_hold_publish_sim = self.sim_time_sec
        if elapsed >= self.gripper_motion_sec + self.gripper_hold_sec:
            position = self.gripper_actual_position
            if position is None or not 0.0 <= position <= 0.045:
                self._finish(False, f'invalid gripper position after close: {position}')
                return
            self._complete_stage('close', abs(position - final_target))
            return
        if elapsed > self.gripper_motion_sec + self.gripper_hold_sec + self.stage_timeout_margin:
            self._finish(False, 'gripper close/hold stage timed out')

    def _complete_stage(self, stage: str, error: Optional[float]) -> None:
        self._stop_base()
        record = {
            'stage': stage,
            'completed_sim_sec': self.sim_time_sec,
            'max_joint_or_gripper_error': error,
            'gripper_position_m': self.gripper_actual_position,
            'gripper_effort_state': self.gripper_actual_effort,
            'selected_fingertip_contact_messages': self.selected_fingertip_contact_messages,
            'front_scan_min_m': self.latest_scan_front_min,
            'front_lidar_robust_clearance_m': self.latest_lidar_robust,
            'source_pose_error_m': self.source_pose_error_m,
            'source_yaw_error_deg': (
                math.degrees(self.source_yaw_error_rad)
                if self.source_yaw_error_rad is not None else None),
        }
        self.stage_records.append(record)
        self.completed_stages.append(stage)
        self._save_stage_image(stage)
        if stage == self.target_stage:
            self._finish(True, f'approved staged grasp experiment reached {stage}')
            return
        next_index = STAGE_ORDER.index(stage) + 1
        if next_index >= len(STAGE_ORDER):
            self._finish(True, 'all staged grasp steps completed')
            return
        self._request_approval(STAGE_ORDER[next_index])

    def _save_stage_image(self, stage: str) -> None:
        if self.latest_frame_bgr is None:
            return
        frame = self.latest_frame_bgr.copy()
        geometry = self.source_result['target_book_geometry']
        bbox = geometry.get('bbox_xywh') or self.source_result.get('target_book_detection', {}).get('bbox_xywh')
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, width, height = (int(value) for value in bbox)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
        lines = [
            'KU SPARCy ERC 2026 - DAY 4 STAGED GRASP',
            f'stage: {stage} | arm: {self.selected_arm}',
            f'gripper position: {self.gripper_actual_position}',
            f'gripper effort state: {self.gripper_actual_effort}',
            f'fingertip contact messages: {self.selected_fingertip_contact_messages}',
            f'front LiDAR min/robust: {self.latest_scan_front_min}/{self.latest_lidar_robust}',
            f'sim time: {self.sim_time_sec}',
            f'UTC: {datetime.now(timezone.utc).isoformat()}',
        ]
        y_text = 22
        for line in lines:
            cv2.putText(frame, line, (8, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y_text += 18
        os.makedirs(self.image_output_dir, exist_ok=True)
        filename = (
            f'day4_grasp_{stage}_{self.selected_arm}_'
            f'sim_{(self.sim_time_sec or 0.0):012.3f}_'
            f'utc_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}.png')
        path = os.path.join(self.image_output_dir, filename)
        temporary = path + '.tmp.png'
        if cv2.imwrite(temporary, frame):
            os.replace(temporary, path)
            self.evidence_paths[stage] = path

    def _write_result(self, passed: bool, reason: str) -> None:
        effort_stats = None
        if self.gripper_effort_samples:
            values = np.asarray(self.gripper_effort_samples, dtype=np.float64)
            effort_stats = {
                'count': int(values.size),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'minimum': float(np.min(values)),
                'maximum': float(np.max(values)),
            }
        result = {
            'day': 4,
            'experiment': 'staged_position_controlled_grasp',
            'passed': bool(passed),
            'reason': reason,
            'source_result_path': self.source_result_path,
            'official_upstream_required_commit': '93554d4f9335b2ee3acb49c6b332611f6ad2a964',
            'target_stage': self.target_stage,
            'state': self.state,
            'manual_approval_required': self.manual_approval_required,
            'selected_arm': self.selected_arm,
            'single_arm_rule_respected': True,
            'torso_commanded': False,
            'effort_command_interface_added': False,
            'effort_commanded': False,
            'gripper_command_mode': 'position',
            'gripper_public_topic': (
                f'/gripper_{self.selected_arm}_controller/joint_trajectory'
                if self.selected_arm else None),
            'arm_commanded': self.arm_commanded,
            'gripper_commanded': self.gripper_commanded,
            'completed_stages': list(self.completed_stages),
            'stage_records': list(self.stage_records),
            'grasp_plan': self.plan.as_dict() if self.plan is not None else None,
            'gripper_actual_position_m': self.gripper_actual_position,
            'gripper_actual_effort_state': self.gripper_actual_effort,
            'gripper_effort_state_statistics': effort_stats,
            'contact_messages_total': self.contact_messages_total,
            'selected_fingertip_contact_messages': self.selected_fingertip_contact_messages,
            'premature_selected_fingertip_contacts': self.premature_selected_fingertip_contacts,
            'unintended_selected_arm_contacts': self.unintended_selected_arm_contacts,
            'contact_pairs': list(self.contact_pairs),
            'front_scan_min_m': self.latest_scan_front_min,
            'front_lidar_robust_clearance_m': self.latest_lidar_robust,
            'lidar_transform_failures': self.lidar_transform_failures,
            'source_pose_error_m': self.source_pose_error_m,
            'source_yaw_error_deg': (
                math.degrees(self.source_yaw_error_rad)
                if self.source_yaw_error_rad is not None else None),
            'evidence_paths': dict(self.evidence_paths),
            'simulation_time_sec': self.sim_time_sec,
            'wall_duration_sec': time.monotonic() - self.started_wall,
            'book_retention_automatic_verification': 'not_yet_reliable; manual visual gate required on Day 4',
        }
        os.makedirs(os.path.dirname(self.grasp_result_path) or '.', exist_ok=True)
        temporary = self.grasp_result_path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, self.grasp_result_path)

    def _finish(self, passed: bool, reason: str) -> None:
        if self.done:
            return
        self._stop_base()
        self.done = True
        self.passed = bool(passed)
        self.reason = reason
        self.state = 'DONE' if passed else 'FAILED'
        self._write_result(passed, reason)
        label = 'PASSED' if passed else 'FAILED'
        self.get_logger().info(f'[DAY4-GRASP][{label}] {reason}')

    def _tick(self) -> None:
        if self.done:
            return
        self._stop_base()
        if time.monotonic() - self.last_clock_wall > self.clock_stall_wall_timeout:
            self._finish(False, 'Gazebo /clock stopped advancing beyond wall-time watchdog')
            return
        if (
            self.latest_lidar is not None
            and self.latest_lidar.minimum_clearance_m
            < self.front_clearance_abort
            and self.latest_lidar.hard_cluster_point_count
            >= self.lidar_hard_cluster_points
        ):
            self._finish(
                False,
                'front LiDAR hard cluster fell below abort threshold: '
                f'min={self.latest_lidar.minimum_clearance_m:.3f} m '
                f'points={self.latest_lidar.hard_cluster_point_count}',
            )
            return
        if self.state == 'WAIT_INPUTS':
            if self._inputs_ready():
                self._transition('PLAN', 'live joint states, RGB, scan and simulation clock are ready')
            elif time.monotonic() - self.started_wall > self.ready_wall_timeout:
                self._finish(False, 'readiness watchdog expired')
            return
        if self.state == 'PLAN':
            self._make_plan()
        elif self.state == 'WAIT_APPROVAL':
            self._approval_tick()
        elif self.state == 'MOVE_PREGRASP':
            self._arm_motion_tick('pregrasp')
        elif self.state == 'OPEN_GRIPPER':
            self._open_tick()
        elif self.state == 'MOVE_NO_CONTACT':
            self._arm_motion_tick('no_contact')
        elif self.state == 'MOVE_CONTACT_POSE':
            self._arm_motion_tick('contact_pose')
        elif self.state == 'CLOSE_GRIPPER':
            self._close_tick()
        elif self.state == 'MOVE_EXTRACT':
            self._arm_motion_tick('extract')
        elif self.state == 'MOVE_LIFT':
            self._arm_motion_tick('lift')
        else:
            self._finish(False, f'unknown state {self.state}')


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[StagedGraspExperiment] = None
    code = 1
    try:
        node = StagedGraspExperiment()
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
            print(f'[DAY4-GRASP][FAILED] {exc}')
        code = 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
