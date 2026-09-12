#!/usr/bin/env python3
"""Day 6: carry the retained book safely back into the recorded start zone.

The selected arm remains in the validated Day 5 lift pose.  The planner treats
its held 0.25 x 0.02 x 0.16 m book as part of the swept radius before any base
rotation, retreats straight away from the shelf, then returns to a live odometry
pose recorded before the mission moved.  No world/model pose is read.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros

try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:  # pragma: no cover
    Contacts = None  # type: ignore

from ku_sparcy_erc.day567_common import (
    any_token,
    atomic_write_json,
    book_half_diagonal_m,
    clamp,
    contact_collision_names,
    directional_lidar_clearance,
    limit_planar,
    load_passed_json,
    meaningful_robot_collision,
    normalize_angle,
    selected_fingertip_tokens,
    vector_odom_to_base,
    yaw_from_odom,
)
from ku_sparcy_erc.range_fusion import LidarClearance


class Day6ReturnHome(Node):
    FRONT_SCAN_TOPIC = '/scan_front_raw'
    REAR_SCAN_TOPIC = '/scan_rear_raw'

    def __init__(self) -> None:
        super().__init__('ku_sparcy_day6_return_home')
        self.declare_parameter(
            'day4_result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day4_result.json')
        self.declare_parameter(
            'day5_result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day5_result.json')
        self.declare_parameter(
            'home_pose_path',
            '/opt/erc_ws/src/ku_sparcy_erc/home_pose.json')
        self.declare_parameter(
            'result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day6_result.json')
        self.declare_parameter('ready_wall_timeout_sec', 45.0)
        self.declare_parameter('clock_stall_wall_timeout_sec', 20.0)
        self.declare_parameter('retreat_distance_m', 0.55)
        self.declare_parameter('retreat_speed_mps', 0.16)
        self.declare_parameter('home_stop_offset_m', 0.28)
        self.declare_parameter('home_position_tolerance_m', 0.10)
        self.declare_parameter('home_settle_samples', 5)
        self.declare_parameter('turn_yaw_tolerance_deg', 2.0)
        self.declare_parameter('turn_settle_samples', 4)
        self.declare_parameter('max_turn_speed_radps', 0.32)
        self.declare_parameter('turn_kp', 1.2)
        self.declare_parameter('max_forward_speed_mps', 0.28)
        self.declare_parameter('max_lateral_speed_mps', 0.14)
        self.declare_parameter('max_planar_speed_mps', 0.30)
        self.declare_parameter('drive_position_kp', 0.55)
        self.declare_parameter('drive_yaw_kp', 0.90)
        self.declare_parameter('drive_yaw_tolerance_deg', 6.0)
        self.declare_parameter('rear_hard_stop_m', 0.55)
        self.declare_parameter('front_hard_stop_m', 0.55)
        self.declare_parameter('scan_stale_timeout_sec', 0.60)
        self.declare_parameter('lidar_corridor_half_width_m', 0.42)
        self.declare_parameter('lidar_robust_percentile', 10.0)
        self.declare_parameter('lidar_hard_cluster_points', 3)
        self.declare_parameter('rotation_clearance_margin_m', 0.20)
        self.declare_parameter('gripper_hold_position_m', 0.010)
        self.declare_parameter('gripper_hold_period_sec', 0.20)
        self.declare_parameter('retention_contact_stale_sec', 0.80)
        self.declare_parameter('source_pose_tolerance_m', 0.05)
        self.declare_parameter('source_yaw_tolerance_deg', 3.0)
        self.declare_parameter('retreat_timeout_sec', 10.0)
        self.declare_parameter('turn_timeout_sec', 12.0)
        self.declare_parameter('return_timeout_sec', 25.0)
        get = self.get_parameter
        self.day4_path = str(get('day4_result_path').value)
        self.day5_path = str(get('day5_result_path').value)
        self.home_path = str(get('home_pose_path').value)
        self.result_path = str(get('result_path').value)
        self.ready_wall_timeout = float(get('ready_wall_timeout_sec').value)
        self.clock_stall_wall_timeout = float(
            get('clock_stall_wall_timeout_sec').value)
        self.retreat_distance = float(get('retreat_distance_m').value)
        self.retreat_speed = float(get('retreat_speed_mps').value)
        self.home_stop_offset = float(get('home_stop_offset_m').value)
        self.home_tolerance = float(get('home_position_tolerance_m').value)
        self.home_settle_required = int(get('home_settle_samples').value)
        self.turn_yaw_tolerance = math.radians(float(
            get('turn_yaw_tolerance_deg').value))
        self.turn_settle_required = int(get('turn_settle_samples').value)
        self.max_turn_speed = float(get('max_turn_speed_radps').value)
        self.turn_kp = float(get('turn_kp').value)
        self.max_forward_speed = float(get('max_forward_speed_mps').value)
        self.max_lateral_speed = float(get('max_lateral_speed_mps').value)
        self.max_planar_speed = float(get('max_planar_speed_mps').value)
        self.drive_position_kp = float(get('drive_position_kp').value)
        self.drive_yaw_kp = float(get('drive_yaw_kp').value)
        self.drive_yaw_tolerance = math.radians(float(
            get('drive_yaw_tolerance_deg').value))
        self.rear_hard_stop = float(get('rear_hard_stop_m').value)
        self.front_hard_stop = float(get('front_hard_stop_m').value)
        self.scan_stale_timeout = float(get('scan_stale_timeout_sec').value)
        self.corridor_half_width = float(
            get('lidar_corridor_half_width_m').value)
        self.lidar_percentile = float(
            get('lidar_robust_percentile').value)
        self.hard_cluster_points = int(
            get('lidar_hard_cluster_points').value)
        self.rotation_margin = float(get('rotation_clearance_margin_m').value)
        self.gripper_hold_position = float(
            get('gripper_hold_position_m').value)
        self.gripper_hold_period = float(
            get('gripper_hold_period_sec').value)
        self.retention_stale = float(
            get('retention_contact_stale_sec').value)
        self.source_pose_tolerance = float(
            get('source_pose_tolerance_m').value)
        self.source_yaw_tolerance = math.radians(float(
            get('source_yaw_tolerance_deg').value))
        self.timeouts = {
            'RETREAT_FROM_SHELF': float(get('retreat_timeout_sec').value),
            'TURN_TOWARD_HOME': float(get('turn_timeout_sec').value),
            'DRIVE_TO_HOME': float(get('return_timeout_sec').value),
        }
        self._validate_parameters()

        self.day4 = load_passed_json(self.day4_path, label='Day 4 result')
        self.day5 = load_passed_json(self.day5_path, label='Day 5 result')
        self.home = load_passed_json(self.home_path, label='home pose')
        if self.day5.get('retention_verified') is not True:
            raise ValueError('Day 5 result did not verify retained book')
        self.selected_arm = str(self.day5.get('selected_arm') or '').lower()
        if self.selected_arm not in {'left', 'right'}:
            raise ValueError('Day 5 result has no selected arm')
        self.home_xy = tuple(float(v) for v in self.home['home_odom_xy'])
        self.home_yaw = float(self.home['home_yaw_rad'])
        self.source_xy = tuple(
            float(v) for v in self.day4['day4_final_base_odom_xy'])
        self.source_yaw = float(self.day4['day4_final_base_yaw_rad'])
        self.lift_joint_names, self.lift_joint_positions = (
            self._lift_joint_target_from_result())
        lift_xyz = self.day5.get('grasp_plan', {}).get('lift_xyz_m')
        if not isinstance(lift_xyz, list) or len(lift_xyz) != 3:
            raise ValueError('Day 5 result has no lift_xyz_m')
        self.lift_xyz = tuple(float(v) for v in lift_xyz)
        half_diagonal = book_half_diagonal_m()
        self.carried_book_radius = (
            math.hypot(self.lift_xyz[0], self.lift_xyz[1])
            + half_diagonal)
        self.carried_book_forward_extent = self.lift_xyz[0] + half_diagonal

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
        self.joint_positions: Dict[str, float] = {}
        self.front_scan: Optional[LidarClearance] = None
        self.rear_scan: Optional[LidarClearance] = None
        self.front_scan_stamp: Optional[float] = None
        self.rear_scan_stamp: Optional[float] = None
        self.scan_frames = {'front': 0, 'rear': 0}
        self.retreat_start_xy: Optional[Tuple[float, float]] = None
        self.return_heading_yaw: Optional[float] = None
        self.return_target_xy: Optional[Tuple[float, float]] = None
        self.turn_settle_count = 0
        self.home_settle_count = 0
        self.last_gripper_hold_sim: Optional[float] = None
        self.last_fingertip_contact_sim: Optional[float] = None
        self.fingertip_contact_messages = 0
        self.unintended_robot_contacts = 0
        self.minimum_front_clearance: Optional[float] = None
        self.minimum_rear_clearance: Optional[float] = None
        self.maximum_abs_command = {'vx': 0.0, 'vy': 0.0, 'wz': 0.0}
        self.path: List[Dict[str, object]] = []
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)
        self.clock_sub = self.create_subscription(
            Clock, '/clock', self._clock_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile_sensor_data)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 20)
        self.front_sub = self.create_subscription(
            LaserScan, self.FRONT_SCAN_TOPIC,
            lambda msg: self._scan_callback(msg, 'front'),
            qos_profile_sensor_data)
        self.rear_sub = self.create_subscription(
            LaserScan, self.REAR_SCAN_TOPIC,
            lambda msg: self._scan_callback(msg, 'rear'),
            qos_profile_sensor_data)
        if Contacts is None:
            raise RuntimeError('ros_gz_interfaces/Contacts is required')
        self.contact_sub = self.create_subscription(
            Contacts, '/contacts', self._contact_callback,
            qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            f'/gripper_{self.selected_arm}_controller/joint_trajectory',
            10)
        self.timer = self.create_timer(0.05, self._tick)
        self.get_logger().info(
            '[DAY6] retained-book return ready; arm=%s radius=%.3f m '
            'forward_extent=%.3f m' % (
                self.selected_arm,
                self.carried_book_radius,
                self.carried_book_forward_extent,
            ))

    def _validate_parameters(self) -> None:
        if not 0.30 <= self.retreat_distance <= 0.90:
            raise ValueError('retreat_distance_m outside conservative range')
        if not 0.05 <= self.retreat_speed <= 0.25:
            raise ValueError('retreat_speed_mps outside conservative range')
        if not 0.0 < self.home_stop_offset < 0.38:
            raise ValueError('home_stop_offset_m must remain inside start zone')
        if self.home_stop_offset + self.home_tolerance > 0.40:
            raise ValueError(
                'home stop offset plus tolerance exceeds 0.40 m '
                'start-zone acceptance radius')
        if not 0.0 <= self.gripper_hold_position <= 0.069:
            raise ValueError('gripper hold position outside public clamp')
        if self.corridor_half_width <= 0.25:
            raise ValueError('LiDAR corridor too narrow')

    def _lift_joint_target_from_result(self) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
        plan = self.day5.get('grasp_plan')
        if not isinstance(plan, dict):
            raise ValueError('Day 5 result has no grasp_plan')
        candidates = plan.get('candidate_plans')
        if not isinstance(candidates, dict):
            raise ValueError('Day 5 grasp plan has no candidates')
        selected = candidates.get(self.selected_arm)
        if not isinstance(selected, dict):
            raise ValueError('selected arm plan missing')
        lift = selected.get('lift')
        if not isinstance(lift, dict) or lift.get('success') is not True:
            raise ValueError('selected arm lift IK did not pass')
        names = selected.get('joint_names')
        positions = lift.get('positions_rad')
        if not isinstance(names, list) or not isinstance(positions, list):
            raise ValueError('selected lift joint target missing')
        if len(names) != 7 or len(positions) != 7:
            raise ValueError('selected lift target must contain seven joints')
        return tuple(str(v) for v in names), tuple(float(v) for v in positions)

    def _clock_callback(self, msg: Clock) -> None:
        value = float(msg.clock.sec) + float(msg.clock.nanosec) * 1.0e-9
        if self.sim_time_sec is None or value > self.sim_time_sec + 1.0e-9:
            self.last_clock_wall = time.monotonic()
        self.sim_time_sec = value

    def _odom_callback(self, msg: Odometry) -> None:
        self.current_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        self.current_yaw = float(yaw_from_odom(msg))

    def _joint_callback(self, msg: JointState) -> None:
        for index, name in enumerate(msg.name):
            if index < len(msg.position):
                self.joint_positions[name] = float(msg.position[index])

    def _lookup_transform_components(
        self,
        source_frame: str,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint', source_frame, Time())
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

    def _scan_callback(self, msg: LaserScan, side: str) -> None:
        transform = self._lookup_transform_components(msg.header.frame_id)
        if transform is None:
            return
        try:
            clearance = directional_lidar_clearance(
                msg.ranges,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                transform[0],
                transform[1],
                direction_sign=(1 if side == 'front' else -1),
                corridor_half_width_m=self.corridor_half_width,
                robust_percentile=self.lidar_percentile,
                hard_stop_m=(
                    self.front_hard_stop if side == 'front'
                    else self.rear_hard_stop),
            )
        except ValueError:
            return
        if clearance is None:
            return
        if side == 'front':
            self.front_scan = clearance
            self.front_scan_stamp = self.sim_time_sec
            value = clearance.minimum_clearance_m
            self.minimum_front_clearance = (
                value if self.minimum_front_clearance is None
                else min(self.minimum_front_clearance, value))
        else:
            self.rear_scan = clearance
            self.rear_scan_stamp = self.sim_time_sec
            value = clearance.minimum_clearance_m
            self.minimum_rear_clearance = (
                value if self.minimum_rear_clearance is None
                else min(self.minimum_rear_clearance, value))
        self.scan_frames[side] += 1

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
                    'unintended robot contact during carried-book return: '
                    + ', '.join(unexpected[:4]))
                return

    def _scan_fresh(self, stamp: Optional[float]) -> bool:
        return (
            stamp is not None
            and self.sim_time_sec is not None
            and self.sim_time_sec - stamp <= self.scan_stale_timeout)

    def _source_pose_ok(self) -> bool:
        if self.current_xy is None or self.current_yaw is None:
            return False
        error = math.hypot(
            self.current_xy[0] - self.source_xy[0],
            self.current_xy[1] - self.source_xy[1])
        yaw_error = abs(normalize_angle(self.current_yaw - self.source_yaw))
        return (
            error <= self.source_pose_tolerance
            and yaw_error <= self.source_yaw_tolerance)

    def _arm_at_lift_pose(self) -> bool:
        if any(name not in self.joint_positions for name in self.lift_joint_names):
            return False
        error = max(
            abs(self.joint_positions[name] - target)
            for name, target in zip(
                self.lift_joint_names, self.lift_joint_positions))
        return error <= 0.06

    def _inputs_ready(self) -> bool:
        return (
            self.sim_time_sec is not None
            and self.current_xy is not None
            and self.current_yaw is not None
            and self.front_scan is not None
            and self.rear_scan is not None
            and self._scan_fresh(self.front_scan_stamp)
            and self._scan_fresh(self.rear_scan_stamp)
            and self._arm_at_lift_pose()
            and self.gripper_pub.get_subscription_count() > 0
        )

    def _publish_velocity(self, vx: float, vy: float, wz: float) -> None:
        message = Twist()
        message.linear.x = float(vx)
        message.linear.y = float(vy)
        message.angular.z = float(wz)
        self.cmd_pub.publish(message)
        self.maximum_abs_command['vx'] = max(
            self.maximum_abs_command['vx'], abs(float(vx)))
        self.maximum_abs_command['vy'] = max(
            self.maximum_abs_command['vy'], abs(float(vy)))
        self.maximum_abs_command['wz'] = max(
            self.maximum_abs_command['wz'], abs(float(wz)))

    def _stop_base(self) -> None:
        for _ in range(3):
            self._publish_velocity(0.0, 0.0, 0.0)

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

    def _transition(self, state: str, reason: str) -> None:
        self.state = state
        self.state_started_sim = self.sim_time_sec
        self.get_logger().info(f'[DAY6] STATE -> {state}: {reason}')
        self._write_result(False, f'running: {reason}')

    def _state_elapsed(self) -> Optional[float]:
        if self.sim_time_sec is None or self.state_started_sim is None:
            return None
        return max(0.0, self.sim_time_sec - self.state_started_sim)

    def _record_path(self) -> None:
        if self.current_xy is None or self.current_yaw is None:
            return
        if self.path and self.sim_time_sec is not None:
            previous = self.path[-1].get('sim_time_sec')
            if isinstance(previous, (int, float)) and self.sim_time_sec - previous < 0.25:
                return
        self.path.append({
            'sim_time_sec': self.sim_time_sec,
            'state': self.state,
            'odom_xy': [self.current_xy[0], self.current_xy[1]],
            'yaw_rad': self.current_yaw,
            'front_min_m': (
                self.front_scan.minimum_clearance_m
                if self.front_scan is not None else None),
            'rear_min_m': (
                self.rear_scan.minimum_clearance_m
                if self.rear_scan is not None else None),
            'retention_contact_age_sec': (
                self.sim_time_sec - self.last_fingertip_contact_sim
                if self.sim_time_sec is not None
                and self.last_fingertip_contact_sim is not None else None),
        })
        if len(self.path) > 500:
            del self.path[:-500]

    def _hard_stop(self, scan: Optional[LidarClearance], threshold: float) -> bool:
        return (
            scan is not None
            and scan.minimum_clearance_m <= threshold
            and scan.hard_cluster_point_count >= self.hard_cluster_points)

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
                if not self._source_pose_ok():
                    self._finish(False, 'base pose differs from Day 4 source geometry')
                    return
                self.retreat_start_xy = self.current_xy
                self._transition(
                    'RETREAT_FROM_SHELF',
                    'retained book and live safety interfaces verified; '
                    'retreating straight away from shelf before rotation')
            elif time.monotonic() - self.started_wall > self.ready_wall_timeout:
                self._finish(False, 'Day 6 readiness timed out')
            return

        elapsed = self._state_elapsed()
        timeout = self.timeouts.get(self.state)
        if timeout is not None and elapsed is not None and elapsed > timeout:
            self._stop_base()
            self._finish(False, f'{self.state} exceeded simulation-time timeout')
            return

        if self.state == 'RETREAT_FROM_SHELF':
            assert self.retreat_start_xy is not None
            assert self.current_xy is not None
            assert self.current_yaw is not None
            if not self._scan_fresh(self.rear_scan_stamp):
                self._stop_base()
                return
            if self._hard_stop(self.rear_scan, self.rear_hard_stop):
                self._stop_base()
                self._finish(False, 'rear LiDAR hard stop during shelf retreat')
                return
            travelled = math.hypot(
                self.current_xy[0] - self.retreat_start_xy[0],
                self.current_xy[1] - self.retreat_start_xy[1])
            if travelled >= self.retreat_distance:
                self._stop_base()
                if self.front_scan is None:
                    return
                required = self.carried_book_radius + self.rotation_margin
                if self.front_scan.robust_clearance_m < required:
                    self._finish(
                        False,
                        'insufficient carried-book swept-radius clearance before '
                        f'rotation: have={self.front_scan.robust_clearance_m:.3f} '
                        f'require={required:.3f}')
                    return
                dx = self.home_xy[0] - self.current_xy[0]
                dy = self.home_xy[1] - self.current_xy[1]
                distance = math.hypot(dx, dy)
                if distance <= 1.0e-6:
                    self.return_heading_yaw = self.current_yaw
                    self.return_target_xy = self.current_xy
                else:
                    ux, uy = dx / distance, dy / distance
                    self.return_heading_yaw = math.atan2(uy, ux)
                    self.return_target_xy = (
                        self.home_xy[0] - ux * self.home_stop_offset,
                        self.home_xy[1] - uy * self.home_stop_offset,
                    )
                self.turn_settle_count = 0
                self._transition(
                    'TURN_TOWARD_HOME',
                    'shelf retreat complete and held-book swept-radius clearance '
                    'verified; rotating toward recorded home pose')
                return
            yaw_error = normalize_angle(self.source_yaw - self.current_yaw)
            self._publish_velocity(
                -self.retreat_speed,
                0.0,
                clamp(0.8 * yaw_error, -0.16, 0.16))
            return

        if self.state == 'TURN_TOWARD_HOME':
            assert self.current_yaw is not None
            assert self.return_heading_yaw is not None
            self._stop_base()
            yaw_error = normalize_angle(
                self.return_heading_yaw - self.current_yaw)
            if abs(yaw_error) <= self.turn_yaw_tolerance:
                self.turn_settle_count += 1
            else:
                self.turn_settle_count = 0
            if self.turn_settle_count >= self.turn_settle_required:
                self._stop_base()
                self.home_settle_count = 0
                self._transition(
                    'DRIVE_TO_HOME',
                    'home heading settled; driving to a safe point inside the '
                    'recorded start zone')
                return
            self._publish_velocity(
                0.0,
                0.0,
                clamp(
                    self.turn_kp * yaw_error,
                    -self.max_turn_speed,
                    self.max_turn_speed))
            return

        if self.state == 'DRIVE_TO_HOME':
            assert self.current_xy is not None
            assert self.current_yaw is not None
            assert self.return_target_xy is not None
            if not self._scan_fresh(self.front_scan_stamp):
                self._stop_base()
                return
            if self._hard_stop(self.front_scan, self.front_hard_stop):
                self._stop_base()
                self._finish(False, 'front LiDAR hard stop while returning home')
                return
            forward, left = vector_odom_to_base(
                self.return_target_xy,
                self.current_xy,
                self.current_yaw)
            distance = math.hypot(forward, left)
            heading = math.atan2(left, max(1.0e-6, forward))
            if distance <= self.home_tolerance:
                self._stop_base()
                self.home_settle_count += 1
                if self.home_settle_count >= self.home_settle_required:
                    self._transition(
                        'VERIFY_HOME',
                        'recorded start-zone target reached with retained book')
                return
            self.home_settle_count = 0
            vx = clamp(
                self.drive_position_kp * forward,
                0.06,
                self.max_forward_speed)
            vy = clamp(
                self.drive_position_kp * left,
                -self.max_lateral_speed,
                self.max_lateral_speed)
            vx, vy = limit_planar(vx, vy, self.max_planar_speed)
            wz = 0.0 if abs(heading) <= self.drive_yaw_tolerance else clamp(
                self.drive_yaw_kp * heading,
                -0.24,
                0.24)
            self._publish_velocity(vx, vy, wz)
            return

        if self.state == 'VERIFY_HOME':
            self._stop_base()
            if self.current_xy is None or self.return_target_xy is None:
                return
            error = math.hypot(
                self.current_xy[0] - self.return_target_xy[0],
                self.current_xy[1] - self.return_target_xy[1])
            if error > self.home_tolerance:
                self._finish(False, 'robot drifted outside home tolerance')
                return
            if not self._retention_ok():
                self._finish(False, 'book was not retained at home')
                return
            self._finish(
                True,
                'retained target book returned to recorded Start/End Zone '
                'with carried-object rotation clearance and LiDAR safety')
            return

        self._finish(False, f'unknown Day 6 state: {self.state}')

    def _write_result(self, passed: bool, reason: str) -> None:
        distance_to_home = None
        if self.current_xy is not None:
            distance_to_home = math.hypot(
                self.current_xy[0] - self.home_xy[0],
                self.current_xy[1] - self.home_xy[1])
        result = {
            'day': 6,
            'passed': bool(passed),
            'reason': reason,
            'state': self.state,
            'selected_arm': self.selected_arm,
            'day4_result_path': self.day4_path,
            'day5_result_path': self.day5_path,
            'home_pose_path': self.home_path,
            'home_odom_xy': list(self.home_xy),
            'home_yaw_rad': self.home_yaw,
            'return_target_odom_xy': (
                list(self.return_target_xy)
                if self.return_target_xy is not None else None),
            'final_odom_xy': (
                list(self.current_xy) if self.current_xy is not None else None),
            'final_yaw_rad': self.current_yaw,
            'distance_to_home_center_m': distance_to_home,
            'retreat_distance_m': self.retreat_distance,
            'home_stop_offset_m': self.home_stop_offset,
            'carried_book_dimensions_m': [0.25, 0.02, 0.16],
            'carried_book_half_diagonal_m': book_half_diagonal_m(),
            'carried_book_swept_radius_m': self.carried_book_radius,
            'carried_book_forward_extent_m': self.carried_book_forward_extent,
            'held_book_geometry_in_safety_reasoning': True,
            'gripper_public_topic': (
                f'/gripper_{self.selected_arm}_controller/joint_trajectory'),
            'gripper_command_mode': 'position',
            'effort_commanded': False,
            'torso_commanded': False,
            'fingertip_contact_messages': self.fingertip_contact_messages,
            'last_fingertip_contact_sim_sec': self.last_fingertip_contact_sim,
            'unintended_robot_contacts': self.unintended_robot_contacts,
            'minimum_front_clearance_m': self.minimum_front_clearance,
            'minimum_rear_clearance_m': self.minimum_rear_clearance,
            'front_scan_frames': self.scan_frames['front'],
            'rear_scan_frames': self.scan_frames['rear'],
            'maximum_abs_command': dict(self.maximum_abs_command),
            'path_trace': list(self.path),
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
        self.get_logger().info(f'[DAY6][{label}] {reason}')


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[Day6ReturnHome] = None
    code = 1
    try:
        node = Day6ReturnHome()
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
            print(f'[DAY6][FAILED] {exc}')
        code = 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
