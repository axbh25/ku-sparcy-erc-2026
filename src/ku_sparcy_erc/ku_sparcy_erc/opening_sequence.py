#!/usr/bin/env python3
"""Day 1 closed-loop opening sequence for KU SPARCy ERC 2026.

This node deliberately implements only the first validated slice of the final
mission: wait for odometry, live RGB frames, and the head controller; tilt the
head upward; then rotate the base clockwise by 90 degrees using odometry
feedback. Later days will replace/extend this with continuous shelf-marker
perception and the complete mission state machine.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw from a geometry_msgs quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class OpeningSequence(Node):
    """Rotate toward the shelves while proving camera and head availability."""

    VALID_COLOURS = {'red', 'green', 'yellow', 'blue'}

    def __init__(self) -> None:
        super().__init__('opening_sequence')

        self.declare_parameter('shelf_column_number', 1)
        self.declare_parameter('book_colour', 'red')
        self.declare_parameter('phase1_fast_start', True)
        self.declare_parameter('enable_motion', True)
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
        self.declare_parameter(
            'result_path', '/opt/erc_ws/src/ku_sparcy_erc/day1_result.json')

        self.shelf_column = int(
            self.get_parameter('shelf_column_number').value)
        self.book_colour = str(
            self.get_parameter('book_colour').value).strip().lower()
        self.phase1_fast_start = bool(
            self.get_parameter('phase1_fast_start').value)
        self.enable_motion = bool(
            self.get_parameter('enable_motion').value)
        self.rotation_deg = float(
            self.get_parameter('clockwise_rotation_deg').value)
        self.max_speed = float(
            self.get_parameter('max_angular_speed').value)
        self.min_speed = float(
            self.get_parameter('min_angular_speed').value)
        self.yaw_kp = float(self.get_parameter('yaw_kp').value)
        self.tolerance_rad = math.radians(float(
            self.get_parameter('yaw_tolerance_deg').value))
        self.settle_samples_required = int(
            self.get_parameter('settle_samples').value)
        self.ready_timeout_sec = float(
            self.get_parameter('ready_timeout_sec').value)
        self.motion_timeout_sec = float(
            self.get_parameter('motion_timeout_sec').value)
        self.head_pan = float(self.get_parameter('head_pan_rad').value)
        self.head_tilt = float(self.get_parameter('head_tilt_rad').value)
        self.head_motion_sec = float(
            self.get_parameter('head_motion_sec').value)
        self.head_tilt_tolerance = float(
            self.get_parameter('head_tilt_tolerance_rad').value)
        self.result_path = str(self.get_parameter('result_path').value)

        self._validate_parameters()

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.head_pub = self.create_publisher(
            JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.status_pub = self.create_publisher(
            String, '/ku_sparcy/opening_status', status_qos)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback,
            qos_profile_sensor_data)

        self.sim_time_sec: Optional[float] = None
        self.clock_sub = self.create_subscription(
            Clock,
            '/clock',
            self._clock_callback,
            qos_profile_sensor_data,
        )

        self.node_started_at = time.monotonic()
        self.motion_started_at: Optional[float] = None
        self.motion_started_sim_sec: Optional[float] = None
        self.current_yaw: Optional[float] = None
        self.initial_yaw: Optional[float] = None
        self.target_yaw: Optional[float] = None
        self.last_image_stamp_ns: Optional[int] = None
        self.actual_head_tilt: Optional[float] = None
        self.camera_frames = 0
        self.frames_at_motion_start = 0
        self.settle_samples = 0
        self.head_command_sent = False
        self.state = 'WAITING_FOR_INPUTS'
        self.done = False
        self.passed = False
        self.failure_reason = ''

        self.timer = self.create_timer(0.05, self._tick)  # 20 Hz
        self._publish_status('WAITING', 'Waiting for odometry, RGB camera, and head controller.')
        self.get_logger().info(
            '[DAY1] Requested marker=%d, colour=%s, fast_start=%s, motion=%s'
            % (
                self.shelf_column,
                self.book_colour,
                self.phase1_fast_start,
                self.enable_motion,
            )
        )

    def _validate_parameters(self) -> None:
        if not 1 <= self.shelf_column <= 5:
            raise ValueError('shelf_column_number must be an integer from 1 to 5')
        if self.book_colour not in self.VALID_COLOURS:
            raise ValueError(
                'book_colour must be one of: blue, green, red, yellow')
        if not self.phase1_fast_start:
            raise ValueError(
                'Day 1 implements only phase1_fast_start:=true; the general '
                'orientation detector is scheduled for Day 2')
        if self.rotation_deg <= 0.0 or self.rotation_deg > 180.0:
            raise ValueError('clockwise_rotation_deg must be in (0, 180]')
        if not 0.0 < self.min_speed <= self.max_speed:
            raise ValueError(
                'angular speeds must satisfy 0 < min_angular_speed <= max_angular_speed')
        if self.yaw_kp <= 0.0:
            raise ValueError('yaw_kp must be positive')
        if self.tolerance_rad <= 0.0:
            raise ValueError('yaw_tolerance_deg must be positive')
        if self.settle_samples_required < 1:
            raise ValueError('settle_samples must be at least 1')
        if self.head_tilt_tolerance <= 0.0:
            raise ValueError('head_tilt_tolerance_rad must be positive')

    def _clock_callback(self, msg: Clock) -> None:
        self.sim_time_sec = (
            float(msg.clock.sec)
            + float(msg.clock.nanosec) * 1.0e-9
        )

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self.current_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def _image_callback(self, msg: Image) -> None:
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return
        self.camera_frames += 1
        self.last_image_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        try:
            index = msg.name.index('head_2_joint')
        except ValueError:
            return
        if index < len(msg.position):
            self.actual_head_tilt = float(msg.position[index])

    def _head_controller_is_ready(self) -> bool:
        return self.head_pub.get_subscription_count() > 0

    def _send_head_command(self) -> None:
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [self.head_pan, self.head_tilt]
        whole_seconds = int(self.head_motion_sec)
        nanoseconds = int((self.head_motion_sec - whole_seconds) * 1e9)
        point.time_from_start.sec = whole_seconds
        point.time_from_start.nanosec = nanoseconds
        msg.points = [point]
        self.head_pub.publish(msg)
        self.head_command_sent = True
        self.get_logger().info(
            '[DAY1] Head command published: pan=%.3f rad, tilt=%.3f rad'
            % (self.head_pan, self.head_tilt)
        )

    def _publish_base_speed(self, angular_z: float) -> None:
        cmd = Twist()
        cmd.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd)

    def stop_base(self) -> None:
        # Send several zero commands so Gazebo does not retain the last velocity.
        for _ in range(3):
            self._publish_base_speed(0.0)

    def _publish_status(self, state: str, detail: str) -> None:
        payload = {
            'state': state,
            'detail': detail,
            'shelf_column_number': self.shelf_column,
            'book_colour': self.book_colour,
        }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(msg)

    def _tick(self) -> None:
        if self.done:
            return

        now = time.monotonic()

        if self.state == 'WAITING_FOR_INPUTS':
            ready = (
                self.current_yaw is not None
                and self.camera_frames >= 1
                and self.sim_time_sec is not None
                and self._head_controller_is_ready()
            )
            if ready:
                self.initial_yaw = self.current_yaw
                # ROS positive angular.z is counter-clockwise, so clockwise is negative.
                self.target_yaw = normalize_angle(
                    self.initial_yaw - math.radians(self.rotation_deg))
                self.frames_at_motion_start = self.camera_frames
                self.motion_started_at = now
                self.motion_started_sim_sec = self.sim_time_sec
                self._send_head_command()
                self.state = 'ROTATING'
                self._publish_status(
                    'ROTATING',
                    'Clockwise odometry-controlled Phase 1 opening turn started.',
                )
                self.get_logger().info(
                    '[DAY1] Clockwise turn started: initial=%.2f deg target=%.2f deg'
                    % (
                        math.degrees(self.initial_yaw),
                        math.degrees(self.target_yaw),
                    )
                )
                return

            if now - self.node_started_at > self.ready_timeout_sec:
                missing = []
                if self.current_yaw is None:
                    missing.append('/odom')
                if self.camera_frames < 1:
                    missing.append('RGB camera frame')
                if self.sim_time_sec is None:
                    missing.append('/clock')
                if not self._head_controller_is_ready():
                    missing.append('head controller subscriber')
                self._finish(False, 'Readiness timeout; missing: ' + ', '.join(missing))
            return

        if self.state == 'ROTATING':
            assert self.current_yaw is not None
            assert self.target_yaw is not None
            assert self.motion_started_at is not None

            error = normalize_angle(self.target_yaw - self.current_yaw)

            if (
                self.sim_time_sec is None
                or self.motion_started_sim_sec is None
            ):
                self._finish(
                    False,
                    'Simulation clock became unavailable during motion')
                return

            sim_elapsed = max(
                0.0,
                self.sim_time_sec - self.motion_started_sim_sec,
            )

            if sim_elapsed > self.motion_timeout_sec:
                self._finish(
                    False,
                    'Opening sequence exceeded simulated motion_timeout_sec')
                return

            if not self.enable_motion:
                self._finish(False, 'enable_motion was false; no Day 1 motion was executed')
                return

            if abs(error) <= self.tolerance_rad:
                self.stop_base()
                self.settle_samples += 1
                if self.settle_samples >= self.settle_samples_required:
                    head_reached = (
                        self.actual_head_tilt is not None
                        and abs(self.actual_head_tilt - self.head_tilt)
                        <= self.head_tilt_tolerance
                    )
                    if head_reached:
                        self._finish(
                            True,
                            'Opening sequence completed within yaw and head tolerances')
                return

            self.settle_samples = 0
            speed = min(self.max_speed, abs(self.yaw_kp * error))
            speed = max(self.min_speed, speed)
            self._publish_base_speed(math.copysign(speed, error))

    def _build_result(self, passed: bool, reason: str) -> Dict[str, Any]:
        now = time.monotonic()

        wall_duration = None
        if self.motion_started_at is not None:
            wall_duration = now - self.motion_started_at

        duration = None
        if (
            self.motion_started_sim_sec is not None
            and self.sim_time_sec is not None
        ):
            duration = max(
                0.0,
                self.sim_time_sec - self.motion_started_sim_sec,
            )

        signed_rotation_deg = None
        final_error_deg = None
        if self.initial_yaw is not None and self.current_yaw is not None:
            signed_rotation_deg = math.degrees(
                normalize_angle(self.current_yaw - self.initial_yaw))
        if self.target_yaw is not None and self.current_yaw is not None:
            final_error_deg = math.degrees(
                normalize_angle(self.target_yaw - self.current_yaw))

        return {
            'passed': bool(passed),
            'reason': reason,
            'state': 'PASSED' if passed else 'FAILED',
            'shelf_column_number': self.shelf_column,
            'book_colour': self.book_colour,
            'phase1_fast_start': self.phase1_fast_start,
            'enable_motion': self.enable_motion,
            'commanded_clockwise_rotation_deg': self.rotation_deg,
            'signed_rotation_deg': signed_rotation_deg,
            'absolute_rotation_deg': (
                abs(signed_rotation_deg) if signed_rotation_deg is not None else None),
            'final_error_deg': final_error_deg,
            'duration_sec': duration,
            'wall_duration_sec': wall_duration,
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
            'camera_frames_total': self.camera_frames,
            'camera_frames_during_motion': max(
                0, self.camera_frames - self.frames_at_motion_start),
            'last_image_stamp_ns': self.last_image_stamp_ns,
            'result_path': self.result_path,
            'generated_unix_time': time.time(),
        }

    def _write_result(self, result: Dict[str, Any]) -> None:
        result_dir = os.path.dirname(self.result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        temporary_path = self.result_path + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary_path, self.result_path)

    def _finish(self, passed: bool, reason: str) -> None:
        if self.done:
            return
        self.stop_base()
        result = self._build_result(passed, reason)
        try:
            self._write_result(result)
        except OSError as exc:
            passed = False
            reason = f'Could not write result file: {exc}'
            result['passed'] = False
            result['state'] = 'FAILED'
            result['reason'] = reason

        self.passed = passed
        self.failure_reason = '' if passed else reason
        self.done = True
        self.state = 'PASSED' if passed else 'FAILED'
        self._publish_status(self.state, reason)

        level = self.get_logger().info if passed else self.get_logger().error
        level('[DAY1][%s] %s | result=%s' % (self.state, reason, self.result_path))


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[OpeningSequence] = None
    exit_code = 1
    try:
        node = OpeningSequence()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.10)
        if node.done and node.passed:
            exit_code = 0
    except KeyboardInterrupt:
        if node is not None:
            node.stop_base()
            node.get_logger().warning('[DAY1] Interrupted; base stop command sent')
        exit_code = 130
    except Exception as exc:  # Fail loudly and leave a concise error in the terminal.
        if node is not None:
            node.stop_base()
            node.get_logger().error('[DAY1][FAILED] Unhandled exception: %s' % exc)
        else:
            print('[DAY1][FAILED] Could not initialize opening_sequence: %s' % exc)
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
