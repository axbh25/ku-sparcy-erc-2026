#!/usr/bin/env python3
"""Record the trial's live initial odometry pose before the mission moves."""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock

from ku_sparcy_erc.day567_common import atomic_write_json, normalize_angle, yaw_from_odom


class HomePoseRecorder(Node):
    def __init__(self) -> None:
        super().__init__('ku_sparcy_home_pose_recorder')
        self.declare_parameter(
            'result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/home_pose.json')
        self.declare_parameter('stable_samples', 4)
        self.declare_parameter('position_tolerance_m', 0.004)
        self.declare_parameter('yaw_tolerance_deg', 0.35)
        self.declare_parameter('ready_wall_timeout_sec', 30.0)
        self.result_path = str(self.get_parameter('result_path').value)
        self.stable_samples_required = int(
            self.get_parameter('stable_samples').value)
        self.position_tolerance = float(
            self.get_parameter('position_tolerance_m').value)
        self.yaw_tolerance = math.radians(float(
            self.get_parameter('yaw_tolerance_deg').value))
        self.ready_wall_timeout = float(
            self.get_parameter('ready_wall_timeout_sec').value)
        if self.stable_samples_required < 3:
            raise ValueError('stable_samples must be at least 3')
        self.started_wall = time.monotonic()
        self.sim_time_sec: Optional[float] = None
        self.samples: List[Tuple[float, float, float, float]] = []
        self.done = False
        self.passed = False
        self.reason = ''
        self.clock_sub = self.create_subscription(
            Clock, '/clock', self._clock_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self._tick)

    def _clock_callback(self, msg: Clock) -> None:
        self.sim_time_sec = (
            float(msg.clock.sec) + float(msg.clock.nanosec) * 1.0e-9)

    def _odom_callback(self, msg: Odometry) -> None:
        if self.done or self.sim_time_sec is None:
            return
        self.samples.append((
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(yaw_from_odom(msg)),
            float(self.sim_time_sec),
        ))
        if len(self.samples) > self.stable_samples_required:
            del self.samples[:-self.stable_samples_required]
        if len(self.samples) < self.stable_samples_required:
            return
        xs = [item[0] for item in self.samples]
        ys = [item[1] for item in self.samples]
        yaws = [item[2] for item in self.samples]
        reference = yaws[0]
        yaw_errors = [abs(normalize_angle(value - reference)) for value in yaws]
        if (
            max(xs) - min(xs) <= self.position_tolerance
            and max(ys) - min(ys) <= self.position_tolerance
            and max(yaw_errors) <= self.yaw_tolerance
        ):
            x = sum(xs) / len(xs)
            y = sum(ys) / len(ys)
            sin_sum = sum(math.sin(value) for value in yaws)
            cos_sum = sum(math.cos(value) for value in yaws)
            yaw = math.atan2(sin_sum, cos_sum)
            self._finish(True, 'stable initial odometry recorded', x, y, yaw)

    def _tick(self) -> None:
        if self.done:
            return
        if time.monotonic() - self.started_wall > self.ready_wall_timeout:
            self._finish(False, 'initial odometry readiness timed out')

    def _finish(
        self,
        passed: bool,
        reason: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> None:
        if self.done:
            return
        self.done = True
        self.passed = bool(passed)
        self.reason = str(reason)
        atomic_write_json(self.result_path, {
            'passed': bool(passed),
            'reason': reason,
            'frame_id': 'odom',
            'home_odom_xy': [float(x), float(y)]
            if x is not None and y is not None else None,
            'home_yaw_rad': float(yaw) if yaw is not None else None,
            'recorded_sim_time_sec': self.sim_time_sec,
            'stable_sample_count': len(self.samples),
            'wall_duration_sec': time.monotonic() - self.started_wall,
        })
        label = 'PASS' if passed else 'FAIL'
        self.get_logger().info(f'[HOME POSE][{label}] {reason}')


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[HomePoseRecorder] = None
    code = 1
    try:
        node = HomePoseRecorder()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = 0 if node.done and node.passed else 1
    except KeyboardInterrupt:
        code = 130
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
