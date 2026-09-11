#!/usr/bin/env python3
"""Position-command and effort-state characterization with no book contact.

This diagnostic deliberately uses only the public, clamped position topic. It
never creates an effort command interface and never interprets free-air effort
as proof that load feedback can maintain a grasp.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class Characterizer(Node):
    def __init__(self, side: str) -> None:
        super().__init__('ku_sparcy_gripper_free_air_characterization')
        self.side = side
        self.joint = f'gripper_{side}_finger_joint'
        self.sim_sec: Optional[float] = None
        self.position: Optional[float] = None
        self.effort: Optional[float] = None
        self.effort_samples: List[float] = []
        self.collecting = False
        self.publisher = self.create_publisher(
            JointTrajectory,
            f'/gripper_{side}_controller/joint_trajectory',
            10,
        )
        self.create_subscription(
            Clock, '/clock', self._clock_cb, qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/joint_states', self._joint_state,
            qos_profile_sensor_data)

    def _clock_cb(self, msg: Clock) -> None:
        self.sim_sec = msg.clock.sec + msg.clock.nanosec * 1.0e-9

    def _joint_state(self, msg: JointState) -> None:
        if self.joint not in msg.name:
            return
        index = msg.name.index(self.joint)
        if index < len(msg.position):
            self.position = float(msg.position[index])
        if index < len(msg.effort):
            self.effort = float(msg.effort[index])
            if self.collecting and math.isfinite(self.effort):
                self.effort_samples.append(self.effort)

    def command(self, target: float, duration: float = 1.5) -> None:
        message = JointTrajectory()
        message.joint_names = [self.joint]
        point = JointTrajectoryPoint()
        point.positions = [float(target)]
        seconds = int(duration)
        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = int(
            round((duration - seconds) * 1.0e9))
        message.points = [point]
        self.publisher.publish(message)


def wait_ready(node: Characterizer, wall_timeout: float) -> bool:
    deadline = time.monotonic() + wall_timeout
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.10)
        if (
            node.sim_sec is not None
            and node.position is not None
            and node.effort is not None
            and node.publisher.get_subscription_count() > 0
        ):
            return True
    return False


def wait_sim(node: Characterizer, duration: float, wall_timeout: float = 60.0) -> bool:
    if node.sim_sec is None:
        return False
    start = node.sim_sec
    wall_deadline = time.monotonic() + wall_timeout
    last_advance = time.monotonic()
    previous = start
    while rclpy.ok() and time.monotonic() < wall_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.sim_sec is not None and node.sim_sec > previous + 1.0e-9:
            previous = node.sim_sec
            last_advance = time.monotonic()
        if node.sim_sec is not None and node.sim_sec - start >= duration:
            return True
        if time.monotonic() - last_advance > 20.0:
            return False
    return False


def collect(node: Characterizer, duration: float) -> List[float]:
    node.effort_samples = []
    node.collecting = True
    ok = wait_sim(node, duration)
    node.collecting = False
    return list(node.effort_samples) if ok else []


def summary(values: List[float]) -> dict:
    if not values:
        return {'count': 0, 'mean': None, 'stdev': None, 'minimum': None, 'maximum': None}
    return {
        'count': len(values),
        'mean': statistics.fmean(values),
        'stdev': statistics.pstdev(values),
        'minimum': min(values),
        'maximum': max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', choices=('left', 'right'), default='right')
    parser.add_argument('--result', required=True)
    parser.add_argument('--open-position', type=float, default=0.040)
    parser.add_argument('--close-position', type=float, default=0.010)
    args = parser.parse_args()
    if not 0.0 <= args.close_position < args.open_position <= 0.069:
        print('[GRIPPER FREE-AIR CHARACTERIZATION][FAIL] invalid positions')
        return 1

    rclpy.init()
    node = Characterizer(args.side)
    result = {
        'side': args.side,
        'joint': node.joint,
        'open_position_command': args.open_position,
        'close_position_command': args.close_position,
        'effort_commanded': False,
        'book_contact_attempted': False,
        'load_discrimination_verified': False,
    }
    try:
        if not wait_ready(node, 25.0):
            print('[GRIPPER FREE-AIR CHARACTERIZATION][FAIL] inputs unavailable')
            return 1

        node.command(args.open_position)
        if not wait_sim(node, 2.0):
            print('[GRIPPER FREE-AIR CHARACTERIZATION][FAIL] open wait failed')
            return 1
        open_position = node.position
        open_effort = collect(node, 0.8)

        node.command(args.close_position)
        if not wait_sim(node, 2.0):
            print('[GRIPPER FREE-AIR CHARACTERIZATION][FAIL] close wait failed')
            return 1
        close_position = node.position
        close_effort = collect(node, 0.8)

        # Leave the diagnostic gripper open and clear after the experiment.
        node.command(args.open_position)
        wait_sim(node, 2.0)

        result.update({
            'open_position_measured': open_position,
            'close_position_measured': close_position,
            'open_effort_free_air': summary(open_effort),
            'close_effort_free_air': summary(close_effort),
            'final_position_measured': node.position,
        })
        temporary = args.result + '.tmp'
        os.makedirs(os.path.dirname(args.result) or '.', exist_ok=True)
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, args.result)

        position_ok = (
            open_position is not None
            and close_position is not None
            and abs(open_position - args.open_position) <= 0.004
            and abs(close_position - args.close_position) <= 0.004
        )
        effort_ok = len(open_effort) >= 5 and len(close_effort) >= 5
        print(f"[{'PASS' if position_ok else 'FAIL'}] public position commands reached their targets")
        print(f"[{'PASS' if effort_ok else 'FAIL'}] effort state produced free-air samples")
        print(json.dumps(result, indent=2, sort_keys=True))
        if position_ok:
            print('[GRIPPER POSITION INTERFACE][PASS]')
        else:
            print('[GRIPPER POSITION INTERFACE][FAIL]')
        if effort_ok:
            print('[EFFORT STATE SAMPLING][PASS]')
            print('[EFFORT LOAD DISCRIMINATION][UNVERIFIED] no loaded book was used')
        else:
            print('[EFFORT STATE SAMPLING][FAIL]')
        if position_ok and effort_ok:
            print('[GRIPPER FREE-AIR CHARACTERIZATION][PASS]')
            return 0
        print('[GRIPPER FREE-AIR CHARACTERIZATION][FAIL]')
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
