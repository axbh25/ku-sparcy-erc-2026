#!/usr/bin/env python3
"""Read-only direct-rclpy probe of the TIAGo gripper interfaces."""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, Optional

import rclpy
from controller_manager_msgs.srv import ListHardwareInterfaces
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class Probe(Node):
    def __init__(self) -> None:
        super().__init__('ku_sparcy_gripper_interface_probe')
        self.sample: Optional[JointState] = None
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state,
            qos_profile_sensor_data)
        self.left_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_left_controller/joint_trajectory',
            10,
        )
        self.right_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_right_controller/joint_trajectory',
            10,
        )
        self.client = self.create_client(
            ListHardwareInterfaces,
            '/controller_manager/list_hardware_interfaces',
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self.sample = msg


def interface_names(items) -> Dict[str, Dict[str, bool]]:
    return {
        str(item.name): {
            'available': bool(item.is_available),
            'claimed': bool(item.is_claimed),
        }
        for item in items
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default='')
    args = parser.parse_args()

    rclpy.init()
    node = Probe()
    try:
        deadline = time.monotonic() + 25.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.10)
            if (
                node.sample is not None
                and node.client.service_is_ready()
                and node.left_pub.get_subscription_count() > 0
                and node.right_pub.get_subscription_count() > 0
            ):
                break
        if node.sample is None or not node.client.service_is_ready():
            print('[GRIPPER INTERFACE PROBE][FAIL] missing joint state or service')
            return 1
        future = node.client.call_async(ListHardwareInterfaces.Request())
        service_deadline = time.monotonic() + 10.0
        while (
            rclpy.ok()
            and not future.done()
            and time.monotonic() < service_deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.10)
        if not future.done() or future.result() is None:
            print('[GRIPPER INTERFACE PROBE][FAIL] service call timed out')
            return 1
        response = future.result()
        commands = interface_names(response.command_interfaces)
        states = interface_names(response.state_interfaces)
        joint_state = node.sample
        assert joint_state is not None
        joints = {}
        for joint in (
            'gripper_left_finger_joint',
            'gripper_right_finger_joint',
        ):
            if joint not in joint_state.name:
                joints[joint] = {'present': False}
                continue
            index = joint_state.name.index(joint)
            joints[joint] = {
                'present': True,
                'position': (
                    float(joint_state.position[index])
                    if index < len(joint_state.position) else None),
                'velocity': (
                    float(joint_state.velocity[index])
                    if index < len(joint_state.velocity) else None),
                'effort': (
                    float(joint_state.effort[index])
                    if index < len(joint_state.effort) else None),
            }
        result = {
            'command_interfaces': commands,
            'state_interfaces': states,
            'joint_state_sample': joints,
            'public_left_topic_subscribers': (
                node.left_pub.get_subscription_count()),
            'public_right_topic_subscribers': (
                node.right_pub.get_subscription_count()),
        }
        if args.result:
            with open(args.result + '.tmp', 'w', encoding='utf-8') as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write('\n')
            import os
            os.replace(args.result + '.tmp', args.result)

        checks = {}
        for side in ('left', 'right'):
            joint = f'gripper_{side}_finger_joint'
            checks[f'{joint} position command exists'] = (
                f'{joint}/position' in commands)
            checks[f'{joint} effort command is absent'] = (
                f'{joint}/effort' not in commands)
            for interface in ('position', 'velocity', 'effort'):
                checks[f'{joint} {interface} state exists'] = (
                    f'{joint}/{interface}' in states)
            checks[f'{joint} effort is present in JointState'] = (
                joints.get(joint, {}).get('effort') is not None)
        checks['public left position topic has subscriber'] = (
            node.left_pub.get_subscription_count() > 0)
        checks['public right position topic has subscriber'] = (
            node.right_pub.get_subscription_count() > 0)

        for label, ok in checks.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        print(json.dumps(result, indent=2, sort_keys=True))
        if all(checks.values()):
            print('[GRIPPER POSITION INTERFACE][PASS]')
            print('[GRIPPER EFFORT STATE INTERFACE][PASS]')
            print('[GRIPPER INTERFACE PROBE][PASS]')
            return 0
        print('[GRIPPER INTERFACE PROBE][FAIL]')
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
