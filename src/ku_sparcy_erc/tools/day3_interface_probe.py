#!/usr/bin/env python3
"""Daemon-independent live Day 3 sensor/interface probe."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Optional

from cv_bridge import CvBridge
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, LaserScan
import tf2_ros


class InterfaceProbe(Node):
    def __init__(self) -> None:
        super().__init__('ku_sparcy_day3_interface_probe')
        self.bridge = CvBridge()
        self.messages: Dict[str, Any] = {}
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False)
        self.create_subscription(
            Clock, '/clock', lambda msg: self._store('clock', msg),
            qos_profile_sensor_data)
        self.create_subscription(
            Odometry, '/odom', lambda msg: self._store('odom', msg),
            qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo,
            '/head_front_camera/head_front_camera/color/camera_info',
            lambda msg: self._store('color_info', msg),
            qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo,
            '/head_front_camera/head_front_camera/depth/camera_info',
            lambda msg: self._store('depth_info', msg),
            qos_profile_sensor_data)
        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/depth/image_rect_raw',
            lambda msg: self._store('depth', msg),
            qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan_front_raw',
            lambda msg: self._store('scan', msg),
            qos_profile_sensor_data)

    def _store(self, name: str, msg: Any) -> None:
        if name not in self.messages:
            self.messages[name] = msg

    def complete(self) -> bool:
        return all(name in self.messages for name in (
            'clock', 'odom', 'color_info', 'depth_info', 'depth', 'scan'))

    def lookup(self, source_frame: str) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                'base_footprint', source_frame, Time())
            return True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return False


def gate(label: str, ok: bool) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    return bool(ok)


def main() -> int:
    rclpy.init()
    node = InterfaceProbe()
    deadline = time.monotonic() + 35.0
    try:
        while rclpy.ok() and not node.complete() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.10)

        results = []
        results.append(gate(
            'all typed Day 3 topics produced messages without ROS CLI daemon',
            node.complete()))
        if not node.complete():
            missing = sorted(set(
                ('clock', 'odom', 'color_info', 'depth_info', 'depth', 'scan')
            ) - set(node.messages))
            print('Missing:', ', '.join(missing))
            print('[DAY3 INTERFACE PROBE][FAIL]')
            return 1

        color: CameraInfo = node.messages['color_info']
        depth_info: CameraInfo = node.messages['depth_info']
        depth: Image = node.messages['depth']
        scan: LaserScan = node.messages['scan']
        odom: Odometry = node.messages['odom']

        tf_deadline = time.monotonic() + 5.0
        depth_tf_ok = False
        scan_tf_ok = False
        while rclpy.ok() and time.monotonic() < tf_deadline:
            depth_tf_ok = node.lookup(depth_info.header.frame_id)
            scan_tf_ok = node.lookup(scan.header.frame_id)
            if depth_tf_ok and scan_tf_ok:
                break
            rclpy.spin_once(node, timeout_sec=0.10)

        results.append(gate(
            'RGB CameraInfo is 640x360 with positive intrinsics',
            color.width == 640 and color.height == 360
            and color.k[0] > 0.0 and color.k[4] > 0.0))
        results.append(gate(
            'depth CameraInfo is 640x360 with positive intrinsics',
            depth_info.width == 640 and depth_info.height == 360
            and depth_info.k[0] > 0.0 and depth_info.k[4] > 0.0))
        results.append(gate(
            'RGB and depth intrinsics are pixel-aligned within tolerance',
            abs(color.k[0] - depth_info.k[0]) <= 2.0
            and abs(color.k[4] - depth_info.k[4]) <= 2.0
            and abs(color.k[2] - depth_info.k[2]) <= 2.0
            and abs(color.k[5] - depth_info.k[5]) <= 2.0))
        results.append(gate(
            'camera headers use optical frames',
            'optical_frame' in color.header.frame_id
            and 'optical_frame' in depth_info.header.frame_id))
        results.append(gate(
            'raw depth image matches CameraInfo and has supported encoding',
            depth.width == depth_info.width
            and depth.height == depth_info.height
            and depth.encoding.upper() in {'32FC1', '16UC1', 'MONO16'}))

        finite_depth = 0
        depth_min: Optional[float] = None
        depth_max: Optional[float] = None
        try:
            array = node.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
            values = np.asarray(array)
            if values.dtype == np.uint16:
                values = values.astype(np.float32) * 0.001
            else:
                values = values.astype(np.float32)
            valid = values[np.isfinite(values) & (values >= 0.2) & (values <= 8.0)]
            finite_depth = int(valid.size)
            if valid.size:
                depth_min = float(np.min(valid))
                depth_max = float(np.max(valid))
        except Exception as exc:  # noqa: BLE001
            print(f'Depth decode exception: {exc}')
        results.append(gate(
            'raw depth frame contains finite values in the 0.2-8.0 m range',
            finite_depth >= 100))

        fov_deg = math.degrees(
            scan.angle_max - scan.angle_min) if scan.angle_max > scan.angle_min else 0.0
        results.append(gate(
            'front LaserScan has approximately 270-degree FoV and dense samples',
            260.0 <= fov_deg <= 275.0 and len(scan.ranges) >= 800))
        results.append(gate(
            'front LaserScan range bounds are valid',
            scan.range_min > 0.0 and scan.range_max >= 10.0))
        results.append(gate(
            'front LaserScan frame is transformable to base_footprint',
            scan_tf_ok))
        results.append(gate(
            'depth optical frame is transformable to base_footprint',
            depth_tf_ok))
        results.append(gate(
            'odometry uses odom parent and a base child frame',
            odom.header.frame_id == 'odom'
            and odom.child_frame_id in {'base_footprint', 'base_link'}))

        summary = {
            'color': {
                'frame_id': color.header.frame_id,
                'width': color.width,
                'height': color.height,
                'fx': color.k[0],
                'fy': color.k[4],
                'cx': color.k[2],
                'cy': color.k[5],
            },
            'depth': {
                'frame_id': depth_info.header.frame_id,
                'width': depth.width,
                'height': depth.height,
                'encoding': depth.encoding,
                'finite_pixels': finite_depth,
                'finite_min_m': depth_min,
                'finite_max_m': depth_max,
            },
            'front_scan': {
                'frame_id': scan.header.frame_id,
                'samples': len(scan.ranges),
                'fov_deg': fov_deg,
                'range_min_m': scan.range_min,
                'range_max_m': scan.range_max,
                'angle_increment_deg': math.degrees(scan.angle_increment),
            },
            'odom': {
                'frame_id': odom.header.frame_id,
                'child_frame_id': odom.child_frame_id,
            },
        }
        print('\nMeasured interfaces:')
        print(json.dumps(summary, indent=2, sort_keys=True))

        if all(results):
            print('\n[CAMERA INTRINSICS][PASS]')
            print('[DEPTH INPUT][PASS]')
            print('[LIDAR INPUT][PASS]')
            print('[TF INTERFACES][PASS]')
            print('[DAY3 INTERFACE PROBE][PASS]')
            return 0
        print('\n[DAY3 INTERFACE PROBE][FAIL]')
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
