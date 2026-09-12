#!/usr/bin/env python3
"""Day 5: autonomous execution of the already-validated Day 4 grasp sequence.

The class reuses ``StagedGraspExperiment`` directly.  It removes operator
approval only after Day 4 demonstrated every stage manually, preserves the
public position-only gripper topics, and adds an automatic post-lift retention
hold based on recent fingertip contact plus a closed gripper state.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros

from ku_sparcy_erc.day567_common import atomic_write_json
from ku_sparcy_erc.range_fusion import CameraModel
from ku_sparcy_erc.staged_grasp_experiment import StagedGraspExperiment


class Day5AutonomousPick(StagedGraspExperiment):
    """Run the tested plan-to-lift sequence without manual checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter('retention_verify_sec', 0.80)
        self.declare_parameter('retention_contact_stale_sec', 0.35)
        self.declare_parameter('retention_min_contact_messages', 3)
        self.declare_parameter('retention_max_gripper_position_m', 0.045)
        self.declare_parameter('retention_visual_frames', 3)
        self.declare_parameter('retention_visual_radius_px', 110.0)
        self.declare_parameter('book_min_saturation', 90)
        self.declare_parameter('book_min_value', 55)
        self.declare_parameter('book_min_area_px', 18)
        self.retention_verify_sec = float(
            self.get_parameter('retention_verify_sec').value)
        self.retention_contact_stale_sec = float(
            self.get_parameter('retention_contact_stale_sec').value)
        self.retention_min_contact_messages = int(
            self.get_parameter('retention_min_contact_messages').value)
        self.retention_max_gripper_position = float(
            self.get_parameter('retention_max_gripper_position_m').value)
        self.retention_visual_required = int(
            self.get_parameter('retention_visual_frames').value)
        self.retention_visual_radius_px = float(
            self.get_parameter('retention_visual_radius_px').value)
        self.book_min_saturation = int(
            self.get_parameter('book_min_saturation').value)
        self.book_min_value = int(
            self.get_parameter('book_min_value').value)
        self.book_min_area = int(
            self.get_parameter('book_min_area_px').value)
        if self.manual_approval_required:
            raise ValueError(
                'Day5AutonomousPick requires manual_approval_required=false')
        if self.target_stage != 'lift':
            raise ValueError('Day5AutonomousPick target_stage must be lift')
        if abs(float(self.lift_distance) - 0.02) > 1.0e-9:
            raise ValueError(
                'validated Day 5 autonomous pick requires lift_distance_m=0.02')
        if abs(float(self.extract_distance) - 0.10) > 1.0e-9:
            raise ValueError(
                'validated Day 5 autonomous pick requires extract_distance_m=0.10')
        if self.retention_verify_sec <= 0.0:
            raise ValueError('retention_verify_sec must be positive')
        self.last_fingertip_contact_sim: Optional[float] = None
        self.retention_started_sim: Optional[float] = None
        self.retention_contact_count_at_start = 0
        self.retention_verified = False
        self.retry_count = 0
        self.retention_camera_model: Optional[CameraModel] = None
        self.retention_visual_frames = 0
        self.retention_visual_first_stamp: Optional[float] = None
        self.retention_visual_last_stamp: Optional[float] = None
        self.retention_visual_bbox = None
        self.requested_colour = str(
            self.source_result.get('book_colour_requested')
            or self.source_result.get('book_colour')
            or '').strip().lower()
        if self.requested_colour not in {'red', 'green', 'yellow', 'blue'}:
            raise ValueError('source Day 4 result has no valid requested colour')
        self.retention_camera_info_sub = self.create_subscription(
            CameraInfo,
            '/head_front_camera/head_front_camera/color/camera_info',
            self._retention_camera_info_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            '[DAY5] autonomous tested grasp enabled; '
            'position-only public gripper path; 0.10 m extract; 0.02 m lift')

    def _retention_camera_info_callback(self, msg: CameraInfo) -> None:
        try:
            self.retention_camera_model = CameraModel.from_arrays(
                msg.width, msg.height, msg.k, msg.p,
                frame_id=msg.header.frame_id,
                prefer_projection=False,
            )
        except ValueError:
            return

    def _project_gripper_pixel(self):
        model = self.retention_camera_model
        if model is None or self.selected_arm is None:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                model.frame_id,
                f'gripper_{self.selected_arm}_grasping_link',
                Time(),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None
        point = transform.transform.translation
        z = float(point.z)
        if z <= 0.05:
            return None
        return (
            model.fx * float(point.x) / z + model.cx,
            model.fy * float(point.y) / z + model.cy,
        )

    def _target_colour_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = self.book_min_saturation
        val = self.book_min_value
        if self.requested_colour == 'red':
            mask = (
                cv2.inRange(hsv, np.array([0, sat, val], dtype=np.uint8),
                            np.array([12, 255, 255], dtype=np.uint8))
                | cv2.inRange(hsv, np.array([168, sat, val], dtype=np.uint8),
                              np.array([179, 255, 255], dtype=np.uint8))
            )
        elif self.requested_colour == 'green':
            mask = cv2.inRange(
                hsv, np.array([35, sat, val], dtype=np.uint8),
                np.array([90, 255, 255], dtype=np.uint8))
        elif self.requested_colour == 'yellow':
            mask = cv2.inRange(
                hsv, np.array([18, sat, val], dtype=np.uint8),
                np.array([38, 255, 255], dtype=np.uint8))
        else:
            mask = cv2.inRange(
                hsv, np.array([90, sat, val], dtype=np.uint8),
                np.array([140, 255, 255], dtype=np.uint8))
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def _image_callback(self, msg: Image) -> None:
        super()._image_callback(msg)
        if self.state != 'VERIFY_RETENTION' or self.latest_frame_bgr is None:
            return
        stamp = self.latest_frame_stamp
        if stamp is None:
            return
        if (
            self.retention_visual_last_stamp is not None
            and abs(stamp - self.retention_visual_last_stamp) <= 1.0e-9
        ):
            return
        pixel = self._project_gripper_pixel()
        if pixel is None:
            return
        mask = self._target_colour_mask(self.latest_frame_bgr)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_distance = math.inf
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.book_min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            center = (x + 0.5 * width, y + 0.5 * height)
            distance = math.hypot(center[0] - pixel[0], center[1] - pixel[1])
            if distance <= self.retention_visual_radius_px and distance < best_distance:
                best = (x, y, width, height)
                best_distance = distance
        if best is None:
            return
        self.retention_visual_bbox = list(best)
        self.retention_visual_frames += 1
        if self.retention_visual_first_stamp is None:
            self.retention_visual_first_stamp = float(stamp)
        self.retention_visual_last_stamp = float(stamp)

    def _contact_callback(self, msg) -> None:
        before = int(self.selected_fingertip_contact_messages)
        super()._contact_callback(msg)
        if (
            not self.done
            and self.selected_fingertip_contact_messages > before
            and self.sim_time_sec is not None
        ):
            self.last_fingertip_contact_sim = float(self.sim_time_sec)

    def _complete_stage(self, stage: str, error: Optional[float]) -> None:
        if stage != 'lift':
            super()._complete_stage(stage, error)
            return

        self._stop_base()
        record: Dict[str, object] = {
            'stage': stage,
            'completed_sim_sec': self.sim_time_sec,
            'max_joint_or_gripper_error': error,
            'gripper_position_m': self.gripper_actual_position,
            'gripper_effort_state': self.gripper_actual_effort,
            'selected_fingertip_contact_messages': (
                self.selected_fingertip_contact_messages),
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
        self.active_stage = 'lift'
        self.retention_started_sim = self.sim_time_sec
        self.retention_contact_count_at_start = int(
            self.selected_fingertip_contact_messages)
        self._transition(
            'VERIFY_RETENTION',
            '2 cm lift settled; holding the public close-position command and '
            'verifying recent fingertip contact plus closed gripper state.',
        )

    def _retention_tick(self) -> None:
        self._stop_base()
        self._maintain_gripper_hold()
        if self.sim_time_sec is None or self.retention_started_sim is None:
            return
        elapsed = max(0.0, self.sim_time_sec - self.retention_started_sim)
        if elapsed < self.retention_verify_sec:
            return
        position = self.gripper_actual_position
        recent_contact = (
            self.last_fingertip_contact_sim is not None
            and self.sim_time_sec - self.last_fingertip_contact_sim
            <= self.retention_contact_stale_sec
        )
        new_contacts = (
            self.selected_fingertip_contact_messages
            - self.retention_contact_count_at_start
        )
        checks = {
            'gripper_position_valid': (
                isinstance(position, (int, float))
                and 0.0 <= float(position)
                <= self.retention_max_gripper_position),
            'recent_fingertip_contact': recent_contact,
            'enough_hold_contact_messages': (
                new_contacts >= self.retention_min_contact_messages),
            'requested_colour_visible_near_gripper': (
                self.retention_visual_frames >= self.retention_visual_required
                and self.retention_visual_first_stamp is not None
                and self.retention_visual_last_stamp is not None
                and self.retention_visual_last_stamp
                - self.retention_visual_first_stamp >= 0.05),
            'no_premature_contact': (
                self.premature_selected_fingertip_contacts == 0),
            'no_unintended_arm_contact': (
                self.unintended_selected_arm_contacts == 0),
            'complete_stage_sequence': all(
                stage in self.completed_stages
                for stage in (
                    'plan', 'pregrasp', 'open', 'no_contact',
                    'contact_pose', 'close', 'extract', 'lift')),
        }
        if all(checks.values()):
            self.retention_verified = True
            self._save_stage_image('retention_verified')
            self._finish(
                True,
                'autonomous position-controlled grasp, 0.10 m extraction, '
                '0.02 m lift and contact-continuity retention verification passed',
            )
            return
        self._finish(
            False,
            'post-lift retention verification failed: '
            + ', '.join(
                name for name, passed in checks.items() if not passed),
        )

    def _tick(self) -> None:
        if self.done:
            return
        if self.state == 'VERIFY_RETENTION':
            if time.monotonic() - self.last_clock_wall > self.clock_stall_wall_timeout:
                self._finish(
                    False,
                    'Gazebo /clock stopped during retention verification')
                return
            self._retention_tick()
            return
        super()._tick()

    def _write_result(self, passed: bool, reason: str) -> None:
        super()._write_result(passed, reason)
        try:
            with open(self.grasp_result_path, 'r', encoding='utf-8') as handle:
                result = __import__('json').load(handle)
        except (OSError, ValueError):
            result = {}
        result.update({
            'day': 5,
            'experiment': 'autonomous_position_controlled_pick',
            'passed': bool(passed),
            'reason': reason,
            'manual_approval_required': False,
            'validated_extract_distance_m': float(self.extract_distance),
            'validated_lift_distance_m': float(self.lift_distance),
            'retention_verify_sec': float(self.retention_verify_sec),
            'retention_contact_stale_sec': float(
                self.retention_contact_stale_sec),
            'retention_min_contact_messages': int(
                self.retention_min_contact_messages),
            'last_fingertip_contact_sim_sec': self.last_fingertip_contact_sim,
            'retention_contact_messages_after_lift': int(
                self.selected_fingertip_contact_messages
                - self.retention_contact_count_at_start),
            'retention_verified': bool(self.retention_verified),
            'requested_book_colour': self.requested_colour,
            'retention_visual_frames': int(self.retention_visual_frames),
            'retention_visual_first_stamp_sec': self.retention_visual_first_stamp,
            'retention_visual_last_stamp_sec': self.retention_visual_last_stamp,
            'retention_visual_bbox_xywh': self.retention_visual_bbox,
            'retry_count': int(self.retry_count),
            'book_retention_automatic_verification': (
                'contact_continuity_closed_position_and_live_colour_near_gripper'),
        })
        atomic_write_json(self.grasp_result_path, result)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[Day5AutonomousPick] = None
    code = 1
    try:
        node = Day5AutonomousPick()
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
            print(f'[DAY5][FAILED] {exc}')
        code = 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
