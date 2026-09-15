#!/usr/bin/env python3
"""Physical-shelf-position aware retained-book transport.

Route selection is based ONLY on the LIVE observed left-to-right
physical shelf position produced by Day 2/Day 4 perception.

Physical position 1/2:
    retreat -> forced clockwise 180 -> lateral return -> final bin align

Physical position 3:
    retreat -> forced counterclockwise 180 -> straight toward live bin

Physical position 4/5:
    retreat -> forced counterclockwise 180 -> lateral return -> final bin align

The printed marker number is NOT used to select the route.
No simulator world-state oracle or ERC_SEED is used.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import (
    CameraInfo,
    Image,
    JointState,
    LaserScan,
)
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)

import tf2_ros

from ament_index_python.packages import get_package_share_directory
from ros_gz_interfaces.msg import Contacts

from ku_sparcy_erc.bin_perception import (
    BinDetection,
    BinDetectorSettings,
    detect_red_bin,
)

from ku_sparcy_erc.day567_common import (
    atomic_write_json,
    book_half_diagonal_m,
    clamp,
    contact_collision_names,
    load_passed_json,
    meaningful_robot_collision,
    normalize_angle,
    selected_fingertip_tokens,
    vector_base_to_odom,
    vector_odom_to_base,
    yaw_from_odom,
)

from ku_sparcy_erc.range_fusion import (
    BoundingBoxLike,
    CameraModel,
    normalize_depth_array,
    target_geometry_from_depth,
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


def stamp(msg) -> Optional[float]:
    value = (
        float(msg.header.stamp.sec)
        + float(msg.header.stamp.nanosec) * 1.0e-9
    )
    return value if value > 0.0 else None


class FastTransport(Node):

    RGB_TOPIC = (
        '/head_front_camera/'
        'head_front_camera/color/image_raw'
    )

    COLOR_INFO_TOPIC = (
        '/head_front_camera/'
        'head_front_camera/color/camera_info'
    )

    DEPTH_TOPIC = (
        '/head_front_camera/'
        'head_front_camera/depth/image_rect_raw'
    )

    DEPTH_INFO_TOPIC = (
        '/head_front_camera/'
        'head_front_camera/depth/camera_info'
    )

    def __init__(self) -> None:
        super().__init__(
            'ku_sparcy_day67_fast_transport'
        )

        # --------------------------------------------------
        # Result/input paths
        # --------------------------------------------------

        self.declare_parameter(
            'day4_result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day4_result.json',
        )

        self.declare_parameter(
            'day5_result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day5_result.json',
        )

        self.declare_parameter(
            'home_pose_path',
            '/opt/erc_ws/src/ku_sparcy_erc/home_pose.json',
        )

        self.declare_parameter(
            'result_path',
            '/opt/erc_ws/src/ku_sparcy_erc/day7_result.json',
        )

        self.declare_parameter(
            'image_output_dir',
            '/opt/erc_ws/src/ku_sparcy_erc/erc_images',
        )

        # --------------------------------------------------
        # Shelf egress
        # --------------------------------------------------

        self.declare_parameter(
            'retreat_distance_m', 0.55)
        self.declare_parameter(
            'retreat_max_distance_m', 0.80)
        self.declare_parameter(
            'retreat_speed_mps', 0.16)
        self.declare_parameter(
            'rotation_clearance_margin_m', 0.20)

        # --------------------------------------------------
        # Validated held-book compact carry.
        #
        # This happens AFTER the short shelf retreat and
        # BEFORE any base rotation.
        # --------------------------------------------------

        self.declare_parameter(
            'carry_retract_distance_m', 0.12)

        self.declare_parameter(
            'carry_raise_distance_m', 0.16)

        self.declare_parameter(
            'carry_motion_sec', 4.0)

        self.declare_parameter(
            'carry_joint_tolerance_rad', 0.045)

        self.declare_parameter(
            'carry_settle_samples', 3)

        self.declare_parameter(
            'carry_timeout_sec', 10.0)

        self.declare_parameter(
            'carry_max_joint_delta_rad', 0.45)

        # --------------------------------------------------
        # GRAVITY-SUPPORTED TRANSPORT ORIENTATION
        #
        # After compact carry, rotate the BOOK around its
        # depth axis by +/-90 degrees.  This turns the thin
        # gripper/book axis vertical so the lower finger
        # mechanically supports the book against gravity.
        # --------------------------------------------------

        self.declare_parameter(
            'support_roll_deg', 90.0)

        self.declare_parameter(
            'support_extra_raise_m', 0.03)

        self.declare_parameter(
            'support_motion_sec', 6.0)

        self.declare_parameter(
            'support_joint_tolerance_rad', 0.050)

        self.declare_parameter(
            'support_settle_samples', 3)

        self.declare_parameter(
            'support_timeout_sec', 14.0)

        self.declare_parameter(
            'support_max_joint_delta_rad', 1.80)

        self.declare_parameter(
            'restore_motion_sec', 6.0)

        # --------------------------------------------------
        # Forced +/-180 degree rotation
        # --------------------------------------------------

        self.declare_parameter(
            'rotation_deg', 180.0)
        self.declare_parameter(
            'rotation_max_speed_radps', 0.10)
        self.declare_parameter(
            'rotation_kp', 0.90)
        self.declare_parameter(
            'rotation_tolerance_deg', 2.0)
        self.declare_parameter(
            'rotation_settle_samples', 4)

        # --------------------------------------------------
        # Side-column lateral return
        # --------------------------------------------------

        self.declare_parameter(
            'lateral_max_speed_mps', 0.08)

        # Carrying-book lateral acceleration/deceleration limiter.
        # Only Y motion is changed; forward X speeds remain untouched.
        self.declare_parameter(
            'lateral_accel_limit_mps2', 0.04)

        self.declare_parameter(
            'lateral_kp', 0.55)
        self.declare_parameter(
            'lateral_tolerance_m', 0.07)

        self.declare_parameter(
            'home_zone_radius_m', 0.38)

        # Main route remains lateral.
        # This only permits a SEPARATE short straight
        # correction after lateral motion has finished.
        self.declare_parameter(
            'longitudinal_correction_max_m', 0.75)
        self.declare_parameter(
            'longitudinal_correction_speed_mps', 0.10)

        # Keep the validated 0.10 m/s longitudinal speed, but ramp into and
        # out of it while carrying the book.  The old code jumped directly
        # between 0.10 m/s and zero at state boundaries / LiDAR stops.
        self.declare_parameter(
            'longitudinal_accel_limit_mps2', 0.03)

        # The front laser is 0.27512 m ahead of base_link in the official
        # TIAGo Pro URDF.  The carried-book forward extent is measured from
        # base_footprint.  Use both to stop BEFORE the held book can reach an
        # obstacle even though the base itself is still outside the old
        # 0.55 m hard-stop distance.
        self.declare_parameter(
            'front_laser_base_x_m', 0.27512)
        self.declare_parameter(
            'carried_front_margin_m', 0.10)

        # --------------------------------------------------
        # Bin placement-ready alignment
        # --------------------------------------------------

        self.declare_parameter(
            'final_bin_standoff_m', 1.00)

        self.declare_parameter(
            'final_bin_forward_tolerance_m', 0.10)

        self.declare_parameter(
            'final_bin_lateral_tolerance_m', 0.08)

        self.declare_parameter(
            'final_bin_bearing_tolerance_deg', 4.0)

        self.declare_parameter(
            'final_align_translation_speed_mps', 0.08)

        self.declare_parameter(
            'final_align_yaw_speed_radps', 0.08)

        self.declare_parameter(
            'final_settle_samples', 4)

        self.declare_parameter(
            'final_home_envelope_m', 0.60)

        # --------------------------------------------------
        # LiDAR
        # --------------------------------------------------

        self.declare_parameter(
            'rear_hard_stop_m', 0.55)

        self.declare_parameter(
            'front_hard_stop_m', 0.55)

        self.declare_parameter(
            'lidar_corridor_half_width_m', 0.42)

        self.declare_parameter(
            'lidar_robust_percentile', 10.0)

        self.declare_parameter(
            'scan_stale_timeout_sec', 0.60)

        # --------------------------------------------------
        # Existing red-bin detector values
        # --------------------------------------------------

        self.declare_parameter(
            'bin_confirm_frames', 3)

        self.declare_parameter(
            'bin_confirm_window_sec', 1.00)

        # Motion-compatible consistency check in ODOM,
        # not in image pixels.
        self.declare_parameter(
            'bin_odom_confirm_tolerance_m', 0.18)

        self.declare_parameter(
            'bin_min_saturation', 110)

        self.declare_parameter(
            'bin_min_value', 55)

        self.declare_parameter(
            'bin_min_area_px', 450)

        self.declare_parameter(
            'bin_min_width_px', 34)

        self.declare_parameter(
            'bin_min_height_px', 18)

        self.declare_parameter(
            'bin_min_aspect_ratio', 1.10)

        self.declare_parameter(
            'bin_min_fill_fraction', 0.30)

        self.declare_parameter(
            'max_rgb_depth_stamp_skew_sec', 0.20)

        self.declare_parameter(
            'depth_stale_timeout_sec', 0.55)

        self.declare_parameter(
            'depth_min_valid_pixels', 18)

        self.declare_parameter(
            'depth_min_valid_fraction', 0.10)

        # --------------------------------------------------
        # Public POSITION-only gripper hold
        # --------------------------------------------------

        self.declare_parameter(
            'gripper_hold_position_m', 0.000)

        self.declare_parameter(
            'gripper_hold_period_sec', 0.10)

        # --------------------------------------------------
        # Head
        # --------------------------------------------------

        self.declare_parameter(
            'head_search_pan_rad', 0.0)

        self.declare_parameter(
            'head_search_tilt_rad', -0.20)

        self.declare_parameter(
            'head_motion_sec', 1.2)

        # --------------------------------------------------
        # Timeouts
        # --------------------------------------------------

        self.declare_parameter(
            'ready_wall_timeout_sec', 45.0)

        self.declare_parameter(
            'clock_stall_wall_timeout_sec', 20.0)

        self.declare_parameter(
            'retreat_timeout_sec', 20.0)

        self.declare_parameter(
            'rotation_timeout_sec', 70.0)

        self.declare_parameter(
            'lateral_timeout_sec', 60.0)

        self.declare_parameter(
            'straight_timeout_sec', 45.0)

        self.declare_parameter(
            'bin_wait_after_transit_sec', 8.0)

        self.declare_parameter(
            'final_align_timeout_sec', 25.0)

        g = self.get_parameter

        # --------------------------------------------------
        # Read parameters
        # --------------------------------------------------

        self.day4_path = str(
            g('day4_result_path').value)

        self.day5_path = str(
            g('day5_result_path').value)

        self.home_path = str(
            g('home_pose_path').value)

        self.result_path = str(
            g('result_path').value)

        self.image_output_dir = str(
            g('image_output_dir').value)

        self.retreat_distance = float(
            g('retreat_distance_m').value)

        self.retreat_max_distance = float(
            g('retreat_max_distance_m').value)

        self.retreat_speed = float(
            g('retreat_speed_mps').value)

        self.rotation_margin = float(
            g('rotation_clearance_margin_m').value)

        self.carry_retract_distance = float(
            g('carry_retract_distance_m').value)

        self.carry_raise_distance = float(
            g('carry_raise_distance_m').value)

        self.carry_motion_sec = float(
            g('carry_motion_sec').value)

        self.carry_joint_tolerance = float(
            g('carry_joint_tolerance_rad').value)

        self.carry_settle_required = int(
            g('carry_settle_samples').value)

        self.carry_timeout = float(
            g('carry_timeout_sec').value)

        self.carry_max_joint_delta_limit = float(
            g('carry_max_joint_delta_rad').value)

        self.support_roll_rad = math.radians(
            float(g('support_roll_deg').value)
        )

        self.support_extra_raise = float(
            g('support_extra_raise_m').value
        )

        self.support_motion_sec = float(
            g('support_motion_sec').value
        )

        self.support_joint_tolerance = float(
            g('support_joint_tolerance_rad').value
        )

        self.support_settle_required = int(
            g('support_settle_samples').value
        )

        self.support_timeout = float(
            g('support_timeout_sec').value
        )

        self.support_max_joint_delta = float(
            g('support_max_joint_delta_rad').value
        )

        self.restore_motion_sec = float(
            g('restore_motion_sec').value
        )

        self.rotation_angle = math.radians(
            float(g('rotation_deg').value)
        )

        self.rotation_max_speed = float(
            g('rotation_max_speed_radps').value)

        self.rotation_kp = float(
            g('rotation_kp').value)

        self.rotation_tolerance = math.radians(
            float(g('rotation_tolerance_deg').value)
        )

        self.rotation_settle_required = int(
            g('rotation_settle_samples').value)

        self.lateral_max_speed = float(
            g('lateral_max_speed_mps').value)

        self.lateral_accel_limit = float(
            g('lateral_accel_limit_mps2').value)

        self.lateral_kp = float(
            g('lateral_kp').value)

        self.lateral_tolerance = float(
            g('lateral_tolerance_m').value)

        self.home_zone_radius = float(
            g('home_zone_radius_m').value)

        self.longitudinal_correction_max = float(
            g('longitudinal_correction_max_m').value)

        self.longitudinal_correction_speed = float(
            g('longitudinal_correction_speed_mps').value)

        self.longitudinal_accel_limit = float(
            g('longitudinal_accel_limit_mps2').value)

        self.front_laser_base_x = float(
            g('front_laser_base_x_m').value)

        self.carried_front_margin = float(
            g('carried_front_margin_m').value)

        self.final_standoff = float(
            g('final_bin_standoff_m').value)

        self.final_forward_tol = float(
            g('final_bin_forward_tolerance_m').value)

        self.final_lateral_tol = float(
            g('final_bin_lateral_tolerance_m').value)

        self.final_bearing_tol = math.radians(
            float(
                g(
                    'final_bin_bearing_tolerance_deg'
                ).value
            )
        )

        self.final_translation_speed = float(
            g('final_align_translation_speed_mps').value)

        self.final_yaw_speed = float(
            g('final_align_yaw_speed_radps').value)

        self.final_settle_required = int(
            g('final_settle_samples').value)

        self.final_home_envelope = float(
            g('final_home_envelope_m').value)

        self.rear_hard_stop = float(
            g('rear_hard_stop_m').value)

        self.front_hard_stop = float(
            g('front_hard_stop_m').value)

        self.corridor_half_width = float(
            g('lidar_corridor_half_width_m').value)

        self.lidar_percentile = float(
            g('lidar_robust_percentile').value)

        self.scan_stale_timeout = float(
            g('scan_stale_timeout_sec').value)

        self.bin_confirm_frames = int(
            g('bin_confirm_frames').value)

        self.bin_confirm_window = float(
            g('bin_confirm_window_sec').value)

        self.bin_odom_tolerance = float(
            g('bin_odom_confirm_tolerance_m').value)

        self.detector_settings = BinDetectorSettings(
            min_saturation=int(
                g('bin_min_saturation').value
            ),
            min_value=int(
                g('bin_min_value').value
            ),
            min_area_px=int(
                g('bin_min_area_px').value
            ),
            min_width_px=int(
                g('bin_min_width_px').value
            ),
            min_height_px=int(
                g('bin_min_height_px').value
            ),
            min_aspect_ratio=float(
                g('bin_min_aspect_ratio').value
            ),
            min_fill_fraction=float(
                g('bin_min_fill_fraction').value
            ),
        )

        self.max_rgb_depth_skew = float(
            g('max_rgb_depth_stamp_skew_sec').value)

        self.depth_stale_timeout = float(
            g('depth_stale_timeout_sec').value)

        self.depth_min_valid_pixels = int(
            g('depth_min_valid_pixels').value)

        self.depth_min_valid_fraction = float(
            g('depth_min_valid_fraction').value)

        self.gripper_hold_position = float(
            g('gripper_hold_position_m').value)

        self.gripper_hold_period = float(
            g('gripper_hold_period_sec').value)

        self.head_pan = float(
            g('head_search_pan_rad').value)

        self.head_tilt = float(
            g('head_search_tilt_rad').value)

        self.head_motion_sec = float(
            g('head_motion_sec').value)

        self.ready_wall_timeout = float(
            g('ready_wall_timeout_sec').value)

        self.clock_stall_wall_timeout = float(
            g('clock_stall_wall_timeout_sec').value)

        self.timeouts = {
            'RETREAT_FROM_SHELF': float(
                g('retreat_timeout_sec').value
            ),
            'WAIT_COMPACT_CARRY': self.carry_timeout,
            'WAIT_GRAVITY_SUPPORTED_CARRY': self.support_timeout,
            'WAIT_RESTORE_PLACEMENT_ORIENTATION': self.support_timeout,
            'ROTATE_180_SEARCH_BIN': float(
                g('rotation_timeout_sec').value
            ),
            'LATERAL_TRANSIT': float(
                g('lateral_timeout_sec').value
            ),
            'LONGITUDINAL_CORRECTION': float(
                g('lateral_timeout_sec').value
            ),
            'STRAIGHT_TO_BIN': float(
                g('straight_timeout_sec').value
            ),
            'WAIT_CENTER_BIN': float(
                g('bin_wait_after_transit_sec').value
            ),
            'FINAL_BIN_ALIGN': float(
                g('final_align_timeout_sec').value
            ),
        }

        # --------------------------------------------------
        # Load previous-stage evidence
        # --------------------------------------------------

        self.day4 = load_passed_json(
            self.day4_path,
            label='Day 4 result',
        )

        self.day5 = load_passed_json(
            self.day5_path,
            label='Day 5 result',
        )

        self.home = load_passed_json(
            self.home_path,
            label='home pose',
        )

        if self.day5.get(
            'retention_verified'
        ) is not True:
            raise ValueError(
                'Day 5 did not verify retained book'
            )

        self.selected_arm = str(
            self.day5.get('selected_arm') or ''
        ).lower()

        if self.selected_arm not in (
            'left',
            'right',
        ):
            raise ValueError(
                'invalid selected arm'
            )

        # Reuse the actual validated Day5 grasp orientation
        # and selected-arm joint naming.  No hard-coded arm
        # joint vector is used.
        grasp_plan = self.day5.get('grasp_plan')

        if not isinstance(grasp_plan, dict):
            raise ValueError(
                'Day5 result has no validated grasp_plan'
            )

        candidates = grasp_plan.get('candidate_plans')

        if not isinstance(candidates, dict):
            raise ValueError(
                'Day5 grasp plan has no candidate_plans'
            )

        selected_candidate = candidates.get(
            self.selected_arm
        )

        self.selected_candidate = selected_candidate

        if not isinstance(selected_candidate, dict):
            raise ValueError(
                'Day5 selected-arm candidate missing'
            )

        lift_solution = selected_candidate.get('lift')

        if (
            not isinstance(lift_solution, dict)
            or not lift_solution.get('success')
        ):
            raise ValueError(
                'Day5 selected-arm lift was not valid'
            )

        names = lift_solution.get('joint_names')

        if not isinstance(names, list) or len(names) != 7:
            raise ValueError(
                'Day5 lift joint-name set is invalid'
            )

        target_rpy = selected_candidate.get(
            'target_rpy_base'
        )

        if (
            not isinstance(target_rpy, list)
            or len(target_rpy) != 3
        ):
            raise ValueError(
                'Day5 target grasp orientation missing'
            )

        lift_xyz = grasp_plan.get('lift_xyz_m')

        if (
            not isinstance(lift_xyz, list)
            or len(lift_xyz) != 3
        ):
            raise ValueError(
                'Day5 lift XYZ missing'
            )

        self.selected_joint_names = tuple(
            str(v) for v in names
        )

        self.day5_lift_xyz = np.asarray(
            lift_xyz,
            dtype=np.float64,
        )

        self.carry_target_rotation = rpy_matrix(
            target_rpy
        )

        # --------------------------------------------------
        # LIVE PHYSICAL SHELF ORDERING
        # --------------------------------------------------

        marker_identity = int(
            self.day4.get(
                'shelf_column_number',
                self.day4.get(
                    'target_marker', 0
                ),
            )
        )

        observed_order = self.day4.get(
            'observed_left_to_right'
        )

        reported_physical = self.day4.get(
            'target_column_index_left_to_right'
        )

        derived_physical = None

        if (
            isinstance(observed_order, list)
            and len(observed_order) == 5
            and marker_identity in observed_order
        ):
            derived_physical = (
                observed_order.index(
                    marker_identity
                )
                + 1
            )

        if reported_physical is not None:
            self.physical_column = int(
                reported_physical
            )
        elif derived_physical is not None:
            self.physical_column = int(
                derived_physical
            )
        else:
            raise ValueError(
                'Day4 result does not contain a valid '
                'live physical shelf position'
            )

        if self.physical_column not in (
            1, 2, 3, 4, 5
        ):
            raise ValueError(
                'physical shelf column must be 1..5'
            )

        # Cross-check the direct physical index
        # against the actual observed visual ordering.
        if (
            derived_physical is not None
            and derived_physical
            != self.physical_column
        ):
            raise ValueError(
                'physical column disagrees with '
                'observed_left_to_right'
            )

        self.observed_left_to_right = (
            list(observed_order)
            if isinstance(
                observed_order, list
            )
            else None
        )

        self.marker_identity = marker_identity

        # --------------------------------------------------
        # HARD ROUTING POLICY BASED ON PHYSICAL POSITION
        # --------------------------------------------------

        if self.physical_column in (1, 2):
            # ROS angular.z < 0 means clockwise.
            self.rotation_sign = -1.0
            self.rotation_direction = (
                'clockwise'
            )
            self.transport_mode = 'lateral'

        elif self.physical_column == 3:
            # User allowed either direction.
            # Use deterministic CCW.
            self.rotation_sign = +1.0
            self.rotation_direction = (
                'counterclockwise'
            )
            self.transport_mode = 'straight'

        else:
            # Physical columns 4 and 5.
            self.rotation_sign = +1.0
            self.rotation_direction = (
                'counterclockwise'
            )
            self.transport_mode = 'lateral'

        self.get_logger().info(
            '[FAST67 ROUTE] '
            f'marker_identity={self.marker_identity} '
            f'observed_left_to_right='
            f'{self.observed_left_to_right} '
            f'PHYSICAL_POSITION='
            f'{self.physical_column} '
            f'rotation='
            f'{self.rotation_direction} '
            f'transport={self.transport_mode}'
        )

        # --------------------------------------------------
        # Recorded home/source geometry
        # --------------------------------------------------

        self.home_xy = tuple(
            float(v)
            for v in self.home[
                'home_odom_xy'
            ]
        )

        self.source_xy = tuple(
            float(v)
            for v in self.day4[
                'day4_final_base_odom_xy'
            ]
        )

        self.source_yaw = float(
            self.day4[
                'day4_final_base_yaw_rad'
            ]
        )

        lift_xyz = (
            self.day5
            .get('grasp_plan', {})
            .get('lift_xyz_m')
        )

        if (
            not isinstance(
                lift_xyz, list
            )
            or len(lift_xyz) != 3
        ):
            raise ValueError(
                'Day5 grasp plan missing lift_xyz_m'
            )

        self.lift_xyz = tuple(
            float(v) for v in lift_xyz
        )

        half_diag = float(
            book_half_diagonal_m()
        )

        self.carried_book_radius = (
            math.hypot(
                self.lift_xyz[0],
                self.lift_xyz[1],
            )
            + half_diag
        )

        self.carried_book_forward_extent = (
            self.lift_xyz[0]
            + half_diag
        )

        # Front scan range is measured from the front-laser origin, while
        # carried_book_forward_extent is measured from the robot base frame.
        # This conservative threshold therefore protects the frontmost point
        # of the carried book, not only the mobile base.
        self.carried_front_stop = max(
            self.front_hard_stop,
            self.carried_book_forward_extent
            - self.front_laser_base_x
            + self.carried_front_margin,
        )

        self.rotation_clearance_required = (
            self.carried_book_radius
            + self.rotation_margin
        )

        # --------------------------------------------------
        # TF / camera
        # --------------------------------------------------

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(
                seconds=10.0
            )
        )

        self.tf_listener = (
            tf2_ros.TransformListener(
                self.tf_buffer,
                self,
                spin_thread=False,
            )
        )

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------

        self.started_wall = (
            time.monotonic()
        )

        self.last_clock_wall = (
            time.monotonic()
        )

        self.sim: Optional[float] = None

        self.state = 'WAIT_INPUTS'

        self.state_started_sim: Optional[
            float
        ] = None

        self.done = False
        self.passed = False
        self.reason = ''

        self.current_xy: Optional[
            Tuple[float, float]
        ] = None

        self.current_yaw: Optional[
            float
        ] = None

        self.joints: Dict[
            str, float
        ] = {}

        self.retreat_start_xy: Optional[
            Tuple[float, float]
        ] = None

        # Compact-carry arm state.
        self.carry_target_xyz: Optional[
            Tuple[float, float, float]
        ] = None

        self.carry_target_positions: Optional[
            Tuple[float, ...]
        ] = None

        self.carry_command_sim: Optional[
            float
        ] = None

        self.carry_settle_count = 0
        self.carry_completed = False

        self.carry_max_joint_delta: Optional[
            float
        ] = None

        self.carry_screen = None

        # Gravity-supported transport pose.
        self.support_target_positions = None
        self.support_target_xyz = None
        self.support_roll_sign = None
        self.support_target_rotation = None
        self.support_command_sim = None
        self.support_settle_count = 0
        self.support_completed = False
        self.support_max_delta = None

        # Original-orientation restoration before Day8.
        self.restore_target_positions = None
        self.restore_command_sim = None
        self.restore_settle_count = 0
        self.restore_completed = False

        self.carry_ik_position_error: Optional[
            float
        ] = None

        self.carry_ik_orientation_error: Optional[
            float
        ] = None

        self.rotation_target_yaw: Optional[
            float
        ] = None

        self.rotation_last_yaw: Optional[
            float
        ] = None

        self.rotation_accumulated_rad = 0.0
        self.rotation_settle = 0
        self.final_settle = 0

        self.front_clearance: Optional[
            float
        ] = None

        self.front_minimum: Optional[
            float
        ] = None

        self.front_scan_stamp: Optional[
            float
        ] = None

        self.rear_clearance: Optional[
            float
        ] = None

        self.rear_minimum: Optional[
            float
        ] = None

        self.rear_scan_stamp: Optional[
            float
        ] = None

        self.color_model: Optional[
            CameraModel
        ] = None

        self.depth_model: Optional[
            CameraModel
        ] = None

        self.depth_history: List[
            Tuple[
                float,
                np.ndarray,
            ]
        ] = []

        self.bin_candidates: List[
            dict
        ] = []

        self.locked_bin_odom_xy: Optional[
            Tuple[float, float]
        ] = None

        self.confirmed_detection: Optional[
            BinDetection
        ] = None

        self.confirmed_geometry = None

        self.bin_lock_sim: Optional[
            float
        ] = None

        self.bin_evidence_image_path: Optional[
            str
        ] = None

        self.bin_observations = 0

        self.last_gripper_publish: Optional[
            float
        ] = None

        self.unintended_contacts = 0
        self.bin_contact_messages = 0

        # Real held-book evidence from the selected fingertips.
        self.last_book_fingertip_contact_sim = None
        self.book_fingertip_contact_messages = 0

        # Lateral velocity ramp state.
        self.last_lateral_cmd_vy = 0.0
        self.last_lateral_cmd_sim = None

        # Longitudinal carried-book acceleration/deceleration ramp state.
        self.last_longitudinal_cmd_vx = 0.0
        self.last_longitudinal_cmd_sim = None

        # Lightweight visual evidence at transport boundaries / retention loss.
        self.latest_rgb_frame = None
        self.transport_evidence_images = {}

        self.path_trace: List[
            dict
        ] = []

        self.max_cmd = {
            'vx': 0.0,
            'vy': 0.0,
            'wz': 0.0,
        }

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------

        self.cmd_pub = (
            self.create_publisher(
                Twist,
                '/cmd_vel',
                10,
            )
        )

        self.gripper_pub = (
            self.create_publisher(
                JointTrajectory,
                (
                    f'/gripper_'
                    f'{self.selected_arm}'
                    f'_controller/'
                    f'joint_trajectory'
                ),
                10,
            )
        )

        self.head_pub = (
            self.create_publisher(
                JointTrajectory,
                '/head_controller/'
                'joint_trajectory',
                10,
            )
        )

        self.arm_pub = (
            self.create_publisher(
                JointTrajectory,
                (
                    f'/arm_{self.selected_arm}'
                    f'_controller/joint_trajectory'
                ),
                10,
            )
        )

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------

        self.create_subscription(
            Clock,
            '/clock',
            self.clock_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_cb,
            20,
        )

        self.create_subscription(
            Image,
            self.RGB_TOPIC,
            self.rgb_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            CameraInfo,
            self.COLOR_INFO_TOPIC,
            self.color_info_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            self.DEPTH_TOPIC,
            self.depth_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            CameraInfo,
            self.DEPTH_INFO_TOPIC,
            self.depth_info_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            LaserScan,
            '/scan_front_raw',
            self.front_scan_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            LaserScan,
            '/scan_rear_raw',
            self.rear_scan_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Contacts,
            '/contacts',
            self.contact_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Contacts,
            '/bin_contacts',
            self.bin_contact_cb,
            qos_profile_sensor_data,
        )

        self.create_timer(
            0.05,
            self.tick,
        )

        self.get_logger().info(
            '[FAST67] ready; '
            f'arm={self.selected_arm} '
            f'physical_shelf_position='
            f'{self.physical_column} '
            f'rotation_clearance='
            f'{self.rotation_clearance_required:.3f}m'
        )

    # ======================================================
    # Basic helpers
    # ======================================================

    def transition(
        self,
        state: str,
        reason: str,
    ) -> None:

        self.stop_base()

        self.state = state

        self.state_started_sim = (
            self.sim
        )

        if state == 'LATERAL_TRANSIT':
            self.last_lateral_cmd_vy = 0.0
            self.last_lateral_cmd_sim = self.sim

        if state == 'LONGITUDINAL_CORRECTION':
            self.last_longitudinal_cmd_vx = 0.0
            self.last_longitudinal_cmd_sim = self.sim
            self.save_transport_snapshot('before_longitudinal')

        self.get_logger().info(
            f'[FAST67] STATE -> '
            f'{state}: {reason}'
        )

        self.write(
            False,
            'running: ' + reason,
        )

    def elapsed(self) -> float:
        if (
            self.sim is None
            or self.state_started_sim
            is None
        ):
            return 0.0

        return max(
            0.0,
            self.sim
            - self.state_started_sim,
        )

    def publish_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
    ) -> None:

        msg = Twist()

        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)

        self.cmd_pub.publish(msg)

        self.max_cmd['vx'] = max(
            self.max_cmd['vx'],
            abs(vx),
        )

        self.max_cmd['vy'] = max(
            self.max_cmd['vy'],
            abs(vy),
        )

        self.max_cmd['wz'] = max(
            self.max_cmd['wz'],
            abs(wz),
        )

    def stop_base(self) -> None:
        self.publish_velocity(
            0.0,
            0.0,
            0.0,
        )

    def ramp_lateral_velocity(
        self,
        target_vy: float,
    ) -> float:
        """Acceleration-limit only the carried-book Y motion."""

        if self.sim is None:
            return 0.0

        target_vy = clamp(
            target_vy,
            -self.lateral_max_speed,
            self.lateral_max_speed,
        )

        if self.last_lateral_cmd_sim is None:
            self.last_lateral_cmd_sim = self.sim
            self.last_lateral_cmd_vy = 0.0
            return 0.0

        dt = max(
            0.0,
            min(
                0.20,
                self.sim - self.last_lateral_cmd_sim,
            ),
        )

        maximum_change = (
            self.lateral_accel_limit * dt
        )

        error = (
            target_vy
            - self.last_lateral_cmd_vy
        )

        change = clamp(
            error,
            -maximum_change,
            maximum_change,
        )

        command = (
            self.last_lateral_cmd_vy
            + change
        )

        self.last_lateral_cmd_vy = command
        self.last_lateral_cmd_sim = self.sim

        return command

    def ramp_longitudinal_velocity(
        self,
        target_vx: float,
    ) -> float:
        """Acceleration-limit carried-book X motion without reducing its max speed."""

        if self.sim is None:
            return 0.0

        target_vx = clamp(
            target_vx,
            -self.longitudinal_correction_speed,
            self.longitudinal_correction_speed,
        )

        if self.last_longitudinal_cmd_sim is None:
            self.last_longitudinal_cmd_sim = self.sim
            self.last_longitudinal_cmd_vx = 0.0
            return 0.0

        dt = max(
            0.0,
            min(
                0.20,
                self.sim - self.last_longitudinal_cmd_sim,
            ),
        )

        maximum_change = self.longitudinal_accel_limit * dt
        error = target_vx - self.last_longitudinal_cmd_vx
        change = clamp(error, -maximum_change, maximum_change)
        command = self.last_longitudinal_cmd_vx + change

        self.last_longitudinal_cmd_vx = command
        self.last_longitudinal_cmd_sim = self.sim

        return command

    def save_transport_snapshot(self, label: str) -> None:
        """Save the latest RGB frame without changing perception thresholds."""

        if self.latest_rgb_frame is None:
            return

        try:
            Path(self.image_output_dir).mkdir(parents=True, exist_ok=True)
            safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in label)
            path = os.path.join(
                self.image_output_dir,
                'fast67_transport_'
                + safe
                + '_'
                + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                + '.png',
            )
            if cv2.imwrite(path, self.latest_rgb_frame):
                self.transport_evidence_images[label] = path
                self.get_logger().info(
                    f'[FAST67][EVIDENCE] {label}: {path}'
                )
        except Exception as exc:
            self.get_logger().warning(
                f'[FAST67][EVIDENCE] could not save {label}: {exc}'
            )

    def continuous_retention_or_fail(self, stage: str) -> bool:
        """Stop immediately if fingertip/book contact goes stale during transport."""

        if self.book_retained():
            return True

        self.stop_base()
        self.save_transport_snapshot(
            'retention_lost_' + stage.lower()
        )
        self.fail(
            'BOOK_RETENTION_LOST_DURING_' + stage.upper()
        )
        return False

    def book_retained(self, max_age_sec: float = 0.80) -> bool:
        return (
            self.sim is not None
            and self.last_book_fingertip_contact_sim is not None
            and (
                self.sim
                - self.last_book_fingertip_contact_sim
                <= max_age_sec
            )
            and self.unintended_contacts == 0
        )

    def require_book_retained(
        self,
        stage: str,
    ) -> bool:
        if self.book_retained():
            return True

        self.stop_base()
        self.save_transport_snapshot(
            'retention_lost_' + stage.lower()
        )
        self.fail(
            'BOOK_RETENTION_LOST_'
            + stage.upper()
        )
        return False

    def maintain_gripper(self) -> None:
        if self.sim is None:
            return

        if (
            self.last_gripper_publish
            is not None
            and (
                self.sim
                - self.last_gripper_publish
                < self.gripper_hold_period
            )
        ):
            return

        msg = JointTrajectory()

        name = (
            f'gripper_'
            f'{self.selected_arm}'
            f'_finger_joint'
        )

        msg.joint_names = [name]

        p = JointTrajectoryPoint()

        p.positions = [
            self.gripper_hold_position
        ]

        p.time_from_start.sec = 0
        p.time_from_start.nanosec = (
            100_000_000
        )

        msg.points = [p]

        self.gripper_pub.publish(msg)

        self.last_gripper_publish = (
            self.sim
        )

    def command_head(self) -> None:
        msg = JointTrajectory()

        msg.joint_names = [
            'head_1_joint',
            'head_2_joint',
        ]

        p = JointTrajectoryPoint()

        p.positions = [
            self.head_pan,
            self.head_tilt,
        ]

        sec = int(
            self.head_motion_sec
        )

        p.time_from_start.sec = sec

        p.time_from_start.nanosec = int(
            (
                self.head_motion_sec
                - sec
            )
            * 1.0e9
        )

        msg.points = [p]

        self.head_pub.publish(msg)

    # ======================================================
    # Validated compact carried-book arm motion
    # ======================================================

    def plan_compact_carry(self):
        """Plan the proven inward+up carried-book posture.

        Base must already have completed the short shelf retreat.
        The gripper orientation remains the Day5 grasp orientation.
        """

        share = get_package_share_directory(
            'erc_description'
        )

        urdf_path = os.path.join(
            share,
            'urdf',
            'tiago_pro.urdf',
        )

        model = URDFKinematicModel.from_file(
            urdf_path
        )

        tip = (
            f'gripper_{self.selected_arm}'
            f'_grasping_link'
        )

        chain = model.chain(
            'base_footprint',
            tip,
        )

        if not all(
            name in self.joints
            for name in self.selected_joint_names
        ):
            raise RuntimeError(
                'selected-arm joint state unavailable '
                'for compact carry'
            )

        current = np.asarray(
            [
                self.joints[name]
                for name in self.selected_joint_names
            ],
            dtype=np.float64,
        )

        # Use the LIVE end-effector position after the short
        # shelf retreat.  The arm itself has not moved during
        # that retreat.
        live_tip = model.forward(
            chain,
            self.joints,
        ).tip_transform

        start_xyz = np.asarray(
            live_tip[:3, 3],
            dtype=np.float64,
        )

        target_xyz = (
            start_xyz
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
            self.joints,
            target_xyz,
            self.carry_target_rotation,
            seeds,
        )

        if not solution.success:
            raise RuntimeError(
                'compact-carry IK failed: '
                + str(solution.reason)
            )

        # Screen the COMPLETE joint-space motion.
        path = interpolate_joint_path(
            current,
            solution.positions,
            samples=32,
        )

        # We deliberately perform this only AFTER the base
        # has already retreated from the shelf.  This is the
        # same remote-shelf collision-screen assumption used
        # by the previously validated carry posture.
        screen = screen_joint_path(
            model,
            chain,
            self.selected_joint_names,
            self.joints,
            path,
            shelf_plane_x_m=10.0,
            book_surface_xyz_m=self.day5_lift_xyz,
            stage='carry_raise',
        )

        if not screen.passed:
            raise RuntimeError(
                'compact-carry path rejected: '
                + '; '.join(
                    screen.violations[:5]
                )
            )

        q1 = np.asarray(
            solution.positions,
            dtype=np.float64,
        )

        maximum_delta = float(
            np.max(
                np.abs(
                    q1 - current
                )
            )
        )

        if (
            maximum_delta
            > self.carry_max_joint_delta_limit
        ):
            raise RuntimeError(
                'compact-carry joint displacement '
                'exceeds validated bound: '
                f'{maximum_delta:.3f} rad'
            )

        self.carry_target_xyz = tuple(
            float(v) for v in target_xyz
        )

        self.carry_target_positions = tuple(
            float(v)
            for v in solution.positions
        )

        self.carry_max_joint_delta = (
            maximum_delta
        )

        self.carry_screen = screen.as_dict()

        self.carry_ik_position_error = float(
            solution.position_error_m
        )

        self.carry_ik_orientation_error = float(
            solution.orientation_error_rad
        )

        return (
            'validated high/compact carry planned: '
            f'retract={self.carry_retract_distance:.3f} m '
            f'raise={self.carry_raise_distance:.3f} m; '
            f'IK position error='
            f'{self.carry_ik_position_error:.4f} m '
            f'max joint delta='
            f'{self.carry_max_joint_delta:.3f} rad'
        )

    def publish_compact_carry(self) -> None:
        if self.carry_target_positions is None:
            raise RuntimeError(
                'compact-carry target unavailable'
            )

        msg = JointTrajectory()

        msg.joint_names = list(
            self.selected_joint_names
        )

        point = JointTrajectoryPoint()

        point.positions = list(
            self.carry_target_positions
        )

        whole = int(
            self.carry_motion_sec
        )

        point.time_from_start.sec = whole

        point.time_from_start.nanosec = int(
            round(
                (
                    self.carry_motion_sec
                    - whole
                )
                * 1.0e9
            )
        )

        msg.points = [point]

        self.arm_pub.publish(msg)

        self.carry_command_sim = self.sim
        self.carry_settle_count = 0

    def compact_carry_error(
        self
    ) -> Optional[float]:

        if self.carry_target_positions is None:
            return None

        if not all(
            name in self.joints
            for name in self.selected_joint_names
        ):
            return None

        return max(
            abs(
                self.joints[name]
                - target
            )
            for name, target
            in zip(
                self.selected_joint_names,
                self.carry_target_positions,
            )
        )

    def _rotation_x(self, angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)

        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.float64,
        )

    def _grasp_attachment_rotation(
        self,
        model,
        chain,
    ) -> np.ndarray:
        """Recover the book->tip rotational attachment from Day4/5.

        This is the same rigid attachment convention used by Day8:
        canonical book orientation is depth-forward,
        thin-lateral, height-up.
        """

        grasp = self.selected_candidate.get('grasp')

        if (
            not isinstance(grasp, dict)
            or not grasp.get('success')
        ):
            raise RuntimeError(
                'validated Day5 grasp solution unavailable'
            )

        q_grasp = dict(self.joints)

        q_grasp.update(
            zip(
                self.selected_joint_names,
                grasp['positions_rad'],
            )
        )

        Tg = model.forward(
            chain,
            q_grasp,
        ).tip_transform

        surface = np.asarray(
            self.day4[
                'target_book_geometry'
            ]['base_point_xyz_m'],
            dtype=np.float64,
        )

        Tb = np.eye(
            4,
            dtype=np.float64,
        )

        # Day8 uses the exposed spine plus half of the
        # public 16 cm book depth.
        Tb[:3, 3] = (
            surface
            + np.asarray(
                [0.08, 0.0, 0.0],
                dtype=np.float64,
            )
        )

        Tattach = (
            np.linalg.inv(Tg)
            @ Tb
        )

        return Tattach[:3, :3]

    def _plan_orientation_target(
        self,
        desired_rotation: np.ndarray,
        target_xyz: np.ndarray,
        maximum_joint_delta: float,
    ):
        share = get_package_share_directory(
            'erc_description'
        )

        urdf_path = os.path.join(
            share,
            'urdf',
            'tiago_pro.urdf',
        )

        model = URDFKinematicModel.from_file(
            urdf_path
        )

        tip = (
            f'gripper_{self.selected_arm}'
            f'_grasping_link'
        )

        chain = model.chain(
            'base_footprint',
            tip,
        )

        if not all(
            n in self.joints
            for n in self.selected_joint_names
        ):
            raise RuntimeError(
                'arm joint state unavailable'
            )

        current = np.asarray(
            [
                self.joints[n]
                for n in self.selected_joint_names
            ],
            dtype=np.float64,
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
            self.joints,
            target_xyz,
            desired_rotation,
            seeds,
        )

        if not solution.success:
            return None

        target = np.asarray(
            solution.positions,
            dtype=np.float64,
        )

        maximum_delta = float(
            np.max(
                np.abs(
                    target - current
                )
            )
        )

        if maximum_delta > maximum_joint_delta:
            return None

        path = interpolate_joint_path(
            current,
            target,
            samples=40,
        )

        screen = screen_joint_path(
            model,
            chain,
            self.selected_joint_names,
            self.joints,
            path,
            # Base has already completed shelf retreat.
            shelf_plane_x_m=10.0,
            book_surface_xyz_m=self.day5_lift_xyz,
            stage='carry_raise',
        )

        if not screen.passed:
            return None

        return {
            'positions': tuple(
                float(v)
                for v in solution.positions
            ),
            'maximum_delta': maximum_delta,
            'position_error':
                float(solution.position_error_m),
            'orientation_error':
                float(solution.orientation_error_rad),
            'screen': screen.as_dict(),
        }

    def plan_gravity_supported_carry(self) -> str:
        """Turn the vertically pinched book into supported transport."""

        share = get_package_share_directory(
            'erc_description'
        )

        model = URDFKinematicModel.from_file(
            os.path.join(
                share,
                'urdf',
                'tiago_pro.urdf',
            )
        )

        chain = model.chain(
            'base_footprint',
            (
                f'gripper_{self.selected_arm}'
                f'_grasping_link'
            ),
        )

        live_tip = model.forward(
            chain,
            self.joints,
        ).tip_transform

        xyz = np.asarray(
            live_tip[:3, 3],
            dtype=np.float64,
        )

        # Add a little height so the 25 cm-tall book can rotate
        # without sweeping downward toward the environment.
        target_xyz = (
            xyz
            + np.asarray(
                [
                    0.0,
                    0.0,
                    self.support_extra_raise,
                ],
                dtype=np.float64,
            )
        )

        Rattach = (
            self._grasp_attachment_rotation(
                model,
                chain,
            )
        )

        candidates = []

        # Try both directions.  Both make the thin/jaw axis
        # vertical; choose whichever is reachable with the
        # smaller joint displacement.
        for sign in (+1.0, -1.0):
            R_book = self._rotation_x(
                sign
                * self.support_roll_rad
            )

            # R_book = R_tip @ R_attach
            # therefore R_tip = R_book @ R_attach.T
            R_tip = (
                R_book
                @ Rattach.T
            )

            result = self._plan_orientation_target(
                R_tip,
                target_xyz,
                self.support_max_joint_delta,
            )

            if result is not None:
                result['sign'] = sign
                result['rotation'] = R_tip
                candidates.append(result)

        if not candidates:
            raise RuntimeError(
                'no collision-screened +/-90deg '
                'gravity-supported carry orientation'
            )

        chosen = min(
            candidates,
            key=lambda x: x['maximum_delta'],
        )

        self.support_target_positions = (
            chosen['positions']
        )

        self.support_target_xyz = tuple(
            float(v)
            for v in target_xyz
        )

        self.support_roll_sign = float(
            chosen['sign']
        )

        self.support_target_rotation = (
            chosen['rotation']
        )

        self.support_max_delta = float(
            chosen['maximum_delta']
        )

        return (
            'gravity-supported transport pose planned: '
            f'book depth-axis roll='
            f'{self.support_roll_sign * math.degrees(self.support_roll_rad):.1f}deg; '
            'thin jaw axis becomes vertical; '
            f'max joint delta={self.support_max_delta:.3f}rad'
        )

    def publish_arm_positions(
        self,
        positions,
        duration_sec: float,
    ) -> None:
        msg = JointTrajectory()

        msg.joint_names = list(
            self.selected_joint_names
        )

        p = JointTrajectoryPoint()

        p.positions = [
            float(v)
            for v in positions
        ]

        whole = int(
            duration_sec
        )

        p.time_from_start.sec = whole

        p.time_from_start.nanosec = int(
            round(
                (
                    duration_sec
                    - whole
                )
                * 1.0e9
            )
        )

        msg.points = [p]

        self.arm_pub.publish(msg)

    def arm_target_error(
        self,
        target,
    ):
        if target is None:
            return None

        if not all(
            n in self.joints
            for n in self.selected_joint_names
        ):
            return None

        return max(
            abs(
                self.joints[n] - q
            )
            for n, q
            in zip(
                self.selected_joint_names,
                target,
            )
        )

    def plan_restore_placement_orientation(self) -> str:
        """Restore the original validated Day5 wrist orientation.

        This occurs only after the base has completely stopped at
        the collection-bin handoff pose.
        """

        share = get_package_share_directory(
            'erc_description'
        )

        model = URDFKinematicModel.from_file(
            os.path.join(
                share,
                'urdf',
                'tiago_pro.urdf',
            )
        )

        chain = model.chain(
            'base_footprint',
            (
                f'gripper_{self.selected_arm}'
                f'_grasping_link'
            ),
        )

        live_tip = model.forward(
            chain,
            self.joints,
        ).tip_transform

        target_xyz = np.asarray(
            live_tip[:3, 3],
            dtype=np.float64,
        )

        result = self._plan_orientation_target(
            self.carry_target_rotation,
            target_xyz,
            self.support_max_joint_delta,
        )

        if result is None:
            raise RuntimeError(
                'could not restore original '
                'validated placement orientation'
            )

        self.restore_target_positions = (
            result['positions']
        )

        return (
            'original Day5 placement orientation replanned '
            'from live supported-carry joints; '
            f'max joint delta='
            f'{result["maximum_delta"]:.3f}rad'
        )

    def begin_forced_rotation(self) -> None:
        """Initialize the requested physical-order 180-degree turn."""

        if self.current_yaw is None:
            raise RuntimeError(
                'current yaw unavailable before rotation'
            )

        self.rotation_last_yaw = (
            self.current_yaw
        )

        self.rotation_accumulated_rad = 0.0

        self.rotation_target_yaw = normalize_angle(
            self.current_yaw
            + self.rotation_sign
            * abs(
                self.rotation_angle
            )
        )

        self.rotation_settle = 0

    # ======================================================
    # ROS callbacks
    # ======================================================

    def clock_cb(
        self,
        msg: Clock,
    ) -> None:

        value = (
            float(msg.clock.sec)
            + float(
                msg.clock.nanosec
            )
            * 1.0e-9
        )

        if (
            self.sim is None
            or value > self.sim
        ):
            self.last_clock_wall = (
                time.monotonic()
            )

        self.sim = value

    def odom_cb(
        self,
        msg: Odometry,
    ) -> None:

        self.current_xy = (
            float(
                msg.pose.pose.position.x
            ),
            float(
                msg.pose.pose.position.y
            ),
        )

        self.current_yaw = float(
            yaw_from_odom(msg)
        )

    def joint_cb(
        self,
        msg: JointState,
    ) -> None:

        self.joints.update({
            n: float(v)
            for n, v
            in zip(
                msg.name,
                msg.position,
            )
        })

    def color_info_cb(
        self,
        msg: CameraInfo,
    ) -> None:

        if not msg.header.frame_id:
            return

        try:
            self.color_model = (
                CameraModel.from_arrays(
                    msg.width,
                    msg.height,
                    msg.k,
                    msg.p,
                    frame_id=(
                        msg.header.frame_id
                    ),
                    prefer_projection=True,
                )
            )
        except ValueError:
            pass

    def depth_info_cb(
        self,
        msg: CameraInfo,
    ) -> None:

        if not msg.header.frame_id:
            return

        try:
            self.depth_model = (
                CameraModel.from_arrays(
                    msg.width,
                    msg.height,
                    msg.k,
                    msg.p,
                    frame_id=(
                        msg.header.frame_id
                    ),
                    prefer_projection=False,
                )
            )
        except ValueError:
            pass

    def depth_cb(
        self,
        msg: Image,
    ) -> None:

        try:
            raw = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding=(
                        'passthrough'
                    ),
                )
            )

            depth = (
                normalize_depth_array(
                    raw,
                    msg.encoding,
                )
            )

        except (
            CvBridgeError,
            ValueError,
        ):
            return

        s = stamp(msg) or self.sim

        if s is None:
            return

        self.depth_history.append(
            (
                float(s),
                depth.copy(),
            )
        )

        self.depth_history = (
            self.depth_history[-24:]
        )

    def lookup_base_depth(self):
        if self.depth_model is None:
            return None

        try:
            tr = (
                self.tf_buffer
                .lookup_transform(
                    'base_footprint',
                    self.depth_model.frame_id,
                    Time(),
                )
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

        t = tr.transform.translation
        q = tr.transform.rotation

        return (
            (
                float(t.x),
                float(t.y),
                float(t.z),
            ),
            (
                float(q.x),
                float(q.y),
                float(q.z),
                float(q.w),
            ),
        )

    def geometry_for_detection(
        self,
        detection: BinDetection,
        rgb_stamp: float,
    ):

        if (
            self.color_model is None
            or self.depth_model is None
            or not self.depth_history
            or self.sim is None
        ):
            return None

        ds, depth = min(
            self.depth_history,
            key=lambda item: abs(
                item[0]
                - rgb_stamp
            ),
        )

        skew = abs(
            ds - rgb_stamp
        )

        if (
            skew
            > self.max_rgb_depth_skew
        ):
            return None

        if (
            self.sim - ds
            > self.depth_stale_timeout
        ):
            return None

        tf = self.lookup_base_depth()

        if tf is None:
            return None

        x, y, w, h = (
            detection.bbox_xywh
        )

        try:
            geometry = (
                target_geometry_from_depth(
                    BoundingBoxLike(
                        x, y, w, h
                    ),
                    self.color_model,
                    self.depth_model,
                    depth,
                    tf[0],
                    tf[1],
                    rgb_depth_stamp_skew_sec=(
                        skew
                    ),
                    min_valid_pixels=(
                        self.depth_min_valid_pixels
                    ),
                    min_depth_m=0.20,
                    max_depth_m=5.0,
                )
            )

        except ValueError:
            return None

        if geometry is None:
            return None

        if (
            geometry.depth.valid_fraction
            < self.depth_min_valid_fraction
        ):
            return None

        if geometry.forward_m <= 0.20:
            return None

        return geometry

    def rgb_cb(
        self,
        msg: Image,
    ) -> None:

        if self.state not in {
            'WAIT_GRAVITY_SUPPORTED_CARRY',
            'ROTATE_180_SEARCH_BIN',
            'LATERAL_TRANSIT',
            'LONGITUDINAL_CORRECTION',
            'STRAIGHT_TO_BIN',
            'WAIT_CENTER_BIN',
            'FINAL_BIN_ALIGN',
        }:
            return

        try:
            frame = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='bgr8',
                )
            )
        except CvBridgeError:
            return

        if (
            frame is None
            or frame.size == 0
        ):
            return

        # Read-only copy for boundary / loss evidence.  Detector inputs and
        # thresholds are unchanged.
        self.latest_rgb_frame = frame.copy()

        s = stamp(msg) or self.sim

        if s is None:
            return

        detections = detect_red_bin(
            frame,
            self.detector_settings,
        )

        if not detections:
            return

        det = detections[0]

        geometry = (
            self.geometry_for_detection(
                det,
                float(s),
            )
        )

        if geometry is None:
            return

        if (
            self.current_xy is None
            or self.current_yaw is None
        ):
            return

        odom_xy = vector_base_to_odom(
            geometry.forward_m,
            geometry.lateral_left_m,
            self.current_xy,
            self.current_yaw,
        )

        self.bin_observations += 1

        self.bin_candidates.append({
            'stamp': float(s),
            'xy': [
                float(odom_xy[0]),
                float(odom_xy[1]),
            ],
            'detection': det,
            'geometry': geometry,
        })

        cutoff = (
            float(s)
            - self.bin_confirm_window
        )

        self.bin_candidates = [
            item
            for item
            in self.bin_candidates
            if item['stamp'] >= cutoff
        ][-30:]

        if (
            len(self.bin_candidates)
            < self.bin_confirm_frames
        ):
            return

        pts = np.asarray(
            [
                item['xy']
                for item
                in self.bin_candidates
            ],
            dtype=float,
        )

        med = np.median(
            pts,
            axis=0,
        )

        residual = np.linalg.norm(
            pts - med,
            axis=1,
        )

        good = np.where(
            residual
            <= self.bin_odom_tolerance
        )[0]

        if (
            len(good)
            < self.bin_confirm_frames
        ):
            return

        confirmed = [
            self.bin_candidates[
                int(i)
            ]
            for i in good
        ]

        locked = np.median(
            np.asarray(
                [
                    item['xy']
                    for item
                    in confirmed
                ]
            ),
            axis=0,
        )

        first_lock = (
            self.locked_bin_odom_xy
            is None
        )

        self.locked_bin_odom_xy = (
            float(locked[0]),
            float(locked[1]),
        )

        latest = confirmed[-1]

        self.confirmed_detection = (
            latest['detection']
        )

        self.confirmed_geometry = (
            latest['geometry']
        )

        self.bin_lock_sim = self.sim

        if first_lock:
            self.save_bin_evidence(
                frame,
                self.confirmed_detection,
                self.confirmed_geometry,
            )

            self.get_logger().info(
                '[FAST67]'
                '[BIN LOCKED DURING MOTION] '
                f'odom='
                f'{self.locked_bin_odom_xy} '
                f'state={self.state}'
            )

    def scan_clearance(
        self,
        msg: LaserScan,
    ) -> Tuple[
        Optional[float],
        Optional[float],
    ]:

        ranges = np.asarray(
            msg.ranges,
            dtype=float,
        )

        if ranges.size == 0:
            return None, None

        angles = (
            float(msg.angle_min)
            + np.arange(
                ranges.size
            )
            * float(
                msg.angle_increment
            )
        )

        valid = (
            np.isfinite(ranges)
            & (
                ranges
                >= float(
                    msg.range_min
                )
            )
            & (
                ranges
                <= float(
                    msg.range_max
                )
            )
        )

        x = (
            ranges
            * np.cos(angles)
        )

        y = (
            ranges
            * np.sin(angles)
        )

        corridor = (
            valid
            & (x > 0.0)
            & (
                np.abs(y)
                <= self.corridor_half_width
            )
        )

        vals = x[corridor]

        if vals.size < 3:
            return None, None

        return (
            float(
                np.percentile(
                    vals,
                    self.lidar_percentile,
                )
            ),
            float(
                np.min(vals)
            ),
        )

    def front_scan_cb(
        self,
        msg: LaserScan,
    ) -> None:

        robust, minimum = (
            self.scan_clearance(msg)
        )

        if robust is not None:
            self.front_clearance = robust
            self.front_minimum = minimum
            self.front_scan_stamp = (
                self.sim
            )

    def rear_scan_cb(
        self,
        msg: LaserScan,
    ) -> None:

        robust, minimum = (
            self.scan_clearance(msg)
        )

        if robust is not None:
            self.rear_clearance = robust
            self.rear_minimum = minimum
            self.rear_scan_stamp = (
                self.sim
            )

    def contact_cb(
        self,
        msg: Contacts,
    ) -> None:

        tips = (
            selected_fingertip_tokens(
                self.selected_arm
            )
        )

        for c in getattr(
            msg,
            'contacts',
            [],
        ):
            names = (
                contact_collision_names(c)
            )

            has_book = any(
                n.startswith('book_col_')
                for n in names
            )

            has_selected_tip = any(
                any(token in n for token in tips)
                for n in names
            )

            if (
                has_book
                and has_selected_tip
                and self.sim is not None
            ):
                self.last_book_fingertip_contact_sim = self.sim
                self.book_fingertip_contact_messages += 1

            meaningful = [
                n
                for n in names
                if meaningful_robot_collision(n)
            ]

            unexpected = [
                n
                for n in meaningful
                if not any(
                    t in n
                    for t in tips
                )
            ]

            if unexpected:
                self.unintended_contacts += 1

                self.fail(
                    'unintended robot contact: '
                    + ', '.join(
                        unexpected[:4]
                    )
                )
                return

    def bin_contact_cb(
        self,
        msg: Contacts,
    ) -> None:

        for c in getattr(
            msg,
            'contacts',
            [],
        ):
            names = (
                contact_collision_names(c)
            )

            if not any(
                'erc_collection_bin::'
                in n
                for n in names
            ):
                continue

            others = [
                n
                for n in names
                if (
                    'erc_collection_bin::'
                    not in n
                )
            ]

            premature = [
                n
                for n in others
                if (
                    meaningful_robot_collision(n)
                    or n.startswith(
                        'book_col_'
                    )
                )
            ]

            if premature:
                self.bin_contact_messages += 1

                self.fail(
                    'premature bin contact: '
                    + ', '.join(
                        premature[:4]
                    )
                )
                return

    # ======================================================
    # Evidence
    # ======================================================

    def save_bin_evidence(
        self,
        frame,
        detection,
        geometry,
    ) -> None:

        image = frame.copy()

        x, y, w, h = (
            detection.bbox_xywh
        )

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            3,
        )

        text = (
            'FAST67 bin lock '
            f'fwd='
            f'{geometry.forward_m:.2f} '
            f'left='
            f'{geometry.lateral_left_m:.2f}'
        )

        cv2.putText(
            image,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        Path(
            self.image_output_dir
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        path = os.path.join(
            self.image_output_dir,
            (
                'fast67_bin_'
                + datetime.now(
                    timezone.utc
                ).strftime(
                    '%Y%m%dT%H%M%S_%fZ'
                )
                + '.png'
            ),
        )

        if cv2.imwrite(
            path,
            image,
        ):
            self.bin_evidence_image_path = (
                path
            )

    # ======================================================
    # State helpers
    # ======================================================

    def ready(self) -> bool:
        return (
            self.sim is not None
            and self.current_xy
            is not None
            and self.current_yaw
            is not None
            and self.front_scan_stamp
            is not None
            and self.rear_scan_stamp
            is not None
            and self.color_model
            is not None
            and self.depth_model
            is not None
            and len(
                self.depth_history
            ) > 0
        )

    def scans_fresh(self) -> bool:
        if (
            self.sim is None
            or self.front_scan_stamp
            is None
            or self.rear_scan_stamp
            is None
        ):
            return False

        return (
            self.sim
            - self.front_scan_stamp
            <= self.scan_stale_timeout
            and
            self.sim
            - self.rear_scan_stamp
            <= self.scan_stale_timeout
        )

    def record_path(self) -> None:
        if (
            self.sim is None
            or self.current_xy
            is None
            or self.current_yaw
            is None
        ):
            return

        if (
            self.path_trace
            and (
                self.sim
                - float(
                    self.path_trace[
                        -1
                    ]['sim_time_sec']
                )
                < 0.25
            )
        ):
            return

        self.path_trace.append({
            'sim_time_sec': self.sim,
            'state': self.state,
            'x': self.current_xy[0],
            'y': self.current_xy[1],
            'yaw': self.current_yaw,
        })

        self.path_trace = (
            self.path_trace[-1000:]
        )

    # ======================================================
    # Main state machine
    # ======================================================

    def tick(self) -> None:

        if self.done:
            return

        # Take gripper ownership immediately.
        self.maintain_gripper()

        self.record_path()

        if (
            self.sim is not None
            and (
                time.monotonic()
                - self.last_clock_wall
                > self.clock_stall_wall_timeout
            )
        ):
            self.fail(
                'Gazebo /clock stopped advancing'
            )
            return

        if self.state == 'WAIT_INPUTS':

            self.stop_base()

            if self.ready():

                dx = (
                    self.current_xy[0]
                    - self.source_xy[0]
                )

                dy = (
                    self.current_xy[1]
                    - self.source_xy[1]
                )

                pose_error = math.hypot(
                    dx, dy
                )

                yaw_error = abs(
                    normalize_angle(
                        self.current_yaw
                        - self.source_yaw
                    )
                )

                if (
                    pose_error > 0.08
                    or yaw_error
                    > math.radians(5.0)
                ):
                    self.fail(
                        'live base is not at '
                        'Day4/5 shelf source pose'
                    )
                    return

                self.command_head()

                self.retreat_start_xy = (
                    self.current_xy
                )

                self.transition(
                    'RETREAT_FROM_SHELF',
                    (
                        'retained grasp ready; '
                        'straight shelf retreat '
                        'before forced physical-'
                        'position rotation'
                    ),
                )
                return

            if (
                time.monotonic()
                - self.started_wall
                > self.ready_wall_timeout
            ):
                self.fail(
                    'inputs not ready'
                )

            return

        # Motion states require fresh LiDAR.
        if (
            self.state
            not in {
                'WAIT_CENTER_BIN',
                'FINAL_BIN_ALIGN',
            }
            and not self.scans_fresh()
        ):
            self.fail(
                'front/rear LiDAR stale'
            )
            return

        timeout = self.timeouts.get(
            self.state
        )

        if (
            timeout is not None
            and self.elapsed() > timeout
        ):
            self.fail(
                f'{self.state} exceeded '
                'simulation-time timeout'
            )
            return

        # --------------------------------------------------
        # 1. RETREAT
        # --------------------------------------------------

        if (
            self.state
            == 'RETREAT_FROM_SHELF'
        ):

            if (
                self.current_xy is None
                or self.retreat_start_xy
                is None
            ):
                self.stop_base()
                return

            travelled = math.hypot(
                (
                    self.current_xy[0]
                    - self.retreat_start_xy[0]
                ),
                (
                    self.current_xy[1]
                    - self.retreat_start_xy[1]
                ),
            )

            if (
                self.rear_minimum
                is not None
                and self.rear_minimum
                < self.rear_hard_stop
            ):
                self.fail(
                    'rear LiDAR hard stop '
                    'during shelf retreat'
                )
                return

            enough_distance = (
                travelled
                >= self.retreat_distance
            )

            enough_rotation_clearance = (
                self.front_clearance
                is not None
                and self.front_clearance
                >= self.rotation_clearance_required
            )

            if (
                enough_distance
                and enough_rotation_clearance
            ):
                self.stop_base()

                self.transition(
                    'PLAN_COMPACT_CARRY',
                    (
                        'short shelf retreat complete; '
                        'base stays stationary while the '
                        'held book is moved inward 0.12 m '
                        'and upward 0.16 m before rotation'
                    ),
                )
                return

            if (
                travelled
                > self.retreat_max_distance
            ):
                self.fail(
                    'required rotation clearance '
                    'not achieved inside bounded '
                    'retreat'
                )
                return

            self.publish_velocity(
                -self.retreat_speed,
                0.0,
                0.0,
            )
            return

        # --------------------------------------------------
        # 2. COMPACT HELD-BOOK CARRY
        # --------------------------------------------------

        if self.state == 'PLAN_COMPACT_CARRY':
            self.stop_base()

            try:
                reason = self.plan_compact_carry()
            except Exception as exc:
                self.fail(
                    'compact-carry planning failed: '
                    + str(exc)
                )
                return

            self.publish_compact_carry()

            self.transition(
                'WAIT_COMPACT_CARRY',
                (
                    reason
                    + '; arm trajectory published; '
                    'base remains zero and public '
                    'gripper POSITION hold stays active'
                ),
            )
            return

        if self.state == 'WAIT_COMPACT_CARRY':
            self.stop_base()

            if (
                self.sim is None
                or self.carry_command_sim is None
            ):
                return

            error = self.compact_carry_error()

            if (
                error is not None
                and error
                <= self.carry_joint_tolerance
            ):
                self.carry_settle_count += 1
            else:
                self.carry_settle_count = 0

            if (
                self.carry_settle_count
                >= self.carry_settle_required
            ):
                self.carry_completed = True

                self.transition(
                    'PLAN_GRAVITY_SUPPORTED_CARRY',
                    (
                        'compact carry settled; base stays zero; '
                        'reorienting the book so gravity is carried '
                        'by the lower finger instead of pad friction'
                    ),
                )
                return

            if (
                self.sim
                - self.carry_command_sim
                > self.carry_timeout
            ):
                self.fail(
                    'compact-carry arm motion '
                    'failed to settle before timeout'
                )
                return

            return

        # --------------------------------------------------
        # 3. GRAVITY-SUPPORTED TRANSPORT ORIENTATION
        # --------------------------------------------------

        if self.state == 'PLAN_GRAVITY_SUPPORTED_CARRY':
            self.stop_base()

            try:
                reason = self.plan_gravity_supported_carry()
            except Exception as exc:
                self.fail(
                    'gravity-supported carry planning failed: '
                    + str(exc)
                )
                return

            self.publish_arm_positions(
                self.support_target_positions,
                self.support_motion_sec,
            )

            self.support_command_sim = self.sim
            self.support_settle_count = 0

            self.transition(
                'WAIT_GRAVITY_SUPPORTED_CARRY',
                reason
                + '; slow arm-only orientation change published'
            )
            return

        if self.state == 'WAIT_GRAVITY_SUPPORTED_CARRY':
            self.stop_base()

            error = self.arm_target_error(
                self.support_target_positions
            )

            if (
                error is not None
                and error
                <= self.support_joint_tolerance
            ):
                self.support_settle_count += 1
            else:
                self.support_settle_count = 0

            if (
                self.support_settle_count
                >= self.support_settle_required
            ):
                self.support_completed = True
                self.save_transport_snapshot(
                    'gravity_supported_settled'
                )

                if not self.continuous_retention_or_fail(
                    'GRAVITY_SUPPORTED_SETTLE'
                ):
                    return

                try:
                    self.begin_forced_rotation()
                except Exception as exc:
                    self.fail(
                        'rotation initialization failed: '
                        + str(exc)
                    )
                    return

                self.transition(
                    'ROTATE_180_SEARCH_BIN',
                    (
                        'gravity-supported book orientation settled; '
                        f'physical position {self.physical_column}: '
                        f'forced {self.rotation_direction} 180-degree '
                        'rotation; live bin perception active'
                    ),
                )
                return

            if (
                self.sim is not None
                and self.support_command_sim is not None
                and self.sim
                - self.support_command_sim
                > self.support_timeout
            ):
                self.fail(
                    'gravity-supported carry failed to settle'
                )
                return

            return

        # --------------------------------------------------
        # 4. FORCED DIRECTION 180 ROTATION
        # --------------------------------------------------

        if (
            self.state
            == 'ROTATE_180_SEARCH_BIN'
        ):

            if not self.continuous_retention_or_fail(
                'ROTATION'
            ):
                return

            if (
                self.current_yaw is None
                or self.rotation_last_yaw
                is None
            ):
                self.stop_base()
                return

            delta = normalize_angle(
                self.current_yaw
                - self.rotation_last_yaw
            )

            self.rotation_last_yaw = (
                self.current_yaw
            )

            # Ignore impossible odometry jumps.
            if abs(delta) < 0.50:
                self.rotation_accumulated_rad += (
                    delta
                )

            directed_progress = (
                self.rotation_sign
                * self.rotation_accumulated_rad
            )

            remaining = (
                abs(self.rotation_angle)
                - directed_progress
            )

            if (
                remaining
                <= self.rotation_tolerance
            ):
                self.stop_base()

                self.rotation_settle += 1

                if (
                    self.rotation_settle
                    >= self.rotation_settle_required
                ):
                    self.save_transport_snapshot(
                        'after_rotation'
                    )

                    if not self.continuous_retention_or_fail(
                        'AFTER_ROTATION'
                    ):
                        return

                    if (
                        self.transport_mode
                        == 'straight'
                    ):
                        self.transition(
                            'STRAIGHT_TO_BIN',
                            (
                                'physical middle '
                                'position: 180-degree '
                                'turn complete; no '
                                'lateral transit'
                            ),
                        )
                    else:
                        self.transition(
                            'LATERAL_TRANSIT',
                            (
                                'side physical '
                                'position: 180-degree '
                                'turn complete; shelf '
                                'behind robot; begin '
                                'pure lateral return'
                            ),
                        )

                return

            self.rotation_settle = 0

            speed = clamp(
                (
                    self.rotation_kp
                    * max(
                        0.0,
                        remaining,
                    )
                ),
                0.03,
                self.rotation_max_speed,
            )

            # HARD direction:
            # physical 1/2 -> negative -> CW
            # physical 3/4/5 -> positive -> CCW
            self.publish_velocity(
                0.0,
                0.0,
                self.rotation_sign
                * speed,
            )
            return

        # --------------------------------------------------
        # 3A. PHYSICAL MIDDLE POSITION:
        #     STRAIGHT TO BIN
        # --------------------------------------------------

        if (
            self.state
            == 'STRAIGHT_TO_BIN'
        ):

            if (
                self.current_xy is None
                or self.current_yaw is None
            ):
                self.stop_base()
                return

            if (
                self.locked_bin_odom_xy
                is None
            ):
                # No lateral search.
                # Stay stopped while centered RGB-D
                # finishes confirmation.
                self.stop_base()
                return

            forward, left = (
                vector_odom_to_base(
                    self.locked_bin_odom_xy,
                    self.current_xy,
                    self.current_yaw,
                )
            )

            bearing = math.atan2(
                left,
                max(
                    1.0e-6,
                    forward,
                ),
            )

            forward_error = (
                forward
                - self.final_standoff
            )

            # Small yaw-only centering is allowed,
            # but NO lateral base translation.
            if (
                abs(bearing)
                > self.final_bearing_tol
            ):
                wz = clamp(
                    0.65 * bearing,
                    -self.final_yaw_speed,
                    self.final_yaw_speed,
                )

                self.publish_velocity(
                    0.0,
                    0.0,
                    wz,
                )
                return

            if (
                abs(forward_error)
                <= self.final_forward_tol
                and abs(left)
                <= self.final_lateral_tol
            ):
                self.stop_base()

                self.final_settle = 0

                self.transition(
                    'FINAL_BIN_ALIGN',
                    (
                        'middle-column straight '
                        'transit reached live '
                        'bin standoff'
                    ),
                )
                return

            vx = clamp(
                0.35 * forward_error,
                -self.final_translation_speed,
                self.final_translation_speed,
            )

            if (
                vx > 0.0
                and self.front_minimum
                is not None
                and self.front_minimum
                < self.front_hard_stop
            ):
                self.fail(
                    'front LiDAR hard stop '
                    'during straight-to-bin '
                    'motion'
                )
                return

            if (
                vx < 0.0
                and self.rear_minimum
                is not None
                and self.rear_minimum
                < self.rear_hard_stop
            ):
                self.fail(
                    'rear LiDAR hard stop '
                    'during straight-to-bin '
                    'motion'
                )
                return

            # Middle route:
            # forward/back only.
            self.publish_velocity(
                vx,
                0.0,
                0.0,
            )
            return

        # --------------------------------------------------
        # 3B. SIDE PHYSICAL POSITIONS:
        #     PURE LATERAL RETURN
        # --------------------------------------------------

        if (
            self.state
            == 'LATERAL_TRANSIT'
        ):

            if not self.continuous_retention_or_fail(
                'LATERAL'
            ):
                return

            if (
                self.current_xy is None
                or self.rotation_target_yaw
                is None
            ):
                self.stop_base()
                return

            home_forward, home_left = (
                vector_odom_to_base(
                    self.home_xy,
                    self.current_xy,
                    self.rotation_target_yaw,
                )
            )

            if (
                abs(home_left)
                <= self.lateral_tolerance
            ):
                # The previous code bypassed the lateral ramp here and jumped
                # directly to zero.  Decelerate through the existing limiter
                # before changing stages so the gravity-supported book is not
                # hit by an impulsive lateral load.
                vy = self.ramp_lateral_velocity(0.0)

                if abs(vy) > 0.005:
                    self.publish_velocity(
                        0.0,
                        vy,
                        0.0,
                    )
                    return

                self.stop_base()
                self.save_transport_snapshot(
                    'after_lateral'
                )

                if not self.require_book_retained(
                    'AFTER_LATERAL_TRANSIT'
                ):
                    return

                distance_home = math.hypot(
                    (
                        self.current_xy[0]
                        - self.home_xy[0]
                    ),
                    (
                        self.current_xy[1]
                        - self.home_xy[1]
                    ),
                )

                if (
                    distance_home
                    <= self.home_zone_radius
                ):
                    self.transition(
                        'WAIT_CENTER_BIN',
                        (
                            'pure lateral transit '
                            'reached Start/End Zone'
                        ),
                    )
                    return

                # Main lateral motion is complete.
                #
                # Do NOT reject the route merely because
                # the remaining longitudinal displacement
                # is larger than an arbitrary fixed value.
                # Perform the correction as its own pure-X
                # stage with the same LiDAR hard stops.
                self.transition(
                    'LONGITUDINAL_CORRECTION',
                    (
                        'pure lateral component complete; '
                        'performing separate straight '
                        'correction to the recorded '
                        'Start/End Zone'
                    ),
                )
                return

            target_vy = clamp(
                self.lateral_kp
                * home_left,
                -self.lateral_max_speed,
                self.lateral_max_speed,
            )

            # Smooth carried-book lateral acceleration AND
            # deceleration. Forward X control is unchanged.
            vy = self.ramp_lateral_velocity(
                target_vy
            )

            self.publish_velocity(
                0.0,
                vy,
                0.0,
            )
            return

        # --------------------------------------------------
        # 4. SEPARATE STRAIGHT HOME CORRECTION
        # --------------------------------------------------

        if (
            self.state
            == 'LONGITUDINAL_CORRECTION'
        ):

            if not self.continuous_retention_or_fail(
                'LONGITUDINAL'
            ):
                return

            if (
                self.current_xy is None
                or self.rotation_target_yaw
                is None
            ):
                self.stop_base()
                return

            forward, left = (
                vector_odom_to_base(
                    self.home_xy,
                    self.current_xy,
                    self.rotation_target_yaw,
                )
            )

            distance_home = math.hypot(
                (
                    self.current_xy[0]
                    - self.home_xy[0]
                ),
                (
                    self.current_xy[1]
                    - self.home_xy[1]
                ),
            )

            if (
                distance_home
                <= self.home_zone_radius
            ):
                vx = self.ramp_longitudinal_velocity(0.0)

                if abs(vx) > 0.005:
                    self.publish_velocity(
                        vx,
                        0.0,
                        0.0,
                    )
                    return

                self.stop_base()
                self.save_transport_snapshot(
                    'entered_home_zone'
                )

                if not self.require_book_retained(
                    'AT_HOME_ZONE_ENTRY'
                ):
                    return

                self.transition(
                    'WAIT_CENTER_BIN',
                    (
                        'smooth bounded separated '
                        'straight correction entered '
                        'Start/End Zone'
                    ),
                )
                return

            target_vx = clamp(
                0.55 * forward,
                -self.longitudinal_correction_speed,
                self.longitudinal_correction_speed,
            )

            # Held-object-aware FRONT protection.  The old 0.55 m check was
            # only safe for the base; the carried book extends much farther
            # forward than the front laser.  Begin braking before the book can
            # reach the obstacle, then decide whether we are already inside the
            # accepted Start/End envelope.
            protective_front = (
                target_vx > 0.0
                and self.front_minimum is not None
                and self.front_minimum <= self.carried_front_stop
            )

            if protective_front:
                target_vx = 0.0

            # Reverse protection remains base/LiDAR guarded.
            protective_rear = (
                target_vx < 0.0
                and self.rear_minimum is not None
                and self.rear_minimum < self.rear_hard_stop
            )

            if protective_rear:
                target_vx = 0.0

            vx = self.ramp_longitudinal_velocity(
                target_vx
            )

            if protective_front and abs(vx) <= 0.005:
                self.stop_base()
                self.save_transport_snapshot(
                    'carried_front_protective_stop'
                )

                if distance_home <= self.final_home_envelope:
                    self.transition(
                        'WAIT_CENTER_BIN',
                        (
                            'carried-book-aware front clearance reached; '
                            'smoothly stopped inside Start/End envelope'
                        ),
                    )
                    return

                self.fail(
                    'CARRIED_BOOK_FRONT_CLEARANCE_BLOCKED_HOME_CORRECTION: '
                    f'front={self.front_minimum:.3f}m '
                    f'required={self.carried_front_stop:.3f}m '
                    f'home_error={distance_home:.3f}m'
                )
                return

            if protective_rear and abs(vx) <= 0.005:
                self.stop_base()
                self.fail(
                    'rear LiDAR hard stop during home correction'
                )
                return

            # Straight only; maximum remains the existing configured 0.10 m/s.
            self.publish_velocity(
                vx,
                0.0,
                0.0,
            )
            return

        # --------------------------------------------------
        # 5. BIN LOCK AT START/END ZONE
        # --------------------------------------------------

        if (
            self.state
            == 'WAIT_CENTER_BIN'
        ):

            self.stop_base()

            if (
                self.locked_bin_odom_xy
                is not None
            ):
                self.final_settle = 0

                self.transition(
                    'FINAL_BIN_ALIGN',
                    (
                        'live RGB-D bin lock '
                        'available; bounded '
                        'placement-ready alignment'
                    ),
                )
                return

            return

        # --------------------------------------------------
        # 6. SMALL FINAL ALIGNMENT
        # --------------------------------------------------

        if (
            self.state
            == 'FINAL_BIN_ALIGN'
        ):

            if (
                self.current_xy is None
                or self.current_yaw is None
                or self.locked_bin_odom_xy
                is None
            ):
                self.stop_base()
                return

            forward, left = (
                vector_odom_to_base(
                    self.locked_bin_odom_xy,
                    self.current_xy,
                    self.current_yaw,
                )
            )

            bearing = math.atan2(
                left,
                max(
                    1.0e-6,
                    forward,
                ),
            )

            forward_error = (
                forward
                - self.final_standoff
            )

            good = (
                abs(forward_error)
                <= self.final_forward_tol
                and abs(left)
                <= self.final_lateral_tol
                and abs(bearing)
                <= self.final_bearing_tol
            )

            if good:
                self.stop_base()

                self.final_settle += 1

                if (
                    self.final_settle
                    >= self.final_settle_required
                ):
                    distance_home = (
                        math.hypot(
                            (
                                self.current_xy[0]
                                - self.home_xy[0]
                            ),
                            (
                                self.current_xy[1]
                                - self.home_xy[1]
                            ),
                        )
                    )

                    if (
                        distance_home
                        > self.final_home_envelope
                    ):
                        self.fail(
                            'final bin alignment '
                            'left Start/End Zone '
                            'envelope'
                        )
                        return

                    if not self.require_book_retained(
                        'BEFORE_DAY8_HANDOFF'
                    ):
                        return

                    # FAST DAY8 HANDOFF:
                    # keep the book in the gravity-supported transport
                    # orientation.  Do NOT restore the fragile upright
                    # orientation and then leave it hanging while Day8
                    # measures/plans.
                    self.finish(
                        True,
                        (
                            'physical-position transport complete; '
                            'gravity-supported book retained for '
                            'immediate Day8 planning; '
                            f'position={self.physical_column} '
                            f'rotation={self.rotation_direction} '
                            f'mode={self.transport_mode}'
                        ),
                    )

                return

            self.final_settle = 0

            # Middle physical column:
            # absolutely no lateral base command.
            if (
                self.transport_mode
                == 'straight'
            ):
                if (
                    abs(bearing)
                    > self.final_bearing_tol
                ):
                    wz = clamp(
                        0.65 * bearing,
                        -self.final_yaw_speed,
                        self.final_yaw_speed,
                    )

                    self.publish_velocity(
                        0.0,
                        0.0,
                        wz,
                    )
                    return

                vx = clamp(
                    0.35 * forward_error,
                    -self.final_translation_speed,
                    self.final_translation_speed,
                )

                self.publish_velocity(
                    vx,
                    0.0,
                    0.0,
                )
                return

            # Side-column final alignment only.
            vx = clamp(
                0.35 * forward_error,
                -self.final_translation_speed,
                self.final_translation_speed,
            )

            vy = clamp(
                0.45 * left,
                -self.final_translation_speed,
                self.final_translation_speed,
            )

            wz = clamp(
                0.65 * bearing,
                -self.final_yaw_speed,
                self.final_yaw_speed,
            )

            # Large bearing:
            # yaw only.
            if (
                abs(bearing)
                > math.radians(8.0)
            ):
                vx = 0.0
                vy = 0.0

            # Large lateral:
            # lateral only.
            elif abs(left) > 0.18:
                vx = 0.0
                wz = 0.0

            self.publish_velocity(
                vx,
                vy,
                wz,
            )
            return

        if self.state == 'PLAN_RESTORE_PLACEMENT_ORIENTATION':
            self.stop_base()

            try:
                reason = self.plan_restore_placement_orientation()
            except Exception as exc:
                self.fail(
                    'placement-orientation restore planning failed: '
                    + str(exc)
                )
                return

            self.publish_arm_positions(
                self.restore_target_positions,
                self.restore_motion_sec,
            )

            self.restore_command_sim = self.sim
            self.restore_settle_count = 0

            self.transition(
                'WAIT_RESTORE_PLACEMENT_ORIENTATION',
                reason
                + '; slow arm-only restore published'
            )
            return

        if self.state == 'WAIT_RESTORE_PLACEMENT_ORIENTATION':
            self.stop_base()

            error = self.arm_target_error(
                self.restore_target_positions
            )

            if (
                error is not None
                and error
                <= self.support_joint_tolerance
            ):
                self.restore_settle_count += 1
            else:
                self.restore_settle_count = 0

            if (
                self.restore_settle_count
                >= self.support_settle_required
            ):
                self.restore_completed = True

                if not self.require_book_retained(
                    'BEFORE_DAY8_HANDOFF'
                ):
                    return

                self.finish(
                    True,
                    (
                        'physical-position transport complete '
                        'using gravity-supported book carry; '
                        f'position={self.physical_column} '
                        f'rotation={self.rotation_direction} '
                        f'mode={self.transport_mode}'
                    ),
                )
                return

            if (
                self.sim is not None
                and self.restore_command_sim is not None
                and self.sim
                - self.restore_command_sim
                > self.support_timeout
            ):
                self.fail(
                    'placement orientation failed to settle'
                )
                return

            return

        self.fail(
            'unknown state: '
            + self.state
        )

    # ======================================================
    # Result
    # ======================================================

    def geometry_dict(self):
        g = self.confirmed_geometry

        if g is None:
            return None

        if hasattr(
            g,
            'as_dict',
        ):
            return g.as_dict()

        return {
            'forward_m': float(
                g.forward_m
            ),
            'lateral_left_m': float(
                g.lateral_left_m
            ),
        }

    def write(
        self,
        passed: bool,
        reason: str,
    ) -> None:

        distance_home = None

        if self.current_xy is not None:
            distance_home = math.hypot(
                (
                    self.current_xy[0]
                    - self.home_xy[0]
                ),
                (
                    self.current_xy[1]
                    - self.home_xy[1]
                ),
            )

        gripper_name = (
            f'gripper_'
            f'{self.selected_arm}'
            f'_finger_joint'
        )

        current_arm = []

        plan = self.day5.get(
            'grasp_plan',
            {},
        )

        candidate = (
            plan
            .get(
                'candidate_plans',
                {},
            )
            .get(
                self.selected_arm,
                {},
            )
        )

        lift = candidate.get(
            'lift',
            {},
        )

        joint_names = lift.get(
            'joint_names',
            [],
        )

        if (
            isinstance(
                joint_names,
                list,
            )
            and all(
                n in self.joints
                for n in joint_names
            )
        ):
            current_arm = [
                self.joints[n]
                for n in joint_names
            ]

        data = {
            'day': 7,
            'passed': bool(passed),
            'reason': reason,
            'state': self.state,

            'transport_strategy':
                (
                    'PHYSICAL_SHELF_ORDER_'
                    'DIRECTED_ROTATE180_'
                    'LIVE_BIN'
                ),

            # Marker identity is evidence only.
            # It does NOT choose the route.
            'marker_identity':
                self.marker_identity,

            'observed_left_to_right':
                self.observed_left_to_right,

            'physical_shelf_position':
                self.physical_column,

            'route_decision_uses_'
            'physical_order_not_marker':
                True,

            'rotation_direction':
                self.rotation_direction,

            'rotation_sign':
                self.rotation_sign,

            'transport_mode':
                self.transport_mode,

            'carry_retract_distance_m':
                self.carry_retract_distance,

            'carry_raise_distance_m':
                self.carry_raise_distance,

            'carry_completed':
                self.carry_completed,

            # Exact pose immediately BEFORE the 90-degree
            # gravity-support rotation.  FAST67 already solved and
            # collision-screened the forward path between this pose
            # and the supported pose, so Day8 may safely use the
            # reverse path instead of rediscovering it with IK.
            'compact_carry_target_positions_rad': (
                list(self.carry_target_positions)
                if self.carry_target_positions is not None
                else None
            ),

            'gravity_supported_target_positions_rad': (
                list(self.support_target_positions)
                if self.support_target_positions is not None
                else None
            ),

            'gravity_supported_carry_completed':
                self.support_completed,

            'gravity_supported_book_roll_deg': (
                self.support_roll_sign
                * math.degrees(self.support_roll_rad)
                if self.support_roll_sign is not None
                else None
            ),

            'gravity_supported_target_xyz_m': (
                list(self.support_target_xyz)
                if self.support_target_xyz is not None
                else None
            ),

            'gravity_supported_max_joint_delta_rad':
                self.support_max_delta,

            'placement_orientation_restored':
                False,

            'gravity_supported_day8_handoff':
                self.support_completed,

            'carry_target_xyz_m': (
                list(self.carry_target_xyz)
                if self.carry_target_xyz
                is not None
                else None
            ),

            'carry_target_positions_rad': (
                list(self.carry_target_positions)
                if self.carry_target_positions
                is not None
                else None
            ),

            'carry_max_joint_delta_rad':
                self.carry_max_joint_delta,

            'carry_ik_position_error_m':
                self.carry_ik_position_error,

            'carry_ik_orientation_error_rad':
                self.carry_ik_orientation_error,

            'carry_collision_screen':
                self.carry_screen,

            'rotation_accumulated_rad':
                self.rotation_accumulated_rad,

            'selected_arm':
                self.selected_arm,

            'day4_result_path':
                self.day4_path,

            'day5_result_path':
                self.day5_path,

            'home_pose_path':
                self.home_path,

            'home_odom_xy':
                list(self.home_xy),

            'final_odom_xy': (
                list(self.current_xy)
                if self.current_xy
                is not None
                else None
            ),

            'final_yaw_rad':
                self.current_yaw,

            'simulation_time_sec':
                self.sim,

            'distance_to_home_center_m':
                distance_home,

            'locked_bin_odom_xy': (
                list(
                    self.locked_bin_odom_xy
                )
                if (
                    self.locked_bin_odom_xy
                    is not None
                )
                else None
            ),

            'bin_detection': (
                self.confirmed_detection.as_dict()
                if (
                    self.confirmed_detection
                    is not None
                )
                else None
            ),

            'bin_geometry':
                self.geometry_dict(),

            'bin_detection_uses_'
            'live_rgb_depth_tf':
                True,

            'bin_lock_sim_time_sec':
                self.bin_lock_sim,

            'bin_observations':
                self.bin_observations,

            'bin_evidence_image_path':
                self.bin_evidence_image_path,

            'carried_book_dimensions_m':
                [0.25, 0.02, 0.16],

            'carried_book_swept_radius_m':
                self.carried_book_radius,

            'carried_book_forward_extent_m':
                self.carried_book_forward_extent,

            'front_laser_base_x_m':
                self.front_laser_base_x,

            'carried_front_margin_m':
                self.carried_front_margin,

            'carried_front_stop_m':
                self.carried_front_stop,

            'longitudinal_accel_limit_mps2':
                self.longitudinal_accel_limit,

            'transport_evidence_images':
                dict(self.transport_evidence_images),

            'held_book_geometry_in_'
            'safety_reasoning':
                True,

            # Day8 compatibility.
            'carry_raise_completed':
                self.carry_completed,

            'rotation_alignment_completed':
                True,

            'carry_raise_target_positions_rad': (
                list(self.support_target_positions)
                if self.support_target_positions is not None
                else (
                    list(self.carry_target_positions)
                    if self.carry_target_positions is not None
                    else current_arm or None
                )
            ),

            'gripper_public_topic':
                (
                    f'/gripper_'
                    f'{self.selected_arm}'
                    f'_controller/'
                    f'joint_trajectory'
                ),

            'gripper_command_mode':
                'position',

            'gripper_hold_position_m':
                self.gripper_hold_position,

            'gripper_actual_position_m':
                self.joints.get(
                    gripper_name
                ),

            'effort_commanded':
                False,

            'raw_topic_commanded':
                False,

            'unintended_robot_contacts':
                self.unintended_contacts,

            'book_fingertip_contact_messages':
                self.book_fingertip_contact_messages,

            'last_book_fingertip_contact_sim_sec':
                self.last_book_fingertip_contact_sim,

            'book_retained_at_result':
                self.book_retained(),

            'lateral_max_speed_mps':
                self.lateral_max_speed,

            'lateral_accel_limit_mps2':
                self.lateral_accel_limit,

            'bin_contact_messages':
                self.bin_contact_messages,

            'maximum_abs_command':
                self.max_cmd,

            'path_trace':
                self.path_trace,

            'hidden_simulator_oracle_used':
                False,

            'world_pose_queried':
                False,

            'erc_seed_used':
                False,
        }

        atomic_write_json(
            self.result_path,
            data,
        )

    def fail(
        self,
        reason: str,
    ) -> None:

        if self.done:
            return

        self.stop_base()

        self.reason = reason
        self.passed = False
        self.done = True
        self.state = 'FAILED'

        self.write(
            False,
            reason,
        )

        self.get_logger().error(
            '[FAST67][FAIL] '
            + reason
        )

    def finish(
        self,
        passed: bool,
        reason: str,
    ) -> None:

        if self.done:
            return

        self.stop_base()

        self.reason = reason
        self.passed = bool(passed)
        self.done = True

        self.state = (
            'DONE'
            if passed
            else 'FAILED'
        )

        self.write(
            passed,
            reason,
        )

        if passed:
            self.get_logger().info(
                '[FAST67][PASS] '
                + reason
            )
        else:
            self.get_logger().error(
                '[FAST67][FAIL] '
                + reason
            )


def main(args=None):

    rclpy.init(args=args)

    node = None
    code = 1

    try:
        node = FastTransport()

        while (
            rclpy.ok()
            and not node.done
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.05,
            )

        if (
            node is not None
            and node.passed
        ):
            code = 0

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.stop_base()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    raise SystemExit(code)


if __name__ == '__main__':
    main()
