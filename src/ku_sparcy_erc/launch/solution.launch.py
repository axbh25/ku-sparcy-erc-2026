"""Competition entry point for KU SPARCy ERC 2026 through Day 3."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('ku_sparcy_erc')
    day2_config = os.path.join(
        package_share, 'config', 'day2_perception.yaml')
    day3_config = os.path.join(
        package_share, 'config', 'day3_approach.yaml')

    shelf_column_number = LaunchConfiguration('shelf_column_number')
    book_colour = LaunchConfiguration('book_colour')
    phase1_fast_start = LaunchConfiguration('phase1_fast_start')
    enable_motion = LaunchConfiguration('enable_motion')
    result_path = LaunchConfiguration('result_path')
    image_output_dir = LaunchConfiguration('image_output_dir')
    validation_require_all_markers = LaunchConfiguration(
        'validation_require_all_markers')
    approach_motion_enabled = LaunchConfiguration(
        'approach_motion_enabled')
    approach_distance_limit_m = LaunchConfiguration(
        'approach_distance_limit_m')
    approach_max_forward_speed_mps = LaunchConfiguration(
        'approach_max_forward_speed_mps')
    approach_standoff_m = LaunchConfiguration('approach_standoff_m')

    return LaunchDescription([
        DeclareLaunchArgument(
            'shelf_column_number',
            description='Requested shelf marker number, integer from 1 to 5.'),
        DeclareLaunchArgument(
            'book_colour',
            description='Requested book colour: red, green, yellow, or blue.'),
        DeclareLaunchArgument(
            'phase1_fast_start',
            default_value='true',
            description=(
                'true: validated Phase 1 clockwise 90-degree opening; '
                'false: Day 2 orientation-independent visual search.')),
        DeclareLaunchArgument(
            'enable_motion',
            default_value='true',
            description='Enable opening/search base motion.'),
        DeclareLaunchArgument(
            'result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day3_result.json',
            description='Absolute path for atomic Day 3 JSON output.'),
        DeclareLaunchArgument(
            'image_output_dir',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/erc_images',
            description='Team-owned live annotated-image directory.'),
        DeclareLaunchArgument(
            'validation_require_all_markers',
            default_value='false',
            description=(
                'Test-only Day 2 complete-set validation; competition default '
                'does not wait after the requested marker is confirmed.')),
        DeclareLaunchArgument(
            'approach_motion_enabled',
            default_value='true',
            description=(
                'false performs opening, marker detection, bearing, raw-depth '
                'and LiDAR ranging without shelf translation.')),
        DeclareLaunchArgument(
            'approach_distance_limit_m',
            default_value='0.0',
            description=(
                'Test-only odometry travel limit; 0.0 means full stand-off '
                'approach.')),
        DeclareLaunchArgument(
            'approach_max_forward_speed_mps',
            default_value='0.30',
            description='Day 3 forward-speed clamp after the opening turn.'),
        DeclareLaunchArgument(
            'approach_standoff_m',
            default_value='1.05',
            description='Desired base-frame target/shelf stand-off distance.'),
        Node(
            package='ku_sparcy_erc',
            executable='day3_mission',
            name='mission_start',
            output='screen',
            emulate_tty=True,
            parameters=[
                day2_config,
                day3_config,
                {
                    'use_sim_time': True,
                    'shelf_column_number': ParameterValue(
                        shelf_column_number, value_type=int),
                    'book_colour': ParameterValue(
                        book_colour, value_type=str),
                    'phase1_fast_start': ParameterValue(
                        phase1_fast_start, value_type=bool),
                    'enable_motion': ParameterValue(
                        enable_motion, value_type=bool),
                    'result_path': ParameterValue(
                        result_path, value_type=str),
                    'image_output_dir': ParameterValue(
                        image_output_dir, value_type=str),
                    'validation_require_all_markers': ParameterValue(
                        validation_require_all_markers, value_type=bool),
                    'approach_motion_enabled': ParameterValue(
                        approach_motion_enabled, value_type=bool),
                    'approach_distance_limit_m': ParameterValue(
                        approach_distance_limit_m, value_type=float),
                    'approach_max_forward_speed_mps': ParameterValue(
                        approach_max_forward_speed_mps, value_type=float),
                    'approach_standoff_m': ParameterValue(
                        approach_standoff_m, value_type=float),
                },
            ],
        ),
    ])
