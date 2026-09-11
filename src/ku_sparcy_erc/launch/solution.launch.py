"""Competition entry point for KU SPARCy ERC 2026 through revised Day 4."""

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
    default_day4_config = os.path.join(
        package_share, 'config', 'day4_books.yaml')

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
    day4_stop_after = LaunchConfiguration('day4_stop_after')
    day4_config_file = LaunchConfiguration('day4_config_file')

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
                'false: frozen Day 2 orientation-independent search.')),
        DeclareLaunchArgument(
            'enable_motion',
            default_value='true',
            description='Enable inherited opening/search base motion.'),
        DeclareLaunchArgument(
            'result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day4_result.json',
            description='Absolute path for atomic Day 4 JSON output.'),
        DeclareLaunchArgument(
            'image_output_dir',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/erc_images',
            description='Team-owned live annotated-image directory.'),
        DeclareLaunchArgument(
            'validation_require_all_markers',
            default_value='false',
            description='Test-only frozen Day 2 complete-marker validation.'),
        DeclareLaunchArgument(
            'approach_motion_enabled',
            default_value='true',
            description=(
                'Enable the validated Day 3 shelf approach. Day 4 first holds '
                'the base stationary for row mapping, then restores +0.35 rad '
                'and enters the unchanged Day 3 approach.')),
        DeclareLaunchArgument(
            'approach_distance_limit_m',
            default_value='0.0',
            description='Day 3 test-only travel limit; 0.0 means full approach.'),
        DeclareLaunchArgument(
            'approach_max_forward_speed_mps',
            default_value='0.30',
            description='Validated Day 3 forward-speed clamp.'),
        DeclareLaunchArgument(
            'approach_standoff_m',
            default_value='1.05',
            description='Validated Day 3 stand-off control value.'),
        DeclareLaunchArgument(
            'day4_stop_after',
            default_value='pregrasp',
            choices=[
                'preapproach_visibility',
                'visibility',
                'perception',
                'geometry',
                'pregrasp',
            ],
            description=(
                'preapproach_visibility runs the experimental base-stationary '
                'selected-column head scan and stops before Day 3 translation; '
                'visibility is a deprecated alias. perception publishes the '
                'row, restores +0.35 rad, runs unchanged Day 3 approach, and '
                'reacquires only the requested colour; geometry adds synchronized '
                '3-D; pregrasp adds a non-executing reach screen.')),
        DeclareLaunchArgument(
            'day4_config_file',
            default_value=default_day4_config,
            description=(
                'Day 4 YAML file. A /tmp copy may be used for the controlled '
                'stationary pre-approach visibility experiment without editing '
                'the competition configuration.')),
        Node(
            package='ku_sparcy_erc',
            executable='day4_mission',
            name='mission_start',
            output='screen',
            emulate_tty=True,
            parameters=[
                day2_config,
                day3_config,
                day4_config_file,
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
                    'day4_stop_after': ParameterValue(
                        day4_stop_after, value_type=str),
                },
            ],
        ),
    ])
