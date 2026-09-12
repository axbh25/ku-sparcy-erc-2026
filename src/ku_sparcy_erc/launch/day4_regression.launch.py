"""Frozen Day 4 regression entry after the full mission launch advances."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('ku_sparcy_erc')
    shelf = LaunchConfiguration('shelf_column_number')
    colour = LaunchConfiguration('book_colour')
    result = LaunchConfiguration('result_path')
    images = LaunchConfiguration('image_output_dir')
    return LaunchDescription([
        DeclareLaunchArgument('shelf_column_number'),
        DeclareLaunchArgument('book_colour'),
        DeclareLaunchArgument(
            'result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day4_regression_result.json'),
        DeclareLaunchArgument(
            'image_output_dir',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/erc_images'),
        Node(
            package='ku_sparcy_erc',
            executable='day4_mission',
            name='mission_start',
            output='screen',
            emulate_tty=True,
            parameters=[
                os.path.join(share, 'config', 'day2_perception.yaml'),
                os.path.join(share, 'config', 'day3_approach.yaml'),
                os.path.join(share, 'config', 'day4_books.yaml'),
                {
                    'use_sim_time': True,
                    'shelf_column_number': ParameterValue(shelf, value_type=int),
                    'book_colour': ParameterValue(colour, value_type=str),
                    'phase1_fast_start': True,
                    'enable_motion': True,
                    'result_path': ParameterValue(result, value_type=str),
                    'image_output_dir': ParameterValue(images, value_type=str),
                    'validation_require_all_markers': False,
                    'approach_motion_enabled': True,
                    'approach_distance_limit_m': 0.0,
                    'approach_max_forward_speed_mps': 0.30,
                    'approach_standoff_m': 1.05,
                    'day4_stop_after': 'pregrasp',
                },
            ],
        ),
    ])
