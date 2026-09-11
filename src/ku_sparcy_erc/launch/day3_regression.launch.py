"""Frozen Day 3 competition-path regression entry point."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('ku_sparcy_erc')
    day2_config = os.path.join(package_share, 'config', 'day2_perception.yaml')
    day3_config = os.path.join(package_share, 'config', 'day3_approach.yaml')

    shelf_column_number = LaunchConfiguration('shelf_column_number')
    book_colour = LaunchConfiguration('book_colour')
    result_path = LaunchConfiguration('result_path')
    image_output_dir = LaunchConfiguration('image_output_dir')

    return LaunchDescription([
        DeclareLaunchArgument('shelf_column_number'),
        DeclareLaunchArgument('book_colour'),
        DeclareLaunchArgument(
            'result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day3_regression_day4.json'),
        DeclareLaunchArgument(
            'image_output_dir',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/erc_images'),
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
                    'book_colour': ParameterValue(book_colour, value_type=str),
                    'phase1_fast_start': True,
                    'enable_motion': True,
                    'approach_motion_enabled': True,
                    'approach_distance_limit_m': 0.0,
                    'result_path': ParameterValue(result_path, value_type=str),
                    'image_output_dir': ParameterValue(
                        image_output_dir, value_type=str),
                    'validation_require_all_markers': False,
                },
            ],
        ),
    ])
