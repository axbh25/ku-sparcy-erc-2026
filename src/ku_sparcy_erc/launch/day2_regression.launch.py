"""Frozen Day 2 competition-node regression entry point.

This launch file deliberately starts the committed Day 2 ``mission_start``
executable with the committed Day 2 configuration.  Day 3 tests use it before
accepting any new approach motion.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('ku_sparcy_erc')
    config_file = os.path.join(
        package_share, 'config', 'day2_perception.yaml')

    shelf_column_number = LaunchConfiguration('shelf_column_number')
    book_colour = LaunchConfiguration('book_colour')
    phase1_fast_start = LaunchConfiguration('phase1_fast_start')
    enable_motion = LaunchConfiguration('enable_motion')
    result_path = LaunchConfiguration('result_path')
    image_output_dir = LaunchConfiguration('image_output_dir')
    validation_require_all_markers = LaunchConfiguration(
        'validation_require_all_markers')

    return LaunchDescription([
        DeclareLaunchArgument('shelf_column_number'),
        DeclareLaunchArgument('book_colour'),
        DeclareLaunchArgument('phase1_fast_start', default_value='true'),
        DeclareLaunchArgument('enable_motion', default_value='true'),
        DeclareLaunchArgument(
            'result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day2_regression.json'),
        DeclareLaunchArgument(
            'image_output_dir',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/erc_images'),
        DeclareLaunchArgument(
            'validation_require_all_markers', default_value='false'),
        Node(
            package='ku_sparcy_erc',
            executable='mission_start',
            name='mission_start',
            output='screen',
            emulate_tty=True,
            parameters=[
                config_file,
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
                },
            ],
        ),
    ])
