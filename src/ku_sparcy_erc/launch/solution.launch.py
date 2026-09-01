from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share = get_package_share_directory('ku_sparcy_erc')
    config_file = os.path.join(package_share, 'config', 'opening_sequence.yaml')

    shelf_column_number = LaunchConfiguration('shelf_column_number')
    book_colour = LaunchConfiguration('book_colour')
    phase1_fast_start = LaunchConfiguration('phase1_fast_start')
    enable_motion = LaunchConfiguration('enable_motion')

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
            description='Use the fixed Phase 1 clockwise 90-degree opening turn.'),
        DeclareLaunchArgument(
            'enable_motion',
            default_value='true',
            description='Enable base and head motion. Set false only for launch validation.'),
        Node(
            package='ku_sparcy_erc',
            executable='opening_sequence',
            name='opening_sequence',
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
                },
            ],
        ),
    ])
