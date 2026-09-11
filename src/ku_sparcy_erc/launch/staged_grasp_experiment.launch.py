"""Explicitly gated Day 4 position-controlled one-arm grasp laboratory."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory('ku_sparcy_erc')
    config = os.path.join(package_share, 'config', 'day4_grasp.yaml')

    source_result_path = LaunchConfiguration('source_result_path')
    grasp_result_path = LaunchConfiguration('grasp_result_path')
    image_output_dir = LaunchConfiguration('image_output_dir')
    target_stage = LaunchConfiguration('target_stage')
    manual_approval_required = LaunchConfiguration(
        'manual_approval_required')

    return LaunchDescription([
        DeclareLaunchArgument(
            'source_result_path',
            default_value=(
                '/opt/erc_ws/src/ku_sparcy_erc/'
                'day4_result_full_gui.json'),
            description=(
                'Passed Day 4 perception/geometry result generated in the '
                'currently running simulator instance.')),
        DeclareLaunchArgument(
            'grasp_result_path',
            default_value=(
                '/opt/erc_ws/src/ku_sparcy_erc/'
                'day4_grasp_result.json'),
            description='Atomic staged-grasp result JSON path.'),
        DeclareLaunchArgument(
            'image_output_dir',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/erc_images',
            description='Team-owned live evidence directory.'),
        DeclareLaunchArgument(
            'target_stage',
            default_value='lift',
            choices=[
                'plan', 'pregrasp', 'open', 'no_contact',
                'contact_pose', 'close', 'extract', 'lift'],
            description=(
                'Last approved development stage. The node pauses before '
                'every stage when manual approval is enabled.')),
        DeclareLaunchArgument(
            'manual_approval_required',
            default_value='true',
            description=(
                'Require std_msgs/Bool true on /ku_sparcy/day4_continue '
                'before every physical manipulation stage.')),
        Node(
            package='ku_sparcy_erc',
            executable='staged_grasp_experiment',
            name='ku_sparcy_day4_grasp_experiment',
            output='screen',
            emulate_tty=True,
            parameters=[
                config,
                {
                    'use_sim_time': True,
                    'source_result_path': ParameterValue(
                        source_result_path, value_type=str),
                    'grasp_result_path': ParameterValue(
                        grasp_result_path, value_type=str),
                    'image_output_dir': ParameterValue(
                        image_output_dir, value_type=str),
                    'target_stage': ParameterValue(
                        target_stage, value_type=str),
                    'manual_approval_required': ParameterValue(
                        manual_approval_required, value_type=bool),
                },
            ],
        ),
    ])
