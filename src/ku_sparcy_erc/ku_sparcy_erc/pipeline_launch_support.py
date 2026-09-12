"""Launch-construction helper for the sequential Day 4 -> Day 7 pipeline."""

from __future__ import annotations

import os
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


VALID_STOPS = ('day5', 'day6', 'day7')


def _after_process(event, context, *, stage: str, next_action, stop_after):
    code = int(getattr(event, 'returncode', 1) or 0)
    if code != 0:
        return [
            LogInfo(msg=f'[PIPELINE][FAIL] {stage} exited with code {code}'),
            EmitEvent(event=Shutdown(reason=f'{stage} failed')),
        ]
    desired = str(stop_after.perform(context)).strip().lower()
    if stage == desired:
        return [
            LogInfo(msg=f'[PIPELINE][PASS] requested stop after {stage}'),
            EmitEvent(event=Shutdown(reason=f'{stage} completed')),
        ]
    if next_action is None:
        return [
            LogInfo(msg='[PIPELINE][PASS] Day 7 completed'),
            EmitEvent(event=Shutdown(reason='Day 7 completed')),
        ]
    return [
        LogInfo(msg=f'[PIPELINE] {stage} passed; starting next stage'),
        next_action,
    ]


def generate_pipeline_description(default_stop_after: str) -> LaunchDescription:
    if default_stop_after not in VALID_STOPS:
        raise ValueError(f'invalid default stop stage: {default_stop_after}')
    share = get_package_share_directory('ku_sparcy_erc')
    config = lambda name: os.path.join(share, 'config', name)

    shelf = LaunchConfiguration('shelf_column_number')
    colour = LaunchConfiguration('book_colour')
    stop_after = LaunchConfiguration('stop_after')
    image_dir = LaunchConfiguration('image_output_dir')
    home_path = LaunchConfiguration('home_pose_path')
    day4_path = LaunchConfiguration('day4_result_path')
    day5_path = LaunchConfiguration('day5_result_path')
    day6_path = LaunchConfiguration('day6_result_path')
    day7_path = LaunchConfiguration('day7_result_path')

    home = Node(
        package='ku_sparcy_erc',
        executable='home_pose_recorder',
        name='ku_sparcy_home_pose_recorder',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'result_path': ParameterValue(home_path, value_type=str),
        }],
    )

    day4 = Node(
        package='ku_sparcy_erc',
        executable='day4_mission',
        name='mission_start',
        output='screen',
        emulate_tty=True,
        parameters=[
            config('day2_perception.yaml'),
            config('day3_approach.yaml'),
            config('day4_books.yaml'),
            {
                'use_sim_time': True,
                'shelf_column_number': ParameterValue(shelf, value_type=int),
                'book_colour': ParameterValue(colour, value_type=str),
                'phase1_fast_start': True,
                'enable_motion': True,
                'result_path': ParameterValue(day4_path, value_type=str),
                'image_output_dir': ParameterValue(image_dir, value_type=str),
                'validation_require_all_markers': False,
                'approach_motion_enabled': True,
                'approach_distance_limit_m': 0.0,
                'approach_max_forward_speed_mps': 0.30,
                'approach_standoff_m': 1.05,
                'day4_stop_after': 'pregrasp',
            },
        ],
    )

    day5 = Node(
        package='ku_sparcy_erc',
        executable='day5_autonomous_pick',
        name='ku_sparcy_day5_autonomous_pick',
        output='screen',
        emulate_tty=True,
        parameters=[
            config('day4_grasp.yaml'),
            config('day5_autonomous_pick.yaml'),
            {
                'use_sim_time': True,
                'source_result_path': ParameterValue(day4_path, value_type=str),
                'grasp_result_path': ParameterValue(day5_path, value_type=str),
                'image_output_dir': ParameterValue(image_dir, value_type=str),
                'target_stage': 'lift',
                'manual_approval_required': False,
            },
        ],
    )

    day6 = Node(
        package='ku_sparcy_erc',
        executable='day6_return_home',
        name='ku_sparcy_day6_return_home',
        output='screen',
        emulate_tty=True,
        parameters=[
            config('day6_return.yaml'),
            {
                'use_sim_time': True,
                'day4_result_path': ParameterValue(day4_path, value_type=str),
                'day5_result_path': ParameterValue(day5_path, value_type=str),
                'home_pose_path': ParameterValue(home_path, value_type=str),
                'result_path': ParameterValue(day6_path, value_type=str),
            },
        ],
    )

    day7 = Node(
        package='ku_sparcy_erc',
        executable='day7_bin_approach',
        name='ku_sparcy_day7_bin_approach',
        output='screen',
        emulate_tty=True,
        parameters=[
            config('day7_bin.yaml'),
            {
                'use_sim_time': True,
                'day5_result_path': ParameterValue(day5_path, value_type=str),
                'day6_result_path': ParameterValue(day6_path, value_type=str),
                'result_path': ParameterValue(day7_path, value_type=str),
                'image_output_dir': ParameterValue(image_dir, value_type=str),
            },
        ],
    )

    handlers = [
        RegisterEventHandler(OnProcessExit(
            target_action=home,
            on_exit=lambda event, context: _after_process(
                event, context, stage='home_pose',
                next_action=day4, stop_after=stop_after),
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=day4,
            on_exit=lambda event, context: _after_process(
                event, context, stage='day4',
                next_action=day5, stop_after=stop_after),
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=day5,
            on_exit=lambda event, context: _after_process(
                event, context, stage='day5',
                next_action=day6, stop_after=stop_after),
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=day6,
            on_exit=lambda event, context: _after_process(
                event, context, stage='day6',
                next_action=day7, stop_after=stop_after),
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=day7,
            on_exit=lambda event, context: _after_process(
                event, context, stage='day7',
                next_action=None, stop_after=stop_after),
        )),
    ]

    actions: List[object] = [
        DeclareLaunchArgument(
            'shelf_column_number',
            description='Requested shelf marker number, integer 1-5.'),
        DeclareLaunchArgument(
            'book_colour',
            description='Requested book colour: red, green, yellow, or blue.'),
        DeclareLaunchArgument(
            'stop_after',
            default_value=default_stop_after,
            choices=list(VALID_STOPS),
            description='Development checkpoint. Competition solution uses day7.'),
        DeclareLaunchArgument(
            'image_output_dir',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/erc_images'),
        DeclareLaunchArgument(
            'home_pose_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/home_pose.json'),
        DeclareLaunchArgument(
            'day4_result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day4_result.json'),
        DeclareLaunchArgument(
            'day5_result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day5_result.json'),
        DeclareLaunchArgument(
            'day6_result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day6_result.json'),
        DeclareLaunchArgument(
            'day7_result_path',
            default_value='/opt/erc_ws/src/ku_sparcy_erc/day7_result.json'),
        *handlers,
        home,
    ]
    return LaunchDescription(actions)
