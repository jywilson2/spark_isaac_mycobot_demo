# Copyright 2026 spark_isaac_mycobot_demo contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('spark_verify_pkg')
    phase1_live_launch = os.path.join(package_share, 'launch', 'phase1_live_ecosystem.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(phase1_live_launch)),
        Node(
            package='spark_verify_pkg',
            executable='block_vision_tracker',
            name='block_vision_tracker',
            output='screen',
            parameters=[{
                'rgb_topic': '/mycobot/camera/rgb',
                'detection_topic': '/mycobot/vision/block_detection',
                'red_threshold': 120,
            }],
        ),
        Node(
            package='spark_verify_pkg',
            executable='rl_observation_bridge',
            name='rl_observation_bridge',
            output='screen',
            parameters=[{
                'detection_topic': '/mycobot/vision/block_detection',
                'joint_states_topic': '/mycobot/joint_states',
                'observation_topic': '/mycobot/rl/observation',
                'end_effector_z': 0.12,
                'is_grasped': False,
            }],
        ),
        Node(
            package='spark_verify_pkg',
            executable='live_mdp_reward_monitor',
            name='live_mdp_reward_monitor',
            output='screen',
            parameters=[{
                'detection_topic': '/mycobot/vision/block_detection',
                'observation_topic': '/mycobot/rl/observation',
                'joint_states_topic': '/mycobot/joint_states',
                'reward_topic': '/mycobot/rl/live_reward',
                'safety_penalty_topic': '/mycobot/rl/live_safety_penalty',
            }],
        ),
    ])
