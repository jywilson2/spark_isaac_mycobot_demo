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

"""Live Phase 2 stack plus PPO trainer (no fixed joint_command_dispatcher)."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    repo_root = os.environ.get(
        'SPARK_REPO_ROOT',
        '/workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo',
    )
    default_checkpoint_dir = os.path.join(repo_root, 'assets', 'checkpoints', 'live_ppo')

    num_episodes_arg = DeclareLaunchArgument(
        'num_episodes',
        default_value='3',
        description='Number of PPO training episodes to collect',
    )
    steps_per_episode_arg = DeclareLaunchArgument(
        'steps_per_episode',
        default_value='8',
        description='Environment steps per episode',
    )
    checkpoint_dir_arg = DeclareLaunchArgument(
        'checkpoint_dir',
        default_value=default_checkpoint_dir,
        description='Directory for policy checkpoints and ONNX-ready export',
    )

    return LaunchDescription([
        num_episodes_arg,
        steps_per_episode_arg,
        checkpoint_dir_arg,
        Node(
            package='spark_verify_pkg',
            executable='live_nitros_camera_bridge',
            name='live_nitros_camera_bridge',
            output='screen',
            parameters=[{
                'rgb_topic': '/mycobot/camera/rgb',
                'nitros_topic': '/mycobot/camera/nitros/rgb',
            }],
        ),
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
        Node(
            package='spark_verify_pkg',
            executable='live_ppo_trainer',
            name='live_ppo_trainer',
            output='screen',
            parameters=[{
                'num_episodes': LaunchConfiguration('num_episodes'),
                'steps_per_episode': LaunchConfiguration('steps_per_episode'),
                'checkpoint_dir': LaunchConfiguration('checkpoint_dir'),
                'export_onnx_weights': True,
                'require_live_sim': True,
                'auto_shutdown': True,
            }],
        ),
    ])
