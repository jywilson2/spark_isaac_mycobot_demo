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
    phase2_mock_launch = os.path.join(package_share, 'launch', 'phase2_mock_ecosystem.launch.py')
    checkpoint_dir = os.path.join(
        os.environ.get('SPARK_REPO_ROOT', '/tmp/spark_mycobot_training_test'),
        'assets',
        'checkpoints',
        'mock_ppo_test',
    )

    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(phase2_mock_launch)),
        Node(
            package='spark_verify_pkg',
            executable='live_ppo_trainer',
            name='live_ppo_trainer',
            output='screen',
            parameters=[{
                'num_episodes': 2,
                'steps_per_episode': 4,
                'checkpoint_dir': checkpoint_dir,
                'export_onnx_weights': True,
                'require_live_sim': False,
                'use_mock_joint_names': True,
                'auto_shutdown': True,
            }],
        ),
    ])
