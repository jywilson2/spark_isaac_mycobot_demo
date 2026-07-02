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

# ---------------------------------------------------------------------------
# LAUNCH COMPOSITION: Each phase's launch file INCLUDES the previous one
# (IncludeLaunchDescription), so phase N always runs the complete phase N-1
# stack plus its own nodes. This layering mirrors the incremental backlog in
# spec.md and means the Phase 4 test implicitly re-verifies Phases 1-3.
#
# Phase 2 adds the vision -> RL observation chain on top of Phase 1:
#
#   /mycobot/camera/rgb --> block_vision_tracker --> /mycobot/vision/block_detection
#   detection + /mycobot/joint_states --> rl_observation_bridge --> /mycobot/rl/observation
# ---------------------------------------------------------------------------

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ament_index resolves a package's INSTALLED share/ directory at
    # runtime. Locating included launch files this way (instead of relative
    # source paths) keeps the file working after installation, where the
    # source tree is not present.
    package_share = get_package_share_directory('spark_verify_pkg')
    phase1_launch = os.path.join(package_share, 'launch', 'phase1_mock_ecosystem.launch.py')

    return LaunchDescription([
        # Pull in the entire Phase 1 stack (articulation bridge, camera,
        # dispatcher) unchanged.
        IncludeLaunchDescription(PythonLaunchDescriptionSource(phase1_launch)),
        Node(
            package='spark_verify_pkg',
            executable='block_vision_tracker',
            name='block_vision_tracker',
            output='screen',
            parameters=[{
                'rgb_topic': '/mycobot/camera/rgb',
                'detection_topic': '/mycobot/vision/block_detection',
                # Must stay below the mock camera's painted red value (230).
                'red_threshold': 180,
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
                # Fixed stand-ins for gripper state; the mock has no FK.
                'end_effector_z': 0.12,
                'is_grasped': False,
            }],
        ),
    ])
