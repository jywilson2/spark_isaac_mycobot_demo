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

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='spark_verify_pkg',
            executable='joint_command_dispatcher',
            name='joint_command_dispatcher',
            output='screen',
            parameters=[{
                'joint_commands_topic': '/mycobot/joint_commands',
                'dispatch_delay_sec': 1.0,
                'target_positions': [0.4, -0.2, 0.6, -0.3, 0.5, -0.1],
                'joint_name_mode': 'urdf',
            }],
        ),
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
    ])
