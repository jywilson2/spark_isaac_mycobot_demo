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
            executable='mock_articulation_bridge',
            name='mock_articulation_bridge',
            output='screen',
            parameters=[{
                'joint_commands_topic': '/mycobot/joint_commands',
                'joint_states_topic': '/mycobot/joint_states',
                'base_frame': 'base_link',
            }],
        ),
        Node(
            package='spark_verify_pkg',
            executable='mock_camera_publisher',
            name='mock_camera_publisher',
            output='screen',
            parameters=[{
                'rgb_topic': '/mycobot/camera/rgb',
                'nitros_topic': '/mycobot/camera/nitros/rgb',
                'frame_id': 'camera_optical_frame',
                'publish_rate_hz': 10.0,
            }],
        ),
        Node(
            package='spark_verify_pkg',
            executable='joint_command_dispatcher',
            name='joint_command_dispatcher',
            output='screen',
            parameters=[{
                'joint_commands_topic': '/mycobot/joint_commands',
                'dispatch_delay_sec': 0.5,
                'target_positions': [0.4, -0.2, 0.6, -0.3, 0.5, -0.1],
            }],
        ),
    ])
