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
    phase3_launch = os.path.join(package_share, 'launch', 'phase3_mock_ecosystem.launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(phase3_launch)),
        Node(
            package='spark_verify_pkg',
            executable='mock_usb_camera_hil',
            name='mock_usb_camera_hil',
            output='screen',
            parameters=[{
                'frame_topic': '/edge/usb_camera/frame',
                'stats_topic': '/edge/usb_camera/stats',
                'publish_rate_hz': 15.0,
            }],
        ),
        Node(
            package='spark_verify_pkg',
            executable='edge_deployment_node',
            name='edge_deployment_node',
            output='screen',
            parameters=[{
                'inference_topic': '/mycobot/policy/inference',
                'serial_topic': '/edge/hardware/serial_command',
                'health_topic': '/edge/health',
                'frame_stats_topic': '/edge/usb_camera/stats',
                'max_inference_latency_ms': 50.0,
                'max_joint_target_rad': 2.8,
            }],
        ),
    ])
