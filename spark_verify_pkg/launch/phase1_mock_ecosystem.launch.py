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
# ROS 2 FUNDAMENTALS — LAUNCH FILES: ROS 2 systems are compositions of many
# small processes; a launch file describes the whole composition. Python
# launch files (this format) must define generate_launch_description()
# returning a LaunchDescription of "actions" — here, three Node actions.
# The launch system starts each Node as a separate OS process, passes its
# parameters, and supervises it (propagating SIGINT on shutdown).
#
# Run manually with:
#   ros2 launch spark_verify_pkg phase1_mock_ecosystem.launch.py
#
# Phase 1 graph (topics flow left to right):
#
#   joint_command_dispatcher --/mycobot/joint_commands--> mock_articulation_bridge
#                                                          |--> /mycobot/joint_states
#                                                          '--> /tf (link transforms)
#   mock_camera_publisher --> /mycobot/camera/rgb, /mycobot/camera/nitros/rgb
# ---------------------------------------------------------------------------

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        # 'executable' is looked up in <install>/lib/spark_verify_pkg/.
        # C++ executables land there via install(TARGETS...), Python wrapper
        # scripts via install(PROGRAMS...). The 'parameters' dict overrides
        # the defaults each node declared with declare_parameter().
        Node(
            package='spark_verify_pkg',
            executable='mock_articulation_bridge',  # C++ node (rclcpp)
            name='mock_articulation_bridge',
            output='screen',  # forward stdout/logs to the launch console
            parameters=[{
                'joint_commands_topic': '/mycobot/joint_commands',
                'joint_states_topic': '/mycobot/joint_states',
                'base_frame': 'base_link',
            }],
        ),
        Node(
            package='spark_verify_pkg',
            executable='mock_camera_publisher',  # Python node (rclpy)
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
                # The deterministic test pose. test_phase1_integration.py
                # asserts these exact values appear in /mycobot/joint_states,
                # so the two files must stay in sync.
                'target_positions': [0.4, -0.2, 0.6, -0.3, 0.5, -0.1],
            }],
        ),
    ])
