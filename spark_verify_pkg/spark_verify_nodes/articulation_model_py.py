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

"""Python mirror of the MyCobot joint naming convention used by the C++ bridge."""

# ROS 2 supports polyglot graphs: this package mixes rclcpp (C++) and rclpy
# (Python) nodes that interoperate over the same DDS topics. What they must
# share is the CONTRACT — message types, topic names, and here, joint names.
# sensor_msgs/JointState consumers match joints by name, so the Python
# dispatcher and the C++ articulation bridge must agree on these strings.
# They mirror the joint names in the elephantrobotics/mycobot_ros2 URDF.
DEFAULT_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
