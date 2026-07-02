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

"""
Dispatches joint position commands to the mock articulation bridge.

TEST-HARNESS PATTERN: Integration tests need a deterministic stimulus. This
tiny node publishes one known 6-DOF pose so the Phase 1 test can assert that
exactly those values appear on /mycobot/joint_states and in TF. In a live
system this role is played by a controller or the RL policy output.
"""

from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from spark_verify_nodes.articulation_model_py import DEFAULT_JOINT_NAMES


class JointCommandDispatcher(Node):
    """Publishes a deterministic joint trajectory for integration verification."""

    def __init__(self) -> None:
        super().__init__('joint_command_dispatcher')
        self.declare_parameter('joint_commands_topic', '/mycobot/joint_commands')
        self.declare_parameter('dispatch_delay_sec', 0.5)
        # Declaring the default as a float list types this parameter as
        # DOUBLE_ARRAY; launch files must then pass a list of floats.
        self.declare_parameter(
            'target_positions',
            [0.4, -0.2, 0.6, -0.3, 0.5, -0.1],
        )

        topic = self.get_parameter('joint_commands_topic').get_parameter_value().string_value
        delay = self.get_parameter('dispatch_delay_sec').get_parameter_value().double_value
        param = self.get_parameter('target_positions').get_parameter_value()
        targets = list(param.double_array_value)

        self._publisher = self.create_publisher(JointState, topic, 10)
        self._targets = targets
        self._timer = self.create_timer(max(delay, 0.1), self._dispatch)
        self._logged = False

    def _dispatch(self) -> None:
        # Re-publish the same deterministic command on every tick: a single
        # shot can be lost if DDS discovery has not completed when the timer
        # first fires, which would stall the entire downstream pipeline.
        #
        # ROS 2 FUNDAMENTALS — DISCOVERY: DDS peers find each other via a
        # decentralized handshake that takes tens to hundreds of ms after
        # startup. A message published before a subscriber has matched is
        # simply never delivered (with VOLATILE durability, the default).
        # Periodic re-publish is the simplest robust pattern; the
        # alternative is a TRANSIENT_LOCAL durability QoS ("latched" topic).
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # JointState is parallel arrays: name[i] describes position[i].
        # Consumers match by name, so both fields must be filled.
        msg.name = list(DEFAULT_JOINT_NAMES)
        msg.position = self._normalize_positions(self._targets)
        self._publisher.publish(msg)
        if not self._logged:
            self._logged = True
            # get_logger() routes through /rosout so the launch process and
            # log files capture it alongside every other node's output.
            self.get_logger().info(f'Dispatching joint command targets: {msg.position}')

    @staticmethod
    def _normalize_positions(targets: List[float]) -> List[float]:
        # Tolerate mis-sized parameter arrays: pad with zeros or truncate to
        # exactly 6 entries so the outgoing message always matches the name
        # array length (a JointState invariant).
        if len(targets) == len(DEFAULT_JOINT_NAMES):
            return targets
        if len(targets) < len(DEFAULT_JOINT_NAMES):
            return targets + [0.0] * (len(DEFAULT_JOINT_NAMES) - len(targets))
        return targets[: len(DEFAULT_JOINT_NAMES)]


def main() -> None:
    rclpy.init()
    node = JointCommandDispatcher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
