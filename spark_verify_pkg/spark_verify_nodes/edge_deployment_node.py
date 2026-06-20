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

"""Standalone edge deployment node with AI Hat latency and safety gating."""

import rclpy
from rclpy.node import Node

from spark_verify_nodes.camera_lens_advisor import recommend_camera_lens
from spark_verify_pkg.msg import EdgeHealth, PolicyInference, SerialCommand
from std_msgs.msg import UInt32


class EdgeDeploymentNode(Node):
    """Mock Raspberry Pi + AI Hat deployment with safety overrides."""

    def __init__(self) -> None:
        super().__init__('edge_deployment_node')
        self.declare_parameter('inference_topic', '/mycobot/policy/inference')
        self.declare_parameter('serial_topic', '/mycobot/hardware/serial_command')
        self.declare_parameter('health_topic', '/edge/health')
        self.declare_parameter('frame_stats_topic', '/edge/usb_camera/stats')
        self.declare_parameter('max_inference_latency_ms', 50.0)
        self.declare_parameter('max_joint_target_rad', 2.8)

        inference_topic = (
            self.get_parameter('inference_topic').get_parameter_value().string_value)
        serial_topic = self.get_parameter('serial_topic').get_parameter_value().string_value
        health_topic = self.get_parameter('health_topic').get_parameter_value().string_value
        frame_stats_topic = (
            self.get_parameter('frame_stats_topic').get_parameter_value().string_value)
        self._max_latency_ms = (
            self.get_parameter('max_inference_latency_ms').get_parameter_value().double_value)
        self._max_joint_target = (
            self.get_parameter('max_joint_target_rad').get_parameter_value().double_value)

        _, lens_category = recommend_camera_lens()
        self._recommended_lens = lens_category
        self._latest_latency_ms = 0.0
        self._frames_received = 0
        self._frames_dropped = 0
        self._last_serial: SerialCommand | None = None

        self._serial_pub = self.create_publisher(SerialCommand, serial_topic, 10)
        self._health_pub = self.create_publisher(EdgeHealth, health_topic, 10)
        self.create_subscription(PolicyInference, inference_topic, self._on_inference, 10)
        self.create_subscription(UInt32, frame_stats_topic, self._on_frame_stats, 10)
        self.create_timer(0.5, self._publish_health)

    def _on_frame_stats(self, stats: UInt32) -> None:
        if stats.data < self._frames_received:
            self._frames_dropped += self._frames_received - stats.data
        self._frames_received = stats.data

    def _on_inference(self, inference: PolicyInference) -> None:
        self._latest_latency_ms = inference.inference_latency_ms
        if inference.inference_latency_ms > self._max_latency_ms:
            self.get_logger().warn('Rejected inference due to latency overrun')
            return
        if any(abs(value) > self._max_joint_target for value in inference.joint_targets_rad):
            self.get_logger().warn('Rejected out-of-bound joint inference')
            return

        payload = self._encode_serial_payload(list(inference.joint_targets_rad))
        serial = SerialCommand()
        serial.header = inference.header
        serial.command_id = 0x22
        serial.payload = payload
        self._serial_pub.publish(serial)
        self._last_serial = serial

    @staticmethod
    def _encode_serial_payload(joint_targets_rad: list[float]) -> list[int]:
        packet = [0xFE, 0xFE, 15, 0x22]
        for angle in joint_targets_rad[:6]:
            degrees_x10 = int(round(angle * 180.0 / 3.141592653589793 * 10.0))
            packet.append((degrees_x10 >> 8) & 0xFF)
            packet.append(degrees_x10 & 0xFF)
        checksum = sum(packet[2:]) & 0xFF
        packet.append(checksum)
        return packet

    def _publish_health(self) -> None:
        health = EdgeHealth()
        health.header.stamp = self.get_clock().now().to_msg()
        health.inference_latency_ms = self._latest_latency_ms
        health.max_latency_threshold_ms = self._max_latency_ms
        health.frames_received = self._frames_received
        health.frames_dropped = self._frames_dropped
        health.latency_ok = self._latest_latency_ms <= self._max_latency_ms
        health.frame_pipeline_ok = self._frames_dropped == 0 and self._frames_received > 0
        health.recommended_lens = self._recommended_lens
        self._health_pub.publish(health)


def main() -> None:
    rclpy.init()
    node = EdgeDeploymentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
