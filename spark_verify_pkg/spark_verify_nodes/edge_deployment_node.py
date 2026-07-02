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
Standalone edge deployment node with AI Hat latency and safety gating.

PROJECT CONTEXT — PHASE 4: On the real system this node runs on a Raspberry
Pi with an AI HAT (Hailo NPU) wired directly to the MyCobot's serial port,
with NO workstation in the loop. Its responsibilities:

1. Latency gate     — reject inferences slower than the control deadline
                      (a late command is a wrong command for a moving arm).
2. Safety override  — "bare-metal" bound check on joint targets, the last
                      software layer between the neural network and motors.
3. Health telemetry — publish an EdgeHealth heartbeat with latency, frame
                      pipeline statistics, and the lens recommendation, so
                      the HIL suite (and a human operator) can watch it.

The mock keeps all three behaviours but takes inferences from the Phase 3
topic instead of an on-device NPU, and camera stats from the mock USB node.
"""

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
        # 50 ms latency budget: at the camera's 15 Hz (66 ms period) an
        # inference slower than this would arrive after the next frame.
        self.declare_parameter('max_inference_latency_ms', 50.0)
        # 2.8 rad ~= 160 deg, just inside the MyCobot's +/-165 deg limits.
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

        # Compute the optics recommendation once at startup — workspace
        # geometry does not change at runtime — and republish it in every
        # health message for operator visibility.
        _, lens_category = recommend_camera_lens()
        self._recommended_lens = lens_category
        self._latest_latency_ms = 0.0
        self._frames_received = 0
        self._frames_dropped = 0
        self._stats_baselined = False
        self._last_serial: SerialCommand | None = None

        self._serial_pub = self.create_publisher(SerialCommand, serial_topic, 10)
        self._health_pub = self.create_publisher(EdgeHealth, health_topic, 10)
        self.create_subscription(PolicyInference, inference_topic, self._on_inference, 10)
        self.create_subscription(UInt32, frame_stats_topic, self._on_frame_stats, 10)
        # Heartbeat timer: 2 Hz health reports regardless of traffic. A
        # periodic heartbeat (rather than event-driven reporting) means a
        # SILENT node is itself a detectable failure.
        self.create_timer(0.5, self._publish_health)

    def _on_frame_stats(self, stats: UInt32) -> None:
        # The camera publishes a cumulative frame counter; after the baseline
        # sample, any jump larger than one frame means frames were dropped
        # before this node could process them. A counter that moves backwards
        # indicates a camera restart, which re-baselines without penalty.
        if self._stats_baselined and stats.data > self._frames_received + 1:
            self._frames_dropped += stats.data - self._frames_received - 1
        self._frames_received = stats.data
        self._stats_baselined = True

    def _on_inference(self, inference: PolicyInference) -> None:
        # Record the latency BEFORE gating so health reports reflect what
        # the NPU is actually doing, including over-budget inferences.
        self._latest_latency_ms = inference.inference_latency_ms
        # Gate 1 — timeliness: a control command computed from a stale
        # observation is dangerous on a moving arm; drop it entirely
        # (the arm simply holds its last commanded pose).
        if inference.inference_latency_ms > self._max_latency_ms:
            self.get_logger().warn('Rejected inference due to latency overrun')
            return
        # Gate 2 — bare-metal bound check: symmetric magnitude limit on
        # every joint target. This duplicates the C++ driver's richer
        # safety evaluator BY DESIGN (defense in depth): on the Pi, this
        # node may be the only gate between the policy and the motors.
        if any(abs(value) > self._max_joint_target for value in inference.joint_targets_rad):
            self.get_logger().warn('Rejected out-of-bound joint inference')
            return

        payload = self._encode_serial_payload(list(inference.joint_targets_rad))
        serial = SerialCommand()
        serial.header = inference.header
        serial.command_id = 0x22  # pymycobot send_angles command id
        serial.payload = payload
        self._serial_pub.publish(serial)
        self._last_serial = serial

    @staticmethod
    def _encode_serial_payload(joint_targets_rad: list[float]) -> list[int]:
        # Python twin of the C++ encode_send_angles_packet (see
        # pymycobot_serial_encoder.cpp for the frame layout). Kept separate
        # because the edge unit deploys WITHOUT the workstation's compiled
        # C++ libraries — but the bytes must match exactly, which the
        # Phase 4 HIL test verifies by decoding them.
        packet = [0xFE, 0xFE, 15, 0x22]
        for angle in joint_targets_rad[:6]:
            # Radians -> tenths of a degree, the pymycobot wire unit.
            degrees_x10 = int(round(angle * 180.0 / 3.141592653589793 * 10.0))
            # Big-endian int16. Python ints are arbitrary precision, so
            # masking with 0xFF after the shift produces the correct two's
            # complement bytes even for negative angles.
            packet.append((degrees_x10 >> 8) & 0xFF)
            packet.append(degrees_x10 & 0xFF)
        # 8-bit checksum over length + command + payload (bytes 2..end).
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
        # Boolean summaries let simple consumers (dashboards, HIL asserts)
        # avoid re-deriving pass/fail from the raw numbers.
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
