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
Mock ONNX inference node for Phase 3 policy export verification.

PIPELINE POSITION: observation -> [this node] -> PolicyInference -> driver.
The node is a thin ROS wrapper around run_mock_onnx_inference(); swapping
the mock for onnxruntime.InferenceSession.run() (or a Hailo-compiled model
on the AI Hat) changes ONE line of this file, because the topic contract —
RlObservation in, PolicyInference out — stays identical. That substitution
point is the whole idea of the Phase 3 mock.
"""

import time

import rclpy
from rclpy.node import Node

from spark_verify_nodes.mock_onnx_policy import run_mock_onnx_inference
# Generated message classes: rosidl turns msg/PolicyInference.msg into the
# Python class spark_verify_pkg.msg.PolicyInference at build time.
from spark_verify_pkg.msg import PolicyInference, RlObservation


class OnnxInferenceNode(Node):
    """Run mock ONNX inference on RL observation tensors."""

    def __init__(self) -> None:
        super().__init__('onnx_inference_node')
        self.declare_parameter('observation_topic', '/mycobot/rl/observation')
        self.declare_parameter('inference_topic', '/mycobot/policy/inference')
        # Additive latency injected into the reported metric so tests can
        # exercise the edge node's latency gate deterministically (set it
        # above max_inference_latency_ms and every inference is rejected).
        self.declare_parameter('simulated_latency_ms', 5.0)

        observation_topic = (
            self.get_parameter('observation_topic').get_parameter_value().string_value)
        inference_topic = (
            self.get_parameter('inference_topic').get_parameter_value().string_value)
        self._simulated_latency_ms = (
            self.get_parameter('simulated_latency_ms').get_parameter_value().double_value)

        self._publisher = self.create_publisher(PolicyInference, inference_topic, 10)
        self.create_subscription(RlObservation, observation_topic, self._on_observation, 10)

    def _on_observation(self, observation_msg: RlObservation) -> None:
        # time.perf_counter() is a monotonic high-resolution clock — the
        # right tool for measuring durations (time.time() can jump if the
        # system clock is adjusted). This wall-clock measurement pattern is
        # exactly what a real inference node does around its runtime call.
        start = time.perf_counter()
        targets = run_mock_onnx_inference(list(observation_msg.observation))
        elapsed_ms = (time.perf_counter() - start) * 1000.0 + self._simulated_latency_ms

        message = PolicyInference()
        # Propagate the observation's header so the original perception
        # timestamp survives the whole chain (camera -> ... -> serial).
        message.header = observation_msg.header
        # joint_targets_rad is float32[6] (fixed size) — rclpy will reject
        # any list that is not exactly 6 elements, which is why the mock
        # policy always pads its output to 6.
        message.joint_targets_rad = targets
        message.inference_latency_ms = elapsed_ms
        # accepted=True marks this inference as valid from this stage's
        # perspective; downstream safety gates may still veto it.
        message.accepted = True
        self._publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = OnnxInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
