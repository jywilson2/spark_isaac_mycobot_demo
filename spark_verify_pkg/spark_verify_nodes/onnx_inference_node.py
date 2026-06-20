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

"""Mock ONNX inference node for Phase 3 policy export verification."""

import time

import rclpy
from rclpy.node import Node

from spark_verify_nodes.mock_onnx_policy import run_mock_onnx_inference
from spark_verify_pkg.msg import PolicyInference, RlObservation


class OnnxInferenceNode(Node):
    """Run mock ONNX inference on RL observation tensors."""

    def __init__(self) -> None:
        super().__init__('onnx_inference_node')
        self.declare_parameter('observation_topic', '/mycobot/rl/observation')
        self.declare_parameter('inference_topic', '/mycobot/policy/inference')
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
        start = time.perf_counter()
        targets = run_mock_onnx_inference(list(observation_msg.observation))
        elapsed_ms = (time.perf_counter() - start) * 1000.0 + self._simulated_latency_ms

        message = PolicyInference()
        message.header = observation_msg.header
        message.joint_targets_rad = targets
        message.inference_latency_ms = elapsed_ms
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
