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

"""Live Phase 2 smoke monitor for MDP reward and safety penalization."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from spark_verify_nodes.isaac_lab_mdp_env import IsaacLabMyCobotPickPlaceEnv
from spark_verify_nodes.safety_boundary_evaluator_py import evaluate_safety_boundaries
from spark_verify_pkg.msg import BlockDetection, RlObservation
from std_msgs.msg import Float32


class LiveMdpRewardMonitor(Node):
    """Publish live reward and safety penalty metrics for integration tests."""

    def __init__(self) -> None:
        super().__init__('live_mdp_reward_monitor')
        self.declare_parameter('detection_topic', '/mycobot/vision/block_detection')
        self.declare_parameter('observation_topic', '/mycobot/rl/observation')
        self.declare_parameter('joint_states_topic', '/mycobot/joint_states')
        self.declare_parameter('reward_topic', '/mycobot/rl/live_reward')
        self.declare_parameter('safety_penalty_topic', '/mycobot/rl/live_safety_penalty')
        self.declare_parameter('end_effector_x', 0.5)
        self.declare_parameter('end_effector_y', 0.5)
        self.declare_parameter('end_effector_z', 0.12)
        self.declare_parameter('block_height', 0.0)
        self.declare_parameter('is_grasped', False)

        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        observation_topic = (
            self.get_parameter('observation_topic').get_parameter_value().string_value)
        joint_states_topic = (
            self.get_parameter('joint_states_topic').get_parameter_value().string_value)
        reward_topic = self.get_parameter('reward_topic').get_parameter_value().string_value
        safety_topic = (
            self.get_parameter('safety_penalty_topic').get_parameter_value().string_value)

        self._end_effector_x = (
            self.get_parameter('end_effector_x').get_parameter_value().double_value)
        self._end_effector_y = (
            self.get_parameter('end_effector_y').get_parameter_value().double_value)
        self._end_effector_z = (
            self.get_parameter('end_effector_z').get_parameter_value().double_value)
        self._block_height = self.get_parameter('block_height').get_parameter_value().double_value
        self._is_grasped = self.get_parameter('is_grasped').get_parameter_value().bool_value

        self._env = IsaacLabMyCobotPickPlaceEnv()
        self._latest_detection: BlockDetection | None = None
        self._latest_observation: RlObservation | None = None
        self._latest_joint_state: JointState | None = None

        self._reward_pub = self.create_publisher(Float32, reward_topic, 10)
        self._safety_pub = self.create_publisher(Float32, safety_topic, 10)
        self.create_subscription(BlockDetection, detection_topic, self._on_detection, 10)
        self.create_subscription(RlObservation, observation_topic, self._on_observation, 10)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)
        self.create_timer(0.2, self._publish_metrics)

    def _on_detection(self, detection: BlockDetection) -> None:
        self._latest_detection = detection

    def _on_observation(self, observation: RlObservation) -> None:
        self._latest_observation = observation

    def _on_joint_state(self, joint_state: JointState) -> None:
        self._latest_joint_state = joint_state

    def _publish_metrics(self) -> None:
        if (
            self._latest_detection is None
            or self._latest_observation is None
            or self._latest_joint_state is None
        ):
            return
        if not self._latest_detection.detected:
            return

        positions = list(self._latest_joint_state.position)
        velocities = list(self._latest_joint_state.velocity)
        if not velocities:
            velocities = [0.0] * len(positions)

        safety = evaluate_safety_boundaries(
            self._env.safety_config,
            positions,
            velocities,
            self._end_effector_x,
            self._end_effector_y,
            self._end_effector_z,
        )
        reward = self._env.compute_reward_from_observation(
            self._latest_observation,
            self._end_effector_x,
            self._end_effector_y,
            self._is_grasped,
            self._block_height,
            safety.total_penalty,
        )

        reward_msg = Float32()
        reward_msg.data = float(reward)
        self._reward_pub.publish(reward_msg)

        safety_msg = Float32()
        safety_msg.data = float(safety.total_penalty)
        self._safety_pub.publish(safety_msg)


def main() -> None:
    rclpy.init()
    node = LiveMdpRewardMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
