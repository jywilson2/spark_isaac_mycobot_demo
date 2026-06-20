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

"""Bridge block detections and joint states into the RL observation vector."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from spark_verify_nodes.mycobot_mdp import MyCobotPickPlaceMDP
from spark_verify_pkg.msg import BlockDetection, RlObservation


class RlObservationBridgeNode(Node):
    """Publish flattened MDP observations for downstream RL consumers."""

    def __init__(self) -> None:
        super().__init__('rl_observation_bridge')
        self.declare_parameter('detection_topic', '/mycobot/vision/block_detection')
        self.declare_parameter('joint_states_topic', '/mycobot/joint_states')
        self.declare_parameter('observation_topic', '/mycobot/rl/observation')
        self.declare_parameter('end_effector_z', 0.12)
        self.declare_parameter('is_grasped', False)

        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        joint_states_topic = (
            self.get_parameter('joint_states_topic').get_parameter_value().string_value)
        observation_topic = (
            self.get_parameter('observation_topic').get_parameter_value().string_value)
        self._end_effector_z = (
            self.get_parameter('end_effector_z').get_parameter_value().double_value)
        self._is_grasped = (
            self.get_parameter('is_grasped').get_parameter_value().bool_value)

        self._mdp = MyCobotPickPlaceMDP()
        self._latest_detection: BlockDetection | None = None
        self._latest_joint_state: JointState | None = None

        self._publisher = self.create_publisher(RlObservation, observation_topic, 10)
        self.create_subscription(BlockDetection, detection_topic, self._on_detection, 10)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)
        self.create_timer(0.1, self._publish_observation)

    def _on_detection(self, detection: BlockDetection) -> None:
        self._latest_detection = detection

    def _on_joint_state(self, joint_state: JointState) -> None:
        self._latest_joint_state = joint_state

    def _publish_observation(self) -> None:
        if self._latest_detection is None or self._latest_joint_state is None:
            return
        if not self._latest_detection.detected:
            return

        observation = self._mdp.build_observation(
            self._latest_detection.centroid_x,
            self._latest_detection.centroid_y,
            list(self._latest_detection.bbox_xyxy),
            list(self._latest_joint_state.position),
            self._end_effector_z,
            self._is_grasped,
        )

        message = RlObservation()
        message.header = self._latest_detection.header
        message.observation_dim = len(observation)
        message.observation = observation
        self._publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = RlObservationBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
