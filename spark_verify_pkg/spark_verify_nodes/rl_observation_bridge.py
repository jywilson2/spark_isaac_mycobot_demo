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
Bridge block detections and joint states into the RL observation vector.

ROS 2 FUNDAMENTALS — SENSOR FUSION VIA LATEST-VALUE CACHING: This node
merges two asynchronous streams (vision detections and joint states) that
arrive at different rates from different processes. The simplest fusion
strategy, used here, is to cache the most recent message from each stream
and let an independent timer snapshot both caches at a fixed rate. More
rigorous systems use message_filters time synchronizers to pair messages by
header timestamp; for the mock ecosystem, latest-value fusion at 10 Hz is
deterministic enough for the tests.
"""

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
        # The mock has no gripper or FK, so end-effector height and grasp
        # state are fixed parameters rather than live estimates.
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

        # The MDP object owns the observation LAYOUT so this node cannot
        # drift out of sync with what the policy was trained on.
        self._mdp = MyCobotPickPlaceMDP()
        # Latest-value caches, None until the first message arrives.
        self._latest_detection: BlockDetection | None = None
        self._latest_joint_state: JointState | None = None

        self._publisher = self.create_publisher(RlObservation, observation_topic, 10)
        # Two independent subscriptions feed the caches. rclpy's default
        # single-threaded executor runs all callbacks sequentially, so no
        # locking is needed around the cache variables.
        self.create_subscription(BlockDetection, detection_topic, self._on_detection, 10)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)
        # Publish observations at a steady 10 Hz, decoupled from the input
        # rates — RL policies expect a fixed control frequency.
        self.create_timer(0.1, self._publish_observation)

    def _on_detection(self, detection: BlockDetection) -> None:
        self._latest_detection = detection

    def _on_joint_state(self, joint_state: JointState) -> None:
        self._latest_joint_state = joint_state

    def _publish_observation(self) -> None:
        # Gate until BOTH inputs have arrived at least once; publishing a
        # half-initialized observation would feed the policy garbage.
        if self._latest_detection is None or self._latest_joint_state is None:
            return
        # Also skip frames where the tracker saw no block: the observation
        # layout has no "not detected" encoding, so silence is safer than
        # zeros that look like a block in the top-left corner.
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
        # Propagate the DETECTION's header: the observation is only as
        # fresh as its perception input, and consumers measuring end-to-end
        # latency need the original capture timestamp.
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
