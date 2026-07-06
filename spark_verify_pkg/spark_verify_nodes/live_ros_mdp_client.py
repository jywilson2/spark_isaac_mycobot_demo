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

"""ROS client helpers for live MDP stepping against Isaac Sim topics."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, TYPE_CHECKING

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from spark_verify_nodes.isaac_lab_mdp_env import IsaacLabMyCobotPickPlaceEnv
from spark_verify_nodes.mycobot_joint_names import mock_joint_names, urdf_joint_names
from spark_verify_nodes.ppo_policy import PpoPolicy
from spark_verify_pkg.msg import BlockDetection, RlObservation

if TYPE_CHECKING:
    from rclpy.executors import Executor


@dataclass(frozen=True)
class LiveStepResult:
    observation: list[float]
    reward: float
    safety_penalty: float
    boundary_violated: bool
    joint_positions: list[float]


class LiveRosMdpClient:
    """Synchronously wait for live ROS observations and publish joint commands."""

    def __init__(
        self,
        node: Node,
        *,
        observation_topic: str = '/mycobot/rl/observation',
        detection_topic: str = '/mycobot/vision/block_detection',
        joint_states_topic: str = '/mycobot/joint_states',
        joint_commands_topic: str = '/mycobot/joint_commands',
        end_effector_x: float = 0.15,
        end_effector_y: float = 0.0,
        end_effector_z: float = 0.12,
        block_height: float = 0.0,
        is_grasped: bool = False,
        use_mock_joint_names: bool = False,
    ) -> None:
        self._node = node
        self._env = IsaacLabMyCobotPickPlaceEnv()
        self._end_effector_x = end_effector_x
        self._end_effector_y = end_effector_y
        self._end_effector_z = end_effector_z
        self._block_height = block_height
        self._is_grasped = is_grasped
        self._joint_names = (
            mock_joint_names() if use_mock_joint_names else urdf_joint_names())

        self._latest_observation: RlObservation | None = None
        self._latest_detection: BlockDetection | None = None
        self._latest_joint_state: JointState | None = None

        self._command_pub = node.create_publisher(JointState, joint_commands_topic, 10)
        node.create_subscription(RlObservation, observation_topic, self._on_observation, 10)
        node.create_subscription(BlockDetection, detection_topic, self._on_detection, 10)
        node.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)

    @property
    def observation_dim(self) -> int:
        return self._env.observation_dim

    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    @property
    def safety_config(self):
        return self._env.safety_config

    def _on_observation(self, observation: RlObservation) -> None:
        self._latest_observation = observation

    def _on_detection(self, detection: BlockDetection) -> None:
        self._latest_detection = detection

    def _on_joint_state(self, joint_state: JointState) -> None:
        self._latest_joint_state = joint_state

    @property
    def latest_joint_state(self) -> JointState | None:
        return self._latest_joint_state

    def _spin_once(self, executor: Executor | None, timeout_sec: float) -> None:
        if executor is not None:
            executor.spin_once(timeout_sec=timeout_sec)
        else:
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def wait_for_observation(
        self,
        timeout_sec: float = 15.0,
        *,
        executor: Executor | None = None,
    ) -> RlObservation:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self._spin_once(executor, 0.05)
            if self._latest_observation is not None:
                return self._latest_observation
        raise TimeoutError(
            f'Timed out waiting for observation on {self._node.get_namespace()}')

    def compute_reward(self) -> LiveStepResult:
        if (
            self._latest_observation is None
            or self._latest_detection is None
            or self._latest_joint_state is None
        ):
            raise RuntimeError('Live MDP inputs not ready for reward computation')
        if not self._latest_detection.detected:
            raise RuntimeError('Latest block detection is not marked detected')

        positions = list(self._latest_joint_state.position)
        velocities = list(self._latest_joint_state.velocity)
        if not velocities:
            velocities = [0.0] * len(positions)

        step = self._env.step_from_live_messages(
            self._latest_detection,
            joint_positions=positions,
            joint_velocities=velocities,
            end_effector_x=self._end_effector_x,
            end_effector_y=self._end_effector_y,
            end_effector_z=self._end_effector_z,
            is_grasped=self._is_grasped,
            block_height=self._block_height,
        )
        return LiveStepResult(
            observation=list(self._latest_observation.observation),
            reward=step.reward,
            safety_penalty=step.safety_penalty,
            boundary_violated=step.boundary_violated,
            joint_positions=positions,
        )

    def publish_joint_targets(self, targets: list[float]) -> None:
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self._joint_names)
        msg.position = [float(value) for value in targets]
        self._command_pub.publish(msg)

    def step_with_policy(
        self,
        policy: PpoPolicy,
        *,
        executor: Executor | None = None,
        timeout_sec: float = 15.0,
        deterministic: bool = False,
    ) -> tuple[LiveStepResult, list[float], float, float, list[float]]:
        observation_msg = self.wait_for_observation(
            timeout_sec=timeout_sec,
            executor=executor,
        )
        step_before = self.compute_reward()
        action_delta, value, log_prob = policy.act(
            observation_msg.observation,
            deterministic=deterministic,
        )
        targets = policy.apply_action_delta(
            step_before.joint_positions,
            action_delta,
            self._env.safety_config.joint_min,
            self._env.safety_config.joint_max,
        )
        self.publish_joint_targets(targets)
        return step_before, targets, value, log_prob, list(action_delta)


def spin_until(
    node: Node,
    predicate: Callable[[], bool],
    timeout_sec: float = 15.0,
) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False
