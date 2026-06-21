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

"""Isaac Lab MDP environment wrapper fed by live Isaac Sim ROS 2 topics."""

from __future__ import annotations

from dataclasses import dataclass

from spark_verify_nodes.mycobot_mdp import MyCobotPickPlaceMDP
from spark_verify_nodes.reward_function import compute_total_reward
from spark_verify_nodes.safety_boundary_evaluator_py import (
    evaluate_safety_boundaries,
    SafetyBoundaryConfig,
)
from spark_verify_pkg.msg import BlockDetection, RlObservation


@dataclass(frozen=True)
class LiveMdpStepResult:
    observation: list[float]
    reward: float
    safety_penalty: float
    boundary_violated: bool


class IsaacLabMyCobotPickPlaceEnv:
    """Live MDP facade for Isaac Lab training against a running Isaac Sim scene."""

    def __init__(
        self,
        mdp: MyCobotPickPlaceMDP | None = None,
        safety_config: SafetyBoundaryConfig | None = None,
    ) -> None:
        self._mdp = mdp or MyCobotPickPlaceMDP()
        self._safety_config = safety_config or SafetyBoundaryConfig()

    @property
    def observation_dim(self) -> int:
        return self._mdp.observation_dim

    @property
    def action_dim(self) -> int:
        return self._mdp.action_dim

    @property
    def safety_config(self) -> SafetyBoundaryConfig:
        return self._safety_config

    def build_observation_from_messages(
        self,
        detection: BlockDetection,
        joint_positions: list[float],
        end_effector_z: float,
        is_grasped: bool,
    ) -> list[float]:
        return self._mdp.build_observation(
            detection.centroid_x,
            detection.centroid_y,
            list(detection.bbox_xyxy),
            joint_positions,
            end_effector_z,
            is_grasped,
        )

    def step_from_live_messages(
        self,
        detection: BlockDetection,
        joint_positions: list[float],
        joint_velocities: list[float],
        end_effector_x: float,
        end_effector_y: float,
        end_effector_z: float,
        is_grasped: bool,
        block_height: float,
    ) -> LiveMdpStepResult:
        safety = evaluate_safety_boundaries(
            self._safety_config,
            joint_positions,
            joint_velocities,
            end_effector_x,
            end_effector_y,
            end_effector_z,
        )
        observation = self.build_observation_from_messages(
            detection,
            joint_positions,
            end_effector_z,
            is_grasped,
        )
        reward = self._mdp.compute_reward(
            centroid_x=detection.centroid_x,
            centroid_y=detection.centroid_y,
            end_effector_x=end_effector_x,
            end_effector_y=end_effector_y,
            is_grasped=is_grasped,
            block_height=block_height,
            safety_penalty=safety.total_penalty,
        )
        return LiveMdpStepResult(
            observation=observation,
            reward=reward,
            safety_penalty=safety.total_penalty,
            boundary_violated=safety.boundary_violated,
        )

    def compute_reward_from_observation(
        self,
        observation_msg: RlObservation,
        end_effector_x: float,
        end_effector_y: float,
        is_grasped: bool,
        block_height: float,
        safety_penalty: float,
    ) -> float:
        if len(observation_msg.observation) < 2:
            return 0.0
        return compute_total_reward(
            observation_msg.observation[0],
            observation_msg.observation[1],
            end_effector_x,
            end_effector_y,
            is_grasped,
            block_height,
            safety_penalty,
        )
