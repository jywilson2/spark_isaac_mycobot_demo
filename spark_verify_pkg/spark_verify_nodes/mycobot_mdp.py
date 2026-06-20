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

"""MyCobot pick-and-place MDP definition for Isaac Lab integration."""

from dataclasses import dataclass
from typing import Sequence

from spark_verify_nodes.articulation_model_py import DEFAULT_JOINT_NAMES
from spark_verify_nodes.reward_function import (
    build_observation_vector,
    compute_total_reward,
    RewardWeights,
)


@dataclass(frozen=True)
class MyCobotMdpConfig:
    target_centroid_x: float = 0.5
    target_centroid_y: float = 0.5
    target_lift_height: float = 0.15
    action_dim: int = 6


class MyCobotPickPlaceMDP:
    """Mock MDP used for pre-training safety and observation verification."""

    OBSERVATION_DIM = 2 + 4 + 2 + len(DEFAULT_JOINT_NAMES)

    def __init__(self, config: MyCobotMdpConfig | None = None) -> None:
        self._config = config or MyCobotMdpConfig()

    @property
    def observation_dim(self) -> int:
        return self.OBSERVATION_DIM

    @property
    def action_dim(self) -> int:
        return self._config.action_dim

    def build_observation(
        self,
        centroid_x: float,
        centroid_y: float,
        bbox_xyxy: Sequence[float],
        joint_positions: Sequence[float],
        end_effector_z: float,
        is_grasped: bool,
    ) -> list[float]:
        return build_observation_vector(
            centroid_x,
            centroid_y,
            bbox_xyxy,
            joint_positions,
            end_effector_z,
            1.0 if is_grasped else 0.0,
        )

    def compute_reward(
        self,
        centroid_x: float,
        centroid_y: float,
        end_effector_x: float,
        end_effector_y: float,
        is_grasped: bool,
        block_height: float,
        safety_penalty: float,
        weights: RewardWeights | None = None,
    ) -> float:
        return compute_total_reward(
            centroid_x,
            centroid_y,
            end_effector_x,
            end_effector_y,
            is_grasped,
            block_height,
            safety_penalty,
            target_x=self._config.target_centroid_x,
            target_y=self._config.target_centroid_y,
            target_height=self._config.target_lift_height,
            weights=weights,
        )
