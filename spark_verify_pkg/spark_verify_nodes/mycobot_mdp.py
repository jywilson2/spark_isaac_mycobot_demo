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
MyCobot pick-and-place MDP definition for Isaac Lab integration.

RL BACKGROUND: An MDP (Markov Decision Process) is the formal frame RL
operates in — at each step the agent sees an OBSERVATION, emits an ACTION,
and receives a REWARD. Isaac Lab environments are built from exactly these
pieces (observation/action/reward "manager" terms). This class collects the
project's definitions in one place:

    observation: 11-dim vector (EE delta to target, flags, 6 joints)
    action:      6 joint position targets (radians)
    reward:      shaped terms from reward_function.py minus safety penalty

It deliberately contains no simulator calls, so the observation layout and
reward math can be verified by fast unit tests (Phase 2) before being bound
to Isaac Lab's GPU-parallel environments for actual training.
"""

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
    """Task-level goal parameters (as opposed to per-term reward weights)."""

    # Where the block should sit in the image, normalized 0..1. (0.5, 0.5)
    # means "centered under the camera".
    target_centroid_x: float = 0.5
    target_centroid_y: float = 0.5
    # Lift goal in meters; the lift reward saturates at this height.
    target_lift_height: float = 0.15
    # One action per joint.
    action_dim: int = 6


class MyCobotPickPlaceMDP:
    """Mock MDP used for pre-training safety and observation verification."""

    # 3 EE delta + 2 flags + 6 joints = 11.
    OBSERVATION_DIM = 3 + 2 + len(DEFAULT_JOINT_NAMES)

    def __init__(self, config: MyCobotMdpConfig | None = None) -> None:
        # "config or default" lets callers construct MyCobotPickPlaceMDP()
        # with no arguments in the common case.
        self._config = config or MyCobotMdpConfig()

    @property
    def observation_dim(self) -> int:
        """Return the size of the flattened observation vector."""
        return self.OBSERVATION_DIM

    @property
    def action_dim(self) -> int:
        """Return the number of policy outputs (joint position targets)."""
        return self._config.action_dim

    def build_observation(
        self,
        end_effector_x: float,
        end_effector_y: float,
        end_effector_z: float,
        target_ee_x: float,
        target_ee_y: float,
        target_ee_z: float,
        target_valid: bool,
        joint_positions: Sequence[float],
        in_contact: bool = False,
    ) -> list[float]:
        """Flatten task state into the motion-policy observation layout."""

        delta = [
            target_ee_x - end_effector_x,
            target_ee_y - end_effector_y,
            target_ee_z - end_effector_z,
        ]
        return build_observation_vector(
            delta,
            1.0 if target_valid else 0.0,
            1.0 if in_contact else 0.0,
            joint_positions,
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
        """
        Evaluate the shaped reward for one step, injecting the task goals.

        The safety_penalty argument is expected to come from the C++
        safety_boundary_evaluator (total_penalty field), demonstrating how a
        deterministic C++ safety layer plugs into a Python RL loop.
        """
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
