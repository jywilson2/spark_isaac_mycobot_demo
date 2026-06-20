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

"""Deterministic reward terms for the MyCobot pick-and-place MDP."""

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RewardWeights:
    tracking: float = 1.0
    alignment: float = 1.5
    grasp: float = 2.0
    lift: float = 3.0
    safety: float = 5.0


def compute_tracking_reward(
    centroid_x: float,
    centroid_y: float,
    target_x: float,
    target_y: float,
) -> float:
    distance = ((centroid_x - target_x) ** 2 + (centroid_y - target_y) ** 2) ** 0.5
    return max(0.0, 1.0 - distance)


def compute_alignment_reward(
    centroid_x: float,
    centroid_y: float,
    end_effector_x: float,
    end_effector_y: float,
) -> float:
    distance = ((centroid_x - end_effector_x) ** 2 + (centroid_y - end_effector_y) ** 2) ** 0.5
    return max(0.0, 1.0 - distance)


def compute_grasp_reward(is_grasped: bool, alignment_reward: float) -> float:
    return alignment_reward if is_grasped else 0.0


def compute_lift_reward(block_height: float, target_height: float) -> float:
    if block_height <= 0.0:
        return 0.0
    return min(1.0, block_height / max(target_height, 1e-6))


def compute_total_reward(
    centroid_x: float,
    centroid_y: float,
    end_effector_x: float,
    end_effector_y: float,
    is_grasped: bool,
    block_height: float,
    safety_penalty: float,
    target_x: float = 0.5,
    target_y: float = 0.5,
    target_height: float = 0.15,
    weights: RewardWeights | None = None,
) -> float:
    active_weights = weights or RewardWeights()
    tracking = compute_tracking_reward(centroid_x, centroid_y, target_x, target_y)
    alignment = compute_alignment_reward(centroid_x, centroid_y, end_effector_x, end_effector_y)
    grasp = compute_grasp_reward(is_grasped, alignment)
    lift = compute_lift_reward(block_height, target_height)
    shaped = (
        active_weights.tracking * tracking
        + active_weights.alignment * alignment
        + active_weights.grasp * grasp
        + active_weights.lift * lift
    )
    return shaped - active_weights.safety * safety_penalty


def build_observation_vector(
    centroid_x: float,
    centroid_y: float,
    bbox_xyxy: Sequence[float],
    joint_positions: Iterable[float],
    end_effector_z: float,
    is_grasped: float,
) -> list[float]:
    observation = [
        centroid_x,
        centroid_y,
        *bbox_xyxy,
        end_effector_z,
        is_grasped,
    ]
    observation.extend(float(value) for value in joint_positions)
    return observation
