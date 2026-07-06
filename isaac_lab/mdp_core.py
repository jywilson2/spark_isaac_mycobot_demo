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

"""Pure-Python MDP contract shared by Isaac Lab training and ROS verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Keep in sync with spark_verify_nodes.mycobot_mdp.MyCobotPickPlaceMDP.OBSERVATION_DIM
OBSERVATION_DIM = 14
ACTION_DIM = 6

REVOLUTE_JOINT_NAMES = (
    'joint2_to_joint1',
    'joint3_to_joint2',
    'joint4_to_joint3',
    'joint5_to_joint4',
    'joint6_to_joint5',
    'joint6output_to_joint6',
)


@dataclass(frozen=True)
class MdpTaskConfig:
    target_centroid_x: float = 0.5
    target_centroid_y: float = 0.5
    target_lift_height: float = 0.15
    action_scale: float = 0.05


def build_observation_vector(
    centroid_x: float,
    centroid_y: float,
    bbox_xyxy: Sequence[float],
    joint_positions: Sequence[float],
    end_effector_z: float,
    is_grasped: bool,
) -> list[float]:
    """Build the 14-dim observation vector used across Isaac Lab and ROS stacks."""

    if len(bbox_xyxy) != 4:
        raise ValueError('bbox_xyxy must contain four values')
    if len(joint_positions) != ACTION_DIM:
        raise ValueError(f'joint_positions must contain {ACTION_DIM} values')
    return [
        float(centroid_x),
        float(centroid_y),
        float(bbox_xyxy[0]),
        float(bbox_xyxy[1]),
        float(bbox_xyxy[2]),
        float(bbox_xyxy[3]),
        float(end_effector_z),
        1.0 if is_grasped else 0.0,
        *[float(value) for value in joint_positions],
    ]


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
    if target_height <= 0.0:
        return 0.0
    return max(0.0, min(1.0, block_height / target_height))


def compute_total_reward(
    centroid_x: float,
    centroid_y: float,
    end_effector_x: float,
    end_effector_y: float,
    is_grasped: bool,
    block_height: float,
    safety_penalty: float,
    *,
    target_centroid_x: float = 0.5,
    target_centroid_y: float = 0.5,
    target_lift_height: float = 0.15,
) -> float:
    tracking = compute_tracking_reward(
        centroid_x, centroid_y, target_centroid_x, target_centroid_y)
    alignment = compute_alignment_reward(
        centroid_x, centroid_y, end_effector_x, end_effector_y)
    grasp = compute_grasp_reward(is_grasped, alignment)
    lift = compute_lift_reward(block_height, target_lift_height)
    return tracking + alignment + grasp + lift - safety_penalty


def apply_action_delta(
    current_joint_positions: Sequence[float],
    action_delta: Sequence[float],
    joint_min: Sequence[float],
    joint_max: Sequence[float],
    *,
    action_scale: float = 0.05,
) -> list[float]:
    targets: list[float] = []
    for current, delta, lower, upper in zip(
        current_joint_positions,
        action_delta,
        joint_min,
        joint_max,
        strict=True,
    ):
        target = float(current) + float(delta) * action_scale
        targets.append(max(lower, min(upper, target)))
    return targets
