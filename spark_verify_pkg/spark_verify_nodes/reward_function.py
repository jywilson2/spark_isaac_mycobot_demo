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
Deterministic reward terms for the MyCobot pick-and-place MDP.

RL BACKGROUND: In reinforcement learning the reward function is the ONLY
training signal — the policy network learns whatever behaviour maximizes it.
A sparse reward ("+1 when the block is lifted") is very hard to learn from,
because random exploration almost never stumbles onto success. This module
instead uses REWARD SHAPING: several dense, graded terms that each pay out
partial credit for progress toward the goal:

    tracking   - keep the block visible near the camera's target point
    alignment  - move the end effector toward the block
    grasp      - alignment credit only while the gripper reports a grasp
    lift       - raise the block toward the target height
    safety     - SUBTRACTED penalty from the safety boundary evaluator

Every term is a pure function of its inputs (no ROS, no randomness), so the
Phase 2 tests can assert exact values before any expensive training run.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


# frozen=True makes the dataclass immutable (hashable, safe to share between
# threads/callbacks) — good practice for configuration objects.
@dataclass(frozen=True)
class RewardWeights:
    """Relative importance of each shaped term; lift dominates, then grasp."""

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
    """
    Reward keeping the block centroid near the target image point.

    Inputs are NORMALIZED image coordinates (0..1, from the vision tracker),
    which makes the reward independent of camera resolution. The reward
    decays linearly with Euclidean distance and clamps at zero so it never
    goes negative (penalties are the safety term's job).
    """
    distance = ((centroid_x - target_x) ** 2 + (centroid_y - target_y) ** 2) ** 0.5
    return max(0.0, 1.0 - distance)


def compute_alignment_reward(
    centroid_x: float,
    centroid_y: float,
    end_effector_x: float,
    end_effector_y: float,
) -> float:
    """Reward moving the end effector toward the block (same linear decay)."""
    distance = ((centroid_x - end_effector_x) ** 2 + (centroid_y - end_effector_y) ** 2) ** 0.5
    return max(0.0, 1.0 - distance)


def compute_grasp_reward(is_grasped: bool, alignment_reward: float) -> float:
    """
    Pay the alignment credit again while grasping.

    Coupling grasp to alignment (instead of a flat bonus) prevents a classic
    reward-hacking failure: closing the gripper far away from the block to
    farm a constant grasp payout.
    """
    return alignment_reward if is_grasped else 0.0


def compute_lift_reward(block_height: float, target_height: float) -> float:
    """
    Reward raising the block, saturating at the target height.

    The max(..., 1e-6) guards against division by zero if a caller ever
    passes target_height=0.
    """
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
    """
    Combine the shaped terms into the scalar reward the RL loop consumes.

    Note the sign convention: shaped terms are added, the safety penalty is
    subtracted with the largest weight (5.0), so no amount of task progress
    can make violating a safety boundary worthwhile.
    """
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
    ee_to_target_delta: Sequence[float],
    target_valid: float,
    in_contact: float,
    joint_positions: Iterable[float],
) -> list[float]:
    """
    Flatten the motion-policy state into the fixed observation layout.

    The policy network consumes a flat float vector:

        index 0-2 : EE-to-target delta (m, normalized by max reach in trainer)
        index 3   : target_valid flag (0.0 / 1.0)
        index 4   : in_contact flag (0.0 / 1.0)
        index 5-10: six joint positions (rad)

    Changing this layout invalidates any previously-trained policy weights.
    """
    observation = [
        float(ee_to_target_delta[0]),
        float(ee_to_target_delta[1]),
        float(ee_to_target_delta[2]),
        float(target_valid),
        float(in_contact),
    ]
    observation.extend(float(value) for value in joint_positions)
    return observation
