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

"""Pure-Python MDP contract shared by Isaac Lab training and ROS verification.

This module is the **single source of truth** for observation layout, reward
semantics, and workspace sampling used by both Isaac Lab (GPU sim) and ROS 2
verification nodes. See spec.md § Phase 2 and README.md for the full pipeline.

External references:
  - Potential-based reward shaping: Ng et al., ICML 1999
    https://people.eecs.berkeley.edu/~pabbeel/papers/ng99policyinvariance.pdf
  - Operational-space / Cartesian control: Khatib, IJRR 1987
    https://doi.org/10.1177/027836498700600304
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import random

# Keep in sync with spark_verify_nodes.mycobot_mdp.MyCobotPickPlaceMDP.OBSERVATION_DIM
# Motion-policy layout: 3 EE-to-target delta + target_valid + reached + 6 joints = 11.
OBSERVATION_DIM = 11

# Phase 2 reach: Cartesian Δx, Δy, Δz in the robot base frame (task-space RL).
REACH_ACTION_DIM = 3

# Phase 6 contact-and-push still uses joint deltas (vision + contact stack).
ACTION_DIM = 6

REVOLUTE_JOINT_NAMES = (
    'joint2_to_joint1',
    'joint3_to_joint2',
    'joint4_to_joint3',
    'joint5_to_joint4',
    'joint6_to_joint5',
    'joint6output_to_joint6',
)

END_EFFECTOR_BODY_NAME = 'joint6_flange'
TARGET_PUSH_DISTANCE_M = 0.005
EE_REACH_TOLERANCE_M = 0.025
DEFAULT_TARGET_REACH_SUCCESS_RATE = 0.99

# Reachable workspace annulus on the table (meters from robot base).
MIN_BLOCK_REACH_M = 0.12
MAX_BLOCK_REACH_M = 0.28
MIN_EE_TARGET_Z_M = 0.08
MAX_EE_TARGET_Z_M = 0.22
BLOCK_SPAWN_Z = 0.021


@dataclass(frozen=True)
class ReachCurriculumStage:
    """One row in the staged reach curriculum (see spec.md § Phase 2)."""

    name: str
    easy_fraction: float
    near_ee_radius_m: float
    min_success_rate_to_advance: float


# Staged curriculum: near current EE → medium annulus → full workspace.
REACH_CURRICULUM_STAGES: tuple[ReachCurriculumStage, ...] = (
    ReachCurriculumStage('near_ee', 1.0, 0.04, 0.90),
    ReachCurriculumStage('medium', 0.55, 0.09, 0.95),
    ReachCurriculumStage('full', 0.15, 0.12, 0.99),
)


@dataclass(frozen=True)
class ReachTaskConfig:
    """Phase 2 EE reach-to-target task parameters.

    ``cartesian_action_scale`` is the maximum EE displacement (meters) per
    control step when the policy outputs ±1.0 on a Cartesian axis. The policy
    learns *which direction* to move; the Jacobian maps that intent to joints.
  """

    reach_tolerance_m: float = EE_REACH_TOLERANCE_M
    min_reach_m: float = MIN_BLOCK_REACH_M
    max_reach_m: float = MAX_BLOCK_REACH_M
    min_target_z_m: float = MIN_EE_TARGET_Z_M
    max_target_z_m: float = MAX_EE_TARGET_Z_M
    cartesian_action_scale: float = 0.025
    jacobian_damping: float = 0.05
    time_penalty: float = 0.04
    reach_bonus: float = 30.0
    progress_scale: float = 40.0
    timeout_penalty: float = 5.0
    # Legacy alias used by joint-space helpers and Phase 6 bridge code.
    action_scale: float = 0.18


@dataclass(frozen=True)
class MdpTaskConfig:
    """Contact-and-push task parameters (grasping not required)."""

    target_push_distance_m: float = TARGET_PUSH_DISTANCE_M
    contact_horizontal_tolerance_m: float = 0.040
    contact_vertical_max_gap_m: float = 0.025
    contact_vertical_min_gap_m: float = -0.025
    max_reach_m: float = MAX_BLOCK_REACH_M
    block_half_size_m: float = 0.02
    action_scale: float = 0.05
    approach_offset_m: float = 0.03
    # Legacy fields kept for ROS bridge compatibility.
    target_centroid_x: float = 0.5
    target_centroid_y: float = 0.5
    target_lift_height: float = 0.15


def sample_reachable_block_xy(
    count: int,
    *,
    min_reach_m: float = MIN_BLOCK_REACH_M,
    max_reach_m: float = MAX_BLOCK_REACH_M,
    robot_base_x: float = 0.0,
    robot_base_y: float = 0.0,
) -> list[tuple[float, float]]:
    """Sample random table (x, y) positions guaranteed within arm reach."""

    samples: list[tuple[float, float]] = []
    for _ in range(max(count, 0)):
        angle = random.uniform(0.0, 2.0 * math.pi)
        radius = random.uniform(min_reach_m, max_reach_m)
        x = robot_base_x + radius * math.cos(angle)
        y = robot_base_y + radius * math.sin(angle)
        samples.append((x, y))
    return samples


def resolve_curriculum_stage(
    rolling_success_rate: float,
    *,
    stages: Sequence[ReachCurriculumStage] = REACH_CURRICULUM_STAGES,
) -> ReachCurriculumStage:
    """Return the active curriculum stage from rolling episode success rate.

    Each stage's ``min_success_rate_to_advance`` is the rolling success required
    to *leave* that stage. Until the threshold is met, targets stay easy (near EE).
    """

    for stage in stages:
        if rolling_success_rate < stage.min_success_rate_to_advance:
            return stage
    return stages[-1]


def sample_reachable_ee_xyz(
    count: int,
    *,
    min_reach_m: float = MIN_BLOCK_REACH_M,
    max_reach_m: float = MAX_BLOCK_REACH_M,
    min_z_m: float = MIN_EE_TARGET_Z_M,
    max_z_m: float = MAX_EE_TARGET_Z_M,
    robot_base_x: float = 0.0,
    robot_base_y: float = 0.0,
    easy_fraction: float = 0.0,
    easy_center: tuple[float, float, float] = (0.20, 0.0, 0.12),
    easy_radius_m: float = 0.05,
    near_ee_center: tuple[float, float, float] | None = None,
    near_ee_radius_m: float = 0.04,
) -> list[tuple[float, float, float]]:
    """Sample random EE target positions within the arm reach envelope.

    When ``near_ee_center`` is set and the easy draw succeeds, targets are
    sampled in a small ball around the **current EE pose** (curriculum stage 1).
    Otherwise targets fall back to ``easy_center`` or the full workspace annulus.
    """

    samples: list[tuple[float, float, float]] = []
    for _ in range(max(count, 0)):
        if easy_fraction > 0.0 and random.random() < easy_fraction:
            if near_ee_center is not None:
                cx, cy, cz = near_ee_center
                radius_limit = near_ee_radius_m
            else:
                cx, cy, cz = easy_center
                radius_limit = easy_radius_m
            angle = random.uniform(0.0, 2.0 * math.pi)
            radius = random.uniform(0.0, radius_limit)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            z = cz + random.uniform(-0.03, 0.03)
            samples.append((x, y, max(min_z_m, min(max_z_m, z))))
            continue
        angle = random.uniform(0.0, 2.0 * math.pi)
        radius = random.uniform(min_reach_m, max_reach_m)
        x = robot_base_x + radius * math.cos(angle)
        y = robot_base_y + radius * math.sin(angle)
        z = random.uniform(min_z_m, max_z_m)
        samples.append((x, y, z))
    return samples


def is_ee_at_target(
    end_effector_x: float,
    end_effector_y: float,
    end_effector_z: float,
    target_x: float,
    target_y: float,
    target_z: float,
    *,
    tolerance_m: float = EE_REACH_TOLERANCE_M,
) -> bool:
    distance = (
        (target_x - end_effector_x) ** 2
        + (target_y - end_effector_y) ** 2
        + (target_z - end_effector_z) ** 2
    ) ** 0.5
    return distance <= tolerance_m


def compute_reach_task_reward(
    end_effector_x: float,
    end_effector_y: float,
    end_effector_z: float,
    target_x: float,
    target_y: float,
    target_z: float,
    *,
    prev_distance_m: float | None = None,
    cfg: ReachTaskConfig | None = None,
) -> tuple[float, float]:
    """Return potential-based reach reward and current EE-to-target distance.

    Shaping uses ``progress_scale * (d_prev - d_now)`` so moving closer always
    earns positive credit; standing still or moving away earns none. A large
    terminal ``reach_bonus`` is added only inside ``reach_tolerance_m``. There
    is no perpetual proximity term — the policy must actually enter the success
    volume to collect the bonus (see Ng et al. 1999, linked in module docstring).
    """

    task = cfg or ReachTaskConfig()
    distance = (
        (target_x - end_effector_x) ** 2
        + (target_y - end_effector_y) ** 2
        + (target_z - end_effector_z) ** 2
    ) ** 0.5
    progress = 0.0
    if prev_distance_m is not None:
        progress = max(0.0, prev_distance_m - distance) * task.progress_scale
    reward = progress - task.time_penalty
    if distance <= task.reach_tolerance_m:
        reward += task.reach_bonus
    return reward, distance


def damped_least_squares_joint_delta(
    jacobian: Sequence[Sequence[float]],
    cartesian_delta: Sequence[float],
    *,
    damping: float = 0.05,
) -> list[float]:
    """Map a 3D Cartesian EE delta to joint deltas via damped least squares (DLS).

    Solves ``dq = J^T (J J^T + λ² I)⁻¹ dx`` for a 3×N positional Jacobian.
    Used in unit tests and as the reference for GPU batched code in
    ``isaac_lab/cartesian_actuation.py``. This is a **low-level actuator map**,
    not analytic IK: the RL policy still chooses ``dx`` each step.

    References:
      - Buss, "Introduction to Inverse Kinematics" (DLS section)
        https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    """

    if len(cartesian_delta) != 3:
        raise ValueError('cartesian_delta must have length 3')
    num_joints = len(jacobian[0]) if jacobian else 0
    if num_joints == 0:
        return []
    for row in jacobian:
        if len(row) != num_joints:
            raise ValueError('Jacobian rows must share the same width')

    lam2 = damping * damping
    # A = J J^T + λ² I  (3×3)
    a00 = a01 = a02 = a11 = a12 = a22 = 0.0
    for col in range(num_joints):
        j0 = jacobian[0][col]
        j1 = jacobian[1][col]
        j2 = jacobian[2][col]
        a00 += j0 * j0
        a01 += j0 * j1
        a02 += j0 * j2
        a11 += j1 * j1
        a12 += j1 * j2
        a22 += j2 * j2
    a00 += lam2
    a11 += lam2
    a22 += lam2

    det = (
        a00 * (a11 * a22 - a12 * a12)
        - a01 * (a01 * a22 - a02 * a12)
        + a02 * (a01 * a12 - a02 * a11)
    )
    if abs(det) < 1e-12:
        return [0.0] * num_joints
    inv_det = 1.0 / det
    i00 = (a11 * a22 - a12 * a12) * inv_det
    i01 = (a02 * a12 - a01 * a22) * inv_det
    i02 = (a01 * a12 - a02 * a11) * inv_det
    i11 = (a00 * a22 - a02 * a02) * inv_det
    i12 = (a02 * a01 - a00 * a12) * inv_det
    i22 = (a00 * a11 - a01 * a01) * inv_det

    dx, dy, dz = (float(v) for v in cartesian_delta)
    y0 = i00 * dx + i01 * dy + i02 * dz
    y1 = i01 * dx + i11 * dy + i12 * dz
    y2 = i02 * dx + i12 * dy + i22 * dz

    joint_delta = [0.0] * num_joints
    for col in range(num_joints):
        joint_delta[col] = (
            jacobian[0][col] * y0
            + jacobian[1][col] * y1
            + jacobian[2][col] * y2
        )
    return joint_delta


def episode_reach_success(
    reached_target: bool,
) -> bool:
    return reached_target


def is_within_arm_reach(
    block_x: float,
    block_y: float,
    *,
    robot_base_x: float = 0.0,
    robot_base_y: float = 0.0,
    min_reach_m: float = MIN_BLOCK_REACH_M,
    max_reach_m: float = MAX_BLOCK_REACH_M,
) -> bool:
    distance = ((block_x - robot_base_x) ** 2 + (block_y - robot_base_y) ** 2) ** 0.5
    return min_reach_m <= distance <= max_reach_m


def compute_target_ee_pose(
    block_x: float,
    block_y: float,
    block_z: float,
    *,
    block_half_size_m: float = 0.02,
    approach_offset_m: float = 0.03,
) -> tuple[float, float, float]:
    """Return the commanded EE pose (base frame) above the detected block top."""

    block_top_z = block_z + block_half_size_m
    return block_x, block_y, block_top_z + approach_offset_m


def build_motion_observation_vector(
    ee_to_target_delta: Sequence[float],
    target_valid: bool,
    reached: bool,
    joint_positions: Sequence[float],
    *,
    max_reach_m: float = MAX_BLOCK_REACH_M,
) -> list[float]:
    """Build the motion-policy observation (target EE known from task generator)."""

    if len(ee_to_target_delta) != 3:
        raise ValueError('ee_to_target_delta must contain three values')
    if len(joint_positions) != ACTION_DIM:
        raise ValueError(f'joint_positions must contain {ACTION_DIM} values')
    scale = max(max_reach_m, 1e-6)
    normalized_delta = [float(value) / scale for value in ee_to_target_delta]
    return [
        *normalized_delta,
        1.0 if target_valid else 0.0,
        1.0 if reached else 0.0,
        *[float(value) for value in joint_positions],
    ]


def build_observation_vector(
    centroid_x: float,
    centroid_y: float,
    bbox_xyxy: Sequence[float],
    joint_positions: Sequence[float],
    end_effector_z: float,
    is_grasped: bool,
    *,
    in_contact: bool | None = None,
    end_effector_x: float = 0.0,
    end_effector_y: float = 0.0,
    target_ee_x: float = 0.0,
    target_ee_y: float = 0.0,
    target_ee_z: float = 0.0,
    target_valid: bool = False,
) -> list[float]:
    """Build the motion-policy observation vector used across Isaac Lab and ROS."""

    _ = (centroid_x, centroid_y, bbox_xyxy, end_effector_z, is_grasped, in_contact)
    delta = [
        target_ee_x - end_effector_x,
        target_ee_y - end_effector_y,
        target_ee_z - end_effector_z,
    ]
    contact_flag = in_contact if in_contact is not None else is_grasped
    return build_motion_observation_vector(
        delta,
        target_valid,
        bool(contact_flag),
        joint_positions,
    )


def compute_motion_approach_reward(
    end_effector_x: float,
    end_effector_y: float,
    end_effector_z: float,
    target_ee_x: float,
    target_ee_y: float,
    target_ee_z: float,
    *,
    target_valid: bool = True,
) -> float:
    """Dense reward for reducing EE distance to the vision-derived target pose."""

    if not target_valid:
        return 0.0
    distance = (
        (target_ee_x - end_effector_x) ** 2
        + (target_ee_y - end_effector_y) ** 2
        + (target_ee_z - end_effector_z) ** 2
    ) ** 0.5
    return max(0.0, 1.0 - distance / 0.15) * 2.5


def compute_reach_reward(block_distance_from_base: float, max_reach_m: float) -> float:
    if block_distance_from_base > max_reach_m:
        return 0.0
    margin = max(max_reach_m - block_distance_from_base, 0.0) / max(max_reach_m, 1e-6)
    return max(0.0, min(1.0, margin))


def compute_alignment_reward(
    centroid_x: float,
    centroid_y: float,
    end_effector_x: float,
    end_effector_y: float,
) -> float:
    distance = ((centroid_x - end_effector_x) ** 2 + (centroid_y - end_effector_y) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / 0.30)


def compute_vertical_approach_reward(
    end_effector_z: float,
    block_top_z: float,
    *,
    tolerance_m: float = 0.05,
) -> float:
    gap = end_effector_z - block_top_z
    if gap < 0.0:
        return 1.0
    return max(0.0, 1.0 - gap / tolerance_m)


def compute_contact_reward(
    horizontal_distance_m: float,
    vertical_gap_m: float,
    *,
    horizontal_tolerance_m: float = 0.025,
    vertical_max_gap_m: float = 0.015,
    vertical_min_gap_m: float = -0.010,
) -> float:
    if horizontal_distance_m > horizontal_tolerance_m:
        return 0.0
    if vertical_gap_m > vertical_max_gap_m or vertical_gap_m < vertical_min_gap_m:
        return 0.0
    return 1.0


def compute_push_reward(push_distance_m: float, target_push_distance_m: float) -> float:
    if push_distance_m <= 0.0 or target_push_distance_m <= 0.0:
        return 0.0
    return max(0.0, min(1.0, push_distance_m / target_push_distance_m))


def compute_contact_push_reward(
    block_x: float,
    block_y: float,
    block_z: float,
    end_effector_x: float,
    end_effector_y: float,
    end_effector_z: float,
    robot_base_x: float,
    robot_base_y: float,
    push_distance_m: float,
    *,
    cfg: MdpTaskConfig | None = None,
) -> float:
    """Reward locating the block, aligning the EE, making contact, and pushing 5 mm."""

    task = cfg or MdpTaskConfig()
    block_top_z = block_z + task.block_half_size_m
    block_dist = (
        (block_x - robot_base_x) ** 2 + (block_y - robot_base_y) ** 2
    ) ** 0.5
    horizontal_dist = (
        (block_x - end_effector_x) ** 2 + (block_y - end_effector_y) ** 2
    ) ** 0.5
    vertical_gap = end_effector_z - block_top_z

    reach = compute_reach_reward(block_dist, task.max_reach_m)
    alignment = compute_alignment_reward(block_x, block_y, end_effector_x, end_effector_y)
    vertical_approach = compute_vertical_approach_reward(end_effector_z, block_top_z)
    if horizontal_dist < 0.12:
        vertical_approach *= 2.5
    contact = compute_contact_reward(
        horizontal_dist,
        vertical_gap,
        horizontal_tolerance_m=task.contact_horizontal_tolerance_m,
        vertical_max_gap_m=task.contact_vertical_max_gap_m,
        vertical_min_gap_m=task.contact_vertical_min_gap_m,
    )
    push = compute_push_reward(push_distance_m, task.target_push_distance_m)
    return reach + alignment + vertical_approach + contact + push


def compute_tracking_reward(
    centroid_x: float,
    centroid_y: float,
    target_x: float,
    target_y: float,
) -> float:
    distance = ((centroid_x - target_x) ** 2 + (centroid_y - target_y) ** 2) ** 0.5
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
    push_distance_m: float = 0.0,
    robot_base_x: float = 0.0,
    robot_base_y: float = 0.0,
    end_effector_z: float = 0.0,
    use_contact_push: bool = True,
) -> float:
    if use_contact_push:
        return (
            compute_contact_push_reward(
                block_x=centroid_x,
                block_y=centroid_y,
                block_z=block_height,
                end_effector_x=end_effector_x,
                end_effector_y=end_effector_y,
                end_effector_z=end_effector_z,
                robot_base_x=robot_base_x,
                robot_base_y=robot_base_y,
                push_distance_m=push_distance_m,
            )
            - safety_penalty
        )

    tracking = compute_tracking_reward(
        centroid_x, centroid_y, target_centroid_x, target_centroid_y)
    alignment = compute_alignment_reward(
        centroid_x, centroid_y, end_effector_x, end_effector_y)
    grasp = compute_grasp_reward(is_grasped, alignment)
    lift = compute_lift_reward(block_height, target_lift_height)
    return tracking + alignment + grasp + lift - safety_penalty


def episode_push_task_success(
    max_push_distance_m: float,
    *,
    target_push_distance_m: float = TARGET_PUSH_DISTANCE_M,
    had_contact: bool = True,
) -> bool:
    return had_contact and max_push_distance_m >= target_push_distance_m


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
