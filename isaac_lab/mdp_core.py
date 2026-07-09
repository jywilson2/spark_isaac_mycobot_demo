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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import random

# Keep in sync with spark_verify_nodes.mycobot_mdp.MyCobotPickPlaceMDP.OBSERVATION_DIM
# Motion-policy layout: 3 EE-to-target delta + target_valid + reached + 6 joints = 11.
OBSERVATION_DIM = 11

# Phase 2 reach: joint deltas — the RL policy learns IK (see spec.md § no IK solvers).
REACH_ACTION_DIM = 6

# Phase 5 contact-and-push uses the same joint-delta layout.
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
# Final EE position tolerance (spec.md): 1 mm.
EE_REACH_TOLERANCE_M = 0.001
# Coarse tolerance for early curriculum stages before precision fine-tune.
EE_REACH_TOLERANCE_COARSE_M = 0.025
# When EE is within this distance, penalize lateral (non-radial) motion steps.
DEFAULT_APPROACH_ZONE_M = 0.05
DEFAULT_TARGET_REACH_SUCCESS_RATE = 0.99

# Reachable workspace annulus on the table (meters from robot base).
MIN_BLOCK_REACH_M = 0.12
MAX_BLOCK_REACH_M = 0.28
MIN_EE_TARGET_Z_M = 0.08
MAX_EE_TARGET_Z_M = 0.22
BLOCK_SPAWN_Z = 0.021

# Demo/showcase: stratified bins + minimum 3D separation between consecutive targets.
DEMO_MIN_TARGET_SEPARATION_M = 0.10
DEMO_AZIMUTH_BINS = 8
DEMO_Z_BINS = 4
DEMO_RADIUS_BINS = 3


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

    ``action_scale`` multiplies normalized policy outputs (≈[-1,1]) into joint
    position deltas (radians per control step). The PPO policy learns how to
    combine these deltas so the EE reaches the target — that coordination is
    the learned IK (spec.md § strategic goal).
    """

    reach_tolerance_m: float = EE_REACH_TOLERANCE_M
    min_reach_m: float = MIN_BLOCK_REACH_M
    max_reach_m: float = MAX_BLOCK_REACH_M
    min_target_z_m: float = MIN_EE_TARGET_Z_M
    max_target_z_m: float = MAX_EE_TARGET_Z_M
    action_scale: float = 0.12
    time_penalty: float = 0.02
    reach_bonus: float = 50.0
    progress_scale: float = 50.0
    timeout_penalty: float = 5.0
    # Penalty per unit of mean |action| — prefers smooth, low-effort motion and
    # removes the incentive to grow the exploration std for its own sake.
    action_penalty: float = 0.05
    # Penalize abrupt changes in commanded joint deltas (jerk) — smooth accel/decel
    # matters more than raw speed for real MyCobot servos (spec.md).
    jerk_penalty: float = 0.08
    # EMA blend for raw actions: higher = more responsive, lower = smoother motion.
    action_smoothing_alpha: float = 0.35
    # Direct-path shaping (spec.md): penalize lateral steps inside ``approach_zone_m``.
    approach_zone_m: float = DEFAULT_APPROACH_ZONE_M
    lateral_penalty: float = 2.0
    direct_path_shaping: bool = False
    # Tiered bonuses when EE enters tighter bands (precision curriculum).
    precision_tier_bonuses: tuple[tuple[float, float], ...] = ()


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


def is_ee_target_in_workspace(
    x: float,
    y: float,
    z: float,
    *,
    min_reach_m: float = MIN_BLOCK_REACH_M,
    max_reach_m: float = MAX_BLOCK_REACH_M,
    min_z_m: float = MIN_EE_TARGET_Z_M,
    max_z_m: float = MAX_EE_TARGET_Z_M,
    robot_base_x: float = 0.0,
    robot_base_y: float = 0.0,
) -> bool:
    """Return True when (x, y, z) lies in the Phase 2 reach envelope (spec.md)."""

    radius = math.hypot(x - robot_base_x, y - robot_base_y)
    return min_reach_m <= radius <= max_reach_m and min_z_m <= z <= max_z_m


def sample_offset_in_ball(radius_m: float) -> tuple[float, float, float]:
    """Uniform random offset inside a 3D ball of the given radius."""

    limit = max(radius_m, 0.0)
    if limit == 0.0:
        return 0.0, 0.0, 0.0
    while True:
        dx = random.uniform(-limit, limit)
        dy = random.uniform(-limit, limit)
        dz = random.uniform(-limit, limit)
        if dx * dx + dy * dy + dz * dz <= limit * limit:
            return dx, dy, dz


def _sample_workspace_xyz(
    *,
    min_reach_m: float,
    max_reach_m: float,
    min_z_m: float,
    max_z_m: float,
    robot_base_x: float,
    robot_base_y: float,
) -> tuple[float, float, float]:
    angle = random.uniform(0.0, 2.0 * math.pi)
    radius = random.uniform(min_reach_m, max_reach_m)
    x = robot_base_x + radius * math.cos(angle)
    y = robot_base_y + radius * math.sin(angle)
    z = random.uniform(min_z_m, max_z_m)
    return x, y, z


def _sample_stratified_workspace_xyz(
    *,
    min_reach_m: float,
    max_reach_m: float,
    min_z_m: float,
    max_z_m: float,
    robot_base_x: float,
    robot_base_y: float,
    azimuth_bin: int | None = None,
    z_bin: int | None = None,
    radius_bin: int | None = None,
) -> tuple[float, float, float]:
    """Pick a target from a random workspace cell (azimuth × radius × height)."""

    az_bin = azimuth_bin if azimuth_bin is not None else random.randrange(DEMO_AZIMUTH_BINS)
    z_cell = z_bin if z_bin is not None else random.randrange(DEMO_Z_BINS)
    r_cell = radius_bin if radius_bin is not None else random.randrange(DEMO_RADIUS_BINS)

    angle = (az_bin + random.uniform(0.15, 0.85)) / DEMO_AZIMUTH_BINS * (2.0 * math.pi)
    r_span = (max_reach_m - min_reach_m) / DEMO_RADIUS_BINS
    radius = random.uniform(
        min_reach_m + r_cell * r_span,
        min_reach_m + (r_cell + 1) * r_span,
    )
    z_span = (max_z_m - min_z_m) / DEMO_Z_BINS
    z = random.uniform(
        min_z_m + z_cell * z_span,
        min_z_m + (z_cell + 1) * z_span,
    )
    x = robot_base_x + radius * math.cos(angle)
    y = robot_base_y + radius * math.sin(angle)
    return x, y, z


def sample_demo_workspace_xyz(
    count: int,
    *,
    previous_targets: Sequence[tuple[float, float, float]] | None = None,
    min_separation_m: float = DEMO_MIN_TARGET_SEPARATION_M,
    min_azimuth_separation_rad: float = math.pi / 4.0,
    min_reach_m: float = MIN_BLOCK_REACH_M,
    max_reach_m: float = MAX_BLOCK_REACH_M,
    min_z_m: float = MIN_EE_TARGET_Z_M,
    max_z_m: float = MAX_EE_TARGET_Z_M,
    robot_base_x: float = 0.0,
    robot_base_y: float = 0.0,
) -> list[tuple[float, float, float]]:
    """High-variability demo targets within the reach envelope.

    Samples uniformly over the workspace (same envelope as training/play), but
    rejects candidates that are too close to the previous target or lie in a
    similar azimuth sector. Stratified cells are used only as a last resort to
    guarantee separation so consecutive demo targets look visibly different
    without forcing extreme corner placements.
    """

    def _azimuth(x: float, y: float) -> float:
        return math.atan2(y - robot_base_y, x - robot_base_x)

    def _angular_delta(a1: float, a2: float) -> float:
        delta = abs(a1 - a2)
        return min(delta, 2.0 * math.pi - delta)

    samples: list[tuple[float, float, float]] = []
    for idx in range(max(count, 0)):
        prev = None
        if previous_targets is not None and idx < len(previous_targets):
            prev = previous_targets[idx]

        chosen: tuple[float, float, float] | None = None
        prev_az = _azimuth(prev[0], prev[1]) if prev is not None else None
        for _ in range(64):
            candidate = _sample_workspace_xyz(
                min_reach_m=min_reach_m,
                max_reach_m=max_reach_m,
                min_z_m=min_z_m,
                max_z_m=max_z_m,
                robot_base_x=robot_base_x,
                robot_base_y=robot_base_y,
            )
            if prev is None:
                chosen = candidate
                break
            sep_ok = math.dist(prev, candidate) >= min_separation_m
            az_ok = (
                prev_az is None
                or _angular_delta(prev_az, _azimuth(candidate[0], candidate[1]))
                >= min_azimuth_separation_rad
            )
            if sep_ok and az_ok:
                chosen = candidate
                break

        if chosen is None:
            for _ in range(32):
                candidate = _sample_stratified_workspace_xyz(
                    min_reach_m=min_reach_m,
                    max_reach_m=max_reach_m,
                    min_z_m=min_z_m,
                    max_z_m=max_z_m,
                    robot_base_x=robot_base_x,
                    robot_base_y=robot_base_y,
                )
                if prev is None or math.dist(prev, candidate) >= min_separation_m * 0.5:
                    chosen = candidate
                    break

        if chosen is None:
            chosen = _sample_workspace_xyz(
                min_reach_m=min_reach_m,
                max_reach_m=max_reach_m,
                min_z_m=min_z_m,
                max_z_m=max_z_m,
                robot_base_x=robot_base_x,
                robot_base_y=robot_base_y,
            )
        samples.append(chosen)
    return samples


def _sample_near_center_xyz(
    center: tuple[float, float, float],
    *,
    radius_m: float,
    min_reach_m: float,
    max_reach_m: float,
    min_z_m: float,
    max_z_m: float,
    robot_base_x: float,
    robot_base_y: float,
    max_attempts: int = 48,
) -> tuple[float, float, float] | None:
    """Sample a target in a 3D ball around ``center`` that also lies in the workspace."""

    cx, cy, cz = center
    for _ in range(max_attempts):
        dx, dy, dz = sample_offset_in_ball(radius_m)
        x, y, z = cx + dx, cy + dy, cz + dz
        if is_ee_target_in_workspace(
            x,
            y,
            z,
            min_reach_m=min_reach_m,
            max_reach_m=max_reach_m,
            min_z_m=min_z_m,
            max_z_m=max_z_m,
            robot_base_x=robot_base_x,
            robot_base_y=robot_base_y,
        ):
            return x, y, z
    return None


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
    sampled uniformly in a 3D ball around the **current EE pose** (curriculum
    stage 1), then validated against the workspace envelope. If no valid point
    is found inside the ball ∩ workspace, sampling falls back to the full
    workspace annulus so targets are always reachable positions per spec.md.
    """

    samples: list[tuple[float, float, float]] = []
    for _ in range(max(count, 0)):
        placed = False
        if easy_fraction > 0.0 and random.random() < easy_fraction:
            if near_ee_center is not None:
                center = near_ee_center
                radius_limit = near_ee_radius_m
            else:
                center = easy_center
                radius_limit = easy_radius_m
            near_sample = _sample_near_center_xyz(
                center,
                radius_m=radius_limit,
                min_reach_m=min_reach_m,
                max_reach_m=max_reach_m,
                min_z_m=min_z_m,
                max_z_m=max_z_m,
                robot_base_x=robot_base_x,
                robot_base_y=robot_base_y,
            )
            if near_sample is not None:
                samples.append(near_sample)
                placed = True
        if not placed:
            samples.append(
                _sample_workspace_xyz(
                    min_reach_m=min_reach_m,
                    max_reach_m=max_reach_m,
                    min_z_m=min_z_m,
                    max_z_m=max_z_m,
                    robot_base_x=robot_base_x,
                    robot_base_y=robot_base_y,
                ),
            )
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


def compute_lateral_step_m(
    step_x: float,
    step_y: float,
    step_z: float,
    to_target_x: float,
    to_target_y: float,
    to_target_z: float,
) -> float:
    """Magnitude of the EE step orthogonal to the direct line toward the target.

    Used to discourage corrective zig-zag when the flange is already near the
  goal (spec.md § direct approach).
    """

    step_len = math.hypot(step_x, step_y, step_z)
    if step_len < 1e-9:
        return 0.0
    tgt_len = math.hypot(to_target_x, to_target_y, to_target_z)
    if tgt_len < 1e-9:
        return 0.0
    cos_align = (
        (step_x * to_target_x + step_y * to_target_y + step_z * to_target_z)
        / (step_len * tgt_len)
    )
    cos_align = max(-1.0, min(1.0, cos_align))
    return step_len * math.sqrt(max(0.0, 1.0 - cos_align * cos_align))


def compute_reach_task_reward(
    end_effector_x: float,
    end_effector_y: float,
    end_effector_z: float,
    target_x: float,
    target_y: float,
    target_z: float,
    *,
    prev_distance_m: float | None = None,
    mean_abs_action: float = 0.0,
    mean_action_jerk: float = 0.0,
    lateral_step_m: float = 0.0,
    cfg: ReachTaskConfig | None = None,
) -> tuple[float, float]:
    """Return potential-based reach reward and current EE-to-target distance.

    Shaping uses the *signed* difference ``progress_scale * (d_prev - d_now)``:
    moving closer earns positive credit, moving away costs exactly as much, so
    oscillating toward/away from the target nets zero (a true potential-based
    term per Ng et al. 1999). Terminal ``reach_bonus`` applies inside tolerance.
    Small ``action_penalty`` and ``jerk_penalty`` terms discourage stuttery motion.
    When ``direct_path_shaping`` is on and distance < ``approach_zone_m``, a
    ``lateral_penalty`` term discourages corrective sideways motion.
    """

    task = cfg or ReachTaskConfig()
    distance = (
        (target_x - end_effector_x) ** 2
        + (target_y - end_effector_y) ** 2
        + (target_z - end_effector_z) ** 2
    ) ** 0.5
    progress = 0.0
    if prev_distance_m is not None:
        progress = (prev_distance_m - distance) * task.progress_scale
    reward = (
        progress
        - task.time_penalty
        - task.action_penalty * abs(mean_abs_action)
        - task.jerk_penalty * abs(mean_action_jerk)
    )
    if task.direct_path_shaping and distance <= task.approach_zone_m:
        reward -= task.lateral_penalty * abs(lateral_step_m)
    for tier_m, bonus in sorted(task.precision_tier_bonuses, key=lambda pair: pair[0]):
        if distance <= tier_m:
            reward += bonus
            break
    if distance <= task.reach_tolerance_m:
        reward += task.reach_bonus
    return reward, distance


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
