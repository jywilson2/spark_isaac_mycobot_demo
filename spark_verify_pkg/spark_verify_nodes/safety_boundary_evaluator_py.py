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

"""Python mirror of spark_verify_pkg safety boundary evaluation for live MDP smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SafetyBoundaryConfig:
    max_joint_velocity_rad_s: float = 2.0
    joint_min: tuple[float, ...] = (
        -2.879793, -1.570796, -2.879793, -2.879793, -2.879793, -2.879793)
    joint_max: tuple[float, ...] = (
        2.879793, 1.570796, 2.879793, 2.879793, 2.879793, 2.879793)
    workspace_x_min: float = -0.28
    workspace_x_max: float = 0.28
    workspace_y_min: float = -0.28
    workspace_y_max: float = 0.28
    workspace_z_min: float = 0.05
    workspace_z_max: float = 0.45
    joint_limit_penalty_weight: float = 10.0
    velocity_penalty_weight: float = 5.0
    workspace_penalty_weight: float = 8.0


@dataclass(frozen=True)
class SafetyPenaltyResult:
    joint_limit_penalty: float = 0.0
    velocity_penalty: float = 0.0
    workspace_penalty: float = 0.0
    total_penalty: float = 0.0
    boundary_violated: bool = False


def evaluate_safety_boundaries(
    config: SafetyBoundaryConfig,
    joint_positions: Sequence[float],
    joint_velocities: Sequence[float],
    end_effector_x: float,
    end_effector_y: float,
    end_effector_z: float,
) -> SafetyPenaltyResult:
    if len(joint_positions) != len(config.joint_min):
        raise ValueError('joint_positions length must match configured DOF')
    if len(joint_velocities) != len(config.joint_min):
        raise ValueError('joint_velocities length must match configured DOF')

    joint_limit_penalty = 0.0
    velocity_penalty = 0.0
    workspace_penalty = 0.0

    for index, position in enumerate(joint_positions):
        lower = config.joint_min[index]
        upper = config.joint_max[index]
        if position < lower:
            joint_limit_penalty += config.joint_limit_penalty_weight * (lower - position)
        elif position > upper:
            joint_limit_penalty += config.joint_limit_penalty_weight * (position - upper)

    for velocity in joint_velocities:
        if abs(velocity) > config.max_joint_velocity_rad_s:
            velocity_penalty += config.velocity_penalty_weight * (
                abs(velocity) - config.max_joint_velocity_rad_s)

    if end_effector_x < config.workspace_x_min:
        workspace_penalty += config.workspace_penalty_weight * (
            config.workspace_x_min - end_effector_x)
    elif end_effector_x > config.workspace_x_max:
        workspace_penalty += config.workspace_penalty_weight * (
            end_effector_x - config.workspace_x_max)

    if end_effector_y < config.workspace_y_min:
        workspace_penalty += config.workspace_penalty_weight * (
            config.workspace_y_min - end_effector_y)
    elif end_effector_y > config.workspace_y_max:
        workspace_penalty += config.workspace_penalty_weight * (
            end_effector_y - config.workspace_y_max)

    if end_effector_z < config.workspace_z_min:
        workspace_penalty += config.workspace_penalty_weight * (
            config.workspace_z_min - end_effector_z)
    elif end_effector_z > config.workspace_z_max:
        workspace_penalty += config.workspace_penalty_weight * (
            end_effector_z - config.workspace_z_max)

    total = joint_limit_penalty + velocity_penalty + workspace_penalty
    return SafetyPenaltyResult(
        joint_limit_penalty=joint_limit_penalty,
        velocity_penalty=velocity_penalty,
        workspace_penalty=workspace_penalty,
        total_penalty=total,
        boundary_violated=total > 0.0,
    )
