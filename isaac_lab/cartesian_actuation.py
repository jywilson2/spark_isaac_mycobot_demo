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

"""Cartesian (task-space) actuation for Phase 2 reach RL.

The PPO policy outputs Δx, Δy, Δz in the robot **base frame**. This module maps
those deltas to **joint position targets** using a damped least-squares (DLS)
Jacobian pseudoinverse. That is operational-space control used as a *low-level
actuator interface* — the policy still learns which Cartesian direction to move
each step; we do **not** solve analytic IK for the episode target.

Tutorial links:
  - spec.md § Phase 2 (Cartesian actions + curriculum)
  - mdp_core.damped_least_squares_joint_delta (pure-Python reference)
  - Khatib, operational-space control: https://doi.org/10.1177/027836498700600304
  - Isaac Lab Articulation API: https://isaac-sim.github.io/IsaacLab/
"""

from __future__ import annotations

from typing import Any

import torch

from isaac_lab.mdp_core import REVOLUTE_JOINT_NAMES


def resolve_arm_joint_indices(robot: Any) -> list[int]:
    """Return PhysX Jacobian column indices for the six revolute arm joints."""

    joint_ids, _joint_names = robot.find_joints(list(REVOLUTE_JOINT_NAMES))
    if hasattr(joint_ids, 'tolist'):
        return list(joint_ids.tolist())
    return list(joint_ids)


def get_ee_position_jacobian(
    robot: Any,
    *,
    ee_body_idx: int,
    joint_indices: list[int],
) -> torch.Tensor:
    """Return batched positional Jacobian J with shape (num_envs, 3, num_joints).

    Uses PhysX ``get_jacobians()`` from the articulation root view. Only the
    first three rows (linear velocity) are retained for reach-in-position tasks.
    """

    jacobians = robot.root_physx_view.get_jacobians()
    joint_cols = torch.tensor(joint_indices, device=jacobians.device, dtype=torch.long)
    # PhysX layout: (num_envs, num_bodies, 6, num_jacobian_cols)
    jacobian = jacobians[:, ee_body_idx, :3, :]
    jacobian = jacobian.index_select(dim=-1, index=joint_cols)
    return jacobian


def damped_least_squares_joint_delta_batched(
    jacobian: torch.Tensor,
    cartesian_delta: torch.Tensor,
    *,
    damping: float,
) -> torch.Tensor:
    """Batched DLS: dq = J^T (J J^T + λ² I)⁻¹ dx.

    Args:
        jacobian: (N, 3, J) positional Jacobian.
        cartesian_delta: (N, 3) desired EE linear delta per env.
        damping: λ in meters; larger values stabilize near singularities.

    Returns:
        (N, J) joint position deltas.
    """

    if jacobian.ndim != 3 or cartesian_delta.ndim != 2:
        raise ValueError('Expected jacobian (N,3,J) and cartesian_delta (N,3)')
    batch = jacobian.shape[0]
    num_joints = jacobian.shape[2]
    identity = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype)
    identity = identity.unsqueeze(0).expand(batch, -1, -1)
    jjt = jacobian @ jacobian.transpose(1, 2)
    damped = jjt + (damping * damping) * identity
    # (N, 3, 3)^{-1} @ (N, 3, 1) via solve for numerical stability
    rhs = cartesian_delta.unsqueeze(-1)
    solved = torch.linalg.solve(damped, rhs).squeeze(-1)
    joint_delta = torch.bmm(jacobian.transpose(1, 2), solved.unsqueeze(-1)).squeeze(-1)
    if joint_delta.shape != (batch, num_joints):
        raise RuntimeError(
            f'Unexpected joint_delta shape {tuple(joint_delta.shape)}; '
            f'expected ({batch}, {num_joints})')
    return joint_delta


def cartesian_actions_to_joint_targets(
    robot: Any,
    *,
    cartesian_actions: torch.Tensor,
    cartesian_action_scale: float,
    ee_body_idx: int,
    joint_indices: list[int],
    damping: float,
) -> torch.Tensor:
    """Convert normalized Cartesian policy actions to joint position targets.

    Each policy dimension is in roughly [-1, 1] (clipped by RSL-RL). We scale
    by ``cartesian_action_scale`` (meters per step) to obtain a task-space delta,
    map through DLS, then add to current joint positions.
    """

    cartesian_delta = cartesian_actions * cartesian_action_scale
    jacobian = get_ee_position_jacobian(
        robot,
        ee_body_idx=ee_body_idx,
        joint_indices=joint_indices,
    )
    joint_delta = damped_least_squares_joint_delta_batched(
        jacobian,
        cartesian_delta,
        damping=damping,
    )
    return robot.data.joint_pos.torch + joint_delta
