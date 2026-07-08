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

"""Deterministic joint scan executed at episode start to locate the block."""

from __future__ import annotations

from typing import Mapping

from isaac_lab.mdp_core import REVOLUTE_JOINT_NAMES

# Base-yaw sweep with a slight shoulder lift so the EE camera covers the table.
INIT_SCAN_JOINT_WAYPOINTS: tuple[Mapping[str, float], ...] = (
    {
        'joint2_to_joint1': -0.85,
        'joint3_to_joint2': 0.35,
        'joint4_to_joint3': 0.55,
        'joint5_to_joint4': 0.0,
        'joint6_to_joint5': 0.0,
        'joint6output_to_joint6': 0.0,
    },
    {
        'joint2_to_joint1': -0.40,
        'joint3_to_joint2': 0.35,
        'joint4_to_joint3': 0.55,
        'joint5_to_joint4': 0.0,
        'joint6_to_joint5': 0.0,
        'joint6output_to_joint6': 0.0,
    },
    {
        'joint2_to_joint1': 0.0,
        'joint3_to_joint2': 0.35,
        'joint4_to_joint3': 0.55,
        'joint5_to_joint4': 0.0,
        'joint6_to_joint5': 0.0,
        'joint6output_to_joint6': 0.0,
    },
    {
        'joint2_to_joint1': 0.40,
        'joint3_to_joint2': 0.35,
        'joint4_to_joint3': 0.55,
        'joint5_to_joint4': 0.0,
        'joint6_to_joint5': 0.0,
        'joint6output_to_joint6': 0.0,
    },
    {
        'joint2_to_joint1': 0.85,
        'joint3_to_joint2': 0.35,
        'joint4_to_joint3': 0.55,
        'joint5_to_joint4': 0.0,
        'joint6_to_joint5': 0.0,
        'joint6output_to_joint6': 0.0,
    },
)

INIT_SCAN_STEPS_PER_WAYPOINT = 6


def waypoint_count() -> int:
    """Return the number of init-scan joint poses."""

    return len(INIT_SCAN_JOINT_WAYPOINTS)


def build_waypoint_tensor(
    waypoint_index: int,
    *,
    device: str,
    dtype,
) -> 'torch.Tensor':
    """Return joint targets for one scan waypoint as (1, num_joints)."""

    import torch  # noqa: WPS433

    if waypoint_index < 0 or waypoint_index >= len(INIT_SCAN_JOINT_WAYPOINTS):
        raise IndexError(f'waypoint_index {waypoint_index} out of range')
    waypoint = INIT_SCAN_JOINT_WAYPOINTS[waypoint_index]
    values = [float(waypoint[name]) for name in REVOLUTE_JOINT_NAMES]
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0)
