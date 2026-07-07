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

"""Map 2D detections to block / target EE poses in the robot base frame."""

from __future__ import annotations

BLOCK_HALF_SIZE_M = 0.02
APPROACH_OFFSET_M = 0.03


def compute_target_ee_pose(
    block_x: float,
    block_y: float,
    block_z: float,
    *,
    block_half_size_m: float = BLOCK_HALF_SIZE_M,
    approach_offset_m: float = APPROACH_OFFSET_M,
) -> tuple[float, float, float]:
    block_top_z = block_z + block_half_size_m
    return block_x, block_y, block_top_z + approach_offset_m


def localize_mock_block_in_base(
    *,
    detected: bool,
    mock_block_x: float,
    mock_block_y: float,
    mock_block_z: float,
    block_half_size_m: float = BLOCK_HALF_SIZE_M,
    approach_offset_m: float = APPROACH_OFFSET_M,
) -> tuple[float, float, float, float, float, float, bool]:
    """Return base-frame block pose, target EE pose, and pose_valid for mock RGB."""

    if not detected:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
    target_x, target_y, target_z = compute_target_ee_pose(
        mock_block_x,
        mock_block_y,
        mock_block_z,
        block_half_size_m=block_half_size_m,
        approach_offset_m=approach_offset_m,
    )
    return (
        mock_block_x,
        mock_block_y,
        mock_block_z,
        target_x,
        target_y,
        target_z,
        True,
    )
