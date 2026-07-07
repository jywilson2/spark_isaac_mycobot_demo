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

"""Unit tests for ROS-side block localization helpers."""

from __future__ import annotations

from spark_verify_nodes.block_localization import compute_target_ee_pose, localize_mock_block_in_base


def test_localize_mock_block_in_base_sets_pose_valid() -> None:
    bx, by, bz, tx, ty, tz, valid = localize_mock_block_in_base(
        detected=True,
        mock_block_x=0.22,
        mock_block_y=0.0,
        mock_block_z=0.021,
    )
    assert valid
    assert bx == 0.22
    assert tz > bz


def test_compute_target_ee_pose() -> None:
    tx, ty, tz = compute_target_ee_pose(0.2, 0.1, 0.02)
    assert tx == 0.2
    assert ty == 0.1
    assert tz == 0.02 + 0.02 + 0.03
