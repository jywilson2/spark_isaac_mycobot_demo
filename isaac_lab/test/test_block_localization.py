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

"""Unit tests for block localization (detector + depth → base frame)."""

from __future__ import annotations

import pytest


@pytest.mark.skipif(
    pytest.importorskip('torch') is None,
    reason='torch required for localization tensor tests',
)
def test_localize_block_from_camera_batch_produces_target_ee() -> None:
    import torch

    from isaac_lab.phase6_red_block.block_localization import localize_block_from_camera_batch

    rgb = torch.zeros((1, 64, 64, 3))
    rgb[0, 28:36, 28:36, 0] = 0.95
    rgb[0, 28:36, 28:36, 1] = 0.05
    rgb[0, 28:36, 28:36, 2] = 0.05
    depth = torch.full((1, 64, 64), 0.25)
    cam_pos = torch.tensor([[0.18, 0.0, 0.20]])
    cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    block_pos, target_ee, detected, pose_valid = localize_block_from_camera_batch(
        rgb,
        depth,
        cam_pos,
        cam_quat,
    )
    assert bool(detected[0].item())
    assert bool(pose_valid[0].item())
    assert float(target_ee[0, 2].item()) > float(block_pos[0, 2].item())


def test_init_scan_has_multiple_waypoints() -> None:
    from isaac_lab.phase6_red_block.init_scan import INIT_SCAN_JOINT_WAYPOINTS, waypoint_count

    assert waypoint_count() == len(INIT_SCAN_JOINT_WAYPOINTS)
    assert waypoint_count() >= 3
