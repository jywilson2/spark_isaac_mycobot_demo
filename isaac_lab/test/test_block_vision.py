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

"""Unit tests for EE camera block detection and reachable spawn sampling."""

from __future__ import annotations

import math

import pytest

from isaac_lab.mdp_core import (
    MAX_BLOCK_REACH_M,
    MIN_BLOCK_REACH_M,
    is_within_arm_reach,
    sample_reachable_block_xy,
)


def test_sample_reachable_block_xy_stays_in_annulus() -> None:
    samples = sample_reachable_block_xy(256)
    for x, y in samples:
        radius = math.hypot(x, y)
        assert MIN_BLOCK_REACH_M - 1e-6 <= radius <= MAX_BLOCK_REACH_M + 1e-6


def test_is_within_arm_reach_boundary() -> None:
    assert is_within_arm_reach(MIN_BLOCK_REACH_M, 0.0)
    assert is_within_arm_reach(MAX_BLOCK_REACH_M, 0.0)
    assert not is_within_arm_reach(MIN_BLOCK_REACH_M - 0.01, 0.0)
    assert not is_within_arm_reach(MAX_BLOCK_REACH_M + 0.01, 0.0)


@pytest.mark.skipif(
    pytest.importorskip('torch') is None,
    reason='torch required for camera tensor tests',
)
def test_detect_red_block_in_camera_batch_finds_center_blob() -> None:
    import torch

    from isaac_lab.phase6_red_block.block_vision import detect_red_block_in_camera_batch

    rgb = torch.zeros((2, 64, 64, 3))
    rgb[0, 28:36, 28:36, 0] = 0.95
    rgb[0, 28:36, 28:36, 1] = 0.05
    rgb[0, 28:36, 28:36, 2] = 0.05
    cx, cy, detected, bbox = detect_red_block_in_camera_batch(rgb)
    assert bool(detected[0].item())
    assert 0.45 <= float(cx[0].item()) <= 0.55
    assert 0.45 <= float(cy[0].item()) <= 0.55
    assert not bool(detected[1].item())
    assert float(bbox[0, 2].item()) > float(bbox[0, 0].item())
