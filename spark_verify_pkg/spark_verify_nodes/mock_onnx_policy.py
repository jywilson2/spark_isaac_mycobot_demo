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

"""Deterministic mock ONNX policy for Phase 3 sim-to-real verification."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MockPolicyWeights:
    centroid_gain: float = 0.35
    joint_bias: tuple[float, ...] = (0.05, -0.05, 0.1, -0.1, 0.05, -0.05)


def run_mock_onnx_inference(observation: Sequence[float]) -> list[float]:
    """Map flattened RL observations to target joint angles in radians."""
    if len(observation) < 8:
        raise ValueError('Observation vector must include centroid and joint positions')

    weights = MockPolicyWeights()
    centroid_x = observation[0]
    centroid_y = observation[1]
    joint_start = 8
    joint_positions = list(observation[joint_start:joint_start + 6])
    if len(joint_positions) < 6:
        joint_positions.extend([0.0] * (6 - len(joint_positions)))

    targets = []
    for idx, joint in enumerate(joint_positions[:6]):
        delta = (centroid_x - 0.5) * weights.centroid_gain
        delta += (centroid_y - 0.5) * weights.centroid_gain * (idx + 1) * 0.1
        targets.append(joint + delta + weights.joint_bias[idx])
    return targets
