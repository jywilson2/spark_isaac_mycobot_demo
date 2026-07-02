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

"""
Deterministic mock ONNX policy for Phase 3 sim-to-real verification.

ONNX BACKGROUND: ONNX (Open Neural Network Exchange) is a portable graph
format for trained networks. The real Phase 3 flow exports the Isaac Lab
policy to .onnx and executes it with an ONNX-compatible runtime (compiled to
the Hailo NPU on the AI Hat for Phase 4). Loading real weights in CI would
make tests slow and non-deterministic, so this module substitutes a tiny
hand-written "policy" with the SAME SIGNATURE as a real inference call:
flat observation vector in, six joint targets out. Everything downstream
(inference node, driver, serial encoder) cannot tell the difference — which
is exactly what lets the pipeline be verified before training completes.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MockPolicyWeights:
    """Stand-in for learned parameters; frozen so tests can rely on them."""

    # Proportional gain applied to the block's offset from image center.
    centroid_gain: float = 0.35
    # Constant per-joint offset, emulating a learned bias term.
    joint_bias: tuple[float, ...] = (0.05, -0.05, 0.1, -0.1, 0.05, -0.05)


def run_mock_onnx_inference(observation: Sequence[float]) -> list[float]:
    """
    Map flattened RL observations to target joint angles in radians.

    The observation layout is the contract defined in
    reward_function.build_observation_vector:
    indices 0-1 centroid, 2-5 bbox, 6 ee height, 7 grasp flag, 8-13 joints.
    """
    # Validate at the boundary: a policy fed a wrong-shaped tensor should
    # fail loudly, not return garbage joint targets.
    if len(observation) < 8:
        raise ValueError('Observation vector must include centroid and joint positions')

    weights = MockPolicyWeights()
    centroid_x = observation[0]
    centroid_y = observation[1]
    # Joint positions start after the 8 perception values (see layout above).
    joint_start = 8
    joint_positions = list(observation[joint_start:joint_start + 6])
    # Zero-pad if the caller supplied fewer than 6 joints so the output is
    # always a full 6-DOF command (PolicyInference.joint_targets_rad is a
    # fixed float32[6] array and rejects shorter assignments).
    if len(joint_positions) < 6:
        joint_positions.extend([0.0] * (6 - len(joint_positions)))

    # A crude visual-servoing rule standing in for the neural network:
    # steer each joint proportionally to how far the block sits from image
    # center (0.5, 0.5), scaling the y-error more for distal joints, plus
    # the fixed bias. Deterministic and differentiable-looking — ideal for
    # byte-exact serial encoding tests downstream.
    targets = []
    for idx, joint in enumerate(joint_positions[:6]):
        delta = (centroid_x - 0.5) * weights.centroid_gain
        delta += (centroid_y - 0.5) * weights.centroid_gain * (idx + 1) * 0.1
        targets.append(joint + delta + weights.joint_bias[idx])
    return targets
