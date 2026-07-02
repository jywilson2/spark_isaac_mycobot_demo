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

# ---------------------------------------------------------------------------
# Pure pytest unit tests for the mock ONNX policy (no ROS required; see
# test_camera_lens_advisor.py for how these are registered). The tests pin
# the mock's DETERMINISM and its exact steering math, because downstream
# serial-byte assertions in the phase 3/4 launch tests depend on the policy
# emitting reproducible joint targets.
# ---------------------------------------------------------------------------

import pytest

from spark_verify_nodes.mock_onnx_policy import MockPolicyWeights, run_mock_onnx_inference


def build_observation(centroid_x=0.5, centroid_y=0.5, joints=None):
    # Helper mirroring the observation layout contract from
    # build_observation_vector: [cx, cy, bbox x4, ee height, grasp, joints].
    joints = joints if joints is not None else [0.4, -0.2, 0.6, -0.3, 0.5, -0.1]
    observation = [centroid_x, centroid_y, 0.4, 0.4, 0.6, 0.6, 0.12, 0.0]
    observation.extend(joints)
    return observation


def test_inference_is_deterministic_and_six_dof():
    observation = build_observation()
    first = run_mock_onnx_inference(observation)
    second = run_mock_onnx_inference(observation)

    assert len(first) == 6
    assert first == second


def test_centered_block_applies_only_joint_bias():
    # With the block exactly at image center the centroid error is zero,
    # so the steering delta vanishes and only the bias term remains —
    # isolating one term of the policy equation.
    joints = [0.4, -0.2, 0.6, -0.3, 0.5, -0.1]
    targets = run_mock_onnx_inference(build_observation(joints=joints))

    weights = MockPolicyWeights()
    for target, joint, bias in zip(targets, joints, weights.joint_bias):
        assert target == pytest.approx(joint + bias, abs=1e-9)


def test_off_center_block_steers_joint_targets():
    # Differential test: compare two runs that differ only in centroid_x,
    # so the assertion isolates the proportional steering term without
    # needing to know the bias values.
    centered = run_mock_onnx_inference(build_observation(centroid_x=0.5))
    off_center = run_mock_onnx_inference(build_observation(centroid_x=0.9))

    weights = MockPolicyWeights()
    expected_delta = (0.9 - 0.5) * weights.centroid_gain
    assert off_center[0] - centered[0] == pytest.approx(expected_delta, abs=1e-9)


def test_truncated_joint_positions_are_zero_padded():
    observation = build_observation(joints=[0.4, -0.2])
    targets = run_mock_onnx_inference(observation)

    weights = MockPolicyWeights()
    assert len(targets) == 6
    for idx in range(2, 6):
        assert targets[idx] == pytest.approx(weights.joint_bias[idx], abs=1e-9)


def test_short_observation_raises_value_error():
    with pytest.raises(ValueError):
        run_mock_onnx_inference([0.5, 0.5, 0.1])
