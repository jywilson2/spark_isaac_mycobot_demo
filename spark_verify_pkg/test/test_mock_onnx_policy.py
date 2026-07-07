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


def build_observation(delta_x=0.0, delta_y=0.0, delta_z=0.0, joints=None):
    joints = joints if joints is not None else [0.4, -0.2, 0.6, -0.3, 0.5, -0.1]
    observation = [delta_x, delta_y, delta_z, 1.0, 0.0]
    observation.extend(joints)
    return observation


def test_inference_is_deterministic_and_six_dof():
    observation = build_observation()
    first = run_mock_onnx_inference(observation)
    second = run_mock_onnx_inference(observation)

    assert len(first) == 6
    assert first == second


def test_zero_delta_applies_only_joint_bias():
    joints = [0.4, -0.2, 0.6, -0.3, 0.5, -0.1]
    targets = run_mock_onnx_inference(build_observation(joints=joints))

    weights = MockPolicyWeights()
    for target, joint, bias in zip(targets, joints, weights.joint_bias):
        assert target == pytest.approx(joint + bias, abs=1e-9)


def test_nonzero_delta_steers_joint_targets():
    zero_delta = run_mock_onnx_inference(build_observation(delta_x=0.0))
    positive_delta = run_mock_onnx_inference(build_observation(delta_x=0.2))

    weights = MockPolicyWeights()
    expected_delta = 0.2 * weights.centroid_gain
    assert positive_delta[0] - zero_delta[0] == pytest.approx(expected_delta, abs=1e-9)


def test_truncated_joint_positions_are_zero_padded():
    observation = build_observation(joints=[0.4, -0.2])
    targets = run_mock_onnx_inference(observation)

    weights = MockPolicyWeights()
    assert len(targets) == 6
    for idx in range(2, 6):
        assert targets[idx] == pytest.approx(weights.joint_bias[idx], abs=1e-9)


def test_short_observation_raises_value_error():
    with pytest.raises(ValueError):
        run_mock_onnx_inference([0.0, 0.0, 0.0, 1.0])
