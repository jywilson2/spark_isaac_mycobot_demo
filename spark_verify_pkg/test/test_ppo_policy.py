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

"""Unit tests for numpy PPO policy."""

from __future__ import annotations

import numpy as np

from spark_verify_nodes.ppo_policy import (
    build_rollout_batch,
    PpoPolicy,
    PpoPolicyConfig,
)


def test_policy_act_returns_action_value_logprob() -> None:
    policy = PpoPolicy(PpoPolicyConfig(observation_dim=14, action_dim=6, seed=1))
    observation = [0.5] * 14
    action, value, log_prob = policy.act(observation)
    assert action.shape == (6,)
    assert isinstance(value, float)
    assert isinstance(log_prob, float)


def test_apply_action_delta_clamps_to_joint_limits() -> None:
    policy = PpoPolicy(PpoPolicyConfig(action_scale=1.0))
    targets = policy.apply_action_delta(
        current_joint_positions=[0.0] * 6,
        action_delta=[10.0] * 6,
        joint_min=[-1.0] * 6,
        joint_max=[1.0] * 6,
    )
    assert targets == [1.0] * 6


def test_ppo_update_runs_without_nan() -> None:
    policy = PpoPolicy(PpoPolicyConfig(observation_dim=4, action_dim=2, hidden_dim=8, seed=2))
    observations = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]
    actions = [[0.01, -0.01], [0.02, -0.02]]
    rewards = [1.0, 0.5]
    values = [0.1, 0.2]
    log_probs = [-0.5, -0.6]
    batch = build_rollout_batch(observations, actions, rewards, values, log_probs)
    metrics = policy.ppo_update(batch, epochs=1)
    assert np.isfinite(metrics['policy_loss'])
    assert np.isfinite(metrics['value_loss'])
