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

"""Tests for policy checkpoint save/load and ONNX-ready export."""

from __future__ import annotations

import json
from pathlib import Path

from spark_verify_nodes.policy_checkpoint import (
    export_policy_weights_for_onnx,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from spark_verify_nodes.ppo_policy import PpoPolicy, PpoPolicyConfig


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    policy = PpoPolicy(PpoPolicyConfig(observation_dim=14, action_dim=6, seed=3))
    path = save_policy_checkpoint(
        policy,
        tmp_path / 'policy.json',
        episode_count=2,
        total_steps=16,
        mean_episode_reward=1.25,
    )
    restored = load_policy_checkpoint(path)
    assert restored.config.observation_dim == policy.config.observation_dim
    assert restored.w1.shape == policy.w1.shape


def test_export_onnx_ready_weights(tmp_path: Path) -> None:
    policy = PpoPolicy(PpoPolicyConfig(observation_dim=14, action_dim=6, seed=4))
    checkpoint = save_policy_checkpoint(
        policy,
        tmp_path / 'policy.json',
        episode_count=1,
        total_steps=8,
        mean_episode_reward=0.5,
    )
    export_path = export_policy_weights_for_onnx(
        checkpoint,
        tmp_path / 'onnx_ready.json',
    )
    payload = json.loads(export_path.read_text(encoding='utf-8'))
    assert payload['format'] == 'spark_mycobot_ppo_v1'
    assert payload['observation_dim'] == 14
    assert len(payload['weights']['w1']) == 14
