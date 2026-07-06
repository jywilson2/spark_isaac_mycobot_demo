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

"""Save/load PPO policy checkpoints and export weights for Phase 3 ONNX pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from spark_verify_nodes.ppo_policy import PpoPolicy, PpoPolicyConfig


@dataclass(frozen=True)
class PolicyCheckpointMetadata:
    observation_dim: int
    action_dim: int
    hidden_dim: int
    episode_count: int
    total_steps: int
    mean_episode_reward: float


def save_policy_checkpoint(
    policy: PpoPolicy,
    path: Path,
    *,
    episode_count: int,
    total_steps: int,
    mean_episode_reward: float,
) -> Path:
    """Persist policy weights and training metadata as JSON + .npz arrays."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PolicyCheckpointMetadata(
        observation_dim=policy.config.observation_dim,
        action_dim=policy.config.action_dim,
        hidden_dim=policy.config.hidden_dim,
        episode_count=episode_count,
        total_steps=total_steps,
        mean_episode_reward=mean_episode_reward,
    )
    payload: dict[str, Any] = {
        'metadata': asdict(metadata),
        'config': asdict(policy.config),
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    np.savez(
        path.with_suffix('.npz'),
        w1=policy.w1,
        b1=policy.b1,
        w2=policy.w2,
        b2=policy.b2,
        log_std=policy.log_std,
    )
    return path


def load_policy_checkpoint(path: Path) -> PpoPolicy:
    """Restore a policy from checkpoint JSON + .npz weights."""
    path = path.resolve()
    meta = json.loads(path.read_text(encoding='utf-8'))
    config = PpoPolicyConfig(**meta['config'])
    policy = PpoPolicy(config)
    arrays = np.load(path.with_suffix('.npz'))
    policy.w1 = np.array(arrays['w1'], dtype=np.float64)
    policy.b1 = np.array(arrays['b1'], dtype=np.float64)
    policy.w2 = np.array(arrays['w2'], dtype=np.float64)
    policy.b2 = np.array(arrays['b2'], dtype=np.float64)
    policy.log_std = np.array(arrays['log_std'], dtype=np.float64)
    return policy


def export_policy_weights_for_onnx(path: Path, export_path: Path) -> Path:
    """Write a flat weight dictionary consumable by Phase 3 export tooling."""
    policy = load_policy_checkpoint(path)
    export_path = export_path.resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_payload = {
        'format': 'spark_mycobot_ppo_v1',
        'observation_dim': policy.config.observation_dim,
        'action_dim': policy.config.action_dim,
        'hidden_dim': policy.config.hidden_dim,
        'weights': {
            'w1': policy.w1.tolist(),
            'b1': policy.b1.tolist(),
            'w2': policy.w2.tolist(),
            'b2': policy.b2.tolist(),
            'log_std': policy.log_std.tolist(),
        },
    }
    export_path.write_text(json.dumps(export_payload, indent=2), encoding='utf-8')
    return export_path
