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

"""Lightweight numpy PPO policy for live ROS training against Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PpoPolicyConfig:
    observation_dim: int = 14
    action_dim: int = 6
    hidden_dim: int = 64
    action_scale: float = 0.05
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    seed: int = 42


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class PpoPolicy:
    """Two-layer Gaussian policy with clipped PPO updates (numpy only)."""

    def __init__(self, config: PpoPolicyConfig | None = None) -> None:
        self.config = config or PpoPolicyConfig()
        rng = np.random.default_rng(self.config.seed)
        hidden = self.config.hidden_dim
        obs = self.config.observation_dim
        act = self.config.action_dim
        scale = 0.1
        self.w1 = rng.normal(0.0, scale, size=(obs, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = rng.normal(0.0, scale, size=(hidden, act))
        self.b2 = np.zeros(act, dtype=np.float64)
        self.log_std = np.full(act, -1.0, dtype=np.float64)

    def _forward(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(observation @ self.w1 + self.b1)
        mean = hidden @ self.w2 + self.b2
        value = float(np.tanh(np.mean(hidden)))
        return mean, np.array([value], dtype=np.float64)

    def _distribution_log_prob(self, mean: np.ndarray, action: np.ndarray) -> float:
        std = np.exp(self.log_std)
        var = std * std
        log_prob = -0.5 * (
            np.sum(((action - mean) ** 2) / var)
            + np.sum(np.log(2.0 * np.pi * var))
        )
        return float(log_prob)

    def act(
        self,
        observation: Sequence[float],
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        obs = np.asarray(observation, dtype=np.float64)
        mean, value = self._forward(obs)
        if deterministic:
            action = mean.copy()
        else:
            std = np.exp(self.log_std)
            action = mean + std * np.random.default_rng().normal(size=mean.shape)
        log_prob = self._distribution_log_prob(mean, action)
        return action, float(value[0]), log_prob

    def apply_action_delta(
        self,
        current_joint_positions: Sequence[float],
        action_delta: Sequence[float],
        joint_min: Sequence[float],
        joint_max: Sequence[float],
    ) -> list[float]:
        targets: list[float] = []
        for current, delta, lower, upper in zip(
            current_joint_positions,
            action_delta,
            joint_min,
            joint_max,
            strict=True,
        ):
            target = current + float(delta) * self.config.action_scale
            targets.append(float(np.clip(target, lower, upper)))
        return targets

    def compute_gae(
        self,
        rewards: Sequence[float],
        values: Sequence[float],
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        values_arr = np.asarray(values, dtype=np.float64)
        advantages = np.zeros_like(rewards_arr)
        last_gae = 0.0
        for step in reversed(range(len(rewards_arr))):
            next_value = values_arr[step + 1] if step + 1 < len(values_arr) else 0.0
            delta = rewards_arr[step] + gamma * next_value - values_arr[step]
            last_gae = delta + gamma * lam * last_gae
            advantages[step] = last_gae
        returns = advantages + values_arr[: len(rewards_arr)]
        return advantages, returns

    def ppo_update(self, batch: RolloutBatch, *, epochs: int = 4) -> dict[str, float]:
        obs = batch.observations
        actions = batch.actions
        old_log_probs = batch.log_probs
        advantages = batch.advantages
        returns = batch.returns
        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages)) or 1.0
        norm_adv = (advantages - adv_mean) / adv_std

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        steps = len(obs)

        for _ in range(epochs):
            for index in range(steps):
                observation = obs[index]
                action = actions[index]
                mean, value = self._forward(observation)
                log_prob = self._distribution_log_prob(mean, action)
                ratio = np.exp(log_prob - old_log_probs[index])
                clipped = np.clip(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                )
                policy_loss = -float(
                    np.minimum(ratio * norm_adv[index], clipped * norm_adv[index]))

                value_error = float(returns[index] - value[0])
                value_loss = value_error * value_error

                std = np.exp(self.log_std)
                entropy = float(np.sum(np.log(std) + 0.5 * np.log(2.0 * np.pi * np.e)))

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                grad_scale = self.config.learning_rate * loss
                hidden = np.tanh(observation @ self.w1 + self.b1)
                action_grad = grad_scale * (action - mean) / (std * std + 1e-8)
                self.w2 += np.outer(hidden, action_grad)
                self.b2 += action_grad
                self.w1 += np.outer(observation, hidden * (1.0 - hidden * hidden) * grad_scale)

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy

        return {
            'policy_loss': total_policy_loss / max(steps * epochs, 1),
            'value_loss': total_value_loss / max(steps * epochs, 1),
            'entropy': total_entropy / max(steps * epochs, 1),
        }


def build_rollout_batch(
    observations: Sequence[Sequence[float]],
    actions: Sequence[Sequence[float]],
    rewards: Sequence[float],
    values: Sequence[float],
    log_probs: Sequence[float],
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> RolloutBatch:
    policy = PpoPolicy()
    advantages, returns = policy.compute_gae(rewards, values, gamma=gamma, lam=lam)
    return RolloutBatch(
        observations=np.asarray(observations, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        log_probs=np.asarray(log_probs, dtype=np.float64),
        advantages=advantages,
        returns=returns,
    )
