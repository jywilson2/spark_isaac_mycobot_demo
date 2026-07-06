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

"""Live PPO trainer node for Isaac Lab Phase 2 against running Isaac Sim."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from rclpy.node import Node

from spark_verify_nodes.live_ros_mdp_client import LiveRosMdpClient
from spark_verify_nodes.live_sim_gate import LiveSimGate, skip_live_tests_enabled
from spark_verify_nodes.policy_checkpoint import (
    export_policy_weights_for_onnx,
    save_policy_checkpoint,
)
from spark_verify_nodes.ppo_policy import PpoPolicy, PpoPolicyConfig, RolloutBatch
from std_msgs.msg import Float32, String

if TYPE_CHECKING:
    from rclpy.executors import Executor


class LivePpoTrainer(Node):
    """Collect live rollouts, run PPO updates, and write checkpoints."""

    def __init__(self) -> None:
        super().__init__('live_ppo_trainer')
        self.declare_parameter('num_episodes', 3)
        self.declare_parameter('steps_per_episode', 8)
        self.declare_parameter('checkpoint_dir', 'assets/checkpoints/live_ppo')
        self.declare_parameter('export_onnx_weights', True)
        self.declare_parameter('training_status_topic', '/mycobot/rl/training_status')
        self.declare_parameter('require_live_sim', True)
        self.declare_parameter('auto_shutdown', True)
        self.declare_parameter('use_mock_joint_names', False)

        self._num_episodes = int(self.get_parameter('num_episodes').value)
        self._steps_per_episode = int(self.get_parameter('steps_per_episode').value)
        checkpoint_dir = self.get_parameter('checkpoint_dir').value
        self._checkpoint_dir = Path(str(checkpoint_dir))
        self._export_onnx = bool(self.get_parameter('export_onnx_weights').value)
        status_topic = self.get_parameter('training_status_topic').value
        self._require_live_sim = bool(self.get_parameter('require_live_sim').value)
        self._auto_shutdown = bool(self.get_parameter('auto_shutdown').value)
        use_mock_joint_names = bool(self.get_parameter('use_mock_joint_names').value)

        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._reward_pub = self.create_publisher(
            Float32, '/mycobot/rl/training_episode_reward', 10)
        self._client = LiveRosMdpClient(
            self,
            use_mock_joint_names=use_mock_joint_names,
        )
        config = PpoPolicyConfig(
            observation_dim=self._client.observation_dim,
            action_dim=self._client.action_dim,
        )
        self._policy = PpoPolicy(config)

        self._total_steps = 0

    def _publish_status(self, message: str) -> None:
        msg = String()
        msg.data = message
        self._status_pub.publish(msg)
        self.get_logger().info(message)

    def run(self, executor: Executor) -> None:
        """Wait for live telemetry, train, publish summary, and optionally shutdown."""
        if self._require_live_sim and not skip_live_tests_enabled():
            if not LiveSimGate.probe(
                required_topics=[
                    '/mycobot/joint_states',
                    '/mycobot/rl/observation',
                    '/mycobot/vision/block_detection',
                ],
                timeout_sec=8.0,
            ):
                self._publish_status('error: live sim not detected')
                return

        deadline = time.time() + 20.0
        ready = False
        while time.time() < deadline and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if self._client.latest_joint_state is not None:
                ready = True
                break
        if not ready:
            self._publish_status('error: joint states unavailable')
            return

        try:
            summary = self._train(executor)
            self._publish_status(json.dumps(summary))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Training failed: {exc}')
            self._publish_status(f'error: {exc}')

    def _train(self, executor: Executor) -> dict[str, float | int | str]:
        all_rewards: list[float] = []
        for episode in range(self._num_episodes):
            episode_reward = self._run_episode(executor, episode)
            all_rewards.append(episode_reward)
            reward_msg = Float32()
            reward_msg.data = float(episode_reward)
            self._reward_pub.publish(reward_msg)
            self._publish_status(
                f'episode {episode + 1}/{self._num_episodes} '
                f'reward={episode_reward:.4f}')

        mean_reward = sum(all_rewards) / max(len(all_rewards), 1)
        checkpoint_path = save_policy_checkpoint(
            self._policy,
            self._checkpoint_dir / 'latest_policy.json',
            episode_count=self._num_episodes,
            total_steps=self._total_steps,
            mean_episode_reward=mean_reward,
        )
        export_path = None
        if self._export_onnx:
            export_path = export_policy_weights_for_onnx(
                checkpoint_path,
                self._checkpoint_dir / 'latest_policy_onnx_ready.json',
            )

        return {
            'episodes': self._num_episodes,
            'total_steps': self._total_steps,
            'mean_episode_reward': mean_reward,
            'checkpoint': str(checkpoint_path),
            'onnx_ready_export': str(export_path) if export_path else '',
        }

    def _run_episode(self, executor: Executor, episode_index: int) -> float:
        observations: list[list[float]] = []
        actions: list[list[float]] = []
        rewards: list[float] = []
        values: list[float] = []
        log_probs: list[float] = []

        for _step in range(self._steps_per_episode):
            step_result, _targets, value, log_prob, action_delta = (
                self._client.step_with_policy(
                    self._policy,
                    executor=executor,
                    timeout_sec=20.0,
                )
            )
            observations.append(step_result.observation)
            actions.append(action_delta)
            rewards.append(step_result.reward)
            values.append(value)
            log_probs.append(log_prob)
            self._total_steps += 1

        batch = RolloutBatch(
            observations=np.array(observations, dtype=np.float64),
            actions=np.array(actions, dtype=np.float64),
            rewards=np.array(rewards, dtype=np.float64),
            values=np.array(values, dtype=np.float64),
            log_probs=np.array(log_probs, dtype=np.float64),
            advantages=np.zeros(len(rewards), dtype=np.float64),
            returns=np.zeros(len(rewards), dtype=np.float64),
        )
        advantages, returns = self._policy.compute_gae(rewards, values)
        batch.advantages = advantages
        batch.returns = returns
        metrics = self._policy.ppo_update(batch, epochs=2)
        episode_reward = float(sum(rewards))
        self.get_logger().info(
            f'Episode {episode_index + 1} reward={episode_reward:.4f} '
            f'policy_loss={metrics["policy_loss"]:.4f}')
        return episode_reward


def main() -> None:
    rclpy.init()
    node = LivePpoTrainer()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    auto_shutdown = node._auto_shutdown  # noqa: SLF001
    try:
        node.run(executor)
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            if auto_shutdown:
                rclpy.shutdown()


if __name__ == '__main__':
    main()
