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

"""Play a trained PPO policy in Isaac Sim (GUI demo: locate + push random cube)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_lab.warning_filters import apply_training_runtime_env

apply_training_runtime_env()


def resolve_checkpoint_path(checkpoint: Path) -> Path:
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    model_files = sorted(checkpoint.glob('model_*.pt'))
    if not model_files:
        raise FileNotFoundError(
            f'No model_*.pt files under {checkpoint}. Train first with train_ppo.py.')
    return model_files[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Play trained MyCobot contact-and-push PPO policy in Isaac Sim',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=REPO_ROOT / 'assets' / 'checkpoints' / 'isaac_lab_ppo' / 'latest_policy',
        help='Policy directory (latest_policy) or model_*.pt file',
    )
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--num-arms', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--robot-usd',
        type=Path,
        default=REPO_ROOT
        / 'assets'
        / 'robots'
        / 'mycobot_280_m5_limo_cobot'
        / 'mycobot_280_m5_limo_cobot'
        / 'mycobot_280_m5_limo_cobot.usda',
    )

    from isaaclab.app import AppLauncher  # noqa: WPS433

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher  # noqa: WPS433

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import importlib.metadata as metadata  # noqa: WPS433
    import torch  # noqa: WPS433
    from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: WPS433

    from isaac_lab.detect_isaac_lab import require_isaac_lab  # noqa: WPS433
    from isaac_lab.mycobot_reach_env import MyCobotReachEnv, make_env_cfg, register_mycobot_env  # noqa: WPS433
    from isaac_lab.rsl_rl_ppo_cfg import MyCobotPPORunnerCfg  # noqa: WPS433
    from isaac_lab.training_success import resolve_task_env  # noqa: WPS433

    require_isaac_lab()

    if not args.robot_usd.is_file():
        print(f'Robot USD missing: {args.robot_usd}', file=sys.stderr)
        simulation_app.close()
        return 1

    try:
        checkpoint_file = resolve_checkpoint_path(args.checkpoint.resolve())
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        simulation_app.close()
        return 1

    register_mycobot_env()
    env_cfg = make_env_cfg(
        num_envs=args.num_arms,
        robot_usd_path=str(args.robot_usd.resolve()),
        seed=args.seed,
    )
    env = MyCobotReachEnv(cfg=env_cfg)

    agent_cfg = MyCobotPPORunnerCfg(max_iterations=1)
    installed_version = metadata.version('rsl-rl-lib')
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    log_dir = args.checkpoint.resolve().parent / 'logs'
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(log_dir),
        device=agent_cfg.device,
    )
    runner.load(str(checkpoint_file), load_optimizer=False)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    task_env = resolve_task_env(runner)
    episodes_completed = 0
    successes = 0
    obs, _ = env.reset()

    print('=== MyCobot EE reach policy demo ===')
    print(f'Checkpoint: {checkpoint_file}')
    print(f'Episodes: {args.episodes}  Arms: {args.num_arms}')
    print('Each episode spawns a random reachable EE target (red marker).\n')

    while simulation_app.is_running() and episodes_completed < args.episodes:
        with torch.inference_mode():
            actions = policy(obs)
        obs, _rewards, dones, _infos = env.step(actions)
        if not torch.any(dones):
            continue

        episodes_completed += 1
        if task_env is not None:
            metrics = task_env.get_task_metrics()
            reach_rate = metrics.get('reach_success_rate', 0.0)
            if reach_rate >= 0.99 or metrics.get('mean_time_to_reach_s', 99.0) < 4.0:
                successes += 1
            print(
                f'Episode {episodes_completed}/{args.episodes}: '
                f'reach_success_rate={reach_rate:.1%}, '
                f'mean_time_to_reach={metrics.get("mean_time_to_reach_s", 0.0):.2f}s'
            )
        else:
            print(f'Episode {episodes_completed}/{args.episodes} complete')

        obs, _ = env.reset()

    print(
        f'\nDemo complete: {episodes_completed} episodes, {successes} reached target quickly.'
    )
    env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
