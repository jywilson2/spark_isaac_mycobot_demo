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

"""Run Isaac Lab PPO training for the MyCobot pick-and-place task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train MyCobot pick-and-place with Isaac Lab PPO')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--max-iterations', type=int, default=20)
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=REPO_ROOT / 'assets' / 'checkpoints' / 'isaac_lab_ppo',
    )
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
    parser.add_argument('--seed', type=int, default=42)

    from isaaclab.app import AppLauncher  # noqa: WPS433

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher  # noqa: WPS433

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch  # noqa: WPS433
    import importlib.metadata as metadata  # noqa: WPS433
    from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: WPS433

    from isaac_lab.detect_isaac_lab import require_isaac_lab  # noqa: WPS433
    from isaac_lab.mycobot_pick_place_env import (  # noqa: WPS433
        MyCobotPickPlaceEnv,
        TASK_ID,
        make_env_cfg,
        register_mycobot_env,
    )
    from isaac_lab.rsl_rl_ppo_cfg import MyCobotPPORunnerCfg  # noqa: WPS433

    require_isaac_lab()

    if not args.robot_usd.is_file():
        print(f'Robot USD missing: {args.robot_usd}', file=sys.stderr)
        print('Run ./scripts/host/iter_build_isaac_scene.sh first.', file=sys.stderr)
        simulation_app.close()
        return 1

    register_mycobot_env()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = checkpoint_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = make_env_cfg(
        num_envs=args.num_envs,
        robot_usd_path=str(args.robot_usd.resolve()),
        checkpoint_dir=str(checkpoint_dir),
    )
    env = MyCobotPickPlaceEnv(cfg=env_cfg)

    agent_cfg = MyCobotPPORunnerCfg(max_iterations=args.max_iterations)
    installed_version = metadata.version('rsl-rl-lib')
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(log_dir),
        device=agent_cfg.device,
    )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    policy_path = checkpoint_dir / 'latest_policy'
    runner.save(str(policy_path))

    summary = {
        'task': TASK_ID,
        'num_envs': args.num_envs,
        'max_iterations': args.max_iterations,
        'checkpoint': str(policy_path),
        'log_dir': str(log_dir),
    }
    summary_path = checkpoint_dir / 'training_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))

    env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
