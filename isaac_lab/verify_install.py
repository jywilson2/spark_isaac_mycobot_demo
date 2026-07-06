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

"""Verify Isaac Lab imports and optional headless env smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify Isaac Lab integration for this repo')
    parser.add_argument('--smoke-env', action='store_true', help='Run a short headless env rollout')
    parser.add_argument('--steps', type=int, default=8)

    from isaaclab.app import AppLauncher  # noqa: WPS433

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def verify_imports() -> None:
    import isaaclab  # noqa: F401,WPS433
    import rsl_rl  # noqa: F401,WPS433

    from isaac_lab.detect_isaac_lab import require_isaac_lab  # noqa: WPS433
    from isaac_lab import mdp_core  # noqa: F401,WPS433
    from isaac_lab.mycobot_pick_place_env import TASK_ID, register_mycobot_env  # noqa: WPS433

    info = require_isaac_lab()
    register_mycobot_env()
    print(f'Isaac Lab root: {info.root}')
    print(f'Isaac Lab task registered: {TASK_ID}')
    print(f'MDP observation dim: {mdp_core.OBSERVATION_DIM}')


def smoke_env(args: argparse.Namespace) -> None:
    import torch  # noqa: WPS433

    from isaac_lab.mycobot_pick_place_env import MyCobotPickPlaceEnv, make_env_cfg  # noqa: WPS433

    env_cfg = make_env_cfg(num_envs=1)
    env = MyCobotPickPlaceEnv(cfg=env_cfg)
    obs, _info = env.reset()
    assert 'policy' in obs
    assert obs['policy'].shape[-1] == env_cfg.observation_space

    for _step in range(args.steps):
        action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
        obs, reward, terminated, truncated, _info = env.step(action)
        assert obs['policy'].shape[-1] == env_cfg.observation_space
        assert reward.shape[0] == env.num_envs
        if bool(torch.any(terminated | truncated)):
            env.reset()

    env.close()
    print(f'Smoke env OK ({args.steps} steps)')


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher  # noqa: WPS433

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        verify_imports()
        if args.smoke_env:
            smoke_env(args)
    finally:
        simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
