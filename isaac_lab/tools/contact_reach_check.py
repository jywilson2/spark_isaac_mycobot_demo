# Copyright 2026 spark_isaac_mycobot_demo contributors
"""Headless check: EE distance to block after reset and after one episode."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(['--headless', '--viz', 'none'])
    app = AppLauncher(args).app

    import torch

    from isaac_lab.mycobot_pick_place_env import MyCobotPickPlaceEnv, make_env_cfg

    env = MyCobotPickPlaceEnv(cfg=make_env_cfg(num_envs=1, seed=7))
    b0, base0, ee0 = env._world_positions()
    print('PRE-RESET joints', [round(x, 3) for x in env._robot.data.joint_pos[0].tolist()])
    print('PRE-RESET base', [round(x, 3) for x in base0[0].tolist()])
    print('PRE-RESET ee', [round(x, 3) for x in ee0[0].tolist()])
    env.reset()
    print('JOINTS', [round(x, 3) for x in env._robot.data.joint_pos[0].tolist()])
    block, base, ee = env._world_positions()
    print('POST-RESET base', [round(x, 3) for x in base[0].tolist()])
    h0 = math.hypot(block[0, 0].item() - ee[0, 0].item(), block[0, 1].item() - ee[0, 1].item())
    v0 = ee[0, 2].item() - (block[0, 2].item() + 0.02)
    print('INIT block', [round(x, 3) for x in block[0].tolist()])
    print('INIT ee', [round(x, 3) for x in ee[0].tolist()])
    print('INIT h', round(h0, 3), 'v', round(v0, 3), 'contact', bool(env._contact_mask(block, ee)[0].item()))

    for step in range(360):
        env.step(torch.zeros((1, 6), device=env.device))
        if step + 1 in (60, 120, 240, 360):
            block, _, ee = env._world_positions()
            h = math.hypot(block[0, 0].item() - ee[0, 0].item(), block[0, 1].item() - ee[0, 1].item())
            v = ee[0, 2].item() - (block[0, 2].item() + 0.02)
            print(
                f'step {step + 1}',
                'ee', [round(x, 3) for x in ee[0].tolist()],
                'h', round(h, 3),
                'v', round(v, 3),
                'contact', bool(env._contact_mask(block, ee)[0].item()),
            )

    print('METRICS', env.get_task_metrics())
    env.close()
    app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
