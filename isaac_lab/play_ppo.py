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

"""Play a trained Phase 2 reach PPO policy in Isaac Sim (GUI).

Tutorial modes:
  * ``--demo`` — continuous showcase: one arm, red target sphere respawned after
    each reach; runs until you close Isaac Sim or press Ctrl+C.
  * ``--episodes N`` — finite regression run (default 10).

See spec.md § Phase 2 and ``./scripts/host/run_isaac_lab_training.sh demo``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_lab.warning_filters import apply_training_runtime_env

apply_training_runtime_env()


def resolve_checkpoint_path(checkpoint: Path) -> Path:
    from isaac_lab.train_ppo import find_newest_checkpoint  # noqa: WPS433

    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    newest = find_newest_checkpoint(checkpoint)
    if newest is None:
        raise FileNotFoundError(
            f'No checkpoint found under {checkpoint}. Train first with train_ppo.py.')
    return newest


def build_parser(*, with_app_launcher: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Play trained MyCobot EE reach PPO policy in Isaac Sim',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Continuous GUI showcase: single arm, red target respawns after each '
        'reach; runs until Isaac Sim exits or Ctrl+C.',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=REPO_ROOT / 'assets' / 'checkpoints' / 'isaac_lab_ppo' / 'latest_policy',
        help='Policy directory (latest_policy) or model_*.pt file',
    )
    parser.add_argument(
        '--demo-max-episodes',
        type=int,
        default=None,
        help='Stop continuous demo after N reach attempts (for headless verification). '
        'GUI demo runs until exit when omitted.',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Finite play episodes (ignored when --demo is set).',
    )
    parser.add_argument(
        '--episode-length-s',
        type=float,
        default=None,
        help='Episode duration in seconds (default matches training: 30 s).',
    )
    parser.add_argument('--num-arms', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--policy-mean',
        action='store_true',
        help='Use the policy mean action instead of sampling (often more consistent '
        'at inference when action std is still wide).',
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
    parser.add_argument(
        '--action-scale',
        type=float,
        default=None,
        help='Joint delta scale; should match the value used during training.',
    )
    if with_app_launcher:
        from isaaclab.app import AppLauncher  # noqa: WPS433

        AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser(with_app_launcher=True).parse_args(argv)
    if args.demo:
        args.num_arms = 1
    if args.episode_length_s is None:
        from isaac_lab.training_defaults import DEFAULT_EPISODE_LENGTH_S  # noqa: WPS433

        args.episode_length_s = DEFAULT_EPISODE_LENGTH_S
    return args


def inference_load_cfg() -> dict[str, bool]:
    """rsl-rl load_cfg for play/demo: policy weights only (no optimizer state)."""

    return {
        'actor': True,
        'critic': True,
        'optimizer': False,
        'iteration': False,
        'rnd': False,
    }


def select_inference_actions(policy: Any, obs: Any, *, stochastic: bool = True) -> Any:
    """Sample actions for play/demo.

    Training rollouts use ``stochastic_output=True`` in ``PPO.act``. Using the
    deterministic mean at inference often under-commands joint deltas when the
    policy still carries significant action std.
    """

    import torch  # noqa: WPS433

    with torch.inference_mode():
        return policy(obs, stochastic_output=stochastic)


def read_episode_outcome(task_env: Any, *, env_idx: int = 0) -> dict[str, Any]:
    """Read reach outcome for one parallel env (fallback when terminal cache is empty)."""

    step_dt = float(task_env.cfg.sim.dt * task_env.cfg.decimation)
    reached = bool(task_env._episode_reached[env_idx].item())
    steps = float(task_env._episode_steps_to_reach[env_idx].item())
    target = task_env._target_ee_pos[env_idx].detach().cpu().tolist()
    ee = task_env._ee_position_base()[env_idx].detach().cpu().tolist()
    distance = float(task_env._distance_to_target(task_env._ee_position_base())[env_idx].item())
    return {
        'reached': reached,
        'time_to_reach_s': steps * step_dt if reached else None,
        'target_xyz': target,
        'ee_xyz': ee,
        'distance_m': distance,
    }


def consume_episode_outcome(task_env: Any, *, env_idx: int = 0) -> dict[str, Any]:
    """Read terminal reach metrics captured before DirectRLEnv auto-reset in ``step()``."""

    pop = getattr(task_env, 'pop_episode_outcome', None)
    if callable(pop):
        outcome = pop(env_idx)
        if outcome is not None:
            return outcome
    return read_episode_outcome(task_env, env_idx=env_idx)


def format_episode_line(episode: int, outcome: dict[str, Any]) -> str:
    target = outcome['target_xyz']
    if outcome['reached']:
        time_s = outcome['time_to_reach_s']
        return (
            f'Episode {episode}: REACH OK in {time_s:.2f}s — '
            f'target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) m'
        )
    return (
        f'Episode {episode}: timeout — distance {outcome["distance_m"] * 1000:.1f} mm, '
        f'target=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) m'
    )


def run_play_loop(
    *,
    simulation_app: Any,
    env: Any,
    policy: Any,
    task_env: Any | None,
    episodes: int,
    demo: bool,
    demo_max_episodes: int | None = None,
    stochastic: bool = True,
) -> tuple[int, int]:
    """Step the policy until episode limit or simulator exit; return (episodes, successes)."""

    import torch  # noqa: WPS433

    episodes_completed = 0
    successes = 0
    obs, _ = env.reset()

    if demo:
        print('=== MyCobot EE reach DEMO (continuous) ===')
        print('Single arm | red sphere target | respawns after each reach')
        print('Close the Isaac Sim window or press Ctrl+C to exit.\n')
    else:
        print('=== MyCobot EE reach policy play ===')
        print(f'Episodes: {episodes}')
        print('Each episode spawns a random reachable EE target (red marker).\n')

    while simulation_app.is_running():
        if not demo and episodes_completed >= episodes:
            break
        if demo and demo_max_episodes is not None and episodes_completed >= demo_max_episodes:
            break
        with torch.inference_mode():
            actions = select_inference_actions(policy, obs, stochastic=stochastic)
        obs, _rewards, dones, _infos = env.step(actions)
        if not torch.any(dones):
            continue

        if task_env is not None:
            num_envs = int(getattr(task_env, 'num_envs', 1))
            for env_idx in range(num_envs):
                if not bool(dones[env_idx].item()):
                    continue
                episodes_completed += 1
                outcome = consume_episode_outcome(task_env, env_idx=env_idx)
                if outcome['reached']:
                    successes += 1
                print(format_episode_line(episodes_completed, outcome))
                if not demo and episodes_completed >= episodes:
                    break
                if demo and demo_max_episodes is not None and episodes_completed >= demo_max_episodes:
                    break
        else:
            episodes_completed += 1
            print(f'Episode {episodes_completed} complete')

        # DirectRLEnv already auto-reset inside step(); obs is post-reset for next episode.

    return episodes_completed, successes


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher  # noqa: WPS433

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import importlib.metadata as metadata  # noqa: WPS433
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
        checkpoint_arg = args.checkpoint
        if not checkpoint_arg.is_absolute():
            checkpoint_arg = (REPO_ROOT / checkpoint_arg).resolve()
        checkpoint_file = resolve_checkpoint_path(checkpoint_arg)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        simulation_app.close()
        return 1

    register_mycobot_env()
    env_cfg = make_env_cfg(
        num_envs=args.num_arms,
        robot_usd_path=str(args.robot_usd.resolve()),
        seed=args.seed,
        target_sampling='demo' if args.demo else 'workspace',
        episode_length_s=args.episode_length_s,
        action_scale=args.action_scale,
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
    runner.load(str(checkpoint_file), load_cfg=inference_load_cfg())
    policy = runner.get_inference_policy(device=agent_cfg.device)

    task_env = resolve_task_env(runner)
    print(f'Checkpoint: {checkpoint_file}\n')

    episodes_completed, successes = run_play_loop(
        simulation_app=simulation_app,
        env=env,
        policy=policy,
        task_env=task_env,
        episodes=args.episodes,
        demo=args.demo,
        demo_max_episodes=args.demo_max_episodes,
        stochastic=not args.policy_mean,
    )

    if args.demo:
        rate = (successes / episodes_completed * 100.0) if episodes_completed else 0.0
        print(
            f'\nDemo ended: {episodes_completed} targets attempted, {successes} reached '
            f'({rate:.1f}%).'
        )
    else:
        print(
            f'\nPlay complete: {episodes_completed} episodes, {successes} reached target.'
        )

    env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
