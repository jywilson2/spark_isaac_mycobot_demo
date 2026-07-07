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

"""Run Isaac Lab PPO training for the MyCobot reach (Phase 2) or red-block (Phase 6) task.

Default training is **duration-bounded** (30 minutes) with early stop at 99% reach
success. Use ``--fixed-iterations`` for short smoke tests. See spec.md § Phase 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_lab.warning_filters import apply_training_runtime_env

apply_training_runtime_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train MyCobot pick-and-place with Isaac Lab PPO')
    parser.add_argument(
        '--num-envs',
        '--num-arms',
        dest='num_envs',
        type=int,
        default=None,
        help='Parallel MyCobot arms. Default: 2 with GUI, 8 headless (DGX Spark).',
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Fixed iteration cap for smoke tests (--fixed-iterations). '
        'Default training uses --max-duration-minutes instead.',
    )
    parser.add_argument(
        '--max-duration-minutes',
        type=float,
        default=None,
        help='Stop after this many minutes (default 30). Set 0 to disable time limit.',
    )
    parser.add_argument(
        '--motion-glossary',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Print tutorial-style MDP/motion glossary at startup and per-iteration '
        'motion notes (default: on). Isaac Lab reserves --verbose for kit logging.',
    )
    parser.add_argument(
        '--target-reach-success-rate',
        type=float,
        default=None,
        help='Stop training when rolling EE reach success rate reaches this target (default 0.99).',
    )
    parser.add_argument(
        '--target-push-success-rate',
        type=float,
        default=0.50,
        help='Phase 6 only: stop when rolling push-success rate reaches this target.',
    )
    parser.add_argument(
        '--target-contact-rate',
        type=float,
        default=0.70,
        help='Phase 6 only: required contact rate paired with push-success target.',
    )
    parser.add_argument(
        '--use-red-block-vision',
        action='store_true',
        help='Enable Phase 6 red-block vision + contact-and-push task (requires --enable_cameras).',
    )
    parser.add_argument(
        '--fixed-iterations',
        action='store_true',
        help='Run exactly --max-iterations instead of stopping on task success.',
    )
    parser.add_argument(
        '--from-scratch',
        action='store_true',
        help='Remove existing checkpoints in --checkpoint-dir before training.',
    )
    parser.add_argument(
        '--no-plateau-abort',
        action='store_true',
        help='Disable early abort when reach success stops improving.',
    )
    parser.add_argument(
        '--plateau-window-iterations',
        type=int,
        default=None,
        help='Abort if reach success does not improve for this many iterations '
        '(default from training_defaults).',
    )
    parser.add_argument(
        '--plateau-warmup-iterations',
        type=int,
        default=None,
        help='Iterations before plateau detection activates (default from training_defaults).',
    )
    parser.add_argument(
        '--min-reach-improvement',
        type=float,
        default=None,
        help='Minimum reach-success increase to reset plateau timer (default 0.01).',
    )
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
    args = parser.parse_args()
    if args.num_envs is None:
        from isaac_lab.training_defaults import default_num_arms  # noqa: WPS433

        headless = bool(getattr(args, 'headless', False))
        viz = getattr(args, 'viz', None)
        if viz == 'none':
            headless = True
        args.num_envs = default_num_arms(headless=headless)
    if args.max_duration_minutes is None:
        from isaac_lab.training_defaults import DEFAULT_MAX_TRAIN_DURATION_MINUTES  # noqa: WPS433

        args.max_duration_minutes = DEFAULT_MAX_TRAIN_DURATION_MINUTES
    if args.target_reach_success_rate is None:
        from isaac_lab.mdp_core import DEFAULT_TARGET_REACH_SUCCESS_RATE  # noqa: WPS433

        args.target_reach_success_rate = DEFAULT_TARGET_REACH_SUCCESS_RATE
    if args.plateau_window_iterations is None:
        from isaac_lab.training_defaults import DEFAULT_PLATEAU_WINDOW_ITERATIONS  # noqa: WPS433

        args.plateau_window_iterations = DEFAULT_PLATEAU_WINDOW_ITERATIONS
    if args.plateau_warmup_iterations is None:
        from isaac_lab.training_defaults import DEFAULT_PLATEAU_WARMUP_ITERATIONS  # noqa: WPS433

        args.plateau_warmup_iterations = DEFAULT_PLATEAU_WARMUP_ITERATIONS
    if args.min_reach_improvement is None:
        from isaac_lab.training_defaults import DEFAULT_MIN_REACH_IMPROVEMENT  # noqa: WPS433

        args.min_reach_improvement = DEFAULT_MIN_REACH_IMPROVEMENT
    return args


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher  # noqa: WPS433

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import importlib.metadata as metadata  # noqa: WPS433
    from rsl_rl.runners import OnPolicyRunner  # noqa: WPS433

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: WPS433

    from isaac_lab.detect_isaac_lab import require_isaac_lab  # noqa: WPS433

    if args.use_red_block_vision:
        from isaac_lab.phase6_red_block.red_block_env import (  # noqa: WPS433
            MyCobotRedBlockEnv,
            PHASE6_TASK_ID as TASK_ID,
            make_env_cfg,
            register_red_block_env,
        )

        register_env = register_red_block_env
        EnvClass = MyCobotRedBlockEnv
        task_mode = 'red_block'
    else:
        from isaac_lab.mycobot_reach_env import (  # noqa: WPS433
            MyCobotReachEnv,
            TASK_ID,
            make_env_cfg,
            register_mycobot_env,
        )

        register_env = register_mycobot_env
        EnvClass = MyCobotReachEnv
        task_mode = 'reach'
    from isaac_lab.rsl_rl_ppo_cfg import MyCobotPPORunnerCfg  # noqa: WPS433
    from isaac_lab.training_success import (  # noqa: WPS433
        TrainingCompletionReport,
        TrainingSuccessCriteria,
        collect_task_metrics,
        resolve_task_env,
        run_training_with_reports,
    )

    require_isaac_lab()

    if not args.robot_usd.is_file():
        print(f'Robot USD missing: {args.robot_usd}', file=sys.stderr)
        print('Run ./scripts/host/iter_build_isaac_scene.sh first.', file=sys.stderr)
        simulation_app.close()
        return 1

    register_env()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if args.from_scratch and checkpoint_dir.exists():
        import shutil

        print(f'--from-scratch: removing {checkpoint_dir}')
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = checkpoint_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = make_env_cfg(
        num_envs=args.num_envs,
        robot_usd_path=str(args.robot_usd.resolve()),
        checkpoint_dir=str(checkpoint_dir),
        seed=args.seed,
    )
    env = EnvClass(cfg=env_cfg)

    if args.motion_glossary and task_mode == 'reach':
        from isaac_lab.training_verbose import print_reach_training_guide  # noqa: WPS433

        print_reach_training_guide(
            num_envs=args.num_envs,
            max_duration_minutes=args.max_duration_minutes,
            target_reach_success_rate=args.target_reach_success_rate,
        )

    planned_iterations = args.max_iterations
    if planned_iterations is None and args.fixed_iterations:
        from isaac_lab.training_defaults import INTEGRATION_TRAIN_MAX_ITERATIONS  # noqa: WPS433

        planned_iterations = INTEGRATION_TRAIN_MAX_ITERATIONS
    agent_cfg = MyCobotPPORunnerCfg(
        max_iterations=planned_iterations or 10_000,
    )
    installed_version = metadata.version('rsl-rl-lib')
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    criteria = TrainingSuccessCriteria(
        task_mode=task_mode,
        target_reach_success_rate=args.target_reach_success_rate,
        target_push_success_rate=args.target_push_success_rate,
        target_contact_rate=args.target_contact_rate,
        abort_on_plateau=not args.no_plateau_abort,
        plateau_warmup_iterations=args.plateau_warmup_iterations,
        plateau_window_iterations=args.plateau_window_iterations,
        min_reach_improvement=args.min_reach_improvement,
    )
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(log_dir),
        device=agent_cfg.device,
    )
    from isaac_lab.training_defaults import default_max_train_duration_s  # noqa: WPS433

    max_duration_s = None
    if args.max_duration_minutes > 0.0:
        max_duration_s = default_max_train_duration_s(minutes=args.max_duration_minutes)

    def _curriculum_stage() -> str:
        task_env = resolve_task_env(runner)
        if task_env is None:
            return 'near_ee'
        metrics = task_env.get_task_metrics()
        return str(metrics.get('curriculum_stage', 'near_ee'))

    num_learning_iterations = planned_iterations if args.fixed_iterations else None
    reports, elapsed_s, stop_reason = run_training_with_reports(
        runner,
        num_learning_iterations=num_learning_iterations,
        criteria=criteria,
        init_at_random_ep_len=True,
        train_until_task_success=not args.fixed_iterations,
        max_duration_s=max_duration_s if not args.fixed_iterations else None,
        verbose=args.motion_glossary,
        verbose_curriculum_fn=_curriculum_stage if task_mode == 'reach' else None,
    )

    policy_path = checkpoint_dir / 'latest_policy'
    runner.save(str(policy_path))

    final_metrics = reports[-1].metrics if reports else None
    final_task = final_metrics.task_metrics if final_metrics else collect_task_metrics(runner)
    task_requirement_met = bool(reports and reports[-1].task_requirement_met)
    completion = TrainingCompletionReport(
        task=TASK_ID,
        completed_iterations=len(reports),
        total_execution_s=elapsed_s,
        final_mean_reward=final_metrics.mean_reward if final_metrics else None,
        task_metrics=final_task,
        target_reach_success_rate=criteria.target_reach_success_rate,
        target_push_success_rate=criteria.target_push_success_rate,
        target_contact_rate=criteria.target_contact_rate,
        task_mode=task_mode,
        checkpoint=str(policy_path),
        all_iterations_stable=all(report.stability_passed for report in reports),
        task_requirement_met=task_requirement_met,
        stop_reason=stop_reason,
        max_duration_minutes=args.max_duration_minutes if args.max_duration_minutes > 0 else None,
        planned_iterations=planned_iterations if args.fixed_iterations else None,
    )
    print(completion.format_with_demo_instructions())

    summary = {
        'task': TASK_ID,
        'num_envs': args.num_envs,
        'max_duration_minutes': args.max_duration_minutes,
        'from_scratch': args.from_scratch,
        'abort_on_plateau': criteria.abort_on_plateau,
        'plateau_window_iterations': criteria.plateau_window_iterations,
        'plateau_warmup_iterations': criteria.plateau_warmup_iterations,
        'min_reach_improvement': criteria.min_reach_improvement,
        'fixed_iterations': args.fixed_iterations,
        'max_iterations': planned_iterations,
        'motion_glossary': args.motion_glossary,
        'completed_iterations': len(reports),
        'stop_reason': stop_reason,
        'total_execution_s': elapsed_s,
        'final_mean_reward': final_metrics.mean_reward if final_metrics else None,
        'reach_success_rate': final_task.reach_success_rate,
        'target_reach_success_rate': criteria.target_reach_success_rate,
        'mean_time_to_reach_s': final_task.mean_time_to_reach_s,
        'target_reach_tolerance_m': final_task.target_reach_tolerance_m,
        'task_mode': task_mode,
        'contact_rate': final_task.contact_rate,
        'target_contact_rate': criteria.target_contact_rate,
        'push_success_rate': final_task.push_success_rate,
        'target_push_success_rate': criteria.target_push_success_rate,
        'mean_push_distance_m': final_task.mean_push_distance_m,
        'target_push_distance_m': final_task.target_push_distance_m,
        'task_requirement_met': task_requirement_met,
        'all_iterations_stable': completion.all_iterations_stable,
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
