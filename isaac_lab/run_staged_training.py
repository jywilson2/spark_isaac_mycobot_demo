#!/usr/bin/env python3
# Copyright 2026 spark_isaac_mycobot_demo contributors
"""Execute the scripted multi-stage training recipe (spec.md)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_lab.training_recipe import STAGED_TRAINING_PHASES  # noqa: E402


def build_train_argv(stage_idx: int, *, headless: bool, extra: list[str]) -> list[str]:
    stage = STAGED_TRAINING_PHASES[stage_idx]
    train_sh = REPO_ROOT / 'scripts' / 'host' / 'run_isaac_lab_training.sh'
    argv = [str(train_sh), 'train']
    if headless:
        argv.append('--headless')
    argv.extend(extra)
    if stage.from_scratch:
        argv.append('--from-scratch')
    elif stage.resume:
        argv.append('--resume')
    argv.extend([
        '--max-duration-minutes', str(stage.minutes),
        '--target-reach-success-rate', str(stage.target_reach_success_rate),
        '--target-sampling', stage.target_sampling,
        '--episode-length-s', str(stage.episode_length_s),
        '--reach-tolerance-m', str(stage.reach_tolerance_m),
    ])
    if stage.no_early_success_stop:
        argv.append('--no-early-success-stop')
    if stage.direct_path_shaping:
        argv.append('--direct-path-shaping')
    if stage.enable_precision_tiers:
        argv.append('--enable-precision-tiers')
    if stage.action_scale is not None:
        argv.extend(['--action-scale', str(stage.action_scale)])
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description='Run staged Phase 2 training from scratch')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--skip-verify', action='store_true')
    parser.add_argument('--start-stage', type=int, default=0)
    parser.add_argument('extra_train_args', nargs='*', help='Extra args forwarded to train')
    args = parser.parse_args()

    total = len(STAGED_TRAINING_PHASES)
    print('=== Staged Phase 2 training ===')
    for idx, stage in enumerate(STAGED_TRAINING_PHASES, start=1):
        print(
            f'  Stage {idx}: {stage.name} | {stage.minutes:.0f} min | '
            f'{stage.target_sampling} | {stage.reach_tolerance_m * 1000:.1f} mm | '
            f'target {stage.target_reach_success_rate:.0%}',
        )
    print()

    for idx in range(args.start_stage, total):
        stage = STAGED_TRAINING_PHASES[idx]
        print(f'--- Stage {idx + 1}/{total}: {stage.name} ---')
        cmd = build_train_argv(idx, headless=args.headless, extra=args.extra_train_args)
        print('Command:', ' '.join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f'Stage {stage.name} failed with exit {result.returncode}', file=sys.stderr)
            return result.returncode
        print()

    if args.skip_verify:
        print('Skipping demo verification (--skip-verify).')
        return 0

    verify_sh = REPO_ROOT / 'scripts' / 'verify_demo_policy.sh'
    verify_cmd = [str(verify_sh), '--reach-tolerance-m', '0.001']
    if args.headless:
        verify_cmd.append('--headless')
    print('--- Demo verification ---')
    return subprocess.run(verify_cmd, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
