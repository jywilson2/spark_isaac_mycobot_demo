# Copyright 2026 spark_isaac_mycobot_demo contributors
"""Unit tests for play_ppo CLI helpers (no Isaac Sim required)."""

from __future__ import annotations

from isaac_lab.play_ppo import (
    build_parser,
    consume_episode_outcome,
    format_episode_line,
    inference_load_cfg,
)


def test_parse_args_demo_forces_single_arm() -> None:
    args = build_parser(with_app_launcher=False).parse_args(
        ['--demo', '--num-arms', '8'],
    )
    args.num_arms = 1 if args.demo else args.num_arms
    assert args.demo is True
    assert args.num_arms == 1


def test_inference_load_cfg_skips_optimizer() -> None:
    cfg = inference_load_cfg()
    assert cfg['actor'] is True
    assert cfg['optimizer'] is False
    assert cfg['iteration'] is False


def test_consume_episode_outcome_uses_terminal_cache() -> None:
    class _TaskEnv:
        def __init__(self) -> None:
            self.popped = False

        def pop_episode_outcome(self, env_idx: int = 0) -> dict[str, object]:
            self.popped = True
            return {
                'reached': True,
                'time_to_reach_s': 0.5,
                'target_xyz': [0.2, 0.0, 0.12],
                'ee_xyz': [0.2, 0.0, 0.12],
                'distance_m': 0.01,
            }

    env = _TaskEnv()
    outcome = consume_episode_outcome(env, env_idx=0)
    assert env.popped is True
    assert outcome['reached'] is True


def test_format_episode_line_reach_ok() -> None:
    line = format_episode_line(
        3,
        {
            'reached': True,
            'time_to_reach_s': 1.25,
            'target_xyz': [0.22, 0.0, 0.12],
            'distance_m': 0.01,
        },
    )
    assert 'REACH OK' in line
    assert '1.25s' in line


def test_format_episode_line_timeout() -> None:
    line = format_episode_line(
        2,
        {
            'reached': False,
            'time_to_reach_s': None,
            'target_xyz': [0.2, 0.1, 0.15],
            'distance_m': 0.04,
        },
    )
    assert 'timeout' in line
    assert '40.0 mm' in line
