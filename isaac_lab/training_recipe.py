# Copyright 2026 spark_isaac_mycobot_demo contributors
"""Reproducible multi-stage Phase 2 training recipe (spec.md)."""

from __future__ import annotations

from dataclasses import dataclass

from isaac_lab.mdp_core import EE_REACH_TOLERANCE_COARSE_M, EE_REACH_TOLERANCE_M
from isaac_lab.training_defaults import DEFAULT_EPISODE_LENGTH_S

PRECISION_TIER_BONUSES: tuple[tuple[float, float], ...] = (
    (0.010, 8.0),
    (0.005, 18.0),
    (0.003, 30.0),
    (0.001, 50.0),
)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    minutes: float
    target_reach_success_rate: float
    target_sampling: str
    reach_tolerance_m: float
    from_scratch: bool = False
    resume: bool = True
    no_early_success_stop: bool = False
    direct_path_shaping: bool = False
    enable_precision_tiers: bool = False
    action_scale: float | None = None
    episode_length_s: float = DEFAULT_EPISODE_LENGTH_S


STAGED_TRAINING_PHASES: tuple[TrainingStage, ...] = (
    TrainingStage(
        name='curriculum_coarse',
        minutes=90.0,
        target_reach_success_rate=0.95,
        target_sampling='curriculum',
        reach_tolerance_m=EE_REACH_TOLERANCE_COARSE_M,
        from_scratch=True,
        resume=False,
    ),
    TrainingStage(
        name='demo_coarse',
        minutes=30.0,
        target_reach_success_rate=0.95,
        target_sampling='demo',
        reach_tolerance_m=EE_REACH_TOLERANCE_COARSE_M,
        no_early_success_stop=True,
    ),
    TrainingStage(
        name='precision_20mm',
        minutes=45.0,
        target_reach_success_rate=0.85,
        target_sampling='precision',
        reach_tolerance_m=0.020,
        enable_precision_tiers=True,
        action_scale=0.10,
        no_early_success_stop=True,
    ),
    TrainingStage(
        name='precision_15mm',
        minutes=45.0,
        target_reach_success_rate=0.85,
        target_sampling='precision',
        reach_tolerance_m=0.015,
        enable_precision_tiers=True,
        action_scale=0.09,
        no_early_success_stop=True,
    ),
    TrainingStage(
        name='precision_12mm',
        minutes=45.0,
        target_reach_success_rate=0.85,
        target_sampling='precision',
        reach_tolerance_m=0.012,
        enable_precision_tiers=True,
        action_scale=0.085,
        no_early_success_stop=True,
    ),
    TrainingStage(
        name='precision_6mm',
        minutes=45.0,
        target_reach_success_rate=0.85,
        target_sampling='precision',
        reach_tolerance_m=0.006,
        enable_precision_tiers=True,
        direct_path_shaping=True,
        action_scale=0.065,
        no_early_success_stop=True,
    ),
    TrainingStage(
        name='precision_3mm',
        minutes=45.0,
        target_reach_success_rate=0.85,
        target_sampling='precision',
        reach_tolerance_m=0.003,
        enable_precision_tiers=True,
        direct_path_shaping=True,
        action_scale=0.055,
        no_early_success_stop=True,
    ),
    TrainingStage(
        name='precision_1mm',
        minutes=60.0,
        target_reach_success_rate=0.85,
        target_sampling='precision',
        reach_tolerance_m=EE_REACH_TOLERANCE_M,
        enable_precision_tiers=True,
        direct_path_shaping=True,
        action_scale=0.05,
        no_early_success_stop=True,
    ),
)

STAGED_VERIFY_REACH_TOLERANCE_M = EE_REACH_TOLERANCE_M
