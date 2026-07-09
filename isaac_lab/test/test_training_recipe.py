# Copyright 2026 spark_isaac_mycobot_demo contributors
"""Unit tests for scripted multi-stage training recipe."""

from __future__ import annotations

from isaac_lab.mdp_core import EE_REACH_TOLERANCE_M
from isaac_lab.training_recipe import STAGED_TRAINING_PHASES, STAGED_VERIFY_REACH_TOLERANCE_M


def test_staged_training_has_eight_progressive_phases() -> None:
    assert len(STAGED_TRAINING_PHASES) == 8
    assert STAGED_TRAINING_PHASES[0].from_scratch is True
    tolerances = [s.reach_tolerance_m for s in STAGED_TRAINING_PHASES]
    assert tolerances == sorted(tolerances, reverse=True)
    assert STAGED_TRAINING_PHASES[-1].reach_tolerance_m == EE_REACH_TOLERANCE_M
    assert STAGED_TRAINING_PHASES[-1].direct_path_shaping is True


def test_final_verify_uses_one_mm_tolerance() -> None:
    assert STAGED_VERIFY_REACH_TOLERANCE_M == 0.001
