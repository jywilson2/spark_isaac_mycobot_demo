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

"""Reproducible multi-stage Phase 2 training recipe (spec.md).

Single source of truth for ``scripts/run_staged_training.sh``. Each stage resumes
from the prior checkpoint so a from-scratch run is fully scripted.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaac_lab.mdp_core import EE_REACH_TOLERANCE_COARSE_M, EE_REACH_TOLERANCE_M
from isaac_lab.training_defaults import DEFAULT_EPISODE_LENGTH_S


@dataclass(frozen=True)
class TrainingStage:
    """One row in the scripted from-scratch training pipeline."""

    name: str
    minutes: float
    target_reach_success_rate: float
    target_sampling: str
    reach_tolerance_m: float
    from_scratch: bool = False
    resume: bool = True
    no_early_success_stop: bool = False
    direct_path_shaping: bool = False
    action_scale: float | None = None
    episode_length_s: float = DEFAULT_EPISODE_LENGTH_S


# Progressive: coarse IK → demo distribution → 5 mm direct approach → 1 mm final.
STAGED_TRAINING_PHASES: tuple[TrainingStage, ...] = (
    TrainingStage(
        name='curriculum_coarse',
        minutes=60.0,
        target_reach_success_rate=0.90,
        target_sampling='curriculum',
        reach_tolerance_m=EE_REACH_TOLERANCE_COARSE_M,
        from_scratch=True,
        resume=False,
    ),
    TrainingStage(
        name='demo_coarse',
        minutes=30.0,
        target_reach_success_rate=0.92,
        target_sampling='demo',
        reach_tolerance_m=EE_REACH_TOLERANCE_COARSE_M,
        no_early_success_stop=True,
    ),
    TrainingStage(
        name='precision_5mm',
        minutes=45.0,
        target_reach_success_rate=0.88,
        target_sampling='precision',
        reach_tolerance_m=0.005,
        direct_path_shaping=True,
        action_scale=0.08,
        no_early_success_stop=True,
    ),
    TrainingStage(
        name='precision_1mm',
        minutes=60.0,
        target_reach_success_rate=0.85,
        target_sampling='precision',
        reach_tolerance_m=EE_REACH_TOLERANCE_M,
        direct_path_shaping=True,
        action_scale=0.05,
        no_early_success_stop=True,
    ),
)

# Demo verify uses the final 1 mm tolerance (spec.md).
STAGED_VERIFY_REACH_TOLERANCE_M = EE_REACH_TOLERANCE_M
