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

"""Tutorial-style console output for Isaac Lab PPO training.

When ``--verbose`` is enabled (default), training prints a glossary that maps
MDP variables to physical motion on the MyCobot arm. This complements inline
source comments (see spec.md § tutorial-quality source code).
"""

from __future__ import annotations

from typing import Any

from isaac_lab.mdp_core import (
    ACTION_DIM,
    EE_REACH_TOLERANCE_M,
    END_EFFECTOR_BODY_NAME,
    MAX_BLOCK_REACH_M,
    MIN_BLOCK_REACH_M,
    OBSERVATION_DIM,
    REACH_CURRICULUM_STAGES,
    REVOLUTE_JOINT_NAMES,
    ReachTaskConfig,
)


def format_reach_motion_glossary(
    *,
    num_envs: int,
    max_duration_minutes: float,
    target_reach_success_rate: float,
    task_cfg: ReachTaskConfig | None = None,
    curriculum_stage: str | None = None,
) -> str:
    """Return a multi-line tutorial glossary for Phase 2 reach training."""

    cfg = task_cfg or ReachTaskConfig()
    stage_text = curriculum_stage or REACH_CURRICULUM_STAGES[0].name
    joint_list = ', '.join(REVOLUTE_JOINT_NAMES)
    lines = [
        '=' * 72,
        'MyCobot Phase 2 — RL reach tutorial (verbose console)',
        '=' * 72,
        '',
        'Goal: PPO learns IK-style reach (joint coordination) — no IK solver in the loop.',
        'Docs: spec.md § Phase 2 | isaac_lab/mdp_core.py | mycobot_reach_env.py',
        '',
        '--- Training budget ---',
        f'  Parallel arms (envs):     {num_envs}',
        f'  Duration limit:           {max_duration_minutes:.1f} min (--max-duration-minutes)',
        f'  Success stop target:      {target_reach_success_rate:.0%} rolling reach rate',
        f'  Active curriculum stage:  {stage_text}',
        '',
        '--- Observations (policy vector, '
        f'{OBSERVATION_DIM}-dim) → what the network sees each step ---',
        '  obs[0:3]  EE-to-target delta (m), normalized by max reach',
        '  obs[3]    target_valid (=1): known 3D target (no vision in Phase 2)',
        '  obs[4]    reached flag: 1 after EE enters success volume',
        f'  obs[5:11] joint positions (rad): {joint_list}',
        '',
        '--- Actions ('
        f'{ACTION_DIM}-dim joint space) → learned IK ---',
        f'  action[0:5]  Δq per revolute joint (scaled by {cfg.action_scale} rad per unit action).',
        '               The policy chooses how each joint moves each step; combined motion',
        f'               should drive {END_EFFECTOR_BODY_NAME} toward the target.',
        '               No analytic/differential IK solver is used (project requirement).',
        '  Joint limits: clamped to URDF soft limits after applying deltas.',
        '',
        '--- Rewards → what PPO optimizes ---',
        f'  Progress:  {cfg.progress_scale} × (d_prev − d_now) when moving closer.',
        f'  Time cost: −{cfg.time_penalty} each step (encourages efficient paths).',
        f'  Success:   +{cfg.reach_bonus} when distance ≤ '
        f'{EE_REACH_TOLERANCE_M * 1000:.0f} mm (early episode end).',
        f'  Timeout:   −{cfg.timeout_penalty} if episode ends without reach.',
        '',
        '--- Workspace & curriculum ---',
        f'  Reach annulus: {MIN_BLOCK_REACH_M:.2f}–{MAX_BLOCK_REACH_M:.2f} m horizontal radius.',
        '  Stages (advance when rolling success exceeds threshold):',
    ]
    for stage in REACH_CURRICULUM_STAGES:
        lines.append(
            f'    • {stage.name}: easy_fraction={stage.easy_fraction:.2f}, '
            f'near_ee_radius={stage.near_ee_radius_m:.2f} m, '
            f'advance≥{stage.min_success_rate_to_advance:.0%}'
        )
    lines.extend(
        [
            '',
            '--- Host execution ---',
            '  Training runs on the DGX Spark host (Isaac Sim). From the container,',
            '  scripts auto-delegate via nsenter — see spec.md § Host vs container.',
            '',
            '--- External references ---',
            '  Isaac Lab:  https://isaac-sim.github.io/IsaacLab/',
            '  RSL-RL PPO: https://github.com/leggedrobotics/rsl_rl',
            '=' * 72,
        ]
    )
    return '\n'.join(lines)


def format_iteration_motion_snapshot(
    *,
    iteration: int,
    task_metrics: Any,
    curriculum_stage: str,
) -> str:
    """Short per-iteration motion summary appended when verbose is on."""

    reach_rate = float(getattr(task_metrics, 'reach_success_rate', 0.0))
    mean_time = float(getattr(task_metrics, 'mean_time_to_reach_s', 0.0))
    return (
        f'[verbose] iter {iteration}: curriculum={curriculum_stage}, '
        f'reach={reach_rate:.1%}, mean_time_to_reach={mean_time:.2f}s '
        f'(lower time ⇒ more efficient learned motion)'
    )


def print_reach_training_guide(**kwargs: Any) -> None:
    """Print the reach motion glossary to stdout."""

    print(format_reach_motion_glossary(**kwargs))
