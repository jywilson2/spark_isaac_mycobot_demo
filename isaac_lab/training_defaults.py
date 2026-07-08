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

"""Default training parameters tuned for DGX Spark (Isaac Sim host)."""

from __future__ import annotations

# GUI + Isaac Sim viewport + EE cameras: keep parallel arms low to avoid OOM kills.
DEFAULT_NUM_ARMS_GUI = 2

# Headless integration / throughput training on DGX Spark (GB10, unified memory).
DEFAULT_NUM_ARMS_HEADLESS = 8

# Backward-compatible alias used by headless integration gates.
DEFAULT_NUM_ARMS = DEFAULT_NUM_ARMS_HEADLESS

# Headless PPO integration train duration (verify_isaac_lab.sh --smoke-train).
DEFAULT_MAX_TRAIN_DURATION_MINUTES = 30.0
INTEGRATION_TRAIN_MAX_DURATION_MINUTES = DEFAULT_MAX_TRAIN_DURATION_MINUTES

# Fast pytest gate: short headless slice proving time-limit stop (not the full 30 min run).
INTEGRATION_PYTEST_MAX_DURATION_MINUTES = 1.0

# Legacy iteration cap retained for --fixed-iterations smoke only.
INTEGRATION_TRAIN_MAX_ITERATIONS = 2

# PPO learns in chunks; the outer loop runs until duration or task success.
DEFAULT_TRAINING_ITERATION_CHUNK = 100

# Training and demo/play share the same episode horizon (spec.md § smooth motion).
DEFAULT_EPISODE_LENGTH_S = 30.0

# Two-phase training recipe (scripts/run_two_phase_training.sh).
TWO_PHASE_CURRICULUM_MINUTES = 90.0
TWO_PHASE_DEMO_MINUTES = 30.0
TWO_PHASE_TARGET_REACH_SUCCESS_RATE = 0.95
DEMO_VERIFY_EPISODE_LENGTHS_S: tuple[float, ...] = (20.0, 30.0, 40.0)
DEMO_VERIFY_MIN_SUCCESS_RATE = 0.95
DEMO_VERIFY_EPISODES = 100

# Plateau abort is opt-in; duration budget is the primary stop (spec.md).
DEFAULT_ABORT_ON_PLATEAU = False
DEFAULT_PLATEAU_WARMUP_ITERATIONS = 40
DEFAULT_PLATEAU_WINDOW_ITERATIONS = 120
DEFAULT_MIN_REACH_IMPROVEMENT = 0.01
# Plateau abort is gated: it cannot fire until best reach crosses this floor,
# so slow-starting runs keep their full --max-duration-minutes budget.
DEFAULT_PLATEAU_MIN_REACH = 0.50


def default_max_train_duration_s(*, minutes: float | None = None) -> float:
    """Return training time budget in seconds."""

    return (minutes if minutes is not None else DEFAULT_MAX_TRAIN_DURATION_MINUTES) * 60.0


def default_num_arms(*, headless: bool) -> int:
    """Return the default parallel arm count for the selected runtime mode."""

    return DEFAULT_NUM_ARMS_HEADLESS if headless else DEFAULT_NUM_ARMS_GUI


def format_policy_demo_instructions(*, checkpoint: str | None = None) -> str:
    """User-facing steps to run the trained policy in the Isaac Sim GUI."""

    ckpt = checkpoint or 'assets/checkpoints/isaac_lab_ppo/latest_policy'
    return '\n'.join(
        [
            '=== Run trained policy (EE reach demo) ===',
            'After training, demonstrate reach-to-target in the Isaac Sim GUI:',
            '',
            '  ./scripts/host/run_isaac_lab_training.sh demo',
            '',
            'Continuous showcase: one arm, red target sphere respawns after each reach.',
            'Runs until you close Isaac Sim or press Ctrl+C.',
            '',
            'Finite regression play:',
            f'  ./scripts/host/run_isaac_lab_training.sh play --checkpoint {ckpt} --episodes 10',
            '',
            'Ongoing use: re-run the play command after retraining; update --checkpoint',
            'when saving to a new directory. See README.md § Ongoing use of trained policies.',
            '=' * 72,
        ]
    )
