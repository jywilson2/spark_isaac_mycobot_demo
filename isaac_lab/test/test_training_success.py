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

"""Unit tests for training success criteria reporting."""

from __future__ import annotations

from isaac_lab.training_success import (
    IterationMetrics,
    TaskMetrics,
    TrainingCompletionReport,
    TrainingSuccessCriteria,
    compute_max_episode_steps,
    evaluate_iteration,
)


def test_compute_max_episode_steps() -> None:
    assert compute_max_episode_steps() == 360


def test_iteration_report_marks_reach_requirement() -> None:
    criteria = TrainingSuccessCriteria(
        task_mode='reach',
        target_reach_success_rate=0.99,
    )
    metrics = IterationMetrics(
        iteration=1,
        total_iterations=50,
        mean_reward=2.0,
        mean_episode_length=120.0,
        value_loss=117.0,
        surrogate_loss=0.01,
        entropy_loss=8.5,
        action_std=1.0,
        task_metrics=TaskMetrics(
            reach_success_rate=0.85,
            mean_time_to_reach_s=2.5,
        ),
    )
    report = evaluate_iteration(metrics, criteria)
    assert report.stability_passed
    assert not report.task_requirement_met
    assert 'reach_success_rate' in report.format()


def test_iteration_report_marks_contact_push_requirement() -> None:
    criteria = TrainingSuccessCriteria(
        task_mode='red_block',
        target_contact_rate=0.70,
        target_push_success_rate=0.50,
    )
    metrics = IterationMetrics(
        iteration=1,
        total_iterations=50,
        mean_reward=2.0,
        mean_episode_length=120.0,
        value_loss=117.0,
        surrogate_loss=0.01,
        entropy_loss=8.5,
        action_std=1.0,
        task_metrics=TaskMetrics(
            contact_rate=0.80,
            push_success_rate=0.25,
            mean_push_distance_m=0.002,
        ),
    )
    report = evaluate_iteration(metrics, criteria)
    assert report.stability_passed
    assert not report.task_requirement_met
    text = report.format()
    assert 'push_success_rate' in text
    assert 'NOT YET ACHIEVED' in text


def test_training_completion_report_formats_duration() -> None:
    report = TrainingCompletionReport(
        task='Spark-MyCobot-PickPlace-Direct-v0',
        completed_iterations=12,
        total_execution_s=95.0,
        final_mean_reward=2.5,
        task_metrics=TaskMetrics(contact_rate=0.8, push_success_rate=0.6),
        target_push_success_rate=0.5,
        target_contact_rate=0.7,
        checkpoint='/tmp/latest_policy',
        all_iterations_stable=True,
        task_requirement_met=True,
        task_mode='reach',
        stop_reason='task_requirement_met',
        max_duration_minutes=30.0,
    )
    text = report.format()
    assert 'Total execution time: 00:01:35' in text
    assert 'task_requirement_met' in text or 'Task requirement met: YES' in text


def test_reach_improvement_tracker_detects_plateau() -> None:
    from isaac_lab.training_success import _ReachImprovementTracker

    tracker = _ReachImprovementTracker(
        warmup_iterations=5,
        plateau_window_iterations=10,
        min_improvement=0.02,
    )
    for it in range(1, 5):
        assert not tracker.update(iteration=it, reach_success_rate=0.50)
    assert not tracker.update(iteration=5, reach_success_rate=0.52)
    for it in range(6, 16):
        assert not tracker.update(iteration=it, reach_success_rate=0.525)
    assert tracker.update(iteration=16, reach_success_rate=0.525) is True


def test_reach_improvement_tracker_gated_below_min_reach() -> None:
    from isaac_lab.training_success import _ReachImprovementTracker

    tracker = _ReachImprovementTracker(
        warmup_iterations=5,
        plateau_window_iterations=10,
        min_improvement=0.01,
        min_reach_to_abort=0.50,
    )
    # Stalls at 5% reach for far longer than the plateau window — the gate
    # must keep training alive because best reach is below the 50% floor.
    for it in range(1, 200):
        assert not tracker.update(iteration=it, reach_success_rate=0.05)

    # Once reach crosses the floor and then stalls, the abort fires again.
    assert not tracker.update(iteration=200, reach_success_rate=0.55)
    for it in range(201, 210):
        assert not tracker.update(iteration=it, reach_success_rate=0.55)
    assert tracker.update(iteration=210, reach_success_rate=0.55) is True


def test_reach_improvement_tracker_resets_on_curriculum_stage_change() -> None:
    """Advancing to a harder stage lowers success by design — not a plateau."""

    from isaac_lab.training_success import _ReachImprovementTracker

    tracker = _ReachImprovementTracker(
        warmup_iterations=5,
        plateau_window_iterations=10,
        min_improvement=0.01,
        min_reach_to_abort=0.50,
    )
    # Learn near_ee up to 92% best.
    assert not tracker.update(iteration=10, reach_success_rate=0.92, curriculum_stage='near_ee')
    # Curriculum advances; measured success drops to 65%. Best must reset so the
    # unbeatable easy-stage 92% cannot trigger the abort on the harder stage.
    assert not tracker.update(iteration=11, reach_success_rate=0.65, curriculum_stage='medium')
    assert tracker.best_reach == 0.65
    # Stalling at 65% within the window keeps training alive...
    for it in range(12, 21):
        assert not tracker.update(
            iteration=it, reach_success_rate=0.65, curriculum_stage='medium',
        )
    # ...but a genuine plateau on the same stage still aborts once the window elapses.
    assert tracker.update(iteration=21, reach_success_rate=0.65, curriculum_stage='medium')


def test_reach_improvement_tracker_first_stage_does_not_reset() -> None:
    from isaac_lab.training_success import _ReachImprovementTracker

    tracker = _ReachImprovementTracker(
        warmup_iterations=0,
        plateau_window_iterations=10,
        min_improvement=0.01,
    )
    # First observed stage initializes tracking without wiping progress.
    assert not tracker.update(iteration=1, reach_success_rate=0.30, curriculum_stage='near_ee')
    assert tracker.best_reach == 0.30
    assert tracker.current_stage == 'near_ee'


def test_reach_improvement_tracker_resets_on_gain() -> None:
    from isaac_lab.training_success import _ReachImprovementTracker

    tracker = _ReachImprovementTracker(
        warmup_iterations=1,
        plateau_window_iterations=5,
        min_improvement=0.01,
    )
    tracker.update(iteration=1, reach_success_rate=0.60)
    for it in range(2, 6):
        tracker.update(iteration=it, reach_success_rate=0.605)
    assert tracker.update(iteration=6, reach_success_rate=0.605) is True
    assert not tracker.update(iteration=7, reach_success_rate=0.62)
    assert tracker.last_improvement_iteration == 7
