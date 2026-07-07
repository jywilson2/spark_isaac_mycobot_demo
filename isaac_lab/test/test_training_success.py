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
