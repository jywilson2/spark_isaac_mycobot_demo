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

"""Training iteration success criteria and user-facing progress reports."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from isaac_lab.mdp_core import (
    DEFAULT_TARGET_REACH_SUCCESS_RATE,
    EE_REACH_TOLERANCE_M,
    TARGET_PUSH_DISTANCE_M,
)


def compute_max_episode_steps(
    *,
    episode_length_s: float = 12.0,
    sim_dt: float = 1.0 / 60.0,
    decimation: int = 2,
) -> int:
    step_dt = sim_dt * decimation
    return max(1, int(round(episode_length_s / step_dt)))


def compute_max_episode_return(
    *,
    max_step_reward: float = 4.0,
    max_episode_steps: int | None = None,
    episode_length_s: float = 12.0,
    sim_dt: float = 1.0 / 60.0,
    decimation: int = 2,
) -> float:
    steps = max_episode_steps or compute_max_episode_steps(
        episode_length_s=episode_length_s,
        sim_dt=sim_dt,
        decimation=decimation,
    )
    return max_step_reward * float(steps)


@dataclass(frozen=True)
class TrainingSuccessCriteria:
    """Per-iteration gates and task targets."""

    task_mode: str = 'reach'  # 'reach' (Phase 2) or 'red_block' (Phase 6)

    max_step_reward: float = 20.0
    max_episode_steps: int = compute_max_episode_steps()
    max_episode_return: float = field(
        default_factory=lambda: compute_max_episode_return(),
    )

    min_mean_reward: float = 0.0
    min_mean_episode_length: float = 8.0
    max_value_loss: float = 500.0
    max_surrogate_loss: float = 1.0
    min_entropy_loss: float = -50.0
    max_entropy_loss: float = 50.0
    min_action_std: float = 0.01
    max_action_std: float = 5.0

    target_reach_success_rate: float = DEFAULT_TARGET_REACH_SUCCESS_RATE
    target_reach_tolerance_m: float = EE_REACH_TOLERANCE_M
    target_push_success_rate: float = 0.50
    target_contact_rate: float = 0.70
    target_push_distance_m: float = TARGET_PUSH_DISTANCE_M
    min_eval_episodes: int = 8


@dataclass(frozen=True)
class TaskMetrics:
    reach_success_rate: float = 0.0
    mean_time_to_reach_s: float = 0.0
    target_reach_tolerance_m: float = EE_REACH_TOLERANCE_M
    contact_rate: float = 0.0
    push_success_rate: float = 0.0
    mean_push_distance_m: float = 0.0
    target_push_distance_m: float = TARGET_PUSH_DISTANCE_M


@dataclass(frozen=True)
class CriterionResult:
    name: str
    actual: float
    target: float
    comparator: str
    passed: bool
    unit: str = ''

    @property
    def status(self) -> str:
        return 'PASS' if self.passed else 'FAIL'

    @property
    def gap(self) -> float:
        if self.comparator in ('>=', '>'):
            return self.target - self.actual
        return self.actual - self.target


@dataclass(frozen=True)
class IterationMetrics:
    iteration: int
    total_iterations: int
    mean_reward: float
    mean_episode_length: float
    value_loss: float
    surrogate_loss: float
    entropy_loss: float
    action_std: float
    task_metrics: TaskMetrics
    metrics_source: str = 'episode'
    mean_step_reward: float | None = None
    completed_episodes: int = 0


@dataclass(frozen=True)
class IterationReport:
    metrics: IterationMetrics
    stability_results: tuple[CriterionResult, ...]
    progress_results: tuple[CriterionResult, ...]
    task_mode: str = 'reach'

    @property
    def stability_passed(self) -> bool:
        return all(result.passed for result in self.stability_results)

    @property
    def progress_passed(self) -> bool:
        return all(result.passed for result in self.progress_results)

    @property
    def task_requirement_met(self) -> bool:
        return self.progress_passed

    def format(self) -> str:
        task = self.metrics.task_metrics
        lines = [
            f'=== Iteration {self.metrics.iteration}/{self.metrics.total_iterations} success criteria ===',
            '',
            'Stability (required each iteration):',
        ]
        for result in self.stability_results:
            lines.append(self._format_criterion(result))
        task_label = (
            'EE reach-to-target (Phase 2)'
            if self.task_mode == 'reach'
            else 'Red-block contact-and-push (Phase 6)'
        )
        lines.extend(['', f'{task_label}:'])
        for result in self.progress_results:
            lines.append(self._format_progress(result))
        if self.task_mode == 'reach':
            lines.extend(
                [
                    '',
                    f'Metrics source: {self.metrics.metrics_source}',
                    f'  reach_success_rate={task.reach_success_rate:.1%}, '
                    f'mean_time_to_reach={task.mean_time_to_reach_s:.2f}s '
                    f'(tolerance {task.target_reach_tolerance_m * 1000.0:.1f} mm)',
                ]
            )
        else:
            lines.extend(
                [
                    '',
                    f'Metrics source: {self.metrics.metrics_source}',
                    f'  contact_rate={task.contact_rate:.1%}, '
                    f'push_success_rate={task.push_success_rate:.1%}, '
                    f'mean_push={task.mean_push_distance_m * 1000.0:.2f} mm '
                    f'(target {task.target_push_distance_m * 1000.0:.1f} mm)',
                ]
            )
        if self.metrics.metrics_source == 'rollout':
            lines.append(
                f'  (No completed episodes yet; using rollout mean step reward '
                f'{self.metrics.mean_step_reward:.4f} scaled to {self.metrics.mean_reward:.2f})'
            )
        lines.extend(
            [
                '',
                f'Iteration stability: {"PASS" if self.stability_passed else "FAIL"}',
                f'Task requirement: {"ACHIEVED" if self.task_requirement_met else "NOT YET ACHIEVED"}',
                '=' * 72,
            ]
        )
        return '\n'.join(lines)

    @staticmethod
    def _format_criterion(result: CriterionResult) -> str:
        return (
            f'  [{result.status}] {result.name}: '
            f'actual={result.actual:.4f} {result.comparator} target={result.target:.4f} '
            f'{result.unit}'.rstrip()
        )

    @staticmethod
    def _format_progress(result: CriterionResult) -> str:
        if result.target <= 0.0:
            pct = 0.0
        else:
            pct = min(100.0, max(0.0, 100.0 * result.actual / result.target))
        gap_text = 'at target' if result.passed else f'gap={abs(result.gap):.4f}'
        unit = result.unit or ''
        return (
            f'  [{result.status}] {result.name}: '
            f'actual={result.actual:.2f}{unit} / target={result.target:.2f}{unit} '
            f'({pct:.1f}%, {gap_text})'
        )


@dataclass(frozen=True)
class TrainingCompletionReport:
    task: str
    completed_iterations: int
    total_execution_s: float
    final_mean_reward: float | None
    task_metrics: TaskMetrics
    checkpoint: str
    all_iterations_stable: bool
    task_requirement_met: bool
    stop_reason: str
    max_duration_minutes: float | None = None
    target_reach_success_rate: float = DEFAULT_TARGET_REACH_SUCCESS_RATE
    target_push_success_rate: float = 0.50
    target_contact_rate: float = 0.70
    task_mode: str = 'reach'
    planned_iterations: int | None = None

    def format(self) -> str:
        reward_text = f'{self.final_mean_reward:.2f}' if self.final_mean_reward is not None else 'n/a'
        task = self.task_metrics
        if self.task_mode == 'reach':
            iter_text = (
                f'Iterations completed: {self.completed_iterations}'
                if self.planned_iterations is None
                else f'Iterations completed: {self.completed_iterations}/{self.planned_iterations}'
            )
            duration_limit = (
                f'{self.max_duration_minutes:.1f} min'
                if self.max_duration_minutes is not None
                else 'none'
            )
            lines = [
                '=== Training sequence complete ===',
                f'Task: {self.task}',
                f'Stop reason: {self.stop_reason}',
                iter_text,
                f'Duration limit: {duration_limit}',
                f'Total execution time: {self._format_duration(self.total_execution_s)} '
                f'({self.total_execution_s:.2f}s)',
                f'Final mean reward: {reward_text}',
                f'Reach success rate: {task.reach_success_rate:.1%} '
                f'(target {self.target_reach_success_rate:.1%})',
                f'Mean time to reach: {task.mean_time_to_reach_s:.2f}s',
                f'Task requirement met: {"YES" if self.task_requirement_met else "NO"}',
                f'All iterations stable: {"YES" if self.all_iterations_stable else "NO"}',
                f'Checkpoint: {self.checkpoint}',
                '=' * 72,
            ]
            return '\n'.join(lines)
        return '\n'.join(
            [
                '=== Training sequence complete ===',
                f'Task: {self.task}',
                f'Stop reason: {self.stop_reason}',
                (
                    f'Iterations completed: {self.completed_iterations}'
                    if self.planned_iterations is None
                    else f'Iterations completed: {self.completed_iterations}/{self.planned_iterations}'
                ),
                f'Total execution time: {self._format_duration(self.total_execution_s)} '
                f'({self.total_execution_s:.2f}s)',
                f'Final mean reward: {reward_text}',
                f'Contact rate: {task.contact_rate:.1%} (target {self.target_contact_rate:.1%})',
                f'Push success rate: {task.push_success_rate:.1%} '
                f'(target {self.target_push_success_rate:.1%})',
                f'Mean push distance: {task.mean_push_distance_m * 1000.0:.2f} mm '
                f'(target {task.target_push_distance_m * 1000.0:.1f} mm)',
                f'Task requirement met: {"YES" if self.task_requirement_met else "NO"}',
                f'All iterations stable: {"YES" if self.all_iterations_stable else "NO"}',
                f'Checkpoint: {self.checkpoint}',
                '=' * 72,
            ]
        )

    def format_with_demo_instructions(self) -> str:
        from isaac_lab.training_defaults import format_policy_demo_instructions  # noqa: WPS433

        return self.format() + '\n' + format_policy_demo_instructions(checkpoint=self.checkpoint)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        whole = int(seconds)
        hours, remainder = divmod(whole, 3600)
        minutes, secs = divmod(remainder, 60)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def resolve_task_env(runner: Any) -> Any | None:
    env = getattr(runner, 'env', None)
    while env is not None:
        if hasattr(env, 'get_task_metrics'):
            return env
        env = getattr(env, 'env', None)
    return None


def collect_task_metrics(runner: Any) -> TaskMetrics:
    task_env = resolve_task_env(runner)
    if task_env is None:
        return TaskMetrics()
    raw = task_env.get_task_metrics()
    return TaskMetrics(
        reach_success_rate=float(raw.get('reach_success_rate', 0.0)),
        mean_time_to_reach_s=float(raw.get('mean_time_to_reach_s', 0.0)),
        target_reach_tolerance_m=float(
            raw.get('target_reach_tolerance_m', EE_REACH_TOLERANCE_M)),
        contact_rate=float(raw.get('contact_rate', 0.0)),
        push_success_rate=float(raw.get('push_success_rate', 0.0)),
        mean_push_distance_m=float(raw.get('mean_push_distance_m', 0.0)),
        target_push_distance_m=float(
            raw.get('target_push_distance_m', TARGET_PUSH_DISTANCE_M)),
    )


def evaluate_iteration(
    metrics: IterationMetrics,
    criteria: TrainingSuccessCriteria,
) -> IterationReport:
    stability = (
        _check(metrics.mean_reward, criteria.min_mean_reward, '>=', 'mean_reward'),
        _check(
            metrics.mean_episode_length,
            criteria.min_mean_episode_length,
            '>=',
            'mean_episode_length',
        ),
        _check(metrics.value_loss, criteria.max_value_loss, '<=', 'value_loss'),
        _check(metrics.surrogate_loss, criteria.max_surrogate_loss, '<=', 'surrogate_loss'),
        _check(metrics.entropy_loss, criteria.min_entropy_loss, '>=', 'entropy_loss'),
        _check(metrics.entropy_loss, criteria.max_entropy_loss, '<=', 'entropy_loss (max)'),
        _check(metrics.action_std, criteria.min_action_std, '>=', 'action_std (min)'),
        _check(metrics.action_std, criteria.max_action_std, '<=', 'action_std (max)'),
    )
    task = metrics.task_metrics
    if criteria.task_mode == 'reach':
        progress = (
            _check(
                task.reach_success_rate,
                criteria.target_reach_success_rate,
                '>=',
                'reach_success_rate',
                unit='',
            ),
        )
    else:
        progress = (
            _check(
                task.contact_rate,
                criteria.target_contact_rate,
                '>=',
                'contact_rate',
                unit='',
            ),
            _check(
                task.push_success_rate,
                criteria.target_push_success_rate,
                '>=',
                'push_success_rate (5 mm after contact)',
                unit='',
            ),
        )
    return IterationReport(
        metrics=metrics,
        stability_results=stability,
        progress_results=progress,
        task_mode=criteria.task_mode,
    )


def _check(
    actual: float,
    target: float,
    comparator: str,
    name: str,
    unit: str = '',
) -> CriterionResult:
    if comparator == '>=':
        passed = actual >= target
    elif comparator == '>':
        passed = actual > target
    elif comparator == '<=':
        passed = actual <= target
    else:
        raise ValueError(f'Unsupported comparator: {comparator}')
    return CriterionResult(
        name=name,
        actual=actual,
        target=target,
        comparator=comparator,
        passed=passed,
        unit=unit,
    )


def collect_metrics_from_logger(
    logger: Any,
    *,
    runner: Any | None = None,
    iteration: int,
    total_iterations: int,
    loss_dict: dict[str, float],
    action_std: Any,
) -> IterationMetrics:
    task_metrics = collect_task_metrics(runner) if runner is not None else TaskMetrics()
    completed_episodes = len(logger.rewbuffer)
    if completed_episodes > 0:
        mean_reward = float(statistics.mean(logger.rewbuffer))
        mean_episode_length = float(statistics.mean(logger.lenbuffer))
        return IterationMetrics(
            iteration=iteration,
            total_iterations=total_iterations,
            mean_reward=mean_reward,
            mean_episode_length=mean_episode_length,
            value_loss=float(loss_dict.get('value', 0.0)),
            surrogate_loss=float(loss_dict.get('surrogate', 0.0)),
            entropy_loss=float(loss_dict.get('entropy', 0.0)),
            action_std=_action_std_value(action_std),
            task_metrics=task_metrics,
            metrics_source='episode',
            completed_episodes=completed_episodes,
        )

    steps_per_env = int(getattr(runner, 'cfg', {}).get('num_steps_per_env', 0)) if runner else 0
    mean_step_reward = 0.0
    if runner is not None and getattr(runner.alg, 'storage', None) is not None:
        storage_rewards = runner.alg.storage.rewards
        if storage_rewards is not None:
            mean_step_reward = float(storage_rewards.mean().item())
    rollout_reward = mean_step_reward * max(steps_per_env, 1)
    return IterationMetrics(
        iteration=iteration,
        total_iterations=total_iterations,
        mean_reward=rollout_reward,
        mean_episode_length=float(steps_per_env),
        value_loss=float(loss_dict.get('value', 0.0)),
        surrogate_loss=float(loss_dict.get('surrogate', 0.0)),
        entropy_loss=float(loss_dict.get('entropy', 0.0)),
        action_std=_action_std_value(action_std),
        task_metrics=task_metrics,
        metrics_source='rollout',
        mean_step_reward=mean_step_reward,
        completed_episodes=0,
    )


def _action_std_value(action_std: Any) -> float:
    if hasattr(action_std, 'mean'):
        return float(action_std.mean().item())
    return float(action_std)


def run_training_with_reports(
    runner: Any,
    *,
    num_learning_iterations: int | None = None,
    criteria: TrainingSuccessCriteria | None = None,
    init_at_random_ep_len: bool = True,
    train_until_task_success: bool = True,
    max_duration_s: float | None = None,
    iteration_chunk: int | None = None,
    verbose: bool = False,
    verbose_curriculum_fn: Callable[[], str] | None = None,
) -> tuple[list[IterationReport], float, str]:
    """Run PPO learning with per-iteration success criteria output.

    When ``num_learning_iterations`` is None, training runs in ``iteration_chunk``
    batches until ``max_duration_s`` elapses or the task requirement is met.
    This is the default duration-bounded mode (spec.md § Phase 2).
    """

    from isaac_lab.training_defaults import DEFAULT_TRAINING_ITERATION_CHUNK  # noqa: WPS433

    criteria = criteria or TrainingSuccessCriteria()
    reports: list[IterationReport] = []
    original_log: Callable[..., None] = runner.logger.log
    chunk = iteration_chunk or DEFAULT_TRAINING_ITERATION_CHUNK
    unlimited = num_learning_iterations is None
    planned_total = (
        runner.current_learning_iteration + chunk
        if unlimited
        else runner.current_learning_iteration + num_learning_iterations
    )
    stop_reason = 'max_duration' if unlimited and max_duration_s else 'max_iterations'

    started = time.perf_counter()
    iterations_remaining = num_learning_iterations

    def wrapped_log(*args: Any, **kwargs: Any) -> None:
        nonlocal stop_reason
        original_log(*args, **kwargs)
        it = kwargs.get('it', args[0] if args else 0)
        loss_dict = kwargs.get('loss_dict', args[5] if len(args) > 5 else {})
        action_std = kwargs.get('action_std', args[7] if len(args) > 7 else 0.0)
        metrics = collect_metrics_from_logger(
            runner.logger,
            runner=runner,
            iteration=it + 1,
            total_iterations=planned_total,
            loss_dict=loss_dict,
            action_std=action_std,
        )
        report = evaluate_iteration(metrics, criteria)
        reports.append(report)
        print(report.format())
        if verbose and criteria.task_mode == 'reach':
            from isaac_lab.training_verbose import format_iteration_motion_snapshot  # noqa: WPS433

            stage = 'near_ee'
            if verbose_curriculum_fn is not None:
                stage = verbose_curriculum_fn()
            print(
                format_iteration_motion_snapshot(
                    iteration=metrics.iteration,
                    task_metrics=metrics.task_metrics,
                    curriculum_stage=stage,
                )
            )
        if train_until_task_success and report.task_requirement_met:
            task_env = resolve_task_env(runner)
            history_len = 0
            if task_env is not None:
                history_len = max(
                    len(getattr(task_env, '_reach_success_history', [])),
                    len(getattr(task_env, '_push_success_history', [])),
                )
            if history_len >= criteria.min_eval_episodes:
                stop_reason = 'task_requirement_met'
                raise _StopTrainingSuccess()
        if max_duration_s is not None and (time.perf_counter() - started) >= max_duration_s:
            stop_reason = 'max_duration'
            raise _StopTrainingTimeLimit()

    runner.logger.log = wrapped_log
    try:
        while True:
            if unlimited:
                if max_duration_s is not None and (time.perf_counter() - started) >= max_duration_s:
                    stop_reason = 'max_duration'
                    break
                learn_steps = chunk
                planned_total = runner.current_learning_iteration + learn_steps
            else:
                if iterations_remaining is None or iterations_remaining <= 0:
                    break
                learn_steps = min(chunk, iterations_remaining)
                planned_total = runner.current_learning_iteration + learn_steps

            try:
                runner.learn(
                    num_learning_iterations=learn_steps,
                    init_at_random_ep_len=init_at_random_ep_len,
                )
            except (_StopTrainingSuccess, _StopTrainingTimeLimit):
                break

            if not unlimited:
                iterations_remaining -= learn_steps
                if iterations_remaining <= 0:
                    stop_reason = 'max_iterations'
                    break
            init_at_random_ep_len = False
    except (_StopTrainingSuccess, _StopTrainingTimeLimit):
        pass
    elapsed = time.perf_counter() - started
    runner.logger.log = original_log
    return reports, elapsed, stop_reason


class _StopTrainingSuccess(Exception):
    """Internal control-flow exception when the push task requirement is met."""


class _StopTrainingTimeLimit(Exception):
    """Internal control-flow exception when max training duration is reached."""
