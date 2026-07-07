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

"""Unit tests for shared Isaac Lab MDP contract."""

from __future__ import annotations

from isaac_lab.mdp_core import (
    ACTION_DIM,
    OBSERVATION_DIM,
    TARGET_PUSH_DISTANCE_M,
    apply_action_delta,
    build_motion_observation_vector,
    compute_contact_push_reward,
    compute_motion_approach_reward,
    compute_target_ee_pose,
    compute_total_reward,
    episode_push_task_success,
)


def test_observation_dim_is_eleven() -> None:
    from isaac_lab.mdp_core import REACH_ACTION_DIM

    assert OBSERVATION_DIM == 11
    assert ACTION_DIM == 6
    assert REACH_ACTION_DIM == 3


def test_build_motion_observation_vector_shape() -> None:
    obs = build_motion_observation_vector(
        ee_to_target_delta=[0.05, -0.02, 0.01],
        target_valid=True,
        reached=False,
        joint_positions=[0.0] * 6,
    )
    assert len(obs) == OBSERVATION_DIM
    assert obs[3] == 1.0
    assert obs[4] == 0.0


def test_sample_reachable_ee_xyz_stays_in_envelope() -> None:
    import math

    from isaac_lab.mdp_core import (
        MAX_EE_TARGET_Z_M,
        MAX_BLOCK_REACH_M,
        MIN_EE_TARGET_Z_M,
        MIN_BLOCK_REACH_M,
        sample_reachable_ee_xyz,
    )

    for x, y, z in sample_reachable_ee_xyz(64):
        radius = math.hypot(x, y)
        assert MIN_BLOCK_REACH_M - 1e-6 <= radius <= MAX_BLOCK_REACH_M + 1e-6
        assert MIN_EE_TARGET_Z_M - 1e-6 <= z <= MAX_EE_TARGET_Z_M + 1e-6


def test_compute_reach_task_reward_bonus_at_target() -> None:
    from isaac_lab.mdp_core import ReachTaskConfig, compute_reach_task_reward

    cfg = ReachTaskConfig()
    reward, dist = compute_reach_task_reward(
        0.22, 0.0, 0.12, 0.22, 0.0, 0.12, prev_distance_m=0.05, cfg=cfg,
    )
    assert dist < 0.001
    assert reward >= cfg.reach_bonus - cfg.time_penalty


def test_compute_reach_task_reward_no_proximity_without_progress() -> None:
    from isaac_lab.mdp_core import compute_reach_task_reward

    reward_still, dist = compute_reach_task_reward(
        0.15, 0.0, 0.12, 0.22, 0.0, 0.12, prev_distance_m=0.07,
    )
    assert dist > 0.025
    assert reward_still == -0.04  # time penalty only when standing still


def test_damped_least_squares_moves_along_jacobian() -> None:
    from isaac_lab.mdp_core import damped_least_squares_joint_delta

    jacobian = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
    dq = damped_least_squares_joint_delta(jacobian, [0.1, 0.0, 0.0], damping=0.01)
    assert abs(dq[0] - 0.1) < 1e-3
    assert abs(dq[1]) < 1e-3


def test_resolve_curriculum_stage_advances_with_success() -> None:
    from isaac_lab.mdp_core import resolve_curriculum_stage

    assert resolve_curriculum_stage(0.0).name == 'near_ee'
    assert resolve_curriculum_stage(0.92).name == 'medium'
    assert resolve_curriculum_stage(0.96).name == 'full'


def test_compute_target_ee_pose_offsets_above_block() -> None:
    tx, ty, tz = compute_target_ee_pose(0.22, 0.0, 0.021, approach_offset_m=0.03)
    assert tx == 0.22
    assert ty == 0.0
    assert tz == 0.021 + 0.02 + 0.03


def test_motion_approach_reward_is_zero_without_target() -> None:
    assert compute_motion_approach_reward(0.0, 0.0, 0.1, 0.2, 0.0, 0.12, target_valid=False) == 0.0


def test_contact_push_reward_is_highest_when_aligned_and_pushed() -> None:
    reward = compute_contact_push_reward(
        block_x=0.25,
        block_y=0.0,
        block_z=0.04,
        end_effector_x=0.25,
        end_effector_y=0.0,
        end_effector_z=0.055,
        robot_base_x=0.0,
        robot_base_y=0.0,
        push_distance_m=TARGET_PUSH_DISTANCE_M,
    )
    assert reward >= 3.0


def test_episode_push_task_success_requires_contact_and_distance() -> None:
    assert episode_push_task_success(0.006, had_contact=True)
    assert not episode_push_task_success(0.006, had_contact=False)
    assert not episode_push_task_success(0.003, had_contact=True)


def test_compute_total_reward_contact_push_mode() -> None:
    reward = compute_total_reward(
        centroid_x=0.25,
        centroid_y=0.0,
        end_effector_x=0.25,
        end_effector_y=0.0,
        is_grasped=False,
        block_height=0.04,
        safety_penalty=0.0,
        end_effector_z=0.055,
        push_distance_m=TARGET_PUSH_DISTANCE_M,
    )
    assert reward >= 3.0


def test_apply_action_delta_clamps() -> None:
    targets = apply_action_delta(
        current_joint_positions=[0.0] * 6,
        action_delta=[10.0] * 6,
        joint_min=[-1.0] * 6,
        joint_max=[1.0] * 6,
        action_scale=1.0,
    )
    assert targets == [1.0] * 6
