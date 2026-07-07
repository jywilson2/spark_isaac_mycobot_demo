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
    assert REACH_ACTION_DIM == 6


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


def test_sample_near_ee_stays_in_ball_and_workspace() -> None:
    import math

    from isaac_lab.mdp_core import sample_reachable_ee_xyz

    center = (0.20, 0.0, 0.12)
    radius_m = 0.04
    for x, y, z in sample_reachable_ee_xyz(
        64,
        easy_fraction=1.0,
        near_ee_center=center,
        near_ee_radius_m=radius_m,
    ):
        dist = math.dist(center, (x, y, z))
        assert dist <= radius_m + 1e-6
        horiz = math.hypot(x, y)
        assert 0.12 - 1e-6 <= horiz <= 0.28 + 1e-6
        assert 0.08 - 1e-6 <= z <= 0.22 + 1e-6


def test_sample_near_ee_does_not_pin_z_to_ceiling() -> None:
    from isaac_lab.mdp_core import MAX_EE_TARGET_Z_M, sample_reachable_ee_xyz

    center = (0.20, 0.0, 0.10)
    samples = sample_reachable_ee_xyz(
        32,
        easy_fraction=1.0,
        near_ee_center=center,
        near_ee_radius_m=0.04,
    )
    z_values = {z for _x, _y, z in samples}
    assert len(z_values) > 1
    assert not all(abs(z - MAX_EE_TARGET_Z_M) < 1e-9 for z in z_values)


def test_is_ee_target_in_workspace() -> None:
    from isaac_lab.mdp_core import is_ee_target_in_workspace

    assert is_ee_target_in_workspace(0.20, 0.0, 0.12)
    assert not is_ee_target_in_workspace(0.05, 0.0, 0.12)
    assert not is_ee_target_in_workspace(0.20, 0.0, 0.05)
    assert not is_ee_target_in_workspace(0.20, 0.0, 0.25)


def test_sample_demo_workspace_spreads_targets() -> None:
    import math

    from isaac_lab.mdp_core import (
        MAX_EE_TARGET_Z_M,
        MIN_EE_TARGET_Z_M,
        sample_demo_workspace_xyz,
    )

    samples = sample_demo_workspace_xyz(24)
    z_values = {round(z, 2) for _x, _y, z in samples}
    radii = {round(math.hypot(x, y), 2) for x, y, _z in samples}
    assert len(z_values) >= 2
    assert len(radii) >= 2
    assert all(MIN_EE_TARGET_Z_M <= z <= MAX_EE_TARGET_Z_M for _x, _y, z in samples)


def test_sample_demo_workspace_enforces_separation() -> None:
    import math

    from isaac_lab.mdp_core import (
        DEMO_MIN_TARGET_SEPARATION_M,
        sample_demo_workspace_xyz,
    )

    previous = [(0.22, 0.0, 0.12)]
    nxt = sample_demo_workspace_xyz(1, previous_targets=previous)[0]
    assert math.dist(previous[0], nxt) >= DEMO_MIN_TARGET_SEPARATION_M - 1e-6


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
    assert reward_still == -0.02  # time penalty only when standing still

def test_compute_reach_task_reward_moving_away_is_penalized() -> None:
    """Signed shaping: retreating must cost exactly what approaching earns."""

    from isaac_lab.mdp_core import ReachTaskConfig, compute_reach_task_reward

    cfg = ReachTaskConfig()
    reward_away, dist = compute_reach_task_reward(
        0.10, 0.0, 0.12, 0.22, 0.0, 0.12, prev_distance_m=0.07, cfg=cfg,
    )
    assert dist > 0.07
    expected = (0.07 - dist) * cfg.progress_scale - cfg.time_penalty
    assert abs(reward_away - expected) < 1e-9
    assert reward_away < 0.0


def test_compute_reach_task_reward_oscillation_nets_zero_progress() -> None:
    """Approach-then-retreat cycles must not farm reward (anti reward hacking)."""

    from isaac_lab.mdp_core import ReachTaskConfig, compute_reach_task_reward

    cfg = ReachTaskConfig()
    target = (0.22, 0.0, 0.12)
    far_x, near_x = 0.10, 0.16
    far_dist = abs(target[0] - far_x)
    # Step 1: approach from far to near. Step 2: retreat back to far.
    reward_in, near_dist = compute_reach_task_reward(
        near_x, 0.0, 0.12, *target, prev_distance_m=far_dist, cfg=cfg,
    )
    reward_out, _ = compute_reach_task_reward(
        far_x, 0.0, 0.12, *target, prev_distance_m=near_dist, cfg=cfg,
    )
    total_progress = (reward_in + cfg.time_penalty) + (reward_out + cfg.time_penalty)
    assert abs(total_progress) < 1e-9


def test_compute_reach_task_reward_action_penalty() -> None:
    from isaac_lab.mdp_core import ReachTaskConfig, compute_reach_task_reward

    cfg = ReachTaskConfig()
    reward_calm, _ = compute_reach_task_reward(
        0.15, 0.0, 0.12, 0.22, 0.0, 0.12,
        prev_distance_m=0.07, mean_abs_action=0.0, cfg=cfg,
    )
    reward_violent, _ = compute_reach_task_reward(
        0.15, 0.0, 0.12, 0.22, 0.0, 0.12,
        prev_distance_m=0.07, mean_abs_action=1.0, cfg=cfg,
    )
    assert reward_violent == reward_calm - cfg.action_penalty


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
