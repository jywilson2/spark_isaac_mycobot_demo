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
    apply_action_delta,
    build_observation_vector,
    compute_total_reward,
)


def test_observation_dim_is_fourteen() -> None:
    assert OBSERVATION_DIM == 14
    assert ACTION_DIM == 6


def test_build_observation_vector_shape() -> None:
    obs = build_observation_vector(
        centroid_x=0.5,
        centroid_y=0.5,
        bbox_xyxy=[0.4, 0.4, 0.6, 0.6],
        joint_positions=[0.0] * 6,
        end_effector_z=0.12,
        is_grasped=False,
    )
    assert len(obs) == OBSERVATION_DIM


def test_compute_total_reward_is_non_negative_without_penalty() -> None:
    reward = compute_total_reward(
        centroid_x=0.5,
        centroid_y=0.5,
        end_effector_x=0.5,
        end_effector_y=0.5,
        is_grasped=False,
        block_height=0.05,
        safety_penalty=0.0,
    )
    assert reward >= 0.0


def test_apply_action_delta_clamps() -> None:
    targets = apply_action_delta(
        current_joint_positions=[0.0] * 6,
        action_delta=[10.0] * 6,
        joint_min=[-1.0] * 6,
        joint_max=[1.0] * 6,
        action_scale=1.0,
    )
    assert targets == [1.0] * 6
