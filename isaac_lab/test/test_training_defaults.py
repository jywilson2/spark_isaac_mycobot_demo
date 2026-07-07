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

"""Unit tests for DGX Spark training defaults and demo instructions."""

from __future__ import annotations

from isaac_lab.training_defaults import (
    DEFAULT_MAX_TRAIN_DURATION_MINUTES,
    DEFAULT_NUM_ARMS,
    DEFAULT_NUM_ARMS_GUI,
    DEFAULT_NUM_ARMS_HEADLESS,
    INTEGRATION_TRAIN_MAX_DURATION_MINUTES,
    default_max_train_duration_s,
    default_num_arms,
    format_policy_demo_instructions,
)


def test_gui_default_is_two_arms() -> None:
    assert DEFAULT_NUM_ARMS_GUI == 2
    assert default_num_arms(headless=False) == 2


def test_headless_default_is_eight_arms_on_dgx_spark() -> None:
    assert DEFAULT_NUM_ARMS_HEADLESS == 8
    assert DEFAULT_NUM_ARMS == 8
    assert default_num_arms(headless=True) == 8


def test_integration_smoke_duration_matches_default() -> None:
    assert INTEGRATION_TRAIN_MAX_DURATION_MINUTES == DEFAULT_MAX_TRAIN_DURATION_MINUTES


def test_default_max_train_duration_is_thirty_minutes() -> None:
    assert DEFAULT_MAX_TRAIN_DURATION_MINUTES == 30.0
    assert default_max_train_duration_s() == 30.0 * 60.0


def test_demo_instructions_reference_play_script() -> None:
    text = format_policy_demo_instructions()
    assert 'run_isaac_lab_training.sh play' in text
    assert 'latest_policy' in text
