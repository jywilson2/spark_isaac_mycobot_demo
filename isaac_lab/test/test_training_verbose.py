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

"""Unit tests for verbose training console output."""

from __future__ import annotations

from isaac_lab.training_verbose import format_reach_motion_glossary


def test_verbose_glossary_mentions_cartesian_actions() -> None:
    text = format_reach_motion_glossary(
        num_envs=8,
        max_duration_minutes=30.0,
        target_reach_success_rate=0.99,
    )
    assert 'Δx, Δy, Δz' in text
    assert 'curriculum' in text.lower()
    assert '30.0 min' in text
