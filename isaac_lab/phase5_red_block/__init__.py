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

"""Phase 5 — red-block vision localization and contact-and-push (opt-in only)."""

from isaac_lab.phase5_red_block.red_block_env import (
    PHASE5_TASK_ID,
    MyCobotRedBlockEnv,
    MyCobotRedBlockEnvCfg,
    register_red_block_env,
)

__all__ = [
    'PHASE5_TASK_ID',
    'MyCobotRedBlockEnv',
    'MyCobotRedBlockEnvCfg',
    'register_red_block_env',
]
