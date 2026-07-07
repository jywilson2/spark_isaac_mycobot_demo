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

"""Minimal runtime environment setup before Isaac Sim / Isaac Lab startup.

Do not suppress warnings here. Meaningful issues are fixed in code; upstream-only
messages are documented in docs/isaac_lab_warnings_audit.md.
"""

from __future__ import annotations

import os


def apply_training_runtime_env() -> None:
    """Set required Omniverse runtime flags only."""

    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')


# Backward-compatible alias used by train_ppo.py imports.
apply_training_warning_filters = apply_training_runtime_env
