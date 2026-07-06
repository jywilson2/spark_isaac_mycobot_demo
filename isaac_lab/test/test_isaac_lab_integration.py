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

"""Host-side gate test for Isaac Lab integration (skips when Lab is absent)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _isaaclab_launcher() -> Path | None:
    lab_path = os.environ.get('ISAACLAB_PATH', os.path.expanduser('~/IsaacLab'))
    launcher = Path(lab_path) / 'isaaclab.sh'
    return launcher if launcher.is_file() else None


@pytest.mark.skipif(_isaaclab_launcher() is None, reason='Isaac Lab not installed on host')
def test_isaac_lab_detect_exits_zero() -> None:
    launcher = _isaaclab_launcher()
    assert launcher is not None
    result = subprocess.run(
        [str(launcher), '-p', str(REPO_ROOT / 'isaac_lab' / 'detect_isaac_lab.py')],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(launcher.parent),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(_isaaclab_launcher() is None, reason='Isaac Lab not installed on host')
def test_isaac_lab_verify_imports() -> None:
    launcher = _isaaclab_launcher()
    assert launcher is not None
    result = subprocess.run(
        [
            str(launcher),
            '-p',
            str(REPO_ROOT / 'isaac_lab' / 'verify_install.py'),
            '--headless',
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(launcher.parent),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Isaac Lab task registered' in result.stdout


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
