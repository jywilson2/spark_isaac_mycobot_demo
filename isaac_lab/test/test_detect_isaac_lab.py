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

"""Tests for Isaac Lab install detection (no Isaac Lab runtime required)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from isaac_lab.detect_isaac_lab import detect_isaac_lab_install


def _scratch_dir() -> Path:
    base = Path(os.environ.get('PYTEST_TMPDIR', tempfile.gettempdir()))
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='spark_detect_', dir=str(base)))


def test_detect_not_installed_when_missing(monkeypatch) -> None:
    tmp_path = _scratch_dir()
    monkeypatch.setattr(
        'isaac_lab.detect_isaac_lab._candidate_roots',
        lambda: [tmp_path / 'missing_isaac_lab'],
    )
    info = detect_isaac_lab_install(isaac_sim_path=tmp_path / 'isaacsim')
    assert info.root is None
    assert info.version_hint == 'not_installed'


def test_detect_finds_launcher(monkeypatch) -> None:
    tmp_path = _scratch_dir()
    lab_root = tmp_path / 'IsaacLab'
    lab_root.mkdir()
    (lab_root / 'isaaclab.sh').write_text('#!/bin/sh\necho ok\n', encoding='utf-8')
    (lab_root / 'isaaclab.sh').chmod(0o755)
    (lab_root / 'VERSION').write_text('test', encoding='utf-8')
    sim_path = tmp_path / 'isaacsim'
    sim_path.mkdir()
    (lab_root / '_isaac_sim').symlink_to(sim_path)

    monkeypatch.setenv('ISAACLAB_PATH', str(lab_root))
    monkeypatch.setenv('SPARK_ISAACLAB_RL_FRAMEWORK', 'rsl_rl')

    info = detect_isaac_lab_install(isaac_sim_path=sim_path)
    assert info.root == lab_root.resolve()
    assert info.launcher == (lab_root / 'isaaclab.sh').resolve()
    assert info.sim_link is not None
    assert info.sim_link.resolve() == sim_path.resolve()
