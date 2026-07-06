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

"""Detect and validate Isaac Lab install for host-side PPO training."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IsaacLabInstallInfo:
    root: Path | None
    launcher: Path | None
    python_wrapper: Path | None
    sim_link: Path | None
    version_hint: str
    compatible_with_isaac_sim: str
    rl_framework_installed: bool


def default_isaac_sim_path() -> Path:
    env_path = os.environ.get('ISAACSIM_PATH')
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / 'isaacsim'


def _candidate_roots() -> list[Path]:
    env_root = os.environ.get('ISAACLAB_PATH') or os.environ.get('SPARK_ISAACLAB_PATH')
    home = Path.home()
    candidates = [
        Path(env_root) if env_root else None,
        home / 'IsaacLab',
        home / 'isaaclab',
        home / 'Isaac-Lab',
    ]
    return [path for path in candidates if path is not None]


def _read_version_hint(root: Path) -> str:
    version_file = root / 'VERSION'
    if version_file.is_file():
        return version_file.read_text(encoding='utf-8').strip()
    return 'unknown'


def _rl_framework_installed(root: Path, framework: str) -> bool:
    launcher = root / 'isaaclab.sh'
    if not launcher.is_file():
        return False
    try:
        result = subprocess.run(
            [str(launcher), '-p', '-c', f'import {framework}; print("ok")'],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and 'ok' in result.stdout


def detect_isaac_lab_install(isaac_sim_path: Path | None = None) -> IsaacLabInstallInfo:
    """Return Isaac Lab location and readiness flags."""

    sim_path = (isaac_sim_path or default_isaac_sim_path()).resolve()
    framework = os.environ.get('SPARK_ISAACLAB_RL_FRAMEWORK', 'rsl_rl')

    for root in _candidate_roots():
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        launcher = root / 'isaaclab.sh'
        sim_link = root / '_isaac_sim'
        python_wrapper = launcher if launcher.is_file() else None
        return IsaacLabInstallInfo(
            root=root,
            launcher=launcher if launcher.is_file() else None,
            python_wrapper=python_wrapper,
            sim_link=sim_link if sim_link.exists() else None,
            version_hint=_read_version_hint(root),
            compatible_with_isaac_sim=str(sim_path),
            rl_framework_installed=_rl_framework_installed(root, framework),
        )

    return IsaacLabInstallInfo(
        root=None,
        launcher=None,
        python_wrapper=None,
        sim_link=None,
        version_hint='not_installed',
        compatible_with_isaac_sim=str(sim_path),
        rl_framework_installed=False,
    )


def require_isaac_lab(reason: str = 'Isaac Lab is required but not installed') -> IsaacLabInstallInfo:
    """Raise RuntimeError when Isaac Lab is missing or incomplete."""

    info = detect_isaac_lab_install()
    if info.root is None or info.launcher is None:
        raise RuntimeError(
            f'{reason}. Run: ./scripts/host/install_isaac_lab.sh')
    if info.sim_link is None:
        raise RuntimeError(
            f'{reason}. Missing _isaac_sim symlink in {info.root}. '
            'Re-run: ./scripts/host/install_isaac_lab.sh')
    if not info.rl_framework_installed:
        raise RuntimeError(
            f'{reason}. RL framework not importable via isaaclab.sh. '
            'Re-run: ./scripts/host/install_isaac_lab.sh')
    return info


def print_isaac_lab_env() -> int:
    info = detect_isaac_lab_install()
    print(f'ISAACLAB_PATH={info.root or ""}')
    print(f'ISAACLAB_LAUNCHER={info.launcher or ""}')
    print(f'ISAACLAB_SIM_LINK={info.sim_link or ""}')
    print(f'ISAACLAB_VERSION={info.version_hint}')
    print(f'ISAAC_SIM_PATH={info.compatible_with_isaac_sim}')
    print(f'ISAACLAB_RL_FRAMEWORK_INSTALLED={info.rl_framework_installed}')
    if info.root is None or info.launcher is None:
        return 1
    if info.sim_link is None:
        return 2
    if not info.rl_framework_installed:
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(print_isaac_lab_env())
