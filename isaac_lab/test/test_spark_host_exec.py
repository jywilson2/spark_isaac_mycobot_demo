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

"""Unit tests for host delegation GUI environment resolution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_EXEC = REPO_ROOT / 'scripts' / 'host' / 'spark_host_exec.sh'


def _bash(snippet: str, *, env: dict[str, str] | None = None) -> str:
    merged = {**os.environ, **(env or {})}
    return subprocess.check_output(
        ['bash', '-lc', f'source "{HOST_EXEC}" && {snippet}'],
        env=merged,
        text=True,
    ).strip()


def _parse_kv_output(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        key, _, value = line.partition('=')
        result[key] = value
    return result


def test_resolve_gui_env_preserves_display() -> None:
    env = _parse_kv_output(
        _bash('spark_resolve_host_gui_env /home/testuser', env={'DISPLAY': ':1'}),
    )
    assert env['DISPLAY'] == ':1'


def test_resolve_gui_env_uses_host_xauthority_when_unset() -> None:
    env = _parse_kv_output(
        _bash(
            'spark_resolve_host_gui_env /home/jywilson',
            env={'DISPLAY': ':1', 'XAUTHORITY': ''},
        ),
    )
    assert env['DISPLAY'] == ':1'
    # In the Isaac ROS container, host cookie is resolved via nsenter.
    assert env['XAUTHORITY'] in {'/home/jywilson/.Xauthority', ''}


def test_resolve_gui_env_prefers_host_home_xauthority() -> None:
    home = Path('/tmp/spark_host_exec_home')
    home.mkdir(parents=True, exist_ok=True)
    xauth = home / '.Xauthority'
    xauth.write_text('dummy', encoding='utf-8')
    try:
        env = _parse_kv_output(
            _bash(
                f'spark_resolve_host_gui_env "{home}"',
                env={'DISPLAY': ':2', 'XAUTHORITY': ''},
            ),
        )
        assert env['DISPLAY'] == ':2'
        assert env['XAUTHORITY'] == str(xauth)
    finally:
        xauth.unlink(missing_ok=True)


def test_require_gui_display_fails_without_display() -> None:
    proc = subprocess.run(
        [
            'bash',
            '-lc',
            (
                f'source "{HOST_EXEC}" && '
                'spark_resolve_host_gui_env() { printf "DISPLAY=\\nXAUTHORITY=\\n"; } && '
                'spark_require_gui_display /home/nobody'
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert 'DISPLAY' in proc.stderr
