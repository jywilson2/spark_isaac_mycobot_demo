"""Tests for Isaac Sim URDF preparation helpers (no Isaac Sim runtime required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac_sim.urdf_utils import (
    default_upstream_urdf,
    resolve_mycobot_280_m5_package_uris,
    write_isaac_ready_urdf,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_resolve_package_uris_to_relative_mesh_names() -> None:
    sample = (
        '<mesh filename="package://mycobot_description/urdf/mycobot_280_m5/joint1.dae"/>'
    )
    resolved = resolve_mycobot_280_m5_package_uris(sample)
    assert resolved == '<mesh filename="joint1.dae"/>'


def test_default_upstream_urdf_exists_after_submodule_init() -> None:
    urdf = default_upstream_urdf(REPO_ROOT)
    if not urdf.is_file():
        pytest.skip("third_party/mycobot_ros2 submodule not initialized")
    assert urdf.name == "mycobot_280_m5.urdf"


def test_write_isaac_ready_urdf_rewrites_mesh_paths(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "joint1.dae").write_text("dummy", encoding="utf-8")

    source_urdf = source_dir / "robot.urdf"
    source_urdf.write_text(
        '<robot><link name="l"><visual><geometry>'
        '<mesh filename="package://mycobot_description/urdf/mycobot_280_m5/joint1.dae"/>'
        "</geometry></visual></link></robot>",
        encoding="utf-8",
    )

    output_urdf = tmp_path / "out" / "robot_isaac.urdf"
    write_isaac_ready_urdf(source_urdf, output_urdf)

    text = output_urdf.read_text(encoding="utf-8")
    assert "package://" not in text
    assert 'filename="joint1.dae"' in text
