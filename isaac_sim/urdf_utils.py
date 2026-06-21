"""Utilities for preparing myCobot URDF files for Isaac Sim import."""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE_URI_PATTERN = re.compile(
    r'package://mycobot_description/urdf/mycobot_280_m5/([^"\']+)'
)


def resolve_mycobot_280_m5_package_uris(urdf_text: str) -> str:
    """Replace package:// mesh URIs with relative filenames for Isaac Sim."""

    def _replace(match: re.Match[str]) -> str:
        return match.group(1)

    return PACKAGE_URI_PATTERN.sub(_replace, urdf_text)


def write_isaac_ready_urdf(
    source_urdf: Path,
    output_urdf: Path,
    *,
    mesh_dir: Path | None = None,
) -> Path:
    """Write a copy of the upstream URDF with Isaac Sim-friendly mesh paths."""

    if not source_urdf.is_file():
        raise FileNotFoundError(f"Source URDF not found: {source_urdf}")

    resolved_mesh_dir = mesh_dir or source_urdf.parent
    if not resolved_mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {resolved_mesh_dir}")

    urdf_text = source_urdf.read_text(encoding="utf-8")
    prepared = resolve_mycobot_280_m5_package_uris(urdf_text)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    output_urdf.write_text(prepared, encoding="utf-8")
    return output_urdf


def default_upstream_urdf(repo_root: Path) -> Path:
    """Return the Limo Cobot arm URDF (myCobot 280 M5) inside the submodule."""

    return (
        repo_root
        / "third_party"
        / "mycobot_ros2"
        / "mycobot_description"
        / "urdf"
        / "mycobot_280_m5"
        / "mycobot_280_m5.urdf"
    )


def default_prepared_urdf(repo_root: Path) -> Path:
    """Return the Isaac-ready URDF path written by the scene builder."""

    return (
        repo_root
        / "assets"
        / "robots"
        / "mycobot_280_m5_limo_cobot"
        / "mycobot_280_m5_limo_cobot.urdf"
    )
