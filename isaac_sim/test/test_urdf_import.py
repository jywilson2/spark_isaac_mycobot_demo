"""Tests for URDF import path helpers (no Isaac Sim runtime required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac_sim.urdf_import import URDF_IMPORTER_EXTENSION, resolve_imported_usd_path


def test_urdf_importer_extension_id() -> None:
    assert URDF_IMPORTER_EXTENSION == 'isaacsim.asset.importer.urdf'


def test_resolve_imported_usd_path_prefers_existing_file(tmp_path: Path) -> None:
    robot_usd = tmp_path / 'robot.usd'
    robot_usd.write_text('#usda', encoding='utf-8')

    resolved = resolve_imported_usd_path(
        imported_path=str(tmp_path / 'other.usd'),
        output_usd=robot_usd,
        search_dir=tmp_path,
    )
    assert resolved == robot_usd.resolve()


def test_resolve_imported_usd_path_falls_back_to_newest_usd(tmp_path: Path) -> None:
    older = tmp_path / 'older.usd'
    newer = tmp_path / 'newer.usd'
    older.write_text('#usda', encoding='utf-8')
    newer.write_text('#usda', encoding='utf-8')
    newer.touch()

    resolved = resolve_imported_usd_path(None, tmp_path / 'missing.usd', tmp_path)
    assert resolved.name == 'newer.usd'


def test_resolve_imported_usd_path_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='did not produce a USD file'):
        resolve_imported_usd_path(None, tmp_path / 'missing.usd', tmp_path)
