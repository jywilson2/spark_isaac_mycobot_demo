# Copyright 2026 spark_isaac_mycobot_demo contributors
"""Unit tests for train_ppo checkpoint-resume helpers (no Isaac Sim required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac_lab.train_ppo import find_newest_checkpoint, latest_model_file, resolve_resume_checkpoint


def _make_policy_dir(tmp_path: Path, iterations: list[int]) -> Path:
    policy_dir = tmp_path / 'latest_policy'
    policy_dir.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        (policy_dir / f'model_{iteration}.pt').write_bytes(b'stub')
    return policy_dir


def test_latest_model_file_sorts_numerically(tmp_path: Path) -> None:
    """model_1000 must beat model_999 — lexicographic order gets this wrong."""

    policy_dir = _make_policy_dir(tmp_path, [10, 999, 1000])
    newest = latest_model_file(policy_dir)
    assert newest is not None
    assert newest.name == 'model_1000.pt'


def test_latest_model_file_missing_dir(tmp_path: Path) -> None:
    assert latest_model_file(tmp_path / 'does_not_exist') is None


def test_latest_model_file_ignores_non_numeric(tmp_path: Path) -> None:
    policy_dir = _make_policy_dir(tmp_path, [5])
    (policy_dir / 'model_final.pt').write_bytes(b'stub')
    newest = latest_model_file(policy_dir)
    assert newest is not None
    assert newest.name == 'model_5.pt'


def test_find_newest_checkpoint_prefers_latest_policy_file(tmp_path: Path) -> None:
    policy_file = tmp_path / 'latest_policy'
    policy_file.write_bytes(b'checkpoint')
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'model_1000.pt').write_bytes(b'older-layout')
    assert find_newest_checkpoint(tmp_path) == policy_file


def test_find_newest_checkpoint_falls_back_to_logs_dir(tmp_path: Path) -> None:
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'model_100.pt').write_bytes(b'a')
    (logs / 'model_200.pt').write_bytes(b'b')
    found = find_newest_checkpoint(tmp_path)
    assert found is not None
    assert found.name == 'model_200.pt'


def test_resolve_resume_auto_resumes_when_checkpoint_exists(tmp_path: Path) -> None:
    _make_policy_dir(tmp_path, [10, 20])
    resumed = resolve_resume_checkpoint(tmp_path, from_scratch=False, resume=None)
    assert resumed is not None
    assert resumed.name == 'model_20.pt'


def test_resolve_resume_auto_fresh_when_no_checkpoint(tmp_path: Path) -> None:
    assert resolve_resume_checkpoint(tmp_path, from_scratch=False, resume=None) is None


def test_resolve_resume_from_scratch_never_resumes(tmp_path: Path) -> None:
    _make_policy_dir(tmp_path, [10])
    assert resolve_resume_checkpoint(tmp_path, from_scratch=True, resume=None) is None


def test_resolve_resume_explicit_no_resume_keeps_fresh_weights(tmp_path: Path) -> None:
    _make_policy_dir(tmp_path, [10])
    assert resolve_resume_checkpoint(tmp_path, from_scratch=False, resume=False) is None


def test_resolve_resume_explicit_resume_requires_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(tmp_path, from_scratch=False, resume=True)


def test_resolve_resume_rejects_scratch_plus_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_resume_checkpoint(tmp_path, from_scratch=True, resume=True)
