"""Tests for the P08 output byte scanner (T305).

The deterministic zero-link regression is the primary seam: it reproduces the
exact ``st_nlink == 0`` observation that the production race produced, without
depending on the race actually winning.  A bounded real concurrency test then
exercises the actual ``EpochMonitor`` atomic snapshot writer.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import pytest

from atlas_sers.evaluation.p08_output_usage import tree_bytes
from atlas_sers.visualization.p08_live_monitor import EpochMonitor

_ORIGINAL_LSTAT = Path.lstat


def _stat_result_like(info: os.stat_result, nlink: int) -> os.stat_result:
    return os.stat_result(
        (
            info.st_mode,
            info.st_ino,
            info.st_dev,
            nlink,
            info.st_uid,
            info.st_gid,
            info.st_size,
            info.st_atime,
            info.st_mtime,
            info.st_ctime,
        )
    )


def test_zero_link_regular_inode_counts_observed_bytes(tmp_path, monkeypatch):
    held = tmp_path / "index.html"
    held.write_bytes(b"0123456789")
    stable = tmp_path / "epochs.jsonl"
    stable.write_bytes(b"abc")
    real = held.lstat()

    def fake_lstat(self):
        if self.name == held.name:
            return _stat_result_like(real, 0)
        return _ORIGINAL_LSTAT(self)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    assert tree_bytes(tmp_path) == 13


def test_vanished_entry_is_skipped(tmp_path, monkeypatch):
    keep = tmp_path / "keep.bin"
    keep.write_bytes(b"12345")
    ghost = tmp_path / "ghost.bin"
    ghost.write_bytes(b"999999999")

    def fake_lstat(self):
        if self.name == ghost.name:
            raise FileNotFoundError(str(self))
        return _ORIGINAL_LSTAT(self)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    assert tree_bytes(tmp_path) == 5


def test_permission_error_propagates(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked.bin"
    blocked.write_bytes(b"123")

    def fake_lstat(self):
        if self.name == blocked.name:
            raise PermissionError(str(self))
        return _ORIGINAL_LSTAT(self)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(PermissionError):
        tree_bytes(tmp_path)


def test_real_hardlink_is_rejected(tmp_path):
    first = tmp_path / "first.bin"
    first.write_bytes(b"abc")
    os.link(first, tmp_path / "second.bin")
    with pytest.raises(ValueError, match="hard link"):
        tree_bytes(tmp_path)


def test_symlink_entry_is_rejected(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"abc")
    os.symlink(target, tmp_path / "link.bin")
    with pytest.raises(ValueError, match="symlink"):
        tree_bytes(tmp_path)


def test_root_symlink_is_refused(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "data.bin").write_bytes(b"abc")
    link = tmp_path / "linked"
    os.symlink(real_dir, link, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        tree_bytes(link)


def test_count_nested_regular_files(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"1234")
    (tmp_path / "nested" / "b.bin").write_bytes(b"12345")
    assert tree_bytes(tmp_path) == 9


def test_str_and_pathlike_accepted(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"123")
    assert tree_bytes(str(tmp_path)) == 3
    assert tree_bytes(os.fspath(tmp_path)) == 3


def test_missing_root_returns_zero(tmp_path):
    assert tree_bytes(tmp_path / "absent") == 0


def test_empty_directory_returns_zero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tree_bytes(empty) == 0


def test_regular_file_root_returns_zero(tmp_path):
    file_root = tmp_path / "file.bin"
    file_root.write_bytes(b"abc")
    assert tree_bytes(file_root) == 0


def test_concurrent_atomic_replacement_scans_cleanly(tmp_path):
    output_dir = tmp_path / "monitor"
    stream = io.StringIO()
    monitor = EpochMonitor(
        os.fspath(output_dir),
        job_id="f" * 64,
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        validation_available=True,
        epoch_budget=200,
        stream=stream,
    )

    def record_for(epoch):
        return {
            "epoch": epoch,
            "chemical_ce": 1.0 / float(epoch),
            "total_loss": 2.0 / float(epoch),
            "supcon_enabled": True,
            "supcon_loss": 0.5 / float(epoch),
            "paired_enabled": False,
            "total_optimizer_steps": epoch,
            "train_nll": 1.0 / float(epoch),
            "validation_nll": 1.1 / float(epoch),
            "train_balanced_accuracy": 0.5,
            "validation_balanced_accuracy": 0.49,
            "best_epoch": 1,
            "nonimproving_epochs": epoch - 1,
        }

    failure = []

    def writer():
        try:
            for epoch in range(1, 31):
                monitor(record_for(epoch))
        except BaseException as exc:  # noqa: BLE001 -- re-reported by the thread-failure assertion below
            failure.append(exc)

    thread = threading.Thread(target=writer, name="p08-monitor-writer")
    thread.start()
    scans = 0
    try:
        while thread.is_alive():
            tree_bytes(tmp_path)
            scans += 1
            if scans >= 4000:
                break
    finally:
        thread.join(timeout=10.0)

    assert not thread.is_alive()
    assert failure == []
    assert scans > 0

    monitor.finish("complete", stop_reason="epoch_limit")

    expected = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    assert tree_bytes(tmp_path) == expected
