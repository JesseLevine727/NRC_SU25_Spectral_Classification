"""Tests for bounded P05 comprehensive storage accounting."""

from __future__ import annotations

import types

import pytest

import atlas_sers.evaluation.p05_comprehensive_storage as storage
from atlas_sers.evaluation.p05_comprehensive_storage import (
    MAX_TRACKED_GROWING,
    MAX_UNIT_FILES,
    RECONCILE_UNITS,
    P05StorageError,
    StorageBudget,
)

_abs = storage._absolute


def write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def roots(tmp_path):
    art = tmp_path / "artifact"
    run = art / "p05comprehensive" / "runs" / "run"
    run.mkdir(parents=True)
    (art / "p05development" / "slot_leases").mkdir(parents=True)
    return art, run


def test_cumulative_arithmetic(tmp_path):
    art, run = roots(tmp_path)
    pilot = write(art / "p05development" / "slot_leases" / "pilot.json", b"P" * 40)
    budget = StorageBudget(art, run)
    assert budget.check() == pilot

    static_path = run / "static.json"
    static_bytes = write(static_path, b"S" * 30)
    assert budget.account_new_file(static_path) == static_bytes
    assert budget.check() == pilot + static_bytes

    journal = run / "journal.jsonl"
    budget.register_growing(journal)
    write(journal, b"J" * 7)
    assert budget.check() == pilot + static_bytes + 7
    with journal.open("ab") as handle:
        handle.write(b"J" * 3)
    assert budget.check() == pilot + static_bytes + 10

    unit = run / "units" / "u0"
    budget.activate_unit(unit)
    unit_bytes = write(unit / "a.bin", b"U" * 100) + write(unit / "b.bin", b"V" * 50)
    assert budget.check() == pilot + static_bytes + 10 + unit_bytes
    assert budget.close_unit() == unit_bytes
    assert budget.check() == pilot + static_bytes + 10 + unit_bytes

    shared = art / "p05development" / "slot_leases" / "new.json"
    shared_bytes = write(shared, b"L" * 25)
    assert budget.account_new_file(shared) == shared_bytes
    assert budget.check() == pilot + static_bytes + 10 + unit_bytes + shared_bytes


def test_reject_double_baseline_closed(tmp_path):
    art, run = roots(tmp_path)
    write(run / "preexisting.json", b"E" * 12)
    budget = StorageBudget(art, run)
    with pytest.raises(P05StorageError, match="file_already_charged"):
        budget.account_new_file(run / "preexisting.json")

    fresh = run / "fresh.json"
    write(fresh, b"N" * 5)
    budget.account_new_file(fresh)
    with pytest.raises(P05StorageError, match="file_already_charged"):
        budget.account_new_file(fresh)

    unit = run / "units" / "u0"
    budget.activate_unit(unit)
    inside = unit / "f.bin"
    write(inside, b"F" * 9)
    budget.close_unit()
    with pytest.raises(P05StorageError, match="closed_unit_mutation"):
        budget.account_new_file(inside)


def test_overlap_and_symlink_rejection(tmp_path):
    art, run = roots(tmp_path)
    budget = StorageBudget(art, run)
    journal = run / "journal.jsonl"
    budget.register_growing(journal)
    with pytest.raises(P05StorageError, match="accounting_overlap"):
        budget.activate_unit(journal)
    unit = run / "units" / "u0"
    budget.activate_unit(unit)
    with pytest.raises(P05StorageError, match="accounting_overlap"):
        budget.register_growing(unit / "inside.jsonl")
    budget.close_unit()

    link = run / "link"
    link.symlink_to(run / "units")
    with pytest.raises(P05StorageError, match="symlink_path_rejected"):
        budget.account_new_file(link / "x.json")

    (art / "p05development" / "broken").symlink_to(art / "p05development" / "missing")
    with pytest.raises(P05StorageError, match="symlink_path_rejected"):
        StorageBudget(art, run)


def test_outside_and_init_validation(tmp_path):
    art, run = roots(tmp_path)
    budget = StorageBudget(art, run)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f").write_bytes(b"x")
    with pytest.raises(P05StorageError, match="path_outside_run"):
        budget.register_growing(outside / "g")
    with pytest.raises(P05StorageError, match="path_outside_run"):
        budget.account_new_file(outside / "f")
    with pytest.raises(P05StorageError, match="path_outside_run"):
        budget.activate_unit(outside / "u")
    with pytest.raises(P05StorageError, match="run_outside_artifact_root"):
        StorageBudget(art, tmp_path / "elsewhere")
    bad = art / "other" / "run"
    bad.mkdir(parents=True)
    with pytest.raises(P05StorageError, match="run_outside_governed_namespace"):
        StorageBudget(art, bad)


def test_cap_headroom_and_disk_reserve(tmp_path, monkeypatch):
    art, run = roots(tmp_path)
    write(run / "seed.json", b"S" * 100)
    budget = StorageBudget(art, run, ceiling=150)
    assert budget.check() == 100
    with pytest.raises(P05StorageError, match="storage_ceiling_exceeded"):
        budget.check(headroom_bytes=51)
    assert budget.check(headroom_bytes=50) == 100
    for bad in (-1, True):
        with pytest.raises(P05StorageError, match="headroom_malformed"):
            budget.check(bad)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=0))
    with pytest.raises(P05StorageError, match="disk_reserve_breached"):
        budget.check()


def test_unit_and_growing_bounds(tmp_path):
    art, run = roots(tmp_path)
    budget = StorageBudget(art, run)
    preexisting = run / "exists.jsonl"
    write(preexisting, b"z")
    with pytest.raises(P05StorageError, match="growing_path_preexists"):
        budget.register_growing(preexisting)
    for index in range(MAX_TRACKED_GROWING):
        budget.register_growing(run / f"g{index}.jsonl")
    with pytest.raises(P05StorageError, match="too_many_tracked_growing"):
        budget.register_growing(run / "extra.jsonl")
    with pytest.raises(P05StorageError, match="growing_path_already_registered"):
        budget.register_growing(run / "g0.jsonl")

    unit = run / "units" / "u0"
    budget.activate_unit(unit)
    for index in range(MAX_UNIT_FILES + 1):
        write(unit / f"f{index}", b"x")
    with pytest.raises(P05StorageError, match="unit_file_bound_exceeded"):
        budget.close_unit()


def test_reconcile_every_64_units(tmp_path):
    art, run = roots(tmp_path)
    budget = StorageBudget(art, run)
    expected = 0
    for index in range(RECONCILE_UNITS):
        unit = run / "units" / f"u{index}"
        budget.activate_unit(unit)
        size = write(unit / "f", b"x" * (index + 1))
        expected += size
        assert budget.close_unit() == size
    assert budget._units_since_reconcile == 0
    assert budget.check() == expected
    assert budget.reconcile() == expected


def test_check_is_bounded(tmp_path, monkeypatch):
    art, run = roots(tmp_path)
    write(art / "p05development" / "slot_leases" / "p.json", b"P" * 20)
    budget = StorageBudget(art, run)

    scans = []
    real_scan = storage._scan_tree

    def spy_scan(root, max_files=None, seen=None):
        scans.append(_abs(root))
        return real_scan(root, max_files=max_files, seen=seen)

    stats = []
    real_current = storage.StorageBudget._current_size

    def spy_current(self, path):
        stats.append(_abs(path))
        return real_current(self, path)

    monkeypatch.setattr(storage, "_scan_tree", spy_scan)
    monkeypatch.setattr(storage.StorageBudget, "_current_size", spy_current)

    static_path = run / "static.json"
    write(static_path, b"S" * 10)
    budget.account_new_file(static_path)
    journal = run / "journal.jsonl"
    budget.register_growing(journal)
    write(journal, b"J" * 5)
    for _ in range(3):
        budget.check()
    assert scans == []
    assert _abs(static_path) not in stats
    assert stats == [_abs(journal)] * 3
    assert len(budget._growing) <= MAX_TRACKED_GROWING

    unit = run / "units" / "u0"
    budget.activate_unit(unit)
    write(unit / "f", b"U" * 4)
    budget.check()
    assert scans == [_abs(unit)]


def test_failure_preserves_partial_unit(tmp_path):
    art, run = roots(tmp_path)
    budget = StorageBudget(art, run, ceiling=10_000)
    unit = run / "units" / "u0"
    budget.activate_unit(unit)
    partial = unit / "partial.bin"
    write(partial, b"X" * 64)
    with pytest.raises(P05StorageError, match="storage_ceiling_exceeded"):
        budget.check(headroom_bytes=10_000)
    assert partial.exists() and partial.read_bytes() == b"X" * 64
    assert budget.close_unit() == 64
