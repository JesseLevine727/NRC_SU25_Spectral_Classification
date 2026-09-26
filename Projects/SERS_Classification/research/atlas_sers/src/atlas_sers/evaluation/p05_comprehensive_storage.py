"""Bounded private-storage accounting for the P05 comprehensive run."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

P05DEVELOPMENT_NAMESPACE = "p05development"
P05COMPREHENSIVE_NAMESPACE = "p05comprehensive"
DISK_RESERVE_BYTES = 32 * 1024 * 1024
MAX_TRACKED_GROWING = 15
MAX_UNIT_FILES = 256
RECONCILE_UNITS = 64
NAMESPACES = (P05DEVELOPMENT_NAMESPACE, P05COMPREHENSIVE_NAMESPACE)


class P05StorageError(RuntimeError):
    """Stable, path-free storage failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _within(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _overlaps(left: Path, right: Path) -> bool:
    left, right = _absolute(left), _absolute(right)
    return _within(left, right) or _within(right, left)


def _assert_disjoint(path: Path, others) -> None:
    for other in others:
        if _overlaps(path, other):
            raise P05StorageError("accounting_overlap")


def _reject_symlink_chain(path: Path) -> None:
    current = _absolute(path)
    while True:
        if current.is_symlink():
            raise P05StorageError("symlink_path_rejected")
        if current.parent == current:
            return
        current = current.parent


def _scan_tree(root: Path, max_files: int | None = None, seen: set | None = None) -> int:
    root = Path(root)
    _reject_symlink_chain(root)
    if not root.exists():
        return 0
    _reject_symlink_chain(root)
    total = 0
    files = 0

    def walk_error(error):
        raise P05StorageError("storage_scan_failed") from error

    for base, dirnames, filenames in os.walk(root, followlinks=False, onerror=walk_error):
        for name in (*dirnames, *filenames):
            entry = Path(base) / name
            if entry.is_symlink():
                raise P05StorageError("symlink_path_rejected")
            if entry.is_file():
                if seen is not None:
                    seen.add(_absolute(entry))
                try:
                    total += entry.stat().st_size
                except OSError as error:
                    raise P05StorageError("storage_stat_failed") from error
                files += 1
                if max_files is not None and files > max_files:
                    raise P05StorageError("unit_file_bound_exceeded")
    return total


class StorageBudget:
    """Conservative accounting against the private-storage ceiling."""

    def __init__(
        self, artifact_root: Path | str, run_dir: Path | str, ceiling: int = 107374182400
    ) -> None:
        if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling <= 0:
            raise P05StorageError("ceiling_malformed")
        self._artifact_root = _absolute(artifact_root)
        self._run_dir = _absolute(run_dir)
        _reject_symlink_chain(self._artifact_root)
        _reject_symlink_chain(self._run_dir)
        if not _within(self._artifact_root, self._run_dir):
            raise P05StorageError("run_outside_artifact_root")
        if not _within(self._artifact_root / P05COMPREHENSIVE_NAMESPACE, self._run_dir):
            raise P05StorageError("run_outside_governed_namespace")
        self._ceiling = ceiling
        self._growing: dict[Path, int] = {}
        self._charged: set[Path] = set()
        self._baseline_files: set[Path] = set()
        self._closed_units: set[Path] = set()
        self._unit: Path | None = None
        self._units_since_reconcile = 0
        self._base = sum(
            _scan_tree(self._artifact_root / name, seen=self._baseline_files) for name in NAMESPACES
        )
        if self._base > self._ceiling:
            raise P05StorageError("storage_ceiling_exceeded")

    def _require_inside_run(self, path: Path | str) -> Path:
        candidate = _absolute(path)
        _reject_symlink_chain(candidate)
        if not _within(self._run_dir, candidate):
            raise P05StorageError("path_outside_run")
        if any(parent in self._closed_units for parent in (candidate, *candidate.parents)):
            raise P05StorageError("closed_unit_mutation")
        return candidate

    def _current_size(self, path: Path) -> int:
        _reject_symlink_chain(path)
        if path.is_symlink():
            raise P05StorageError("symlink_path_rejected")
        try:
            return path.stat().st_size if path.exists() else 0
        except OSError as error:
            raise P05StorageError("storage_stat_failed") from error

    def _live_usage(self) -> int:
        total = self._base
        for path, baseline in self._growing.items():
            size = self._current_size(path)
            if size > baseline:
                total += size - baseline
        if self._unit is not None:
            total += _scan_tree(self._unit, MAX_UNIT_FILES)
        return total

    def check(self, headroom_bytes: int = 0) -> int:
        if (
            isinstance(headroom_bytes, bool)
            or not isinstance(headroom_bytes, int)
            or headroom_bytes < 0
        ):
            raise P05StorageError("headroom_malformed")
        live = self._live_usage()
        if live + headroom_bytes > self._ceiling:
            raise P05StorageError("storage_ceiling_exceeded")
        try:
            free = int(shutil.disk_usage(os.fspath(self._artifact_root)).free)
        except OSError as error:
            raise P05StorageError("disk_usage_unavailable") from error
        if free < DISK_RESERVE_BYTES + headroom_bytes:
            raise P05StorageError("disk_reserve_breached")
        return live

    def register_growing(self, path: Path | str) -> Path:
        candidate = self._require_inside_run(path)
        if candidate in self._growing:
            raise P05StorageError("growing_path_already_registered")
        if len(self._growing) >= MAX_TRACKED_GROWING:
            raise P05StorageError("too_many_tracked_growing")
        if candidate.is_symlink() or candidate.exists():
            raise P05StorageError("growing_path_preexists")
        _assert_disjoint(candidate, self._growing)
        if self._unit is not None:
            _assert_disjoint(candidate, (self._unit,))
        _assert_disjoint(candidate, self._charged)
        self._growing[candidate] = 0
        return candidate

    def account_new_file(self, path: Path | str) -> int:
        candidate = _absolute(path)
        _reject_symlink_chain(candidate)
        shared = self._artifact_root / P05DEVELOPMENT_NAMESPACE / "slot_leases"
        if not (_within(self._run_dir, candidate) or _within(shared, candidate)):
            raise P05StorageError("path_outside_run")
        if candidate in self._charged or candidate in self._baseline_files:
            raise P05StorageError("file_already_charged")
        if any(parent in self._closed_units for parent in candidate.parents):
            raise P05StorageError("closed_unit_mutation")
        if candidate in self._growing:
            raise P05StorageError("accounting_overlap")
        if self._unit is not None:
            _assert_disjoint(candidate, (self._unit,))
        _assert_disjoint(candidate, self._growing)
        if candidate.is_symlink() or not candidate.is_file():
            raise P05StorageError("new_file_invalid")
        try:
            size = candidate.stat().st_size
        except OSError as error:
            raise P05StorageError("storage_stat_failed") from error
        self._base += size
        self._charged.add(candidate)
        return size

    def activate_unit(self, path: Path | str) -> Path:
        if self._unit is not None:
            raise P05StorageError("unit_already_active")
        candidate = self._require_inside_run(path)
        _assert_disjoint(candidate, self._growing)
        _reject_symlink_chain(candidate.parent)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(candidate, 0o700)
        except FileExistsError as error:
            raise P05StorageError("unit_already_exists") from error
        except OSError as error:
            raise P05StorageError("unit_create_failed") from error
        self._unit = candidate
        return candidate

    def close_unit(self) -> int:
        if self._unit is None:
            raise P05StorageError("no_active_unit")
        size = _scan_tree(self._unit, MAX_UNIT_FILES)
        self._base += size
        self._closed_units.add(self._unit)
        self._unit = None
        self._units_since_reconcile += 1
        if self._units_since_reconcile >= RECONCILE_UNITS:
            self.reconcile()
        return size

    def reconcile(self) -> int:
        if self._unit is not None:
            raise P05StorageError("unit_active_during_reconcile")
        self._baseline_files = set()
        measured = sum(
            _scan_tree(self._artifact_root / name, seen=self._baseline_files) for name in NAMESPACES
        )
        if measured > self._ceiling:
            raise P05StorageError("storage_ceiling_exceeded")
        self._base = measured
        for path in self._growing:
            self._growing[path] = self._current_size(path)
        self._units_since_reconcile = 0
        return measured
