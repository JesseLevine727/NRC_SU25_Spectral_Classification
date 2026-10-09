"""Read-only byte accounting for a P08 run output directory.

T305 correction for the supervisor resource scanner.

The production scanner walked an *active* monitor directory.  While it was
running, the monitor replaced ``index.html`` through
``atlas_sers.visualization.p08_live_monitor._atomic_write_private``: a private
temporary file is written, flushed and fsync'ed, then ``os.replace`` atomically
moves it over the previous snapshot.  A scan can therefore resolve the
directory entry of the *old* inode and then ``lstat`` that inode after the
``os.replace`` unlinked it.  The kernel reports the former inode with
``st_nlink == 0``.  The original scanner treated any ``st_nlink != 1`` as
``"Unexpected hard link in run output."`` and aborted, even though no hard
link existed and atomic replacement is expected and permitted for monitor
snapshots.

This module keeps the scanner read-only and changes one decision only:
``st_nlink == 0`` for a regular file is the legal concurrent-unlink snapshot,
so its *observed* ``st_size`` is counted conservatively.  A real hard link has
``st_nlink > 1`` and is still rejected.  Symlinks are still rejected and
``FileNotFoundError`` from a vanished entry is still skipped.  The scanner is
not a transactional disk snapshot: it reports the bytes it observed across
atomic replacements, and the caller keeps run-level high-water tracking
separately.  Immutable artifact receipt authentication is a different check
that still requires exactly one link; it is not weakened here.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

__all__ = ["tree_bytes"]


def _walk_error(error: OSError) -> None:
    """Propagate scandir failures except vanished entries.

    ``os.walk`` silently swallows errors when ``onerror`` is ``None``.  The
    scanner must keep skipping a vanished entry (``FileNotFoundError``) but
    must not mask ``PermissionError`` or any other filesystem failure.
    """
    if isinstance(error, FileNotFoundError):
        return
    raise error


def tree_bytes(root: str | os.PathLike[str]) -> int:
    """Return a conservative count of regular-file bytes under ``root``.

    The root must not be a symlink; a symlink root is refused rather than
    silently traversed.  A missing root yields ``0`` and a non-directory root
    yields ``0``, matching the previous behaviour.  Individual entries that
    vanish between the directory walk and ``lstat`` are skipped, symlinks are
    rejected, and real hard links (``st_nlink > 1``) are rejected.  A regular
    inode observed with zero remaining links is the legal result of a
    concurrent atomic replacement and is counted using its observed size.
    """
    raw_root = os.fspath(root)
    if isinstance(raw_root, bytes):
        raw_root = os.fsdecode(raw_root)
    root_path = Path(raw_root)

    try:
        root_info = root_path.lstat()
    except FileNotFoundError:
        return 0
    if stat.S_ISLNK(root_info.st_mode):
        raise ValueError("Unexpected symlink in run output.")
    if not stat.S_ISDIR(root_info.st_mode):
        return 0

    total = 0
    for directory, dirs, files in os.walk(root_path, followlinks=False, onerror=_walk_error):
        for name in list(dirs) + list(files):
            path = Path(directory) / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("Unexpected symlink in run output.")
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink < 0:
                    raise ValueError("Unexpected negative link count.")
                if info.st_nlink > 1:
                    raise ValueError("Unexpected hard link in run output.")
                total += info.st_size
    return total
