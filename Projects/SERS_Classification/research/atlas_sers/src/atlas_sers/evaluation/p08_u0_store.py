"""P08-T050 bounded POSIX U0 journal persistence and exclusive ownership.

This leaf joins exact-U0 prospective candidate checks to invented, hash-linked
metadata persisted under one cooperative controller lease.  It is not a model
runner, scientific permit, live monitor, checkpoint authenticator or recovery
lease.  Metadata success never implies receipt-file authenticity, wall-clock
freshness, real hardware readings or permission.  There is no automatic
recovery, retry, stale-lock removal, evidence deletion, truncation or repair.
Scientific execution entry always denies.

POSIX/Linux is the explicit supported platform.  Only the standard library and
the existing replay kernel plus admission guard are used.  Roots must be
absolute, symlinks are never followed, metadata files are size-bounded and
hardlinked metadata is rejected.  The returned store is a cooperative,
single-coordinator journal writer; it proves nothing about durable storage,
live owners, clocks or scientific admissibility.
"""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import pathlib
import stat

from . import p08_u0_admission as admission
from .p08_attempt_journal import JournalError, replay_attempt_journal

__all__ = [
    "StoreError",
    "Store",
    "create_store",
    "open_store",
    "inspect_store",
    "require_scientific_execution",
]

_HEAD_SCHEMA = "nato-sers-p08-u0-head-v1"

_MANIFEST_NAME = "manifest.json"
_HEAD_NAME = "head.json"
_PENDING_NAME = "head.pending"
_EVENTS_NAME = "events"
_LOCK_NAME = ".lock"

_ALLOWED_ROOT_NAMES = frozenset({_MANIFEST_NAME, _HEAD_NAME, _EVENTS_NAME, _LOCK_NAME})
_MAX_ROOT_ENTRIES = len(_ALLOWED_ROOT_NAMES) + 1

MAX_EVENTS = 16384
MAX_MANIFEST_BYTES = 1048576
MAX_EVENT_BYTES = 65536
MAX_HEAD_BYTES = 4096

_REASON_CODES = frozenset(
    {
        "invalid_store_input",
        "invalid_path",
        "store_exists",
        "invalid_layout",
        "symlink_or_nonregular",
        "lease_unavailable",
        "invalid_json",
        "size_limit",
        "invalid_manifest",
        "invalid_journal",
        "head_mismatch",
        "incomplete_write",
        "incomplete_session_requires_review",
        "closed_store",
        "poisoned_store",
        "fresh_progress_required",
        "invalid_resources",
        "candidate_blocked",
        "storage_io_error",
        "scientific_execution_not_authorized",
    }
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_DIGITS = frozenset("0123456789")

_HEAD_KEYS = frozenset(
    {
        "schema_version",
        "manifest_sha256",
        "head_sha256",
        "event_count",
    }
)

_EVENT_KEYS = frozenset(
    {
        "seq",
        "previous_sha256",
        "session_id",
        "event_type",
        "elapsed_ns",
        "artifact_bytes",
        "job_id",
        "status",
        "receipt_sha256",
        "event_sha256",
    }
)


class StoreError(ValueError):
    """Static, data-free store failure exposing ``reason_code``."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_store_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code):
    raise StoreError(reason_code) from None


def _is_lower64(value):
    if type(value) is not str or len(value) != 64:
        return False
    for char in value:
        if char not in _HEX_DIGITS:
            return False
    return True


def _is_ascii_digits(value):
    if type(value) is not str or len(value) != 8:
        return False
    for char in value:
        if char not in _DIGITS:
            return False
    return True


def _has_exact_keys(obj, expected):
    if type(obj) is not dict:
        return False
    count = 0
    for key in obj:
        if type(key) is not str:
            return False
        count += 1
    if count != len(expected):
        return False
    return set(obj) == expected


def _close_fds(*fds):
    first_exc = None
    for fd in fds:
        if type(fd) is int and fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
            except BaseException as exc:
                if first_exc is None:
                    first_exc = exc
    if first_exc is not None:
        raise first_exc


# ---------------------------------------------------------------------------
# Canonical encoding and strict parsing
# ---------------------------------------------------------------------------


def _encode_json(value):
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (ValueError, TypeError, RecursionError):
        raise StoreError("invalid_json") from None
    try:
        return (text + "\n").encode("utf-8")
    except UnicodeEncodeError:
        raise StoreError("invalid_json") from None


def _pairs_hook(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate key")
        obj[key] = value
    return obj


def _reject_constant(_token):
    raise ValueError("non-finite constant")


def _ensure_finite(value):
    if type(value) is float:
        if not math.isfinite(value):
            _fail("invalid_json")
        return
    if type(value) is dict:
        for item in value.values():
            _ensure_finite(item)
        return
    if type(value) is list:
        for item in value:
            _ensure_finite(item)
        return


def _parse_json(raw):
    try:
        text = raw.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        _fail("invalid_json")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_hook,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError, RecursionError):
        _fail("invalid_json")
    _ensure_finite(value)
    return value


# ---------------------------------------------------------------------------
# Path handling (no symlink following anywhere)
# ---------------------------------------------------------------------------


def _root_components(root):
    if isinstance(root, pathlib.PurePath):
        try:
            text = os.fspath(root)
        except TypeError:
            _fail("invalid_path")
        if type(text) is not str:
            _fail("invalid_path")
    elif type(root) is str:
        text = root
    else:
        _fail("invalid_path")

    if text == "" or "\x00" in text:
        _fail("invalid_path")
    if not text.startswith("/"):
        _fail("invalid_path")

    parts = text.split("/")
    components = parts[1:]
    if not components:
        _fail("invalid_path")
    for component in components:
        if component == "" or component == "." or component == "..":
            _fail("invalid_path")
    return components


def _resolve_parent(root):
    components = _root_components(root)
    root_name = components[-1]
    dirs = components[:-1]
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        _fail("storage_io_error")
    try:
        for component in dirs:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    _fail("symlink_or_nonregular")
                if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                    _fail("invalid_path")
                _fail("storage_io_error")
            old_fd = fd
            fd = next_fd
            _close_fds(old_fd)
    except BaseException:
        _close_fds(fd)
        raise
    return fd, root_name


# ---------------------------------------------------------------------------
# Bounded file IO relative to held descriptors
# ---------------------------------------------------------------------------


def _read_file(dir_fd, name, max_bytes):
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            _fail("symlink_or_nonregular")
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.EISDIR):
            _fail("invalid_layout")
        _fail("storage_io_error")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _fail("symlink_or_nonregular")
        initial_size = info.st_size
        if initial_size > max_bytes:
            _fail("size_limit")
        chunks = []
        total = 0
        read_limit = max_bytes + 1
        while total < read_limit:
            chunk = os.read(fd, min(65536, read_limit - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > max_bytes:
            _fail("size_limit")
        final = os.fstat(fd)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or final.st_size != initial_size
            or total != initial_size
        ):
            _fail("invalid_layout")
        return b"".join(chunks)
    except OSError:
        _fail("storage_io_error")
    finally:
        os.close(fd)


def _write_all(fd, data):
    view = memoryview(data)
    total = 0
    while total < len(view):
        written = os.write(fd, view[total:])
        if written <= 0:
            raise OSError(errno.EIO, "write failed")
        total += written


def _write_exclusive(dir_fd, name, data):
    fd = os.open(
        name,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
        0o600,
        dir_fd=dir_fd,
    )
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Directory and lock helpers
# ---------------------------------------------------------------------------


def _list_dir_bounded(dir_fd, limit):
    names = set()
    try:
        with os.scandir(dir_fd) as entries:
            for entry in entries:
                names.add(entry.name)
                if len(names) > limit:
                    _fail("size_limit")
    except OSError:
        _fail("invalid_layout")
    return names


def _open_dir(dir_fd, name):
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            _fail("symlink_or_nonregular")
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            _fail("invalid_layout")
        _fail("storage_io_error")


def _acquire_lock(lock_fd):
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            _fail("lease_unavailable")
        _fail("storage_io_error")


def _create_lock(root_fd):
    try:
        fd = os.open(
            _LOCK_NAME,
            os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            _fail("invalid_layout")
        _fail("storage_io_error")
    try:
        try:
            info = os.fstat(fd)
        except OSError:
            _fail("storage_io_error")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _fail("symlink_or_nonregular")
        _acquire_lock(fd)
        try:
            os.fsync(fd)
        except OSError:
            _fail("storage_io_error")
    except BaseException:
        _close_fds(fd)
        raise
    return fd


def _open_lock(root_fd):
    try:
        fd = os.open(
            _LOCK_NAME,
            os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=root_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            _fail("symlink_or_nonregular")
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.EISDIR):
            _fail("invalid_layout")
        _fail("storage_io_error")
    try:
        try:
            info = os.fstat(fd)
        except OSError:
            _fail("storage_io_error")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _fail("symlink_or_nonregular")
    except BaseException:
        _close_fds(fd)
        raise
    return fd


def _verify_identities(parent_fd, root_name, root_fd, events_fd, lock_fd):
    try:
        root_info = os.fstat(root_fd)
        entry = os.stat(root_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        _fail("invalid_layout")
    if not stat.S_ISDIR(entry.st_mode):
        _fail("invalid_layout")
    if (entry.st_dev, entry.st_ino) != (root_info.st_dev, root_info.st_ino):
        _fail("invalid_layout")

    try:
        events_info = os.fstat(events_fd)
        entry = os.stat(_EVENTS_NAME, dir_fd=root_fd, follow_symlinks=False)
    except OSError:
        _fail("invalid_layout")
    if not stat.S_ISDIR(entry.st_mode):
        _fail("invalid_layout")
    if (entry.st_dev, entry.st_ino) != (events_info.st_dev, events_info.st_ino):
        _fail("invalid_layout")

    try:
        lock_info = os.fstat(lock_fd)
        entry = os.stat(_LOCK_NAME, dir_fd=root_fd, follow_symlinks=False)
    except OSError:
        _fail("invalid_layout")
    if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
        _fail("symlink_or_nonregular")
    if (entry.st_dev, entry.st_ino) != (lock_info.st_dev, lock_info.st_ino):
        _fail("invalid_layout")


# ---------------------------------------------------------------------------
# Head and coherent loading
# ---------------------------------------------------------------------------


def _make_head(head_sha256, event_count):
    return {
        "schema_version": _HEAD_SCHEMA,
        "manifest_sha256": admission.U0_MANIFEST_SHA256,
        "head_sha256": head_sha256,
        "event_count": event_count,
    }


def _validate_head(head, expected_head):
    if not _has_exact_keys(head, _HEAD_KEYS):
        _fail("invalid_layout")
    if head["schema_version"] != _HEAD_SCHEMA:
        _fail("invalid_layout")
    if head["manifest_sha256"] != admission.U0_MANIFEST_SHA256:
        _fail("head_mismatch")
    if not _is_lower64(head["head_sha256"]):
        _fail("invalid_layout")
    count = head["event_count"]
    if type(count) is not int or count < 0:
        _fail("invalid_layout")
    if count > MAX_EVENTS:
        _fail("size_limit")
    if not _is_lower64(expected_head):
        _fail("invalid_store_input")
    if head["head_sha256"] != expected_head:
        _fail("head_mismatch")


def _replay_error(reason_code):
    if reason_code in ("invalid_manifest", "manifest_hash_mismatch"):
        _fail("invalid_manifest")
    if reason_code == "journal_head_mismatch":
        _fail("head_mismatch")
    _fail("invalid_journal")


def _validate_new_manifest(manifest):
    if type(manifest) is not dict:
        _fail("invalid_manifest")
    try:
        replay_attempt_journal(
            manifest,
            [],
            expected_manifest_sha256=admission.U0_MANIFEST_SHA256,
            expected_head_sha256=admission.U0_MANIFEST_SHA256,
        )
    except JournalError as exc:
        _replay_error(getattr(exc, "reason_code", ""))
    if manifest.get("proposal_sha256") != admission.U0_PROPOSAL_SHA256:
        _fail("invalid_manifest")
    try:
        encoded = _encode_json(manifest)
    except StoreError:
        _fail("invalid_manifest")
    if len(encoded) > MAX_MANIFEST_BYTES:
        _fail("size_limit")


def _load_state(root_fd, events_fd, expected_head):
    names = _list_dir_bounded(root_fd, _MAX_ROOT_ENTRIES)
    if _PENDING_NAME in names:
        _fail("incomplete_write")
    if names != _ALLOWED_ROOT_NAMES:
        _fail("invalid_layout")

    raw = _read_file(root_fd, _MANIFEST_NAME, MAX_MANIFEST_BYTES)
    manifest = _parse_json(raw)
    if type(manifest) is not dict:
        _fail("invalid_manifest")

    raw = _read_file(root_fd, _HEAD_NAME, MAX_HEAD_BYTES)
    head = _parse_json(raw)
    if type(head) is not dict:
        _fail("invalid_layout")
    _validate_head(head, expected_head)
    event_count = head["event_count"]

    event_names = _list_dir_bounded(events_fd, MAX_EVENTS)
    expected_names = set()
    for index in range(1, event_count + 1):
        expected_names.add(f"{index:08d}.json")
    for entry in event_names:
        if (
            type(entry) is not str
            or len(entry) != 13
            or entry[8:] != ".json"
            or not _is_ascii_digits(entry[:8])
        ):
            _fail("invalid_layout")
    if event_names != expected_names:
        _fail("incomplete_write")

    events = []
    for index in range(1, event_count + 1):
        raw = _read_file(events_fd, f"{index:08d}.json", MAX_EVENT_BYTES)
        events.append(_parse_json(raw))

    try:
        summary = replay_attempt_journal(
            manifest,
            events,
            expected_manifest_sha256=admission.U0_MANIFEST_SHA256,
            expected_head_sha256=head["head_sha256"],
        )
    except JournalError as exc:
        _replay_error(getattr(exc, "reason_code", ""))

    if manifest.get("proposal_sha256") != admission.U0_PROPOSAL_SHA256:
        _fail("invalid_manifest")
    if summary["head_sha256"] != head["head_sha256"]:
        _fail("head_mismatch")
    if summary["manifest_sha256"] != head["manifest_sha256"]:
        _fail("invalid_journal")
    if head["event_count"] != len(events):
        _fail("invalid_journal")

    return {"manifest": manifest, "events": events, "summary": summary}


# ---------------------------------------------------------------------------
# Store object
# ---------------------------------------------------------------------------


class Store:
    """Single-coordinator journal writer holding one exclusive lease."""

    def __init__(self, parent_fd, root_fd, events_fd, lock_fd, root_name, accepted_head):
        self._parent_fd = parent_fd
        self._root_fd = root_fd
        self._events_fd = events_fd
        self._lock_fd = lock_fd
        self._root_name = root_name
        self._accepted_head = accepted_head
        self._closed = False
        self._poisoned = False

    def _require_usable(self):
        if self._closed:
            _fail("closed_store")
        if self._poisoned:
            _fail("poisoned_store")

    def _poison(self):
        self._poisoned = True

    def _reload(self, expected_head):
        try:
            _verify_identities(
                self._parent_fd,
                self._root_name,
                self._root_fd,
                self._events_fd,
                self._lock_fd,
            )
            return _load_state(self._root_fd, self._events_fd, expected_head)
        except BaseException:
            self._poison()
            raise

    def snapshot(self):
        """Return a fresh coherent ``{manifest, events, summary}`` dict."""
        try:
            self._require_usable()
            return self._reload(self._accepted_head)
        except StoreError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            raise StoreError("storage_io_error") from None
        except Exception:
            raise StoreError("invalid_store_input") from None

    def _check_candidate(self, manifest, events, resources, job_id, current_head):
        try:
            candidate = admission.evaluate_u0_candidate(
                manifest,
                events,
                resources,
                job_id=job_id,
                expected_head_sha256=current_head,
            )
        except admission.AdmissionError as exc:
            code = getattr(exc, "reason_code", None)
            if code == "invalid_resources":
                _fail("invalid_resources")
            if code == "invalid_journal":
                _fail("invalid_journal")
            _fail("candidate_blocked")
        except (KeyboardInterrupt, SystemExit):
            raise
        if candidate.get("proposed_candidate_admissible") is not True:
            _fail("candidate_blocked")

    def append_event(self, event, *, expected_head_sha256, resources=None):
        """Validate and durably append one invented event; return its summary."""
        try:
            return self._append_event(event, expected_head_sha256, resources)
        except StoreError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            raise StoreError("storage_io_error") from None
        except Exception:
            raise StoreError("invalid_store_input") from None

    def _append_event(self, event, expected_head_sha256, resources):
        self._require_usable()
        if not _is_lower64(expected_head_sha256):
            _fail("invalid_store_input")

        state = self._reload(self._accepted_head)
        summary = state["summary"]
        manifest = state["manifest"]
        events = state["events"]
        current_head = summary["head_sha256"]
        if expected_head_sha256 != current_head:
            _fail("head_mismatch")

        if not _has_exact_keys(event, _EVENT_KEYS):
            _fail("invalid_journal")
        declared = event["event_sha256"]
        if not _is_lower64(declared):
            _fail("invalid_journal")
        if len(events) + 1 > MAX_EVENTS:
            _fail("size_limit")

        try:
            encoded = _encode_json(event)
        except StoreError:
            _fail("invalid_journal")
        if len(encoded) > MAX_EVENT_BYTES:
            _fail("size_limit")

        try:
            replay_attempt_journal(
                manifest,
                events + [event],
                expected_manifest_sha256=admission.U0_MANIFEST_SHA256,
                expected_head_sha256=declared,
            )
        except JournalError as exc:
            _replay_error(getattr(exc, "reason_code", ""))

        if event["event_type"] == "attempt_start":
            if resources is None:
                _fail("invalid_resources")
            if event["elapsed_ns"] != summary["active_session_elapsed_ns"]:
                _fail("fresh_progress_required")
            if event["artifact_bytes"] != summary["new_artifact_bytes"]:
                _fail("fresh_progress_required")
            self._check_candidate(manifest, events, resources, event["job_id"], current_head)
        elif resources is not None:
            _fail("invalid_resources")

        next_index = len(events) + 1
        event_name = f"{next_index:08d}.json"
        try:
            head_bytes = _encode_json(_make_head(declared, next_index))
        except StoreError:
            _fail("invalid_journal")
        if len(head_bytes) > MAX_HEAD_BYTES:
            _fail("size_limit")

        try:
            _write_exclusive(self._events_fd, event_name, encoded)
            os.fsync(self._events_fd)
            _write_exclusive(self._root_fd, _PENDING_NAME, head_bytes)
            os.replace(
                _PENDING_NAME,
                _HEAD_NAME,
                src_dir_fd=self._root_fd,
                dst_dir_fd=self._root_fd,
            )
            os.fsync(self._root_fd)
        except FileExistsError:
            self._poison()
            _fail("incomplete_write")
        except OSError:
            self._poison()
            _fail("storage_io_error")
        except BaseException:
            self._poison()
            raise

        self._accepted_head = declared
        try:
            fresh = self._reload(declared)
        except BaseException:
            self._poison()
            raise
        return fresh["summary"]

    def close(self):
        """Release the descriptor and lease; never writes a close event."""
        if self._closed:
            return
        self._closed = True
        try:
            _close_fds(
                self._events_fd,
                self._lock_fd,
                self._root_fd,
                self._parent_fd,
            )
        finally:
            self._events_fd = -1
            self._lock_fd = -1
            self._root_fd = -1
            self._parent_fd = -1

    def __enter__(self):
        self._require_usable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def _create_store(root, manifest):
    _validate_new_manifest(manifest)
    parent_fd, root_name = _resolve_parent(root)
    root_fd = -1
    events_fd = -1
    lock_fd = -1
    try:
        try:
            os.mkdir(root_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            _fail("store_exists")
        except OSError:
            _fail("storage_io_error")
        try:
            os.fsync(parent_fd)
        except OSError:
            _fail("storage_io_error")
        root_fd = _open_dir(parent_fd, root_name)
        lock_fd = _create_lock(root_fd)
        manifest_bytes = _encode_json(manifest)
        _write_exclusive(root_fd, _MANIFEST_NAME, manifest_bytes)
        try:
            os.mkdir(_EVENTS_NAME, 0o700, dir_fd=root_fd)
        except OSError:
            _fail("storage_io_error")
        events_fd = _open_dir(root_fd, _EVENTS_NAME)
        head_bytes = _encode_json(_make_head(admission.U0_MANIFEST_SHA256, 0))
        _write_exclusive(root_fd, _HEAD_NAME, head_bytes)
        try:
            os.fsync(events_fd)
            os.fsync(root_fd)
        except OSError:
            _fail("storage_io_error")
        store = Store(
            parent_fd,
            root_fd,
            events_fd,
            lock_fd,
            root_name,
            admission.U0_MANIFEST_SHA256,
        )
    except BaseException as exc:
        _close_fds(root_fd, events_fd, lock_fd, parent_fd)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, StoreError):
            raise
        if isinstance(exc, OSError):
            raise StoreError("storage_io_error") from None
        raise

    try:
        store._reload(admission.U0_MANIFEST_SHA256)
    except BaseException:
        store.close()
        raise
    return store


def _open_store(root, expected_head):
    if not _is_lower64(expected_head):
        _fail("invalid_store_input")
    parent_fd, root_name = _resolve_parent(root)
    root_fd = -1
    events_fd = -1
    lock_fd = -1
    try:
        root_fd = _open_dir(parent_fd, root_name)
        lock_fd = _open_lock(root_fd)
        _acquire_lock(lock_fd)
        events_fd = _open_dir(root_fd, _EVENTS_NAME)
        _verify_identities(parent_fd, root_name, root_fd, events_fd, lock_fd)
        state = _load_state(root_fd, events_fd, expected_head)
        if state["summary"]["journal_state"] not in ("not_started", "closed"):
            _fail("incomplete_session_requires_review")
        return Store(
            parent_fd,
            root_fd,
            events_fd,
            lock_fd,
            root_name,
            expected_head,
        )
    except BaseException as exc:
        _close_fds(root_fd, events_fd, lock_fd, parent_fd)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, StoreError):
            raise
        if isinstance(exc, OSError):
            raise StoreError("storage_io_error") from None
        raise


def _inspect_store(root, expected_head):
    if not _is_lower64(expected_head):
        _fail("invalid_store_input")
    parent_fd, root_name = _resolve_parent(root)
    root_fd = -1
    events_fd = -1
    lock_fd = -1
    try:
        root_fd = _open_dir(parent_fd, root_name)
        lock_fd = _open_lock(root_fd)
        _acquire_lock(lock_fd)
        events_fd = _open_dir(root_fd, _EVENTS_NAME)
        _verify_identities(parent_fd, root_name, root_fd, events_fd, lock_fd)
        return _load_state(root_fd, events_fd, expected_head)
    finally:
        _close_fds(root_fd, events_fd, lock_fd, parent_fd)


def create_store(root, manifest):
    """Create a new store root and hold its exclusive controller lease."""
    try:
        return _create_store(root, manifest)
    except StoreError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except OSError:
        raise StoreError("storage_io_error") from None
    except Exception:
        raise StoreError("invalid_store_input") from None


def open_store(root, *, expected_head_sha256):
    """Take ownership of a coherent not_started/closed store root."""
    try:
        return _open_store(root, expected_head_sha256)
    except StoreError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except OSError:
        raise StoreError("storage_io_error") from None
    except Exception:
        raise StoreError("invalid_store_input") from None


def inspect_store(root, *, expected_head_sha256):
    """Read a coherent store under a temporary exclusive lease, then release."""
    try:
        return _inspect_store(root, expected_head_sha256)
    except StoreError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except OSError:
        raise StoreError("storage_io_error") from None
    except Exception:
        raise StoreError("invalid_store_input") from None


def require_scientific_execution(*args, **kwargs):
    """Always deny: this leaf grants no scientific execution authority."""
    raise StoreError("scientific_execution_not_authorized") from None
