"""P08-T116 source-session IO persistence half.

This leaf implements the exclusive, single-coordinator IO surface used by the
integrated P08 source-session controller: it owns one private flat artifact
root, appends session/progress/start/finish/close events through an existing
``p08_u0_store.Store``, and verifies terminal receipt bytes with the existing
read-only terminal verifier.  It is not a model runner, live monitor, permit,
checkpoint authenticator or recovery lease.  Byte integrity never implies
scientific semantics, freshness, a live owner or permission.  There is no
retry, deletion, repair, truncation or hidden fit.  Scientific execution
always denies.  POSIX/Linux only; roots are absolute, symlinks are never
followed and metadata is size-bounded.

Accounting covers all three roots (journal, artifact root, launch-control
directory) only while ``launch.json`` is the sole fixed control file.  Any
terminal record written after session close is owned and charged by the
caller/outer controller; the byte pin is not a permit.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time

from . import p08_resources as resources_guard
from . import p08_u0_admission as admission
from . import p08_u0_store as store
from .p08_qc_blocks import canonical_sha256
from .p08_terminal_receipt import ReceiptError, verify_terminal_receipt

__all__ = [
    "SessionIOError",
    "_SessionIO",
    "require_scientific_execution",
]

_RECEIPT_SCHEMA = "nato-sers-p08-terminal-receipt-v1"

_MAX_ARTIFACTS = 8
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 65536
_MAX_LAUNCH_RECORD_BYTES = 65536
_MAX_ARTIFACT_ENTRIES = 1024
_FRESH_NS = 10**9
_MAX_FIXED_POINT = 16

_LAUNCH_RECORD_NAME = "launch.json"
_HEX64_PATTERN = re.compile(r"[0-9a-f]{64}")

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "interrupted"})

_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")

_RESERVED_NAMES = frozenset(
    {
        store._MANIFEST_NAME,
        store._HEAD_NAME,
        store._PENDING_NAME,
        store._EVENTS_NAME,
        store._LOCK_NAME,
    }
)

_REASON_CODES = frozenset(
    {
        "invalid_input",
        "not_owned",
        "invalid_path",
        "owner_unavailable",
        "state_conflict",
        "parent_mismatch",
        "artifact_exists",
        "layout_invalid",
        "symlink_or_nonregular",
        "io_error",
        "identity_mismatch",
        "storage_limited",
        "head_mismatch",
        "pending_required",
        "stale_measurement",
        "invalid_artifact",
        "incomplete_write",
        "receipt_error",
        "not_running",
        "artifact_mismatch",
        "control_mismatch",
        "running_job",
        "closed",
        "poisoned",
        "scientific_execution_not_authorized",
    }
)


class SessionIOError(ValueError):
    """Static, data-free session IO failure exposing ``reason_code``."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code):
    raise SessionIOError(reason_code) from None


def _is_safe_name(value):
    if type(value) is not str:
        return False
    return _NAME_PATTERN.fullmatch(value) is not None


def _event_spec(event_type, job_id=None, status=None, receipt_sha256=None):
    return {
        "event_type": event_type,
        "job_id": job_id,
        "status": status,
        "receipt_sha256": receipt_sha256,
    }


def _build_event(
    session_id,
    event_type,
    elapsed_ns,
    artifact_bytes,
    job_id,
    status,
    receipt_sha256,
    seq,
    previous_sha256,
):
    event = {
        "seq": seq,
        "previous_sha256": previous_sha256,
        "session_id": session_id,
        "event_type": event_type,
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
        "job_id": job_id,
        "status": status,
        "receipt_sha256": receipt_sha256,
    }
    event["event_sha256"] = canonical_sha256(event)
    return event


class _SessionIO:
    """One exclusive artifact root bound to one coherent store session.

    Observed bytes cover the journal, the artifact root and the launch-control
    directory only while ``launch.json`` is that directory's sole fixed file.
    A terminal record written after close belongs to the caller and is charged
    separately; the byte pin is not a permit.
    """

    def __init__(
        self,
        owner,
        artifact_root,
        *,
        started_monotonic_ns,
        launch_control_root=None,
        launch_record_sha256=None,
    ):
        self._owner = None
        self.artifact_root = None
        self._root_name = None
        self._parent_fd = -1
        self._root_fd = -1
        self._control_fd = -1
        self._control_name = None
        self._control_info = None
        self._control_sha256 = None
        self._control_bytes = 0
        self._started_ns = 0
        self._closed = False
        self._poisoned = False
        self._session_id = None
        self._pending = None
        self._running = None
        try:
            self._initialize(
                owner,
                artifact_root,
                started_monotonic_ns,
                launch_control_root,
                launch_record_sha256,
            )
        except BaseException as exc:
            try:
                store._close_fds(
                    self._root_fd, self._parent_fd, self._control_fd
                )
            except BaseException:
                pass
            self._root_fd = -1
            self._parent_fd = -1
            self._control_fd = -1
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, SessionIOError):
                raise
            if isinstance(exc, store.StoreError):
                raise SessionIOError("owner_unavailable") from None
            if isinstance(exc, OSError):
                raise SessionIOError("io_error") from None
            raise SessionIOError("invalid_input") from None

    def __repr__(self):
        return "<_SessionIO>"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _initialize(
        self,
        owner,
        artifact_root,
        started_monotonic_ns,
        launch_control_root,
        launch_record_sha256,
    ):
        if type(owner) is not store.Store:
            _fail("not_owned")
        if type(started_monotonic_ns) is not int or started_monotonic_ns < 0:
            _fail("invalid_input")
        if started_monotonic_ns > time.monotonic_ns():
            _fail("invalid_input")
        if (launch_control_root is None) != (launch_record_sha256 is None):
            _fail("invalid_input")
        if launch_control_root is not None:
            if type(launch_control_root) is not str or launch_control_root == "":
                _fail("invalid_path")
            if (
                type(launch_record_sha256) is not str
                or _HEX64_PATTERN.fullmatch(launch_record_sha256) is None
            ):
                _fail("invalid_input")
        try:
            state = owner.snapshot()
        except store.StoreError:
            _fail("owner_unavailable")
        if type(state) is not dict:
            _fail("owner_unavailable")
        summary = state.get("summary")
        if type(summary) is not dict or summary.get("journal_state") != "not_started":
            _fail("state_conflict")

        try:
            parent_fd, root_name = store._resolve_parent(artifact_root)
        except store.StoreError as exc:
            if getattr(exc, "reason_code", "") == "invalid_path":
                _fail("invalid_path")
            _fail("io_error")
        except OSError:
            _fail("io_error")

        self._owner = owner
        self.artifact_root = artifact_root
        self._parent_fd = parent_fd
        self._root_name = root_name
        self._started_ns = started_monotonic_ns

        try:
            ours = os.fstat(parent_fd)
            theirs = os.fstat(owner._parent_fd)
        except OSError:
            _fail("parent_mismatch")
        if (ours.st_dev, ours.st_ino) != (theirs.st_dev, theirs.st_ino):
            _fail("parent_mismatch")
        if root_name == getattr(owner, "_root_name", None):
            _fail("artifact_exists")

        if launch_control_root is not None:
            self._initialize_control(
                parent_fd, root_name, launch_control_root, launch_record_sha256
            )

        try:
            os.mkdir(root_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            _fail("artifact_exists")
        except OSError:
            _fail("io_error")
        try:
            os.fsync(parent_fd)
        except OSError:
            _fail("io_error")
        try:
            self._root_fd = store._open_dir(parent_fd, root_name)
        except store.StoreError as exc:
            if getattr(exc, "reason_code", "") == "symlink_or_nonregular":
                _fail("symlink_or_nonregular")
            _fail("io_error")
        self._check_identity()

    # ------------------------------------------------------------------
    # Shared integrity guards
    # ------------------------------------------------------------------

    def _poison(self):
        self._poisoned = True

    def _mutate(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            self._poison()
            raise

    def _require_usable(self):
        if self._closed:
            _fail("closed")
        if self._poisoned:
            _fail("poisoned")

    def _guard(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SessionIOError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except store.StoreError:
            self._poison()
            _fail("owner_unavailable")
        except ReceiptError:
            _fail("receipt_error")
        except OSError:
            self._poison()
            _fail("io_error")
        except Exception:
            _fail("invalid_input")

    def _fresh_state(self):
        try:
            state = self._owner.snapshot()
        except store.StoreError:
            self._poison()
            _fail("owner_unavailable")
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            self._poison()
            _fail("io_error")
        if type(state) is not dict:
            self._poison()
            _fail("owner_unavailable")
        return state

    def _check_identity(self):
        try:
            parent_info = os.fstat(self._parent_fd)
            owned_parent = os.fstat(self._owner._parent_fd)
        except OSError:
            self._poison()
            _fail("identity_mismatch")
        if (parent_info.st_dev, parent_info.st_ino) != (
            owned_parent.st_dev,
            owned_parent.st_ino,
        ):
            self._poison()
            _fail("identity_mismatch")
        try:
            root_info = os.fstat(self._root_fd)
            entry = os.stat(
                self._root_name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            self._poison()
            _fail("identity_mismatch")
        if not stat.S_ISDIR(entry.st_mode):
            self._poison()
            _fail("identity_mismatch")
        if (entry.st_dev, entry.st_ino) != (root_info.st_dev, root_info.st_ino):
            self._poison()
            _fail("identity_mismatch")
        if self._control_fd >= 0:
            try:
                control_info = os.fstat(self._control_fd)
                journal_info = os.fstat(self._owner._root_fd)
            except OSError:
                self._poison()
                _fail("identity_mismatch")
            control_id = (control_info.st_dev, control_info.st_ino)
            if control_id == (journal_info.st_dev, journal_info.st_ino):
                self._poison()
                _fail("identity_mismatch")
            if control_id == (root_info.st_dev, root_info.st_ino):
                self._poison()
                _fail("identity_mismatch")
        self._check_control_identity()

    def _map_store_error(self, exc):
        code = getattr(exc, "reason_code", "")
        if code in ("incomplete_write", "storage_io_error"):
            self._poison()
            _fail("io_error")
        if code == "head_mismatch":
            self._poison()
            _fail("head_mismatch")
        if code == "fresh_progress_required":
            _fail("head_mismatch")
        if code == "size_limit":
            _fail("storage_limited")
        if code in ("symlink_or_nonregular", "invalid_layout"):
            self._poison()
            _fail("layout_invalid")
        _fail("invalid_input")

    # ------------------------------------------------------------------
    # Bounded observation
    # ------------------------------------------------------------------

    def _scan_journal(self):
        total = 0
        try:
            with os.scandir(self._owner._root_fd) as entries:
                for entry in entries:
                    name = entry.name
                    try:
                        info = os.stat(
                            name,
                            dir_fd=self._owner._root_fd,
                            follow_symlinks=False,
                        )
                    except OSError:
                        self._poison()
                        _fail("io_error")
                    if stat.S_ISDIR(info.st_mode):
                        if name != store._EVENTS_NAME:
                            self._poison()
                            _fail("layout_invalid")
                        continue
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        self._poison()
                        _fail("symlink_or_nonregular")
                    total += info.st_size
        except OSError:
            self._poison()
            _fail("io_error")
        count = 0
        try:
            with os.scandir(self._owner._events_fd) as entries:
                for entry in entries:
                    count += 1
                    if count > store.MAX_EVENTS:
                        self._poison()
                        _fail("layout_invalid")
                    try:
                        info = os.stat(
                            entry.name,
                            dir_fd=self._owner._events_fd,
                            follow_symlinks=False,
                        )
                    except OSError:
                        self._poison()
                        _fail("io_error")
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        self._poison()
                        _fail("symlink_or_nonregular")
                    total += info.st_size
        except OSError:
            self._poison()
            _fail("io_error")
        return total

    def _scan_artifacts(self):
        total = 0
        count = 0
        try:
            with os.scandir(self._root_fd) as entries:
                for entry in entries:
                    count += 1
                    if count > _MAX_ARTIFACT_ENTRIES:
                        self._poison()
                        _fail("layout_invalid")
                    try:
                        info = os.stat(
                            entry.name,
                            dir_fd=self._root_fd,
                            follow_symlinks=False,
                        )
                    except OSError:
                        self._poison()
                        _fail("io_error")
                    if stat.S_ISDIR(info.st_mode):
                        self._poison()
                        _fail("layout_invalid")
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        self._poison()
                        _fail("symlink_or_nonregular")
                    total += info.st_size
        except OSError:
            self._poison()
            _fail("io_error")
        return total

    # ------------------------------------------------------------------
    # Launch control directory (read-only, authenticated, never created)
    # ------------------------------------------------------------------

    def _initialize_control(self, parent_fd, root_name, control_root, expected_sha):
        owner_root_name = getattr(self._owner, "_root_name", None)
        try:
            control_parent_fd, control_name = store._resolve_parent(control_root)
        except store.StoreError as exc:
            if getattr(exc, "reason_code", "") == "invalid_path":
                _fail("invalid_path")
            _fail("io_error")
        except OSError:
            _fail("io_error")
        primary_error = None
        try:
            try:
                theirs = os.fstat(control_parent_fd)
                ours = os.fstat(parent_fd)
            except OSError:
                _fail("parent_mismatch")
            if (theirs.st_dev, theirs.st_ino) != (ours.st_dev, ours.st_ino):
                _fail("parent_mismatch")
            if control_name == root_name or (
                owner_root_name is not None and control_name == owner_root_name
            ):
                _fail("artifact_exists")
            try:
                self._control_fd = store._open_dir(parent_fd, control_name)
            except store.StoreError as exc:
                if getattr(exc, "reason_code", "") == "symlink_or_nonregular":
                    _fail("symlink_or_nonregular")
                _fail("io_error")
            except OSError:
                _fail("io_error")
            self._control_name = control_name
            payload, info = self._read_control_payload()
            try:
                digest = hashlib.sha256(payload).hexdigest()
            except Exception:
                self._poison()
                _fail("control_mismatch")
            if digest != expected_sha:
                self._poison()
                _fail("control_mismatch")
            self._control_info = info
            self._control_sha256 = expected_sha
            self._control_bytes = len(payload)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                os.close(control_parent_fd)
            except OSError:
                pass
            except BaseException:
                # Preserve the primary failure through cleanup: a close-time
                # interruption must not mask an in-flight BaseException from
                # the control read.  With no primary failure the close
                # interruption is the failure and propagates unchanged.
                if primary_error is None:
                    raise

    def _check_control_identity(self):
        if self._control_fd < 0:
            return
        try:
            info = os.fstat(self._control_fd)
            entry = os.stat(
                self._control_name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            self._poison()
            _fail("identity_mismatch")
        if not stat.S_ISDIR(entry.st_mode):
            self._poison()
            _fail("identity_mismatch")
        if (entry.st_dev, entry.st_ino) != (info.st_dev, info.st_ino):
            self._poison()
            _fail("identity_mismatch")

    def _read_control_payload(self):
        # Immutable identity carries mtime/ctime alongside dev/ino/size so an
        # equal-length in-place rewrite that lands during observation is caught
        # as an accidental concurrent change.  This is not a defence against a
        # malicious kernel or interpreter and does not claim to be one.
        self._check_control_identity()
        try:
            before = store._list_dir_bounded(self._control_fd, 1)
        except store.StoreError:
            self._poison()
            _fail("control_mismatch")
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            self._poison()
            _fail("io_error")
        if before != {_LAUNCH_RECORD_NAME}:
            self._poison()
            _fail("control_mismatch")
        try:
            info = os.stat(
                _LAUNCH_RECORD_NAME,
                dir_fd=self._control_fd,
                follow_symlinks=False,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            self._poison()
            _fail("io_error")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            self._poison()
            _fail("symlink_or_nonregular")
        if info.st_size <= 0 or info.st_size > _MAX_LAUNCH_RECORD_BYTES:
            self._poison()
            _fail("control_mismatch")
        info_identity = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        recorded = self._control_info
        if recorded is not None and info_identity != recorded:
            self._poison()
            _fail("control_mismatch")
        try:
            payload = store._read_file(
                self._control_fd, _LAUNCH_RECORD_NAME, _MAX_LAUNCH_RECORD_BYTES
            )
        except store.StoreError as exc:
            if getattr(exc, "reason_code", "") == "symlink_or_nonregular":
                self._poison()
                _fail("symlink_or_nonregular")
            self._poison()
            _fail("control_mismatch")
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            self._poison()
            _fail("io_error")
        self._check_control_identity()
        try:
            after = store._list_dir_bounded(self._control_fd, 1)
        except store.StoreError:
            self._poison()
            _fail("control_mismatch")
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            self._poison()
            _fail("io_error")
        if after != {_LAUNCH_RECORD_NAME}:
            self._poison()
            _fail("control_mismatch")
        try:
            post = os.stat(
                _LAUNCH_RECORD_NAME,
                dir_fd=self._control_fd,
                follow_symlinks=False,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            self._poison()
            _fail("io_error")
        if not stat.S_ISREG(post.st_mode) or post.st_nlink != 1:
            self._poison()
            _fail("symlink_or_nonregular")
        post_identity = (
            post.st_dev,
            post.st_ino,
            post.st_size,
            post.st_mtime_ns,
            post.st_ctime_ns,
        )
        if post_identity != info_identity:
            self._poison()
            _fail("control_mismatch")
        if recorded is not None and post_identity != recorded:
            self._poison()
            _fail("control_mismatch")
        if type(payload) is not bytes or len(payload) != post.st_size:
            self._poison()
            _fail("control_mismatch")
        return payload, info_identity

    def _scan_control(self):
        if self._control_fd < 0:
            return 0
        payload, _ = self._read_control_payload()
        try:
            digest = hashlib.sha256(payload).hexdigest()
        except Exception:
            self._poison()
            _fail("control_mismatch")
        if digest != self._control_sha256:
            self._poison()
            _fail("control_mismatch")
        return len(payload)

    def _observed(self):
        self._require_usable()
        self._check_identity()
        self._fresh_state()
        return (
            self._scan_journal()
            + self._scan_artifacts()
            + self._scan_control()
        )

    # ------------------------------------------------------------------
    # Storage ceiling gate (proposed, unapproved U0 guard values)
    # ------------------------------------------------------------------

    def _limits(self):
        try:
            limits = resources_guard.proposed_limits("U0")
        except resources_guard.ResourceGuardError:
            _fail("invalid_input")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_input")
        return limits["filesystem_reserve_bytes"], limits["new_artifact_bytes"]

    def _storage_gate(self, state, growth):
        reserve, ceiling = self._limits()
        summary = state.get("summary") if type(state) is dict else None
        previous = summary.get("new_artifact_bytes") if type(summary) is dict else None
        if type(previous) is not int or previous < 0:
            self._poison()
            _fail("owner_unavailable")
        observed = self._observed()
        try:
            info = os.fstatvfs(self._root_fd)
            free = info.f_bavail * info.f_frsize
        except OSError:
            self._poison()
            _fail("io_error")
        charged = observed + growth
        charge = previous if previous >= charged else charged
        if charge >= ceiling:
            _fail("storage_limited")
        gap = ceiling - observed
        need_free = growth if growth >= gap else gap
        if free < reserve + need_free:
            _fail("storage_limited")

    # ------------------------------------------------------------------
    # Event charging and append
    # ------------------------------------------------------------------

    def _compute_charge(self, state, session_id, elapsed_ns, specs):
        summary = state["summary"]
        events = state["events"]
        recorded = summary.get("new_artifact_bytes")
        head = summary.get("head_sha256")
        if type(recorded) is not int or recorded < 0 or type(head) is not str:
            self._poison()
            _fail("owner_unavailable")
        observed = self._observed()
        candidate = recorded if recorded >= observed else observed
        built = None
        for _ in range(_MAX_FIXED_POINT):
            built = []
            seq = len(events)
            previous = head
            total = 0
            for spec in specs:
                seq += 1
                try:
                    event = _build_event(
                        session_id,
                        spec["event_type"],
                        elapsed_ns,
                        candidate,
                        spec["job_id"],
                        spec["status"],
                        spec["receipt_sha256"],
                        seq,
                        previous,
                    )
                    encoded = store._encode_json(event)
                    head_bytes = store._encode_json(
                        store._make_head(event["event_sha256"], seq)
                    )
                except store.StoreError:
                    self._poison()
                    _fail("invalid_input")
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    self._poison()
                    _fail("invalid_input")
                total += len(encoded) + len(head_bytes)
                previous = event["event_sha256"]
                built.append((event, encoded, head_bytes))
            new_candidate = recorded if recorded >= observed + total else observed + total
            if new_candidate == candidate:
                return candidate, built
            candidate = new_candidate
        self._poison()
        _fail("storage_limited")

    def _append_checked(
        self,
        state,
        event,
        encoded,
        head_bytes,
        *,
        resources,
        growth=None,
        measurement_started_ns=None,
    ):
        if growth is None:
            growth = len(encoded) + len(head_bytes)
        _, ceiling = self._limits()
        if event["artifact_bytes"] >= ceiling:
            self._poison()
            _fail("storage_limited")
        self._storage_gate(state, growth)
        self._check_identity()
        expected = state["summary"]["head_sha256"]
        ready = self._fresh_state()
        if ready["summary"]["head_sha256"] != expected:
            self._poison()
            _fail("head_mismatch")
        if event.get("event_type") == "attempt_start":
            if type(measurement_started_ns) is not int or measurement_started_ns < 0:
                _fail("stale_measurement")
            now = time.monotonic_ns()
            if (
                measurement_started_ns > now
                or now - measurement_started_ns > _FRESH_NS
            ):
                _fail("stale_measurement")
            if measurement_started_ns < self._started_ns + event["elapsed_ns"]:
                _fail("stale_measurement")
        try:
            return self._mutate(
                self._owner.append_event,
                event,
                expected_head_sha256=expected,
                resources=resources,
            )
        except store.StoreError as exc:
            self._map_store_error(exc)

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def snapshot(self):
        """Return a fresh owner ``{manifest, events, summary}`` dict."""
        return self._guard(self._snapshot)

    def _snapshot(self):
        self._require_usable()
        self._check_identity()
        return self._fresh_state()

    def elapsed_ns(self):
        """Nonnegative monotonic elapsed since the supplied start."""
        return self._guard(self._elapsed)

    def _elapsed(self):
        self._require_usable()
        now = time.monotonic_ns()
        if now < self._started_ns:
            _fail("invalid_input")
        return now - self._started_ns

    def observed_bytes(self):
        """Total logical bytes under the journal, artifact and control roots.

        The launch-control directory is counted only while ``launch.json`` is
        its sole fixed file.  A terminal record written after session close is
        charged by the caller; the byte pin is not a permit.
        """
        return self._guard(self._observed)

    def open_session(self):
        """Append exactly one ``session_open`` under a ``not_started`` store."""
        return self._guard(self._open_session)

    def _open_session(self):
        self._require_usable()
        state = self._fresh_state()
        summary = state["summary"]
        if summary.get("journal_state") != "not_started":
            _fail("state_conflict")
        count = summary.get("session_count")
        if type(count) is not int or count < 0:
            _fail("state_conflict")
        session_id = count + 1
        seq = len(state["events"]) + 1
        event = _build_event(
            session_id,
            "session_open",
            0,
            0,
            None,
            None,
            None,
            seq,
            summary["head_sha256"],
        )
        encoded = store._encode_json(event)
        head_bytes = store._encode_json(store._make_head(event["event_sha256"], seq))
        out = self._append_checked(
            state, event, encoded, head_bytes, resources=None, growth=None
        )
        self._session_id = session_id
        return out

    def progress(self, *, next_job_id=None):
        """Append ``progress``; optionally precharge its following start."""
        return self._guard(self._progress, next_job_id)

    def _progress(self, next_job_id):
        self._require_usable()
        state = self._fresh_state()
        summary = state["summary"]
        session_id = summary.get("active_session_id")
        if type(session_id) is not int or session_id <= 0:
            _fail("state_conflict")
        elapsed = self._elapsed()
        specs = [_event_spec("progress")]
        if next_job_id is not None:
            if type(next_job_id) is not str or next_job_id == "":
                _fail("invalid_input")
            specs.append(_event_spec("attempt_start", job_id=next_job_id))
        charge, built = self._compute_charge(state, session_id, elapsed, specs)
        total_growth = 0
        for _, encoded, head_bytes in built:
            total_growth += len(encoded) + len(head_bytes)
        event, encoded, head_bytes = built[0]
        out = self._append_checked(
            state, event, encoded, head_bytes, resources=None, growth=total_growth
        )
        if len(built) > 1:
            self._pending = {
                "job_id": next_job_id,
                "elapsed_ns": elapsed,
                "artifact_bytes": charge,
                "expected_head": event["event_sha256"],
                "start_event": built[1][0],
            }
        else:
            self._pending = None
        return out

    def start(self, job_id, resources, *, measurement_started_ns):
        """Append the cached ``attempt_start`` with exact promised counters."""
        return self._guard(self._start, job_id, resources, measurement_started_ns)

    def _start(self, job_id, resources, measurement_started_ns):
        self._require_usable()
        if type(measurement_started_ns) is not int or measurement_started_ns < 0:
            _fail("stale_measurement")
        pending = self._pending
        if pending is None or pending.get("job_id") != job_id:
            _fail("pending_required")
        if self._running is not None:
            _fail("state_conflict")
        if measurement_started_ns < self._started_ns + pending["elapsed_ns"]:
            _fail("stale_measurement")
        now = time.monotonic_ns()
        if measurement_started_ns > now or now - measurement_started_ns > _FRESH_NS:
            _fail("stale_measurement")
        state = self._fresh_state()
        if state["summary"]["head_sha256"] != pending["expected_head"]:
            self._poison()
            _fail("head_mismatch")
        event = pending["start_event"]
        encoded = store._encode_json(event)
        head_bytes = store._encode_json(
            store._make_head(event["event_sha256"], event["seq"])
        )
        now = time.monotonic_ns()
        if measurement_started_ns > now or now - measurement_started_ns > _FRESH_NS:
            _fail("stale_measurement")
        out = self._append_checked(
            state, event, encoded, head_bytes, resources=resources, growth=None,
            measurement_started_ns=measurement_started_ns,
        )
        self._pending = None
        self._running = {
            "job_id": job_id,
            "session_id": event["session_id"],
            "start_event": event,
        }
        return out

    def write(self, name, payload):
        """Exclusively persist one bounded flat artifact with directory sync."""
        return self._guard(self._write, name, payload)

    def _write(self, name, payload):
        self._require_usable()
        if not _is_safe_name(name) or name in _RESERVED_NAMES:
            _fail("invalid_artifact")
        if type(payload) is not bytes:
            _fail("invalid_artifact")
        size = len(payload)
        if size > _MAX_ARTIFACT_BYTES:
            _fail("invalid_artifact")
        state = self._fresh_state()
        self._storage_gate(state, size)
        self._check_identity()
        count = 0
        try:
            with os.scandir(self._root_fd) as entries:
                for _ in entries:
                    count += 1
                    if count >= _MAX_ARTIFACT_ENTRIES:
                        _fail("layout_invalid")
        except OSError:
            self._poison()
            _fail("io_error")
        try:
            self._mutate(store._write_exclusive, self._root_fd, name, payload)
            self._mutate(os.fsync, self._root_fd)
        except FileExistsError:
            _fail("artifact_exists")
        except OSError:
            self._poison()
            _fail("io_error")
        try:
            info = os.stat(name, dir_fd=self._root_fd, follow_symlinks=False)
        except OSError:
            self._poison()
            _fail("incomplete_write")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != size:
            self._poison()
            _fail("incomplete_write")
        self._check_identity()
        return name

    def read(self, name):
        """Read one bounded flat artifact; authenticates filesystem kind only."""
        return self._guard(self._read, name)

    def _read(self, name):
        self._require_usable()
        if not _is_safe_name(name):
            _fail("invalid_artifact")
        self._fresh_state()
        return self._read_checked(name)

    def _read_checked(self, name):
        self._require_usable()
        self._check_identity()
        try:
            return store._read_file(self._root_fd, name, _MAX_ARTIFACT_BYTES)
        except store.StoreError as exc:
            code = getattr(exc, "reason_code", "")
            if code == "storage_io_error":
                self._poison()
                _fail("io_error")
            if code == "symlink_or_nonregular":
                self._poison()
                _fail("symlink_or_nonregular")
            _fail("artifact_mismatch")
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            self._poison()
            _fail("io_error")

    def finish(self, job_id, status, artifacts):
        """Verify saved artifacts and append exactly the verified finish event."""
        return self._guard(self._finish, job_id, status, artifacts)

    def _finish(self, job_id, status, artifacts):
        self._require_usable()
        running = self._running
        if running is None or running.get("job_id") != job_id:
            _fail("not_running")
        if type(status) is not str or status not in _TERMINAL_STATUSES:
            _fail("invalid_artifact")
        self._validate_artifacts(artifacts)
        receipt_name = job_id + "-receipt.json"
        if not _is_safe_name(receipt_name) or receipt_name in artifacts:
            _fail("invalid_artifact")
        start_event = running["start_event"]
        session_id = running["session_id"]
        state = self._fresh_state()
        if not self._check_running_summary(state, job_id, session_id, start_event):
            self._poison()
            _fail("state_conflict")
        current_head = state["summary"]["head_sha256"]
        manifest = state["manifest"]
        manifest_job = self._find_job(manifest, job_id)
        receipt = {
            "schema_version": _RECEIPT_SCHEMA,
            "execution_authorized": False,
            "proposal_sha256": admission.U0_PROPOSAL_SHA256,
            "manifest_sha256": admission.U0_MANIFEST_SHA256,
            "job_id": job_id,
            "stage": manifest_job["stage"],
            "worker": manifest_job["worker"],
            "session_id": session_id,
            "start_event_sha256": start_event["event_sha256"],
            "status": status,
            "artifacts": [
                {
                    "name": name,
                    "size_bytes": len(artifacts[name]),
                    "sha256": hashlib.sha256(artifacts[name]).hexdigest(),
                }
                for name in sorted(artifacts)
            ],
        }
        try:
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            receipt_bytes = store._encode_json(receipt)
        except (store.StoreError, ValueError, TypeError):
            _fail("invalid_artifact")
        if len(receipt_bytes) > _MAX_RECEIPT_BYTES:
            _fail("invalid_artifact")
        self._storage_gate(state, len(receipt_bytes))
        self._check_identity()
        try:
            self._mutate(
                store._write_exclusive, self._root_fd, receipt_name, receipt_bytes
            )
            self._mutate(os.fsync, self._root_fd)
        except FileExistsError:
            _fail("artifact_exists")
        except OSError:
            self._poison()
            _fail("io_error")
        self._check_identity()
        for name in sorted(artifacts):
            raw = self._read_checked(name)
            if raw != artifacts[name]:
                _fail("artifact_mismatch")
        state2 = self._fresh_state()
        if state2["summary"]["head_sha256"] != current_head:
            self._poison()
            _fail("head_mismatch")
        elapsed = self._elapsed()
        spec = _event_spec(
            "attempt_finish",
            job_id=job_id,
            status=status,
            receipt_sha256=receipt["receipt_sha256"],
        )
        _, built = self._compute_charge(state2, session_id, elapsed, [spec])
        event, encoded, head_bytes = built[0]
        self._storage_gate(state2, len(encoded) + len(head_bytes))
        self._check_identity()
        try:
            verify_terminal_receipt(
                self.artifact_root,
                receipt_name,
                manifest,
                state2["events"],
                event,
                expected_head_sha256=state2["summary"]["head_sha256"],
                expected_artifact_names=sorted(artifacts),
            )
        except ReceiptError:
            _fail("receipt_error")
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError:
            self._poison()
            _fail("io_error")
        self._check_identity()
        out = self._append_checked(
            state2, event, encoded, head_bytes, resources=None, growth=None
        )
        self._running = None
        return out

    def _validate_artifacts(self, artifacts):
        if type(artifacts) is not dict:
            _fail("invalid_artifact")
        names = list(artifacts.keys())
        if len(names) < 1 or len(names) > _MAX_ARTIFACTS:
            _fail("invalid_artifact")
        for name in names:
            if not _is_safe_name(name):
                _fail("invalid_artifact")
            payload = artifacts[name]
            if type(payload) is not bytes or len(payload) > _MAX_ARTIFACT_BYTES:
                _fail("invalid_artifact")

    def _find_job(self, manifest, job_id):
        jobs = manifest.get("jobs") if type(manifest) is dict else None
        if type(jobs) is not list:
            _fail("layout_invalid")
        for job in jobs:
            if type(job) is dict and job.get("job_id") == job_id:
                stage = job.get("stage")
                worker = job.get("worker")
                if type(stage) is not str or type(worker) is not str:
                    _fail("layout_invalid")
                return job
        _fail("not_running")

    def _check_running_summary(self, state, job_id, session_id, start_event):
        summary = state.get("summary") if type(state) is dict else None
        if type(summary) is not dict:
            return False
        if summary.get("active_session_id") != session_id:
            return False
        if summary.get("in_flight_job_ids") != [job_id]:
            return False
        attempts = summary.get("attempts")
        if type(attempts) is not dict:
            return False
        attempt = attempts.get(job_id)
        if type(attempt) is not dict:
            return False
        if attempt.get("status") != "running":
            return False
        if attempt.get("session_id") != session_id:
            return False
        events = state.get("events")
        if type(events) is not list:
            return False
        found = None
        for ev in events:
            if type(ev) is not dict:
                return False
            if found is None:
                if (
                    ev.get("event_type") == "attempt_start"
                    and ev.get("job_id") == job_id
                ):
                    found = ev
                continue
            if ev.get("event_type") != "progress":
                return False
        if found is None or found != start_event:
            return False
        return True

    def close_session(self):
        """Append ``session_close``; no running job may remain."""
        return self._guard(self._close_session)

    def _close_session(self):
        self._require_usable()
        if self._running is not None:
            _fail("running_job")
        state = self._fresh_state()
        session_id = state["summary"].get("active_session_id")
        if type(session_id) is not int or session_id <= 0:
            _fail("state_conflict")
        elapsed = self._elapsed()
        charge, built = self._compute_charge(
            state, session_id, elapsed, [_event_spec("session_close")]
        )
        event, encoded, head_bytes = built[0]
        out = self._append_checked(
            state, event, encoded, head_bytes, resources=None, growth=None
        )
        self._pending = None
        return out

    def close(self):
        """Release this IO's three artifact descriptors; idempotent.

        Accounting covers the journal, artifact and launch-control roots only
        while ``launch.json`` is the sole fixed control file.  A terminal
        record written after this close is owned and charged by the caller;
        the byte pin is not a permit.
        """
        if self._closed:
            return
        self._closed = True
        root_fd = self._root_fd
        parent_fd = self._parent_fd
        control_fd = self._control_fd
        self._root_fd = -1
        self._parent_fd = -1
        self._control_fd = -1
        store._close_fds(root_fd, parent_fd, control_fd)


def require_scientific_execution(*args, **kwargs):
    """Always deny: this leaf grants no scientific execution authority."""
    raise SessionIOError("scientific_execution_not_authorized") from None
