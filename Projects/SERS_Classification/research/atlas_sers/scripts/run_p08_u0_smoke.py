#!/usr/bin/env python3
"""P08 U0 permit-bound outer launcher for the approved source-only pilot.

This is the single outer entry point that connects independent authority, an
exclusive private destination, the reviewer-authenticated project runtime, the
fixed input adapters, one existing serial session and a residual evidence
record.  It is an implementation specification, not an execution permit.

Production release
------------------
This deployment sets ``APPROVED_PERMIT_SHA256`` to the exact SHA-256 of the
independently approved 2026-10-08 U0 private permit, bound to one destination.
Only a byte-identical copy of that permit is accepted: an absent, unset or
changed permit still denies before any scientific work begins.  There is no CLI
replacement hash, no new destination and no retry.  Setting the pin is a
reviewed deployment decision, not a public mode: the command line has no
authority flag, no alternate output, no job subset and no environment bypass.
If the pin is left unset (``None``), every invocation denies with the static
reason ``scientific_execution_not_authorized`` before the permit file is even
opened.

Trust limits
------------
The interpreter, standard library, reviewed bootstrap and external numerical
dependencies are trusted rather than byte-authenticated by the project source
catalog.  The catalog authenticates project source identity only.  No positive
permit is generated here.  Classically, native estimator calls remain
cooperatively bounded, never hard-preempted.  Resource samples are
point-in-time observations; they do not promise hard preemption or continuous
monitoring.  A successful source run still requires independent supervisor
review: the report never auto-grants ``live_runtime_accepted``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import stat
import sys
import time
import types

__all__ = ["launch", "main"]

# ---------------------------------------------------------------------------
# Independent release pin (the ONLY production authority gate)
# ---------------------------------------------------------------------------

# Owner-approved 2026-10-08 U0 private permit, bound to one destination.
# If this pin is left unset (None) the launcher denies all invocations.
APPROVED_PERMIT_SHA256 = "a0ddf4adbbdad560de83941e5ca8f5330b98e000928f887ac88a69afdb04947d"
BOOTSTRAP_SHA256 = "fa7c48b766b5522eb5008ed7faf2fc92e5013b2a96376f83167ab77d307dc784"
PROPOSAL_SHA256 = "6639a32c1dd930612ead6ae59adf9aff5831904e883081f16c3490721af5089a"
MANIFEST_SHA256 = "aac8523a1614610cf99cf1ab548d8970b4f8054fe8821b5250c347376e30b28a"

PERMIT_SCHEMA = "nato-sers-p08-u0-execution-permit-v1"
REPORT_SCHEMA = "nato-sers-p08-u0-launch-report-v1"
LAUNCH_RECORD_SCHEMA = "nato-sers-p08-u0-launch-record-v1"
TERMINAL_SCHEMA = "nato-sers-p08-u0-terminal-v1"

BOOTSTRAP_RELATIVE_PATH = "scripts/check_p08_u0_runtime.py"
TERMINAL_FILE_NAME = "terminal.json"
LAUNCH_FILE_NAME = "launch.json"

NAMESPACE = "atlas_sers"

MAX_PERMIT_BYTES = 65536
MAX_CONTROL_BYTES = 65536
MAX_BOOTSTRAP_BYTES = 4 * 1024 * 1024
MAX_SOURCE_SNAPSHOT_BYTES = 1 * 1024 * 1024
MAX_ROOT_ENTRIES = 3
MAX_CONTROL_ENTRIES = 2
MAX_JOURNAL_ENTRIES = 5
MAX_ARTIFACT_FILES = 1024
MAX_EVENT_FILES = 16384

_JOURNAL_FIXED_FILES = frozenset({"manifest.json", "head.json", ".lock"})
_JOURNAL_PENDING_FILE = "head.pending"
_JOURNAL_EVENTS_DIR = "events"
_ROOT_ALLOWED_NAMES = frozenset({"control", "journal", "artifacts"})
_CONTROL_ALLOWED_NAMES = frozenset({LAUNCH_FILE_NAME, TERMINAL_FILE_NAME})

_EXPECTED_PAIRS = 78
_EXPECTED_JOBS = 156
_EXPECTED_FIT_JOBS = 78
_EXPECTED_PREDICTION_JOBS = 78
_EXPECTED_SPECIFICATION_SOURCES = 18

_LIMIT_KEYS = frozenset(
    {
        "model_fits",
        "source_predictions",
        "scalar_calibrations",
        "automatic_retries",
        "wall_ns",
        "artifact_bytes",
        "process_tree_rss_bytes",
        "allocated_gpu_bytes",
        "filesystem_reserve_bytes",
        "cpu_threads",
        "maximum_cpu_workers",
        "maximum_gpu_workers",
    }
)

_EXPECTED_LIMITS = {
    "model_fits": 78,
    "source_predictions": 78,
    "scalar_calibrations": 0,
    "automatic_retries": 0,
    "wall_ns": 5400000000000,
    "artifact_bytes": 8589934592,
    "process_tree_rss_bytes": 17179869184,
    "allocated_gpu_bytes": 8589934592,
    "filesystem_reserve_bytes": 32212254720,
    "cpu_threads": 1,
    "maximum_cpu_workers": 2,
    "maximum_gpu_workers": 1,
}

_PERMIT_KEYS = frozenset(
    {
        "schema_version",
        "stage",
        "execution_authorized",
        "proposal_sha256",
        "manifest_sha256",
        "catalog_sha256",
        "source_revision",
        "limits",
        "output_root",
        "metadata_paths",
        "action_paths",
        "specification_audit_path",
        "candidate_registry_path",
    }
)

_METADATA_KEYS = frozenset(
    {
        "proposal_bytes",
        "attempt_manifest_bytes",
        "manifest_bytes",
        "contexts_bytes",
        "roles_bytes",
    }
)

_METADATA_READ_CAPS = {
    "proposal_bytes": 1 * 1024 * 1024,
    "attempt_manifest_bytes": 1 * 1024 * 1024,
    "manifest_bytes": 2 * 1024 * 1024,
    "contexts_bytes": 2 * 1024 * 1024,
    "roles_bytes": 32 * 1024 * 1024,
}

_ACTION_KEYS = ("R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800")
_ACTION_KEYS_SET = frozenset(_ACTION_KEYS)
_ACTION_READ_CAP = 8 * 1024 * 1024
_AUDIT_READ_CAP = 1 * 1024 * 1024
_CANDIDATE_READ_CAP = 1 * 1024 * 1024

_NON_SRC_SOURCES = frozenset(
    {
        "plan/contracts/hyperparameter_registry.json",
        "plan/contracts/p03_governance_contract.json",
        "plan/contracts/p04_execution_contract.json",
        "plan/contracts/p05_core_contract.json",
    }
)

_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_HEX = frozenset("0123456789abcdef")

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "unsupported_platform",
        "not_isolated",
        "bytecode_writes_enabled",
        "preexisting_project_modules",
        "permit_pin_invalid",
        "package_root_invalid",
        "permit_read_failed",
        "permit_hash_mismatch",
        "permit_parse_failed",
        "permit_schema_invalid",
        "permit_unknown_field",
        "permit_stage_invalid",
        "permit_not_authorized",
        "permit_proposal_mismatch",
        "permit_manifest_mismatch",
        "permit_limits_invalid",
        "permit_path_invalid",
        "permit_output_inside_package",
        "output_parent_missing",
        "output_exists",
        "output_create_failed",
        "output_identity_changed",
        "control_write_failed",
        "bootstrap_read_failed",
        "bootstrap_hash_mismatch",
        "bootstrap_compile_failed",
        "bootstrap_missing_api",
        "permit_catalog_mismatch",
        "permit_revision_mismatch",
        "input_read_failed",
        "audit_hash_mismatch",
        "audit_invalid",
        "source_pin_conflict",
        "specification_sources_incomplete",
        "input_preparation_failed",
        "manifest_binding_mismatch",
        "invalid_pair_binding",
        "resource_setup_failed",
        "cuda_unavailable",
        "resource_limit_exceeded",
        "session_incomplete",
        "internal_error",
        "interrupted",
    }
)


class LaunchError(ValueError):
    """Static, path-free launcher failure carrying only ``reason_code``."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "internal_error"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code):
    raise LaunchError(reason_code) from None


def _close_fds(descriptors):
    """Close every owned descriptor, returning the first failure (if any)."""
    first_error = None
    for descriptor in descriptors:
        if type(descriptor) is not int or descriptor < 0:
            continue
        try:
            os.close(descriptor)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    return first_error


def _measured_elapsed_ns(outer_start_ns):
    """Validate and return a nonnegative measured nanosecond interval."""
    elapsed = time.monotonic_ns() - outer_start_ns
    if type(elapsed) is not int or elapsed < 0:
        _fail("resource_limit_exceeded")
    return elapsed


# ---------------------------------------------------------------------------
# Strict JSON helpers
# ---------------------------------------------------------------------------


def _duplicate_hook(reason):
    def hook(pairs):
        seen = {}
        for key, value in pairs:
            if key in seen:
                _fail(reason)
            seen[key] = value
        return seen

    return hook


def _constant_hook(reason):
    def hook(name):
        _fail(reason)

    return hook


def _float_hook(reason):
    def hook(text):
        value = float(text)
        if value != value:
            _fail(reason)
        if value == float("inf") or value == float("-inf"):
            _fail(reason)
        return value

    return hook


def _load_json_bytes(raw, reason):
    if type(raw) is not bytes or not raw:
        _fail(reason)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail(reason)
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_hook(reason),
            parse_constant=_constant_hook(reason),
            parse_float=_float_hook(reason),
        )
    except LaunchError:
        raise
    except Exception:
        _fail(reason)


def _load_json_text(text, reason):
    if type(text) is not str or text == "":
        _fail(reason)
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_hook(reason),
            parse_constant=_constant_hook(reason),
            parse_float=_float_hook(reason),
        )
    except LaunchError:
        raise
    except Exception:
        _fail(reason)


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


# ---------------------------------------------------------------------------
# Environment and filesystem trust primitives
# ---------------------------------------------------------------------------


def _require_environment():
    if os.name != "posix":
        _fail("unsupported_platform")
    if sys.flags.isolated != 1:
        _fail("not_isolated")
    if not getattr(sys, "dont_write_bytecode", False):
        _fail("bytecode_writes_enabled")
    for name in list(sys.modules):
        if name == NAMESPACE or name.startswith(NAMESPACE + "."):
            _fail("preexisting_project_modules")
    for flag_name in ("O_RDONLY", "O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK"):
        if not hasattr(os, flag_name):
            _fail("unsupported_platform")
    if not hasattr(os, "fstatvfs"):
        _fail("unsupported_platform")
    supports = getattr(os, "supports_dir_fd", None)
    if supports is None or os.open not in supports or os.stat not in supports:
        _fail("unsupported_platform")


def _validate_hex(value, length, reason):
    if type(value) is not str or len(value) != length:
        _fail(reason)
    for char in value:
        if char not in _HEX:
            _fail(reason)
    return value


def _canonical_absolute(value, reason):
    """Validate absolute-path syntax only; real reads use no-follow holders."""
    if type(value) is not str or value == "" or value[0] != "/" or "\\" in value:
        _fail(reason)
    if os.path.normpath(value) != value:
        _fail(reason)
    parts = [part for part in value.split("/") if part]
    if not parts or any(part in (".", "..") for part in parts):
        _fail(reason)
    return value


def _open_absolute(path, reason, final_directory=False):
    """Descriptor-walk an absolute path, rejecting every symlinked component."""
    if type(path) is not str or not path or path[0] != "/":
        _fail(reason)
    parts = [part for part in path.split("/") if part]
    if not parts or any(part in (".", "..") for part in parts):
        _fail(reason)
    owned = []
    try:
        descriptor = os.open(
            "/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        owned.append(descriptor)
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            if last and not final_directory:
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            else:
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                following = os.open(part, flags, dir_fd=descriptor)
            except OSError:
                _fail(reason)
            owned.append(following)
            descriptor = following
    except BaseException:
        _close_fds(owned)
        raise
    result = owned[-1]
    error = _close_fds(owned[:-1])
    if error is not None:
        try:
            os.close(result)
        except BaseException:
            pass
        raise error
    return result


def _require_directory(path, reason):
    descriptor = _open_absolute(path, reason, final_directory=True)
    error = _close_fds((descriptor,))
    if error is not None:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise error
        _fail(reason)


def _read_fd(descriptor, max_bytes, reason):
    try:
        before = os.fstat(descriptor)
    except OSError:
        _fail(reason)
    if not stat.S_ISREG(before.st_mode):
        _fail(reason)
    if before.st_nlink != 1:
        _fail(reason)
    if before.st_size > max_bytes:
        _fail(reason)
    chunks = []
    remaining = max_bytes + 1
    while remaining > 0:
        try:
            chunk = os.read(descriptor, min(65536, remaining))
        except OSError:
            _fail(reason)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if remaining <= 0:
        _fail(reason)
    data = b"".join(chunks)
    try:
        after = os.fstat(descriptor)
    except OSError:
        _fail(reason)
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_nlink,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ):
        _fail(reason)
    if len(data) != before.st_size:
        _fail(reason)
    return data


def _read_absolute_file(path, max_bytes, reason):
    descriptor = _open_absolute(path, reason)
    primary_error = None
    try:
        try:
            before = os.fstat(descriptor)
        except OSError:
            _fail(reason)
        data = _read_fd(descriptor, max_bytes, reason)
        try:
            held = os.fstat(descriptor)
        except OSError:
            _fail(reason)
        if _stat_signature(held) != _stat_signature(before):
            _fail(reason)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        error = _close_fds((descriptor,))
        if error is not None and primary_error is None:
            raise error
    check = _open_absolute(path, reason)
    primary_error = None
    try:
        try:
            check_info = os.fstat(check)
        except OSError:
            _fail(reason)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        error2 = _close_fds((check,))
        if error2 is not None and primary_error is None:
            raise error2
    if _stat_signature(check_info) != _stat_signature(held):
        _fail(reason)
    return data


# ---------------------------------------------------------------------------
# Permit validation
# ---------------------------------------------------------------------------


def _read_permit(permit_file, expected_sha256):
    path = _canonical_absolute(permit_file, "permit_read_failed")
    raw = _read_absolute_file(path, MAX_PERMIT_BYTES, "permit_read_failed")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail("permit_hash_mismatch")
    return raw


def _validate_limits(limits):
    if type(limits) is not dict or set(limits) != _LIMIT_KEYS:
        _fail("permit_limits_invalid")
    for key in _LIMIT_KEYS:
        value = limits[key]
        if type(value) is not int or value < 0:
            _fail("permit_limits_invalid")
    if limits != _EXPECTED_LIMITS:
        _fail("permit_limits_invalid")


def _validate_permit(permit_bytes, package_root):
    document = _load_json_bytes(permit_bytes, "permit_parse_failed")
    if type(document) is not dict:
        _fail("permit_schema_invalid")
    keys = set(document)
    if not keys.issubset(_PERMIT_KEYS):
        _fail("permit_unknown_field")
    if keys != _PERMIT_KEYS:
        _fail("permit_schema_invalid")
    if document["schema_version"] != PERMIT_SCHEMA:
        _fail("permit_schema_invalid")
    if document["stage"] != "U0":
        _fail("permit_stage_invalid")
    if document["execution_authorized"] is not True:
        _fail("permit_not_authorized")
    if document["proposal_sha256"] != PROPOSAL_SHA256:
        _fail("permit_proposal_mismatch")
    if document["manifest_sha256"] != MANIFEST_SHA256:
        _fail("permit_manifest_mismatch")
    _validate_hex(document["catalog_sha256"], 64, "permit_schema_invalid")
    _validate_hex(document["source_revision"], 40, "permit_schema_invalid")
    _validate_limits(document["limits"])

    output_root = _canonical_absolute(document["output_root"], "permit_path_invalid")
    if output_root == package_root or output_root.startswith(package_root + os.sep):
        _fail("permit_output_inside_package")

    metadata_paths = document["metadata_paths"]
    if type(metadata_paths) is not dict or set(metadata_paths) != _METADATA_KEYS:
        _fail("permit_path_invalid")
    for key in _METADATA_READ_CAPS:
        _canonical_absolute(metadata_paths.get(key), "permit_path_invalid")

    action_paths = document["action_paths"]
    if type(action_paths) is not dict or set(action_paths) != _ACTION_KEYS_SET:
        _fail("permit_path_invalid")
    for key in _ACTION_KEYS:
        _canonical_absolute(action_paths.get(key), "permit_path_invalid")

    _canonical_absolute(document["specification_audit_path"], "permit_path_invalid")
    _canonical_absolute(document["candidate_registry_path"], "permit_path_invalid")
    return document


# ---------------------------------------------------------------------------
# Setup budget: wall clock and physical/artifact capacity across all stages
# ---------------------------------------------------------------------------


class _Budget:
    """Cumulative, non-resettable setup/finalization budget."""

    __slots__ = ("permit", "outer_start_ns", "high_water_bytes")

    def __init__(self, permit, outer_start_ns):
        self.permit = permit
        self.outer_start_ns = outer_start_ns
        self.high_water_bytes = 0

    def check_setup(self, descriptor, actual_bytes, proposed_growth):
        limits = self.permit["limits"]
        elapsed = _measured_elapsed_ns(self.outer_start_ns)
        if elapsed >= limits["wall_ns"]:
            _fail("resource_limit_exceeded")
        if type(actual_bytes) is not int or actual_bytes < 0:
            _fail("resource_limit_exceeded")
        if type(proposed_growth) is not int or proposed_growth < 0:
            _fail("resource_limit_exceeded")
        allowance = limits["artifact_bytes"]
        prospective = actual_bytes + proposed_growth
        high = prospective
        if self.high_water_bytes > high:
            high = self.high_water_bytes
        self.high_water_bytes = high
        if high >= allowance:
            _fail("resource_limit_exceeded")
        unallocated = allowance - actual_bytes
        if unallocated < 0:
            unallocated = 0
        required = limits["filesystem_reserve_bytes"] + max(unallocated, proposed_growth)
        try:
            info = os.fstatvfs(descriptor)
        except OSError:
            _fail("output_identity_changed")
        free_bytes = info.f_bavail * info.f_frsize
        if free_bytes < required:
            _fail("resource_limit_exceeded")

    def check_layout(self, output, proposed_growth):
        actual = output.verify()
        self.check_setup(output._root_fd, actual, proposed_growth)
        return actual


# ---------------------------------------------------------------------------
# Bounded descriptor-relative outer layout inspection
# ---------------------------------------------------------------------------


def _verify_fd(descriptor, expected_identity):
    try:
        info = os.fstat(descriptor)
    except OSError:
        _fail("output_identity_changed")
    if (info.st_dev, info.st_ino) != expected_identity:
        _fail("output_identity_changed")


def _stat_signature(info):
    """Full stable identity, deliberately excluding atime (a read side effect)."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_signature(descriptor, reason):
    try:
        info = os.fstat(descriptor)
    except OSError:
        _fail(reason)
    if not stat.S_ISDIR(info.st_mode):
        _fail(reason)
    return _stat_signature(info)


@contextlib.contextmanager
def _scandir_bounded(descriptor, reason):
    """Yield one explicit, always-closed descriptor-relative iterator."""
    try:
        iterator = os.scandir(descriptor)
    except OSError:
        _fail(reason)
    primary_error = None
    try:
        yield iterator
    except BaseException as error:
        primary_error = error
        raise
    finally:
        closer = getattr(iterator, "close", None)
        if closer is not None:
            try:
                closer()
            except BaseException:
                if primary_error is None:
                    raise


def _list_fd(descriptor, max_entries, reason):
    """Bounded name inventory; rejects duplicates and invalid names."""
    names = []
    seen = set()
    with _scandir_bounded(descriptor, reason) as iterator:
        try:
            for entry in iterator:
                try:
                    name = entry.name
                except AttributeError:
                    _fail(reason)
                if type(name) is not str or name in ("", ".", ".."):
                    _fail(reason)
                if name in seen:
                    _fail(reason)
                seen.add(name)
                if len(names) + 1 > max_entries:
                    _fail(reason)
                names.append(name)
        except LaunchError:
            raise
        except OSError:
            _fail(reason)
    return names


def _stat_at(descriptor, name, reason):
    try:
        return os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except OSError:
        _fail(reason)


def _inventory_fd(descriptor, max_entries, reason):
    """Directory signature plus bounded, exact per-entry identities."""
    directory = _directory_signature(descriptor, reason)
    entries = []
    for name in _list_fd(descriptor, max_entries, reason):
        info = _stat_at(descriptor, name, reason)
        entries.append((name, _stat_signature(info)))
    entries.sort(key=lambda item: item[0])
    return directory, tuple(entries)


def _verify_child_binding(parent_fd, name, child_fd, reason):
    """Require the parent name and the retained child descriptor to agree."""
    parent_info = _stat_at(parent_fd, name, reason)
    try:
        child_info = os.fstat(child_fd)
    except OSError:
        _fail(reason)
    if (parent_info.st_dev, parent_info.st_ino) != (
        child_info.st_dev,
        child_info.st_ino,
    ):
        _fail(reason)


def _open_child_directory(descriptor, name, reason):
    before = _stat_at(descriptor, name, reason)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        _fail(reason)
    try:
        child = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=descriptor,
        )
    except OSError:
        _fail(reason)
    primary_error = None
    try:
        try:
            info = os.fstat(child)
        except OSError:
            _fail(reason)
        if not stat.S_ISDIR(info.st_mode):
            _fail(reason)
        if (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino):
            _fail(reason)
        after = _stat_at(descriptor, name, reason)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            _fail(reason)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if primary_error is not None:
            _close_fds((child,))
    return child


def _scan_flat(descriptor, max_files, reason, per_file_cap=None):
    before_dir, before_entries = _inventory_fd(descriptor, max_files, reason)
    total = 0
    for _name, signature in before_entries:
        mode = signature[2]
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            _fail(reason)
        if signature[3] != 1:
            _fail(reason)
        if per_file_cap is not None and signature[4] > per_file_cap:
            _fail(reason)
        total += signature[4]
    after_dir, after_entries = _inventory_fd(descriptor, max_files, reason)
    if before_dir != after_dir or before_entries != after_entries:
        _fail(reason)
    return total


def _scan_control(output):
    reason = "output_identity_changed"
    before_dir, before_entries = _inventory_fd(
        output._control_fd, MAX_CONTROL_ENTRIES, reason
    )
    total = 0
    for name, signature in before_entries:
        if name not in _CONTROL_ALLOWED_NAMES:
            _fail(reason)
        mode = signature[2]
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            _fail(reason)
        if signature[3] != 1 or signature[4] > MAX_CONTROL_BYTES:
            _fail(reason)
        total += signature[4]
    after_dir, after_entries = _inventory_fd(
        output._control_fd, MAX_CONTROL_ENTRIES, reason
    )
    if before_dir != after_dir or before_entries != after_entries:
        _fail(reason)
    return total


def _scan_journal(descriptor):
    reason = "output_identity_changed"
    before_dir, before_entries = _inventory_fd(
        descriptor, MAX_JOURNAL_ENTRIES, reason
    )
    total = 0
    events_name = None
    for name, signature in before_entries:
        if name in _JOURNAL_FIXED_FILES or name == _JOURNAL_PENDING_FILE:
            mode = signature[2]
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                _fail(reason)
            if signature[3] != 1:
                _fail(reason)
            total += signature[4]
        elif name == _JOURNAL_EVENTS_DIR:
            mode = signature[2]
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _fail(reason)
            events_name = name
        else:
            _fail(reason)
    if events_name is not None:
        events_fd = _open_child_directory(descriptor, events_name, reason)
        error = None
        try:
            total += _scan_flat(events_fd, MAX_EVENT_FILES, reason)
            _verify_child_binding(descriptor, events_name, events_fd, reason)
        finally:
            error = _close_fds((events_fd,))
        if error is not None:
            raise error
    after_dir, after_entries = _inventory_fd(
        descriptor, MAX_JOURNAL_ENTRIES, reason
    )
    if before_dir != after_dir or before_entries != after_entries:
        _fail(reason)
    return total


class _Output:
    """Owned private destination with retained identity-checked descriptors."""

    __slots__ = (
        "root",
        "parent_path",
        "name",
        "control_root",
        "artifacts_root",
        "journal_root",
        "_parent_fd",
        "_root_fd",
        "_control_fd",
        "_parent_id",
        "_root_id",
        "_control_id",
        "_terminal_attempted",
        "_closed",
    )

    def __init__(
        self,
        root,
        parent_path,
        name,
        control_root,
        artifacts_root,
        journal_root,
        parent_fd,
        root_fd,
        control_fd,
        parent_id,
        root_id,
        control_id,
    ):
        self.root = root
        self.parent_path = parent_path
        self.name = name
        self.control_root = control_root
        self.artifacts_root = artifacts_root
        self.journal_root = journal_root
        self._parent_fd = parent_fd
        self._root_fd = root_fd
        self._control_fd = control_fd
        self._parent_id = parent_id
        self._root_id = root_id
        self._control_id = control_id
        self._terminal_attempted = False
        self._closed = False

    def verify_identities(self):
        if self._closed:
            _fail("output_identity_changed")
        _verify_fd(self._parent_fd, self._parent_id)
        _verify_fd(self._root_fd, self._root_id)
        _verify_fd(self._control_fd, self._control_id)
        # Reopen/walk the absolute parent no-follow and require the same inode.
        reopened = _open_absolute(
            self.parent_path, "output_identity_changed", final_directory=True
        )
        error = None
        try:
            info = os.fstat(reopened)
            if (info.st_dev, info.st_ino) != self._parent_id:
                _fail("output_identity_changed")
        finally:
            error = _close_fds((reopened,))
        if error is not None:
            raise error
        # The root name under the retained parent must still be our root inode.
        root_info = _stat_at(self._parent_fd, self.name, "output_identity_changed")
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            _fail("output_identity_changed")
        if (root_info.st_dev, root_info.st_ino) != self._root_id:
            _fail("output_identity_changed")
        # The control entry under the retained root must be our control inode.
        control_info = _stat_at(self._root_fd, "control", "output_identity_changed")
        if stat.S_ISLNK(control_info.st_mode) or not stat.S_ISDIR(control_info.st_mode):
            _fail("output_identity_changed")
        if (control_info.st_dev, control_info.st_ino) != self._control_id:
            _fail("output_identity_changed")

    def verify(self):
        self.verify_identities()
        reason = "output_identity_changed"
        total = 0
        before_dir, before_entries = _inventory_fd(
            self._root_fd, MAX_ROOT_ENTRIES, reason
        )
        root_names = {name for name, _signature in before_entries}
        for name, signature in before_entries:
            if name not in _ROOT_ALLOWED_NAMES:
                _fail(reason)
            mode = signature[2]
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _fail(reason)
        total += _scan_control(self)
        if "journal" in root_names:
            journal_fd = _open_child_directory(self._root_fd, "journal", reason)
            error = None
            try:
                total += _scan_journal(journal_fd)
                _verify_child_binding(self._root_fd, "journal", journal_fd, reason)
            finally:
                error = _close_fds((journal_fd,))
            if error is not None:
                raise error
        if "artifacts" in root_names:
            artifacts_fd = _open_child_directory(self._root_fd, "artifacts", reason)
            error = None
            try:
                total += _scan_flat(artifacts_fd, MAX_ARTIFACT_FILES, reason)
                _verify_child_binding(self._root_fd, "artifacts", artifacts_fd, reason)
            finally:
                error = _close_fds((artifacts_fd,))
            if error is not None:
                raise error
        after_dir, after_entries = _inventory_fd(
            self._root_fd, MAX_ROOT_ENTRIES, reason
        )
        if before_dir != after_dir or before_entries != after_entries:
            _fail(reason)
        self.verify_identities()
        return total

    def close(self):
        if self._closed:
            return
        self._closed = True
        error = _close_fds((self._control_fd, self._root_fd, self._parent_fd))
        if error is not None:
            raise error


def _claim_output(output_root, budget):
    parent = os.path.dirname(output_root)
    name = os.path.basename(output_root)
    if parent == "" or name in ("", ".", ".."):
        _fail("permit_path_invalid")
    parent_fd = _open_absolute(parent, "output_parent_missing", final_directory=True)
    root_fd = -1
    control_fd = -1
    try:
        parent_info = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_info.st_mode):
            _fail("output_parent_missing")
        parent_id = (parent_info.st_dev, parent_info.st_ino)
        # Refuse an already-exhausted setup budget before creating anything.
        budget.check_setup(parent_fd, 0, 0)
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            _fail("output_exists")
        except OSError:
            _fail("output_create_failed")
        try:
            os.fsync(parent_fd)
        except OSError:
            _fail("output_create_failed")
        try:
            root_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
        except OSError:
            _fail("output_create_failed")
        root_info = os.fstat(root_fd)
        if not stat.S_ISDIR(root_info.st_mode):
            _fail("output_create_failed")
        root_id = (root_info.st_dev, root_info.st_ino)
        # Only output_root and control/ are created here; the store and session
        # exclusively create journal/ and artifacts/ under their own contract.
        try:
            os.mkdir("control", 0o700, dir_fd=root_fd)
        except OSError:
            _fail("output_create_failed")
        try:
            os.fsync(root_fd)
        except OSError:
            _fail("output_create_failed")
        try:
            control_fd = os.open(
                "control",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=root_fd,
            )
        except OSError:
            _fail("output_create_failed")
        control_info = os.fstat(control_fd)
        if not stat.S_ISDIR(control_info.st_mode):
            _fail("output_create_failed")
        control_id = (control_info.st_dev, control_info.st_ino)
    except BaseException:
        _close_fds((control_fd, root_fd, parent_fd))
        raise
    return _Output(
        output_root,
        parent,
        name,
        os.path.join(output_root, "control"),
        os.path.join(output_root, "artifacts"),
        os.path.join(output_root, "journal"),
        parent_fd,
        root_fd,
        control_fd,
        parent_id,
        root_id,
        control_id,
    )


def _write_control_file(output, name, data):
    if type(data) is not bytes or len(data) == 0 or len(data) > MAX_CONTROL_BYTES:
        _fail("control_write_failed")
    if name not in _CONTROL_ALLOWED_NAMES:
        _fail("control_write_failed")
    if name == TERMINAL_FILE_NAME:
        if output._terminal_attempted:
            _fail("control_write_failed")
        # Marked before any byte is attempted: one attempt, no retry.
        output._terminal_attempted = True
    output.verify()
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=output._control_fd,
        )
    except OSError:
        _fail("control_write_failed")
    primary_error = None
    try:
        written = 0
        size = len(data)
        while written < size:
            try:
                result = os.write(descriptor, data[written:])
            except OSError:
                _fail("control_write_failed")
            if type(result) is not int or result <= 0 or result > size - written:
                _fail("control_write_failed")
            written += result
        try:
            os.fsync(descriptor)
            os.fsync(output._control_fd)
        except OSError:
            _fail("control_write_failed")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            os.close(descriptor)
        except BaseException:
            if primary_error is None:
                raise


def _launch_record_bytes(permit_sha256, outer_start_ns, permit):
    record = {
        "schema_version": LAUNCH_RECORD_SCHEMA,
        "permit_sha256": permit_sha256,
        "outer_start_monotonic_ns": outer_start_ns,
        "catalog_sha256": permit["catalog_sha256"],
        "process_id": os.getpid(),
    }
    return _canonical_bytes(record)


# ---------------------------------------------------------------------------
# Trusted bootstrap load
# ---------------------------------------------------------------------------


def _load_bootstrap(package_root):
    path = os.path.join(package_root, *BOOTSTRAP_RELATIVE_PATH.split("/"))
    data = _read_absolute_file(path, MAX_BOOTSTRAP_BYTES, "bootstrap_read_failed")
    if hashlib.sha256(data).hexdigest() != BOOTSTRAP_SHA256:
        _fail("bootstrap_hash_mismatch")
    module = types.ModuleType("p08_u0_bootstrap")
    module.__file__ = path
    try:
        code = compile(data, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        _fail("bootstrap_compile_failed")
    try:
        exec(code, module.__dict__)
    except Exception:
        _fail("bootstrap_compile_failed")
    if not callable(getattr(module, "authenticated_runtime", None)):
        _fail("bootstrap_missing_api")
    return module


def _set_thread_environment():
    for name in _THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------


def _assemble_specification_sources(runtime, package_root, audit_bytes):
    audit = _load_json_bytes(audit_bytes, "audit_invalid")
    if type(audit) is not dict:
        _fail("audit_invalid")
    specification_inputs = audit.get("specification_inputs")
    if type(specification_inputs) is not dict:
        _fail("audit_invalid")

    source_pins = {}
    for entry in specification_inputs.values():
        if type(entry) is not dict:
            _fail("audit_invalid")
        inherited = entry.get("inherited_sources")
        if type(inherited) is not dict or not inherited:
            _fail("audit_invalid")
        for path, pin in inherited.items():
            if type(path) is not str or not path:
                _fail("audit_invalid")
            if type(pin) is not str or len(pin) != 64 or any(char not in _HEX for char in pin):
                _fail("audit_invalid")
            previous = source_pins.get(path)
            if previous is None:
                source_pins[path] = pin
            elif previous != pin:
                _fail("source_pin_conflict")

    if len(source_pins) != _EXPECTED_SPECIFICATION_SOURCES:
        _fail("specification_sources_incomplete")

    result = {}
    for path, pin in source_pins.items():
        if path.startswith("src/"):
            try:
                data = runtime.source_bytes(path)
            except Exception:
                _fail("specification_sources_incomplete")
        else:
            if path not in _NON_SRC_SOURCES:
                _fail("specification_sources_incomplete")
            absolute = os.path.join(package_root, *path.split("/"))
            data = _read_absolute_file(absolute, MAX_SOURCE_SNAPSHOT_BYTES, "input_read_failed")
        if type(data) is not bytes or hashlib.sha256(data).hexdigest() != pin:
            _fail("source_pin_conflict")
        result[path] = data
    return result


def _prepare_runtime_inputs(runtime, package_root, permit, inputs_module):
    metadata_snapshot = {}
    for key, cap in _METADATA_READ_CAPS.items():
        metadata_snapshot[key] = _read_absolute_file(
            permit["metadata_paths"][key], cap, "input_read_failed"
        )
    action_snapshot = {}
    for key in _ACTION_KEYS:
        action_snapshot[key] = _read_absolute_file(
            permit["action_paths"][key], _ACTION_READ_CAP, "input_read_failed"
        )
    audit_bytes = _read_absolute_file(
        permit["specification_audit_path"], _AUDIT_READ_CAP, "input_read_failed"
    )
    candidate_bytes = _read_absolute_file(
        permit["candidate_registry_path"], _CANDIDATE_READ_CAP, "input_read_failed"
    )

    expected_audit = getattr(inputs_module, "_SPEC_AUDIT_SHA256", None)
    if type(expected_audit) is not str:
        _fail("audit_invalid")
    if hashlib.sha256(audit_bytes).hexdigest() != expected_audit:
        _fail("audit_hash_mismatch")

    specification_source_bytes = _assemble_specification_sources(
        runtime, package_root, audit_bytes
    )

    try:
        inputs = inputs_module.prepare_u0_runtime_inputs(
            metadata_bytes=metadata_snapshot,
            action_bytes=action_snapshot,
            specification_audit_bytes=audit_bytes,
            specification_source_bytes=specification_source_bytes,
            candidate_registry_bytes=candidate_bytes,
        )
    except Exception:
        _fail("input_preparation_failed")

    manifest = _bind_manifest(permit, metadata_snapshot["attempt_manifest_bytes"])
    return inputs, manifest


def _bind_manifest(permit, manifest_bytes):
    manifest = _load_json_bytes(manifest_bytes, "manifest_binding_mismatch")
    if type(manifest) is not dict:
        _fail("manifest_binding_mismatch")
    if manifest.get("manifest_sha256") != MANIFEST_SHA256:
        _fail("manifest_binding_mismatch")
    if permit.get("manifest_sha256") != MANIFEST_SHA256:
        _fail("manifest_binding_mismatch")
    jobs = manifest.get("jobs")
    if type(jobs) not in (list, tuple) or len(jobs) != _EXPECTED_JOBS:
        _fail("manifest_binding_mismatch")
    fit_jobs = 0
    prediction_jobs = 0
    for job in jobs:
        if type(job) is not dict:
            _fail("manifest_binding_mismatch")
        stage = job.get("stage")
        if stage == "source_fit":
            fit_jobs += 1
        elif stage == "source_validation_prediction":
            prediction_jobs += 1
        else:
            _fail("manifest_binding_mismatch")
    if fit_jobs != _EXPECTED_FIT_JOBS or prediction_jobs != _EXPECTED_PREDICTION_JOBS:
        _fail("manifest_binding_mismatch")
    return manifest


def _fit_job_id(pair):
    prepared = getattr(pair, "prepared_pair", None)
    if prepared is None:
        _fail("invalid_pair_binding")
    job = _load_json_text(getattr(prepared, "fit_job_json", None), "invalid_pair_binding")
    if type(job) is not dict:
        _fail("invalid_pair_binding")
    job_id = job.get("job_id")
    if type(job_id) is not str or job_id == "":
        _fail("invalid_pair_binding")
    return job_id


# ---------------------------------------------------------------------------
# Numerical runtime setup and serial resource observation
# ---------------------------------------------------------------------------


def _configure_cpu_threads():
    """Authenticated CPU-only configuration; no CUDA query or initialization."""
    try:
        torch_module = importlib.import_module("torch")
    except Exception:
        _fail("resource_setup_failed")
    try:
        torch_module.set_num_threads(1)
    except Exception:
        _fail("resource_setup_failed")
    try:
        torch_module.set_num_interop_threads(1)
    except Exception:
        try:
            if torch_module.get_num_interop_threads() != 1:
                _fail("resource_setup_failed")
        except LaunchError:
            raise
        except Exception:
            _fail("resource_setup_failed")
    return torch_module


def _setup_cuda(torch_module):
    """Initialize CUDA only after authority, ownership, code and input auth."""
    try:
        available = torch_module.cuda.is_available()
    except Exception:
        _fail("cuda_unavailable")
    if available is not True:
        _fail("cuda_unavailable")
    try:
        torch_module.cuda.init()
    except Exception:
        _fail("cuda_unavailable")
    return torch_module


_GPU_PACKET_KEYS = frozenset(
    {"observed", "initialized", "allocated", "reserved", "device_used", "peak"}
)

_GPU_EVIDENCE_KEYS = frozenset(
    {
        "gpu_initialized",
        "gpu_observed",
        "gpu_allocated_bytes",
        "gpu_reserved_bytes",
        "gpu_device_used_bytes",
        "gpu_peak_bytes",
        "gpu_peak_limit_bytes",
        "gpu_peak_within_limit",
        "resource_observation_scope",
        "resource_observation_monotonic_ns",
    }
)


def _gpu_count(value):
    """Exact nonnegative int from the observer packet; reject bool/float/None."""
    if type(value) is not int or type(value) is bool or value < 0:
        _fail("resource_setup_failed")
    return value


def _gpu_resource_evidence(cuda, scope, observed_at_ns):
    """Scalar evidence reconstructed from one exact ``_probe_cuda`` packet."""
    if type(scope) is not str or scope == "":
        _fail("resource_setup_failed")
    if (
        type(observed_at_ns) is not int
        or type(observed_at_ns) is bool
        or observed_at_ns < 0
    ):
        _fail("resource_setup_failed")
    if type(cuda) is not dict or set(cuda) != _GPU_PACKET_KEYS:
        _fail("resource_setup_failed")
    initialized = cuda["initialized"]
    observed = cuda["observed"]
    if type(initialized) is not bool or type(observed) is not bool:
        _fail("resource_setup_failed")
    allocated = _gpu_count(cuda["allocated"])
    reserved = _gpu_count(cuda["reserved"])
    device_used = _gpu_count(cuda["device_used"])
    peak = _gpu_count(cuda["peak"])
    limit = _EXPECTED_LIMITS["allocated_gpu_bytes"]
    if initialized is False:
        if observed is not False:
            _fail("resource_setup_failed")
        # The authenticated ``_probe_cuda`` contract reports exact zero for
        # every uninitialized quantity; any nonzero value is contradictory and
        # is rejected before the unknown-quantity conversion below.
        if allocated != 0 or reserved != 0 or device_used != 0 or peak != 0:
            _fail("resource_setup_failed")
        # Unobserved GPU/global usage stays unknown rather than measured zero.
        allocated = None
        reserved = None
        device_used = None
        peak = None
        within_limit = None
    else:
        if observed is not True:
            _fail("resource_setup_failed")
        if allocated > reserved or allocated > peak:
            _fail("resource_setup_failed")
        if peak > limit:
            _fail("resource_limit_exceeded")
        within_limit = peak <= limit
    return {
        "gpu_initialized": initialized,
        "gpu_observed": observed,
        "gpu_allocated_bytes": allocated,
        "gpu_reserved_bytes": reserved,
        "gpu_device_used_bytes": device_used,
        "gpu_peak_bytes": peak,
        "gpu_peak_limit_bytes": limit,
        "gpu_peak_within_limit": within_limit,
        "resource_observation_scope": scope,
        "resource_observation_monotonic_ns": observed_at_ns,
    }


def _observe_resources(
    serial_module,
    resources_module,
    torch_module,
    target_root,
    usage,
    scope="resource_observation",
):
    try:
        started, finished, resources, cuda = serial_module._sample(
            target_root, torch_module, 1
        )
    except Exception:
        _fail("resource_setup_failed")
    if (
        type(started) is not int
        or type(finished) is not int
        or started < 0
        or finished < 0
        or finished < started
    ):
        _fail("resource_setup_failed")
    observed_at_ns = finished
    try:
        record = resources_module.evaluate_resource_snapshot("U0", usage, resources)
    except Exception:
        _fail("resource_setup_failed")
    if type(record) is not dict or record.get("within_proposed_limits") is not True:
        _fail("resource_limit_exceeded")
    evidence = _gpu_resource_evidence(cuda, scope, observed_at_ns)
    # The authenticated snapshot is never mutated in place; the additional
    # lifetime evidence travels beside it in a separate wrapper.
    return {"snapshot": record, "evidence": evidence}


def _final_resource_gate(
    serial_module, resources_module, output, torch_module, session_report, outer_start_ns, budget
):
    if type(session_report) is not dict:
        _fail("session_incomplete")
    recorded = session_report.get("recorded_artifact_bytes")
    observed = session_report.get("observed_logical_bytes")
    if type(recorded) is not int or recorded < 0:
        _fail("session_incomplete")
    if type(observed) is not int or observed < 0:
        _fail("session_incomplete")
    actual = budget.check_layout(output, 0)
    high = max(recorded, observed, actual, budget.high_water_bytes)
    budget.high_water_bytes = high
    usage = {
        "model_fit_attempts": session_report["fit_attempt_count"],
        "scalar_calibration_attempts": 0,
        "active_wall_ns": _measured_elapsed_ns(outer_start_ns),
        "new_artifact_bytes": high,
    }
    return _observe_resources(
        serial_module,
        resources_module,
        torch_module,
        output.root,
        usage,
        "post_session_pre_terminal_persistence",
    )


# ---------------------------------------------------------------------------
# Session execution
# ---------------------------------------------------------------------------


def _verify_session_report(report):
    if type(report) is not dict:
        _fail("session_incomplete")
    if report.get("closed") is not True:
        _fail("session_incomplete")
    if report.get("observation_current") is not True:
        _fail("session_incomplete")
    if report.get("incomplete") is not False:
        _fail("session_incomplete")
    if report.get("completed_pair_count") != _EXPECTED_PAIRS:
        _fail("session_incomplete")
    if report.get("fit_attempt_count") != _EXPECTED_FIT_JOBS:
        _fail("session_incomplete")
    if report.get("prediction_attempt_count") != _EXPECTED_PREDICTION_JOBS:
        _fail("session_incomplete")
    for key in ("recorded_artifact_bytes", "observed_logical_bytes"):
        value = report.get(key)
        if type(value) is not int or value < 0:
            _fail("session_incomplete")


def _run(package_root, permit, outer_start_ns, launch_sha, output, budget):
    bootstrap = _load_bootstrap(package_root)
    budget.check_layout(output, 0)
    session_report = None
    resource_record = None
    with bootstrap.authenticated_runtime(package_root) as runtime:
        metadata = runtime.verify_loaded()
        if metadata.get("catalog_sha256") != permit["catalog_sha256"]:
            _fail("permit_catalog_mismatch")
        if metadata.get("source_revision") != permit["source_revision"]:
            _fail("permit_revision_mismatch")

        inputs_module = importlib.import_module("atlas_sers.evaluation.p08_u0_runtime_inputs")
        serial_module = importlib.import_module("atlas_sers.evaluation.p08_serial_resources")
        resources_module = importlib.import_module("atlas_sers.evaluation.p08_resources")
        store_module = importlib.import_module("atlas_sers.evaluation.p08_u0_store")
        session_module = importlib.import_module("atlas_sers.evaluation.p08_u0_session")
        budget.check_layout(output, 0)

        # CPU thread configuration before input preparation; no CUDA here.
        torch_module = _configure_cpu_threads()
        budget.check_layout(output, 0)

        inputs, manifest = _prepare_runtime_inputs(runtime, package_root, permit, inputs_module)
        budget.check_layout(output, 0)

        _setup_cuda(torch_module)
        budget.check_layout(output, 0)

        # Full point-in-time resource observation before any fit/session work,
        # through the serial observer on the existing output root.
        initial_bytes = budget.check_layout(output, 0)
        _observe_resources(
            serial_module,
            resources_module,
            torch_module,
            output.root,
            {
                "model_fit_attempts": 0,
                "scalar_calibration_attempts": 0,
                "active_wall_ns": _measured_elapsed_ns(outer_start_ns),
                "new_artifact_bytes": initial_bytes,
            },
            "pre_session_setup",
        )

        owner = None
        session = None
        session_close_attempted = False
        primary_error = None
        try:
            owner = store_module.create_store(output.journal_root, manifest)
            session = session_module._SourceSession(
                owner,
                inputs,
                artifact_root=output.artifacts_root,
                torch_module=torch_module,
                started_monotonic_ns=outer_start_ns,
                launch_control_root=output.control_root,
                launch_record_sha256=launch_sha,
            )
            for pair in inputs.pairs:
                runtime.verify_loaded()
                session.run_pair(_fit_job_id(pair))
            session_close_attempted = True
            session_report = session.close()
        except BaseException as exc:
            primary_error = exc
            if session is not None and not session_close_attempted:
                session_close_attempted = True
                try:
                    session.close()
                except BaseException:
                    pass
            raise
        finally:
            if owner is not None:
                try:
                    owner.close()
                except BaseException:
                    if primary_error is None:
                        raise

        _verify_session_report(session_report)
        runtime.verify_loaded()
        resource_record = _final_resource_gate(
            serial_module,
            resources_module,
            output,
            torch_module,
            session_report,
            outer_start_ns,
            budget,
        )
    return {"session_report": session_report, "resource_record": resource_record}


# ---------------------------------------------------------------------------
# Terminal records
# ---------------------------------------------------------------------------


def _reason_for_exception(exc):
    if isinstance(exc, LaunchError) and exc.reason_code in _REASON_CODES:
        return exc.reason_code
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "interrupted"
    return "internal_error"


def _resource_evidence(data):
    """Validate and reconstruct exact scalar GPU evidence for finalization."""
    if type(data) is not dict:
        _fail("resource_setup_failed")
    record = data.get("resource_record")
    if type(record) is not dict:
        _fail("resource_setup_failed")
    snapshot = record.get("snapshot")
    if type(snapshot) is not dict or snapshot.get("within_proposed_limits") is not True:
        _fail("resource_setup_failed")
    evidence = record.get("evidence")
    if type(evidence) is not dict or set(evidence) != _GPU_EVIDENCE_KEYS:
        _fail("resource_setup_failed")

    initialized = evidence["gpu_initialized"]
    observed = evidence["gpu_observed"]
    if type(initialized) is not bool or type(observed) is not bool:
        _fail("resource_setup_failed")
    scope = evidence["resource_observation_scope"]
    observed_at_ns = evidence["resource_observation_monotonic_ns"]
    if type(scope) is not str or scope == "":
        _fail("resource_setup_failed")
    if (
        type(observed_at_ns) is not int
        or type(observed_at_ns) is bool
        or observed_at_ns < 0
    ):
        _fail("resource_setup_failed")
    limit = evidence["gpu_peak_limit_bytes"]
    if type(limit) is not int or type(limit) is bool or limit < 0:
        _fail("resource_setup_failed")
    if limit != _EXPECTED_LIMITS["allocated_gpu_bytes"]:
        _fail("resource_setup_failed")

    if initialized is False:
        if observed is not False:
            _fail("resource_setup_failed")
        for key in (
            "gpu_allocated_bytes",
            "gpu_reserved_bytes",
            "gpu_device_used_bytes",
            "gpu_peak_bytes",
            "gpu_peak_within_limit",
        ):
            if evidence[key] is not None:
                _fail("resource_setup_failed")
        allocated = None
        reserved = None
        device_used = None
        peak = None
        within_limit = None
    else:
        if observed is not True:
            _fail("resource_setup_failed")
        allocated = evidence["gpu_allocated_bytes"]
        reserved = evidence["gpu_reserved_bytes"]
        device_used = evidence["gpu_device_used_bytes"]
        peak = evidence["gpu_peak_bytes"]
        for value in (allocated, reserved, device_used, peak):
            if type(value) is not int or type(value) is bool or value < 0:
                _fail("resource_setup_failed")
        if allocated > reserved or allocated > peak:
            _fail("resource_setup_failed")
        if peak > limit:
            _fail("resource_setup_failed")
        if evidence["gpu_peak_within_limit"] is not True:
            _fail("resource_setup_failed")
        within_limit = True
    return {
        "gpu_initialized": initialized,
        "gpu_observed": observed,
        "gpu_allocated_bytes": allocated,
        "gpu_reserved_bytes": reserved,
        "gpu_device_used_bytes": device_used,
        "gpu_peak_bytes": peak,
        "gpu_peak_limit_bytes": limit,
        "gpu_peak_within_limit": within_limit,
        "resource_observation_scope": scope,
        "resource_observation_monotonic_ns": observed_at_ns,
    }


def _finalize_success(output, permit, permit_sha256, outer_start_ns, data, budget):
    session_report = data["session_report"]
    if type(session_report) is not dict:
        _fail("session_incomplete")
    completed = session_report.get("completed_pair_count")
    fit = session_report.get("fit_attempt_count")
    prediction = session_report.get("prediction_attempt_count")
    recorded = session_report.get("recorded_artifact_bytes")
    observed = session_report.get("observed_logical_bytes")
    for value in (completed, fit, prediction, recorded, observed):
        if type(value) is not int or value < 0:
            _fail("session_incomplete")

    retained_before = budget.check_layout(output, 0)
    # Fold observed/logical high-water into the retained maximum before any
    # prospective terminal admission, not only afterwards.
    if observed > budget.high_water_bytes:
        budget.high_water_bytes = observed
    if recorded > budget.high_water_bytes:
        budget.high_water_bytes = recorded
    terminal_elapsed = _measured_elapsed_ns(outer_start_ns)
    evidence = _resource_evidence(data)
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "status": "source_execution_complete_pending_finalization",
        "measurement_scope": "pre_terminal_persistence",
        "reason_code": None,
        "execution_authorized": True,
        "review_acceptance_granted": False,
        "future_execution_authorized": False,
        "permit_sha256": permit_sha256,
        "proposal_sha256": PROPOSAL_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "catalog_sha256": permit["catalog_sha256"],
        "source_revision": permit["source_revision"],
        "counters_known": True,
        "completed_pair_count": completed,
        "fit_attempt_count": fit,
        "prediction_attempt_count": prediction,
        "retained_bytes": retained_before,
        "outer_elapsed_ns": terminal_elapsed,
    }
    terminal.update(evidence)
    terminal_bytes = _canonical_bytes(terminal)
    # Time and exact prospective terminal payload before writing, never after.
    budget.check_layout(output, len(terminal_bytes))
    _write_control_file(output, TERMINAL_FILE_NAME, terminal_bytes)

    # Retained accounting after the write, then owned-descriptor cleanup.
    retained_after = budget.check_layout(output, 0)
    if retained_after > budget.high_water_bytes:
        budget.high_water_bytes = retained_after
    close_started = time.monotonic_ns()
    output.close()
    close_finished = time.monotonic_ns()

    # An exhausted finalization must fail here, after the single terminal
    # attempt, so no second terminal record is ever written or retried and
    # the pending terminal record is preserved.
    close_elapsed = close_finished - outer_start_ns
    if type(close_elapsed) is not int or close_elapsed < 0:
        _fail("resource_limit_exceeded")
    if close_elapsed >= permit["limits"]["wall_ns"]:
        _fail("resource_limit_exceeded")
    if budget.high_water_bytes >= permit["limits"]["artifact_bytes"]:
        _fail("resource_limit_exceeded")

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "succeeded",
        "execution_authorized": True,
        "review_acceptance_granted": False,
        "future_execution_authorized": False,
        "external_dependencies_authenticated": False,
        "project_sources_verified": True,
        "permit_sha256": permit_sha256,
        "proposal_sha256": PROPOSAL_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "catalog_sha256": permit["catalog_sha256"],
        "source_revision": permit["source_revision"],
        "completed_pair_count": completed,
        "fit_attempt_count": fit,
        "prediction_attempt_count": prediction,
        "retained_bytes": retained_after,
        "high_water_bytes": budget.high_water_bytes,
        "outer_elapsed_ns": close_elapsed,
        "pre_terminal_elapsed_ns": terminal_elapsed,
        "cleanup_elapsed_ns": close_finished - close_started,
        "terminal_written": True,
        "raw_artifact_bytes": recorded,
        "observed_logical_bytes": observed,
    }
    report.update(evidence)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def _finalize_failure(output, permit, permit_sha256, outer_start_ns, exc, budget):
    if output._terminal_attempted or output._closed:
        return
    reason_code = _reason_for_exception(exc)
    try:
        retained_bytes = output.verify()
    except BaseException:
        retained_bytes = None
    elapsed = time.monotonic_ns() - outer_start_ns
    catalog_sha256 = permit.get("catalog_sha256") if type(permit) is dict else None
    source_revision = permit.get("source_revision") if type(permit) is dict else None
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "status": "failed",
        "measurement_scope": "pre_terminal_persistence",
        "reason_code": reason_code,
        "execution_authorized": False,
        "review_acceptance_granted": False,
        "future_execution_authorized": False,
        "permit_sha256": permit_sha256,
        "proposal_sha256": PROPOSAL_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "catalog_sha256": catalog_sha256,
        "source_revision": source_revision,
        "counters_known": False,
        "completed_pair_count": None,
        "fit_attempt_count": None,
        "prediction_attempt_count": None,
        "retained_bytes": retained_bytes,
        "outer_elapsed_ns": elapsed,
    }
    payload = _canonical_bytes(terminal)
    if budget is not None:
        try:
            budget.check_layout(output, len(payload))
        except BaseException:
            # Budget exhausted: keep existing evidence, spend no more.
            return
    try:
        _write_control_file(output, TERMINAL_FILE_NAME, payload)
    except BaseException:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def launch(package_root, permit_file):
    """Validate independent authority, claim output, run one fixed U0 smoke.

    An unset production pin (``APPROVED_PERMIT_SHA256`` is ``None``) denies with
    ``scientific_execution_not_authorized`` before the permit file is read.  This
    release accepts only the exact bytes of the independently approved private
    permit.  A successful return remains scalar-only.  Any original failure still
    propagates through all diagnostic and close failures.
    """
    outer_start_ns = time.monotonic_ns()
    _require_environment()
    if APPROVED_PERMIT_SHA256 is None:
        _fail("scientific_execution_not_authorized")
    permit_sha256 = _validate_hex(APPROVED_PERMIT_SHA256, 64, "permit_pin_invalid")
    package_root = _canonical_absolute(package_root, "package_root_invalid")
    _require_directory(package_root, "package_root_invalid")

    permit_bytes = _read_permit(permit_file, permit_sha256)
    permit = _validate_permit(permit_bytes, package_root)

    budget = _Budget(permit, outer_start_ns)
    output = None
    primary_error = None
    try:
        output = _claim_output(permit["output_root"], budget)
        _set_thread_environment()
        launch_bytes = _launch_record_bytes(permit_sha256, outer_start_ns, permit)
        budget.check_layout(output, len(launch_bytes))
        _write_control_file(output, LAUNCH_FILE_NAME, launch_bytes)
        launch_sha = hashlib.sha256(launch_bytes).hexdigest()
        data = _run(package_root, permit, outer_start_ns, launch_sha, output, budget)
        return _finalize_success(output, permit, permit_sha256, outer_start_ns, data, budget)
    except BaseException as exc:
        primary_error = exc
        if output is not None:
            try:
                exc._p08_authorized_work = True
            except BaseException:
                pass
            try:
                _finalize_failure(output, permit, permit_sha256, outer_start_ns, exc, budget)
            except BaseException:
                pass
        raise
    finally:
        if output is not None:
            try:
                output.close()
            except BaseException:
                if primary_error is None:
                    raise


def _emit(payload):
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _emit_error(reason_code, status):
    _emit(
        {
            "schema_version": REPORT_SCHEMA,
            "status": status,
            "execution_authorized": False,
            "review_acceptance_granted": False,
            "future_execution_authorized": False,
            "error": reason_code,
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--permit", required=True)
    args = parser.parse_args(argv)
    try:
        report = launch(args.package_root, args.permit)
    except LaunchError as exc:
        authorized = getattr(exc, "_p08_authorized_work", False)
        _emit_error(exc.reason_code, "failed" if authorized else "denied")
        return 2
    except KeyboardInterrupt:
        _emit_error("interrupted", "failed")
        return 130
    except BaseException:
        _emit_error("internal_error", "failed")
        return 2
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
