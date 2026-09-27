"""Authority loading and operational resource guards for the P05 recovery run.

This module validates the public P05 recovery permit and exposes fail-closed
operational resource checks.  It performs no recovery execution, creates no
leases, and never imports torch or numpy.  Torch is accepted as an injected
object by :func:`check_resources`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "RecoveryAuthorityError",
    "RECOVERY_SCHEMA_VERSION",
    "RECOVERY_PROTOCOL_VERSION",
    "RECOVERY_PERMIT_SHA256",
    "BASECOMPREHENSIVE_PERMIT_SHA256",
    "ORIGINAL_EVIDENCE_ANCHOR_SHA256",
    "MAXIMUM_CUDA_ALLOCATED_BYTES",
    "FROZEN_SOURCE_FIT_CUDA_CAP_BYTES",
    "MINIMUM_FREE_CUDA_BYTES",
    "MINIMUM_HOST_AVAILABLE_BYTES_BEFORE_LAUNCH",
    "MINIMUM_HOST_AVAILABLE_BYTES_DURING_EXECUTION",
    "MAXIMUM_TOTAL_SECONDS",
    "PRIVATE_STORAGE_CEILING_BYTES",
    "PRIOR_SCIENTIFIC_SECONDS_CHARGED",
    "PRELAUNCH_AUDIT_RESERVE_SECONDS",
    "MAXIMUM_NEW_NEURAL_EXECUTIONS",
    "MAXIMUM_NEW_OPTIMIZER_STEPS",
    "validate_recovery_permit",
    "load_recovery_permit",
    "read_host_available_bytes",
    "check_resources",
]

RECOVERY_SCHEMA_VERSION = "nato-sers-p05-comprehensive-recovery-v1"
RECOVERY_PROTOCOL_VERSION = "nato-sers-p05-comprehensive-recovery-20260927-v1"
RECOVERY_PERMIT_SHA256 = "dfd19af6546de2e891f35da27f496c3e4ebc8e4fd000c4dc35fac8a9d75e913e"
BASECOMPREHENSIVE_PERMIT_SHA256 = "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8"
ORIGINAL_EVIDENCE_ANCHOR_SHA256 = "419e31c24ae4c2eae1d3de72697b124522928ddf18d211210646f4308116af3b"

MAXIMUM_CUDA_ALLOCATED_BYTES = 8589934592
FROZEN_SOURCE_FIT_CUDA_CAP_BYTES = 4294967296
MINIMUM_FREE_CUDA_BYTES = 9663676416
MINIMUM_HOST_AVAILABLE_BYTES_BEFORE_LAUNCH = 17179869184
MINIMUM_HOST_AVAILABLE_BYTES_DURING_EXECUTION = 8589934592
MAXIMUM_TOTAL_SECONDS = 172800
PRIVATE_STORAGE_CEILING_BYTES = 107374182400
PRIOR_SCIENTIFIC_SECONDS_CHARGED = 36000
PRELAUNCH_AUDIT_RESERVE_SECONDS = 3600
MAXIMUM_NEW_NEURAL_EXECUTIONS = 17785
MAXIMUM_NEW_OPTIMIZER_STEPS = 14228000

_MAX_PERMIT_BYTES = 64 * 1024
_MAX_MEMINFO_BYTES = 1024 * 1024
_VALID_PHASES = ("launch", "fit", "epoch")
_MEMINFO_PATTERN = re.compile(r"MemAvailable:[ \t]+([0-9]{1,20})[ \t]+kB")

_RESOURCE_FIELD_EXPECTATIONS = {
    "maximum_cuda_allocated_bytes": MAXIMUM_CUDA_ALLOCATED_BYTES,
    "frozen_source_fit_cuda_cap_bytes": FROZEN_SOURCE_FIT_CUDA_CAP_BYTES,
    "minimum_free_cuda_bytes_before_launch": MINIMUM_FREE_CUDA_BYTES,
    "minimum_host_available_bytes_before_launch": MINIMUM_HOST_AVAILABLE_BYTES_BEFORE_LAUNCH,
    "minimum_host_available_bytes_during_execution": MINIMUM_HOST_AVAILABLE_BYTES_DURING_EXECUTION,
    "maximum_total_seconds": MAXIMUM_TOTAL_SECONDS,
    "private_storage_ceiling_bytes": PRIVATE_STORAGE_CEILING_BYTES,
    "prior_scientific_seconds_charged_upper_bound": PRIOR_SCIENTIFIC_SECONDS_CHARGED,
    "prelaunch_audit_reserve_seconds": PRELAUNCH_AUDIT_RESERVE_SECONDS,
    "maximum_new_neural_executions": MAXIMUM_NEW_NEURAL_EXECUTIONS,
    "maximum_new_optimizer_steps": MAXIMUM_NEW_OPTIMIZER_STEPS,
    "maximum_fit_seconds": 120,
    "authorized_replay_attempts": 1,
    "automatic_retries": 0,
}

_PINNED_FIELD_EXPECTATIONS = {
    "base_comprehensive_permit_sha256": BASECOMPREHENSIVE_PERMIT_SHA256,
    "original_evidence_anchor_sha256": ORIGINAL_EVIDENCE_ANCHOR_SHA256,
}


class RecoveryAuthorityError(ValueError):
    """Stable, path-free authority/guard failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_json_structure(value: Any, *, depth: int = 0) -> None:
    if depth > 128:
        raise RecoveryAuthorityError("input_nesting_exceeded")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecoveryAuthorityError("non_finite_number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecoveryAuthorityError("non_string_key")
            _validate_json_structure(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_structure(item, depth=depth + 1)
        return
    raise RecoveryAuthorityError("invalid_value_type")


def _validate_versions(payload: Mapping) -> None:
    if payload.get("schema_version") != RECOVERY_SCHEMA_VERSION:
        raise RecoveryAuthorityError("schema_version_mismatch")
    if payload.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise RecoveryAuthorityError("protocol_version_mismatch")


def _validate_explicit_fields(payload: Mapping) -> None:
    for field, expected in _RESOURCE_FIELD_EXPECTATIONS.items():
        value = payload.get(field)
        if type(value) is not int or value != expected:
            raise RecoveryAuthorityError("invalid_resource_field")
    for field, expected in _PINNED_FIELD_EXPECTATIONS.items():
        value = payload.get(field)
        if not isinstance(value, str) or value != expected:
            raise RecoveryAuthorityError("invalid_pinned_field")


def validate_recovery_permit(payload: Mapping) -> dict:
    """Validate the exact public recovery permit and return an independent copy.

    The caller cannot supply a digest: production accepts only the pinned
    canonical payload. Replay consumption belongs to an exclusive filesystem
    lease; this loader does not inspect or create that lease. An extra caller
    flag cannot change the pinned authority.
    """

    if not isinstance(payload, Mapping):
        raise RecoveryAuthorityError("not_a_mapping")
    _validate_json_structure(payload)
    _validate_versions(payload)
    _validate_explicit_fields(payload)
    try:
        raw = _canonical_bytes(payload)
    except (TypeError, ValueError, RecursionError) as exc:
        raise RecoveryAuthorityError("permit_not_json") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RECOVERY_PERMIT_SHA256:
        raise RecoveryAuthorityError("permit_digest_mismatch")
    return json.loads(raw.decode("utf-8"))


def _coerce_path(path: Path | str) -> Path:
    if not isinstance(path, (Path, str)):
        raise RecoveryAuthorityError("path_invalid")
    if not str(path) or "\x00" in str(path):
        raise RecoveryAuthorityError("path_invalid")
    target = Path(path)
    if ".." in target.parts:
        raise RecoveryAuthorityError("path_traversal_rejected")
    return target


def _reject_symlink_ancestors(path: Path, reason_code: str) -> None:
    current = Path(os.path.abspath(os.fspath(path)))
    while True:
        if os.path.islink(current):
            raise RecoveryAuthorityError(reason_code)
        parent = current.parent
        if parent == current:
            return
        current = parent


def _object_pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryAuthorityError("permit_duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(token):
    raise RecoveryAuthorityError("permit_nonfinite_json")


def _read_bounded(path: Path, limit: int, too_large_code: str, unreadable_code: str) -> bytes:
    try:
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
        with os.fdopen(os.open(path, flags), "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RecoveryAuthorityError(unreadable_code)
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise RecoveryAuthorityError(unreadable_code) from exc
    if len(raw) > limit:
        raise RecoveryAuthorityError(too_large_code)
    return raw


def load_recovery_permit(path: Path | str) -> dict:
    """Load and validate the public recovery permit from a bounded regular file."""

    target = _coerce_path(path)
    _reject_symlink_ancestors(target, "permit_symlink")
    try:
        info = os.stat(target)
    except OSError as exc:
        raise RecoveryAuthorityError("permit_unreadable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RecoveryAuthorityError("permit_not_regular_file")
    if info.st_size > _MAX_PERMIT_BYTES:
        raise RecoveryAuthorityError("permit_too_large")
    raw = _read_bounded(target, _MAX_PERMIT_BYTES, "permit_too_large", "permit_unreadable")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecoveryAuthorityError("permit_not_utf8") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_object_pairs_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except RecoveryAuthorityError:
        raise
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError) as exc:
        raise RecoveryAuthorityError("permit_invalid_json") from exc
    return validate_recovery_permit(payload)


def read_host_available_bytes(path: Path | str = "/proc/meminfo") -> int:
    """Return ``MemAvailable`` in bytes from a bounded, fail-closed meminfo read."""

    target = _coerce_path(path)
    _reject_symlink_ancestors(target, "meminfo_symlink")
    try:
        info = os.stat(target)
    except OSError as exc:
        raise RecoveryAuthorityError("meminfo_unreadable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RecoveryAuthorityError("meminfo_not_regular_file")
    raw = _read_bounded(target, _MAX_MEMINFO_BYTES, "meminfo_too_large", "meminfo_unreadable")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecoveryAuthorityError("meminfo_not_utf8") from exc
    matches = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("MemAvailable:"):
            matches.append(stripped)
    if not matches:
        raise RecoveryAuthorityError("meminfo_missing")
    if len(matches) > 1:
        raise RecoveryAuthorityError("meminfo_duplicate")
    match = _MEMINFO_PATTERN.fullmatch(matches[0])
    if match is None:
        raise RecoveryAuthorityError("meminfo_malformed")
    return int(match.group(1)) * 1024


def _strict_nonneg_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def check_resources(
    torch: Any,
    *,
    phase: str,
    meminfo_path: Path | str = "/proc/meminfo",
) -> dict:
    """Detect low headroom; no tensors, allocator reset or cache-clearing calls.

    CUDA queries can initialize the device context. The caller must invoke this
    only during an authorized numerical-stage preflight or execution.
    """

    if phase not in _VALID_PHASES:
        raise RecoveryAuthorityError("invalid_phase")

    host_available = read_host_available_bytes(meminfo_path)
    minimum_host = (
        MINIMUM_HOST_AVAILABLE_BYTES_BEFORE_LAUNCH
        if phase == "launch"
        else MINIMUM_HOST_AVAILABLE_BYTES_DURING_EXECUTION
    )
    if host_available < minimum_host:
        raise RecoveryAuthorityError("insufficient_host_memory")

    try:
        available = torch.cuda.is_available()
    except (OSError, RuntimeError) as exc:
        raise RecoveryAuthorityError("cuda_query_failed") from exc
    if available is not True:
        raise RecoveryAuthorityError("cuda_unavailable")

    try:
        allocated = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        info = torch.cuda.mem_get_info() if phase == "launch" else None
    except (OSError, RuntimeError) as exc:
        raise RecoveryAuthorityError("cuda_query_failed") from exc

    if not _strict_nonneg_int(allocated) or not _strict_nonneg_int(peak):
        raise RecoveryAuthorityError("invalid_cuda_metric")
    if peak < allocated:
        raise RecoveryAuthorityError("inconsistent_cuda_memory")
    if allocated > MAXIMUM_CUDA_ALLOCATED_BYTES or peak > MAXIMUM_CUDA_ALLOCATED_BYTES:
        raise RecoveryAuthorityError("cuda_allocation_exceeded")

    result = {
        "phase": phase,
        "host_available_bytes": host_available,
        "cuda_allocated_bytes": allocated,
        "cuda_max_allocated_bytes": peak,
    }

    if phase == "launch":
        if not isinstance(info, (tuple, list)) or len(info) != 2:
            raise RecoveryAuthorityError("invalid_cuda_metric")
        free, total = info
        if not _strict_nonneg_int(free) or not _strict_nonneg_int(total):
            raise RecoveryAuthorityError("invalid_cuda_metric")
        if free > total or allocated + free > total:
            raise RecoveryAuthorityError("inconsistent_cuda_memory")
        if free < MINIMUM_FREE_CUDA_BYTES:
            raise RecoveryAuthorityError("insufficient_cuda_free")
        result["cuda_free_bytes"] = free
        result["cuda_total_bytes"] = total

    return result
