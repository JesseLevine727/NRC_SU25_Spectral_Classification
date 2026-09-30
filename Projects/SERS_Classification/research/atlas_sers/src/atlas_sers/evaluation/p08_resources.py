"""Pure, no-execution checker for the PROPOSED P08 resource ceilings.

This module only compares caller-supplied, caller-authenticated cumulative
usage and caller-supplied measured resources against three *proposed*
ceilings.  It is not a live monitor, persistent journal, reservation,
job-admission authority or scientific permit.  Callers must authenticate the
truth of their own inputs; a hash does not authenticate truth.  None of the
proposed limits is approved: returning or passing them is not permission.
U1 is expected to include accepted U0 usage; this leaf cannot reset or prove
that transfer.  Examinations never consult real hardware, mutate inputs, fit
models or authorize scientific execution.
"""

from __future__ import annotations

from .p08_qc_blocks import canonical_sha256

_GIB = 1024**3
_NANOS_PER_SECOND = 10**9

_SCHEMA_VERSION = "nato-sers-p08-resource-snapshot-v1"

_REASON_CODES = frozenset(
    {
        "invalid_resource_input",
        "invalid_stage",
        "invalid_usage",
        "invalid_resources",
        "scientific_execution_not_authorized",
    }
)

_USAGE_KEYS = (
    "model_fit_attempts",
    "scalar_calibration_attempts",
    "active_wall_ns",
    "new_artifact_bytes",
)

_RESOURCE_KEYS = (
    "filesystem_free_bytes",
    "process_tree_rss_bytes",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "cuda_device_used_bytes",
    "active_cpu_workers",
    "active_gpu_workers",
    "model_threads",
    "blas_threads",
    "torch_threads",
)

_LIMIT_KEYS = (
    "stage",
    "model_fit_attempts",
    "scalar_calibration_attempts",
    "active_wall_ns",
    "new_artifact_bytes",
    "process_tree_rss_bytes",
    "cuda_allocated_bytes",
    "max_cpu_workers",
    "max_gpu_workers",
    "worker_threads",
    "filesystem_reserve_bytes",
    "automatic_retries",
)


def _stage_limits(model_fit, scalars, wall_seconds, artifact_gib, rss_gib, cuda_gib, cpu):
    return {
        "model_fit_attempts": model_fit,
        "scalar_calibration_attempts": scalars,
        "active_wall_ns": wall_seconds * _NANOS_PER_SECOND,
        "new_artifact_bytes": artifact_gib * _GIB,
        "process_tree_rss_bytes": rss_gib * _GIB,
        "cuda_allocated_bytes": cuda_gib * _GIB,
        "max_cpu_workers": cpu,
        "max_gpu_workers": 1,
        "worker_threads": 1,
        "filesystem_reserve_bytes": 30 * _GIB,
        "automatic_retries": 0,
    }


_STAGE_LIMITS = {
    "U0": _stage_limits(78, 0, 5400, 8, 16, 8, 2),
    "U1": _stage_limits(195202, 3354, 172800, 80, 24, 8, 4),
    "Q1": _stage_limits(1630980, 53880, 864000, 400, 24, 8, 4),
}


class ResourceGuardError(ValueError):
    """Static, non-echoing resource-guard error."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_resource_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _validate_mapping(value, expected_keys, reason):
    if type(value) is not dict:
        raise ResourceGuardError(reason)
    if set(value) != set(expected_keys):
        raise ResourceGuardError(reason)
    for key in expected_keys:
        item = value[key]
        if type(item) is not int or item < 0:
            raise ResourceGuardError(reason)


def proposed_limits(stage):
    """Return a fresh plain dict of proposed ceilings; never permission."""
    try:
        if type(stage) is not str or stage not in _STAGE_LIMITS:
            raise ResourceGuardError("invalid_stage")
        base = _STAGE_LIMITS[stage]
        limits = {"stage": stage}
        for key in _LIMIT_KEYS:
            if key != "stage":
                limits[key] = base[key]
        return limits
    except ResourceGuardError:
        raise
    except Exception:
        raise ResourceGuardError("invalid_resource_input") from None


def _evaluate_resource_snapshot(stage, usage, resources):
    limits = proposed_limits(stage)
    _validate_mapping(usage, _USAGE_KEYS, "invalid_usage")
    _validate_mapping(resources, _RESOURCE_KEYS, "invalid_resources")

    used_fit = usage["model_fit_attempts"]
    used_scalars = usage["scalar_calibration_attempts"]
    used_wall = usage["active_wall_ns"]
    used_artifacts = usage["new_artifact_bytes"]

    remaining_fit = max(0, limits["model_fit_attempts"] - used_fit)
    remaining_scalars = max(0, limits["scalar_calibration_attempts"] - used_scalars)
    remaining_wall = max(0, limits["active_wall_ns"] - used_wall)
    remaining_artifacts = max(0, limits["new_artifact_bytes"] - used_artifacts)
    required_filesystem_free = limits["filesystem_reserve_bytes"] + remaining_artifacts

    breaches = []
    if used_fit > limits["model_fit_attempts"]:
        breaches.append("model_fit_budget_exceeded")
    if used_scalars > limits["scalar_calibration_attempts"]:
        breaches.append("scalar_calibration_budget_exceeded")
    if used_wall >= limits["active_wall_ns"]:
        breaches.append("active_wall_budget_exhausted")
    if used_artifacts >= limits["new_artifact_bytes"]:
        breaches.append("artifact_budget_exhausted")
    if resources["process_tree_rss_bytes"] > limits["process_tree_rss_bytes"]:
        breaches.append("process_tree_memory_exceeded")
    if resources["cuda_allocated_bytes"] > limits["cuda_allocated_bytes"]:
        breaches.append("cuda_allocated_memory_exceeded")
    if resources["active_cpu_workers"] > limits["max_cpu_workers"]:
        breaches.append("cpu_worker_limit_exceeded")
    if resources["active_gpu_workers"] > limits["max_gpu_workers"]:
        breaches.append("gpu_worker_limit_exceeded")
    if (
        resources["model_threads"] != 1
        or resources["blas_threads"] != 1
        or resources["torch_threads"] != 1
    ):
        breaches.append("worker_thread_limit_violated")
    if resources["filesystem_free_bytes"] < required_filesystem_free:
        breaches.append("filesystem_reserve_insufficient")

    record = {
        "schema_version": _SCHEMA_VERSION,
        "execution_authorized": False,
        "stage": limits["stage"],
        "limits": limits,
        "usage": dict(usage),
        "resources": dict(resources),
        "remaining_model_fit_attempts": remaining_fit,
        "remaining_scalar_calibration_attempts": remaining_scalars,
        "remaining_active_wall_ns": remaining_wall,
        "remaining_artifact_bytes": remaining_artifacts,
        "required_filesystem_free_bytes": required_filesystem_free,
        "within_proposed_limits": len(breaches) == 0,
        "breaches": breaches,
    }
    record["snapshot_sha256"] = canonical_sha256(record)
    return record


def evaluate_resource_snapshot(stage, usage, resources):
    """Compare authenticated usage and measured resources to proposed limits."""
    try:
        return _evaluate_resource_snapshot(stage, usage, resources)
    except ResourceGuardError:
        raise
    except Exception:
        raise ResourceGuardError("invalid_resource_input") from None


def require_scientific_execution(*args, **kwargs):
    """Always refuse: this leaf grants no scientific execution authority."""
    raise ResourceGuardError("scientific_execution_not_authorized") from None
