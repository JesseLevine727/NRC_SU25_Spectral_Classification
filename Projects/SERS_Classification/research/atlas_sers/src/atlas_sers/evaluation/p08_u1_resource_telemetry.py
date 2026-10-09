"""T314 bounded resource-monitor correction (transport-adjacent, no science).

Answers one question for a supervisor-owned recovery launcher: what fresh,
conservative resource evidence can be asserted about a worker whose in-process
heartbeat has stalled?  It never restarts, fits, loads checkpoints, or touches a
running study.

Integration boundary (documented, not implemented here): the supervisor will
separately wire ``sample_worker_resources`` into a private recovery launcher and
persist each fallback's worker PID/kind/age/source/evidence.  Whole-tree RSS
accounting, the inner 4 GiB / 120 s guard, and the outer 8 GiB GPU / 24 GiB RAM /
48 h / 80 GiB caps remain enforced by that launcher.  This module neither fixes
nor restarts the running study and performs no automatic recovery.
"""

import csv
import subprocess
import time

from .p08_u1_process import ProcessWorkerError, StaleTelemetryError

_MIB = 1024 * 1024


def query_gpu_process_bytes():
    """Return a conservative ``{pid: bytes}`` device-process memory bound.

    ``nvidia-smi`` compute-apps accounting includes context and reserved device
    memory, so this is a device-process upper bound rather than an
    allocator-equivalent claim.  Empty output yields an empty mapping so a
    missing stale GPU worker is refused, never charged zero.
    """
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=2,
        check=True,
        text=True,
        capture_output=True,
    )
    return _parse_compute_apps(completed.stdout)


def _parse_compute_apps(text):
    mapping = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        row = next(csv.reader([line]))
        if len(row) != 2:
            raise ProcessWorkerError("malformed nvidia-smi row")
        pid_field, mem_field = row[0].strip(), row[1].strip()
        try:
            pid = int(pid_field)
            mebibytes = int(mem_field)
        except ValueError:
            raise ProcessWorkerError("malformed nvidia-smi value") from None
        if pid <= 0:
            raise ProcessWorkerError("invalid nvidia-smi pid")
        if mebibytes < 0:
            raise ProcessWorkerError("invalid nvidia-smi memory")
        if pid in mapping:
            raise ProcessWorkerError("duplicate nvidia-smi pid")
        # Integer MiB output has coarser precision than allocator bytes. Charge
        # one extra MiB so truncation/rounding cannot understate the bound.
        mapping[pid] = (mebibytes + 1) * _MIB
    return mapping


_DEFAULT_GPU_QUERY = query_gpu_process_bytes


def sample_worker_resources(worker, *, query_gpu_process_bytes=None):
    """Return bounded resource evidence for ``worker`` or raise.

    Fresh in-process telemetry passes through untouched.  Only a validated
    :class:`StaleTelemetryError` triggers bounded fallback evidence; unhealthy,
    invalid, unavailable, locked, or dead workers propagate their error.
    """
    query = _DEFAULT_GPU_QUERY if query_gpu_process_bytes is None else query_gpu_process_bytes
    try:
        sample = worker.telemetry()
    except StaleTelemetryError as stale:
        return _resolve_stale(worker, stale, query)
    if not isinstance(sample, dict):
        raise ProcessWorkerError("telemetry sample must be a mapping")
    return {
        "timestamp": sample["timestamp"],
        "allocated_gpu_bytes": sample["allocated_gpu_bytes"],
        "reserved_gpu_bytes": sample["reserved_gpu_bytes"],
        "source": "allocator_heartbeat",
        "upper_bound": False,
        "stale_age_seconds": None,
    }


def _resolve_stale(worker, stale, query):
    if type(stale) is not StaleTelemetryError:
        raise ProcessWorkerError("unexpected stale telemetry type")
    if not worker.alive():
        raise ProcessWorkerError("stale worker is not alive")
    if stale.worker_id != worker.worker_id or stale.worker_kind != worker.kind:
        raise ProcessWorkerError("stale telemetry identity mismatch")
    if not isinstance(stale.pid, int) or isinstance(stale.pid, bool) or stale.pid <= 0:
        raise ProcessWorkerError("invalid stale telemetry pid")
    if stale.pid != worker.pid:
        raise ProcessWorkerError("stale telemetry pid mismatch")

    if worker.kind == "CPU":
        if stale.last_allocated_gpu_bytes == 0 and stale.last_reserved_gpu_bytes == 0:
            return {
                "timestamp": time.monotonic(),
                "allocated_gpu_bytes": 0,
                "reserved_gpu_bytes": 0,
                "source": "cpu_device_contract",
                "upper_bound": True,
                "stale_age_seconds": stale.age_seconds,
            }
        raise ProcessWorkerError("stale CPU worker reports nonzero GPU bytes")

    if worker.kind == "GPU":
        bound = _query_device_bound(query, stale.pid)
        # Both cells are charged with the same conservative device bound; the
        # reserved cell is a diagnostic bound, not allocator-reserved bytes.
        return {
            "timestamp": time.monotonic(),
            "allocated_gpu_bytes": bound,
            "reserved_gpu_bytes": bound,
            "source": "device_process_upper_bound",
            "upper_bound": True,
            "stale_age_seconds": stale.age_seconds,
        }

    raise ProcessWorkerError("unsupported worker kind")


def _query_device_bound(query, pid):
    if not callable(query):
        raise ProcessWorkerError("gpu memory query must be callable")
    try:
        mapping = query()
    except ProcessWorkerError:
        raise
    except Exception as exc:
        raise ProcessWorkerError("gpu process memory query failed") from exc
    if not isinstance(mapping, dict):
        raise ProcessWorkerError("gpu process memory query returned invalid mapping")
    bound = mapping.get(pid)
    if bound is None:
        raise ProcessWorkerError("stale gpu worker missing from device query")
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
        raise ProcessWorkerError("invalid gpu device memory bound")
    return bound
