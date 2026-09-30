"""P08-T066 bounded serial U0 resource/admission composition (no-fit readiness).

This leaf composes the accepted U0 attempt-journal replay with one fresh,
point-in-time measurement of this Python process and a caller-injected
in-process torch module, then delegates admissibility to the existing
prospective U0 candidate guard.  It samples at most one serial candidate and
does not support, model or validate later parallel execution.

The returned evidence is a transient snapshot only.  It is not persistent
lease ownership, not fresh durable progress under a live owner, not semantic
checkpoint validation, not a resource reservation, not a permit and not proof
of hard resource isolation.  The child-free ``process_tree_rss_bytes`` value
is this process' own ``VmRSS`` and is only valid under the explicit serial
assumption checked at two instants; short-lived processes are not excluded.
When CUDA is unobserved the ``cuda_device_used_bytes`` placeholder is 0 and
must not be read as zero global GPU use.  The GPU lifetime peak is additive
telemetry and never resets allocator statistics.  Estimator configuration is
audited only by querying loaded thread pools and the injected module; this
inspection cannot prove how threads will be used later.  Scientific execution
entry always denies.

The proposed serial smoke schedule keeps the unchanged 78 fit + 78
source-prediction U0 jobs and leaves the proposed scientific panel/limits
untouched.

The ``model_threads`` field is caller-declared: this leaf only requires it to be
exactly 1 and never measures it.  By contrast, ``blas_threads`` and
``torch_threads`` are independently inspected at sample time from loaded
thread pools and the injected torch module.  Neither the caller declaration nor
the independent inspection proves how threads will actually be used later, and
neither adds any authority or new execution entry point.
"""

from __future__ import annotations

import os
import time

import threadpoolctl

from . import p08_resources as resource_guard
from . import p08_u0_admission as admission
from . import p08_u0_store as store
from .p08_attempt_journal import JournalError, replay_attempt_journal
from .p08_qc_blocks import canonical_sha256

__all__ = [
    "SCHEMA_VERSION",
    "MAXIMUM_AGE_NS",
    "SerialResourceError",
    "check_serial_u0_candidate",
    "require_scientific_execution",
]

SCHEMA_VERSION = "nato-sers-p08-serial-resource-report-v1"
MAXIMUM_AGE_NS = 1_000_000_000

MAX_STATUS_BYTES = 65536
MAX_CHILDREN_BYTES = 65536
MAX_TASKS = 4096

_PROC_SELF_STATUS = "/proc/self/status"
_PROC_SELF_TASK = "/proc/self/task"

_DIGITS = frozenset("0123456789")

_REASON_CODES = frozenset(
    {
        "invalid_serial_input",
        "invalid_journal",
        "smoke_binding_mismatch",
        "invalid_candidate",
        "unregistered_job",
        "invalid_model_threads",
        "session_not_open",
        "serial_active_jobs_present",
        "prior_failed_or_interrupted_attempt_requires_review",
        "invalid_output_directory",
        "invalid_path",
        "symlink_or_nonregular",
        "invalid_layout",
        "storage_io_error",
        "invalid_proc_status",
        "invalid_proc_children",
        "child_process_detected",
        "invalid_threadpool_info",
        "invalid_torch_threads",
        "invalid_cuda_probe",
        "invalid_cuda_memory",
        "invalid_filesystem",
        "invalid_measurement_window",
        "stale_measurement",
        "invalid_resources",
        "candidate_blocked",
        "scientific_execution_not_authorized",
    }
)


class SerialResourceError(ValueError):
    """Static, data-free serial readiness failure exposing ``reason_code``."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_serial_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code):
    raise SerialResourceError(reason_code) from None


def _ascii_uint(value):
    if type(value) is not str or value == "":
        return None
    for char in value:
        if char not in _DIGITS:
            return None
    return int(value)


# ---------------------------------------------------------------------------
# Bounded process introspection
# ---------------------------------------------------------------------------


def _read_bounded(path, max_bytes, reason):
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        _fail(reason)
    body_exc = None
    result = None
    try:
        try:
            chunks = []
            total = 0
            limit = max_bytes + 1
            while total < limit:
                chunk = os.read(fd, min(65536, limit - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            if total > max_bytes:
                _fail(reason)
            result = b"".join(chunks)
        except OSError:
            _fail(reason)
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        except BaseException:
            if body_exc is None:
                raise
    return result


def _read_vm_rss_bytes():
    raw = _read_bounded(_PROC_SELF_STATUS, MAX_STATUS_BYTES, "invalid_proc_status")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail("invalid_proc_status")
    found = None
    count = 0
    for line in text.split("\n"):
        if line.startswith("VmRSS:"):
            count += 1
            parts = line.split()
            if len(parts) != 3 or parts[0] != "VmRSS:" or parts[2] != "kB":
                _fail("invalid_proc_status")
            kilobytes = _ascii_uint(parts[1])
            if kilobytes is None:
                _fail("invalid_proc_status")
            found = kilobytes
    if count != 1 or found is None:
        _fail("invalid_proc_status")
    return found * 1024


def _assert_no_children():
    names = []
    try:
        with os.scandir(_PROC_SELF_TASK) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > MAX_TASKS:
                    _fail("invalid_proc_children")
    except SerialResourceError:
        raise
    except OSError:
        _fail("invalid_proc_children")
    if not names:
        _fail("invalid_proc_children")
    for tid in names:
        if _ascii_uint(tid) is None:
            _fail("invalid_proc_children")
        raw = _read_bounded(
            f"{_PROC_SELF_TASK}/{tid}/children",
            MAX_CHILDREN_BYTES,
            "invalid_proc_children",
        )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            _fail("invalid_proc_children")
        if text.strip():
            _fail("child_process_detected")


# ---------------------------------------------------------------------------
# Bounded external measurements
# ---------------------------------------------------------------------------


def _measure_filesystem(output_directory):
    parent_fd = -1
    root_fd = -1
    info = None
    body_exc = None
    try:
        parent_fd, root_name = store._resolve_parent(output_directory)
        root_fd = store._open_dir(parent_fd, root_name)
        info = os.fstatvfs(root_fd)
    except BaseException as exc:
        body_exc = exc
        raise
    finally:
        try:
            store._close_fds(root_fd, parent_fd)
        except BaseException:
            if body_exc is None:
                raise
    fragment_size = info.f_frsize
    available = info.f_bavail
    if type(fragment_size) is not int or fragment_size <= 0:
        _fail("invalid_filesystem")
    if type(available) is not int or available < 0:
        _fail("invalid_filesystem")
    return available * fragment_size


def _measure_blas_threads():
    try:
        pools = threadpoolctl.threadpool_info()
    except Exception:
        _fail("invalid_threadpool_info")
    if type(pools) is not list:
        _fail("invalid_threadpool_info")
    maximum = 0
    for pool in pools:
        if type(pool) is not dict:
            _fail("invalid_threadpool_info")
        count = pool.get("num_threads")
        if type(count) is not int or count <= 0:
            _fail("invalid_threadpool_info")
        if count > maximum:
            maximum = count
    return maximum if maximum > 0 else 1


def _measure_torch_threads(torch_module):
    try:
        num_threads = torch_module.get_num_threads()
        interop_threads = torch_module.get_num_interop_threads()
    except Exception:
        _fail("invalid_torch_threads")
    if type(num_threads) is not int or num_threads <= 0:
        _fail("invalid_torch_threads")
    if type(interop_threads) is not int or interop_threads <= 0:
        _fail("invalid_torch_threads")
    return max(num_threads, interop_threads)


def _probe_cuda(torch_module):
    cuda = getattr(torch_module, "cuda", None)
    try:
        initialized = cuda.is_initialized()
    except Exception:
        _fail("invalid_cuda_probe")
    if type(initialized) is not bool:
        _fail("invalid_cuda_probe")
    if initialized is False:
        return {
            "observed": False,
            "initialized": False,
            "allocated": 0,
            "reserved": 0,
            "device_used": 0,
            "peak": 0,
        }
    try:
        device = cuda.current_device()
    except Exception:
        _fail("invalid_cuda_probe")
    if type(device) is not int or device < 0:
        _fail("invalid_cuda_probe")
    try:
        allocated = cuda.memory_allocated(device)
        reserved = cuda.memory_reserved(device)
        peak = cuda.max_memory_allocated(device)
        memory_info = cuda.mem_get_info(device)
    except Exception:
        _fail("invalid_cuda_probe")
    for value in (allocated, reserved, peak):
        if type(value) is not int or value < 0:
            _fail("invalid_cuda_memory")
    if type(memory_info) is not tuple and type(memory_info) is not list:
        _fail("invalid_cuda_memory")
    if len(memory_info) != 2:
        _fail("invalid_cuda_memory")
    free_bytes, total_bytes = memory_info
    if type(free_bytes) is not int or type(total_bytes) is not int:
        _fail("invalid_cuda_memory")
    if free_bytes < 0 or total_bytes < 0:
        _fail("invalid_cuda_memory")
    if free_bytes > total_bytes:
        _fail("invalid_cuda_memory")
    if allocated > reserved or allocated > peak or reserved > total_bytes:
        _fail("invalid_cuda_memory")
    return {
        "observed": True,
        "initialized": True,
        "allocated": allocated,
        "reserved": reserved,
        "device_used": total_bytes - free_bytes,
        "peak": peak,
    }


def _sample(output_directory, torch_module, model_threads):
    started = time.monotonic_ns()
    _assert_no_children()
    rss_bytes = _read_vm_rss_bytes()
    filesystem_free = _measure_filesystem(output_directory)
    blas_threads = _measure_blas_threads()
    torch_threads = _measure_torch_threads(torch_module)
    cuda = _probe_cuda(torch_module)
    _assert_no_children()
    finished = time.monotonic_ns()
    resources = {
        "filesystem_free_bytes": filesystem_free,
        "process_tree_rss_bytes": rss_bytes,
        "cuda_allocated_bytes": cuda["allocated"],
        "cuda_reserved_bytes": cuda["reserved"],
        "cuda_device_used_bytes": cuda["device_used"],
        "active_cpu_workers": 0,
        "active_gpu_workers": 0,
        "model_threads": model_threads,
        "blas_threads": blas_threads,
        "torch_threads": torch_threads,
    }
    return started, finished, resources, cuda


def _store_reason(reason_code):
    if reason_code in (
        "invalid_path",
        "symlink_or_nonregular",
        "invalid_layout",
        "storage_io_error",
    ):
        return reason_code
    return "invalid_output_directory"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _check_serial(
    manifest,
    events,
    *,
    job_id,
    expected_head_sha256,
    output_directory,
    torch_module,
    model_threads,
):
    try:
        summary = replay_attempt_journal(
            manifest,
            events,
            expected_manifest_sha256=admission.U0_MANIFEST_SHA256,
            expected_head_sha256=expected_head_sha256,
        )
    except JournalError:
        _fail("invalid_journal")

    if not admission._u0_binding_ok(manifest):
        _fail("smoke_binding_mismatch")

    try:
        admission._validate_candidate_id(job_id)
    except admission.AdmissionError:
        _fail("invalid_candidate")
    jobs_by_id = {job["job_id"]: job for job in manifest["jobs"]}
    job = jobs_by_id.get(job_id)
    if job is None:
        _fail("unregistered_job")

    if type(model_threads) is not int or model_threads != 1:
        _fail("invalid_model_threads")

    if summary["journal_state"] != "open":
        _fail("session_not_open")
    if summary["active_cpu_workers"] != 0 or summary["active_gpu_workers"] != 0:
        _fail("serial_active_jobs_present")
    if summary["failed_job_ids"] or summary["interrupted_job_ids"]:
        _fail("prior_failed_or_interrupted_attempt_requires_review")

    started, finished, resources, cuda = _sample(output_directory, torch_module, model_threads)

    try:
        candidate_check = admission.evaluate_u0_candidate(
            manifest,
            events,
            resources,
            job_id=job_id,
            expected_head_sha256=expected_head_sha256,
        )
    except admission.AdmissionError as exc:
        code = getattr(exc, "reason_code", "")
        if code == "invalid_resources":
            _fail("invalid_resources")
        if code == "invalid_journal":
            _fail("invalid_journal")
        _fail("candidate_blocked")

    checked = time.monotonic_ns()
    if (
        type(started) is not int
        or type(finished) is not int
        or type(checked) is not int
        or started < 0
        or finished < 0
        or checked < 0
    ):
        _fail("invalid_measurement_window")
    if not started <= finished <= checked:
        _fail("invalid_measurement_window")
    if checked - started > MAXIMUM_AGE_NS:
        _fail("stale_measurement")

    try:
        ceiling = resource_guard.proposed_limits("U0")["cuda_allocated_bytes"]
    except resource_guard.ResourceGuardError:
        _fail("invalid_serial_input")

    reasons = []
    if job["worker"] == "gpu" and cuda["initialized"] is False:
        reasons.append("cuda_not_initialized")
    if cuda["peak"] > ceiling:
        reasons.append("cuda_peak_exceeds_proposed_limit")
    reasons.extend(candidate_check["reasons"])

    admissible = candidate_check["proposed_candidate_admissible"] is True and len(reasons) == 0
    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "measurement_started_monotonic_ns": started,
        "measurement_finished_monotonic_ns": finished,
        "checked_monotonic_ns": checked,
        "maximum_age_ns": MAXIMUM_AGE_NS,
        "resources": resources,
        "cuda_observed": cuda["observed"],
        "cuda_peak_allocated_bytes": cuda["peak"],
        "candidate_check": candidate_check,
        "proposed_serial_candidate_admissible": admissible,
        "reasons": reasons,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def check_serial_u0_candidate(
    manifest,
    events,
    *,
    job_id,
    expected_head_sha256,
    output_directory,
    torch_module,
    model_threads,
):
    """Fresh point-in-time serial U0 resource/candidate evidence; never a permit.

    This call only combines an accepted-journal replay with measured process
    resources and the existing prospective candidate guard.  A true
    ``proposed_serial_candidate_admissible`` value establishes nothing about
    persistent lease ownership, fresh durable progress, semantic checkpoint
    validity, reservation or execution authority, and nothing about hard
    resource isolation outside the two sampled child checks.
    """
    try:
        return _check_serial(
            manifest,
            events,
            job_id=job_id,
            expected_head_sha256=expected_head_sha256,
            output_directory=output_directory,
            torch_module=torch_module,
            model_threads=model_threads,
        )
    except SerialResourceError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except store.StoreError as exc:
        raise SerialResourceError(_store_reason(getattr(exc, "reason_code", ""))) from None
    except admission.AdmissionError:
        raise SerialResourceError("candidate_blocked") from None
    except OSError:
        raise SerialResourceError("storage_io_error") from None
    except Exception:
        raise SerialResourceError("invalid_serial_input") from None


def require_scientific_execution(*args, **kwargs):
    """Always deny: this leaf grants no scientific execution authority."""
    raise SerialResourceError("scientific_execution_not_authorized") from None
