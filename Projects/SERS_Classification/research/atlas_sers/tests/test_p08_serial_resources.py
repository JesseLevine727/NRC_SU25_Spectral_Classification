"""Synthetic, no-execution tests for the T066 serial U0 resource leaf.

Every manifest, event, process file, filesystem metric and torch object used
here is invented.  These tests never run jobs, read real hardware, fit models or
authorize scientific execution.
"""

from __future__ import annotations

import copy
import errno
import os
import pathlib

import pytest

from atlas_sers.evaluation import p08_resources as resource_guard
from atlas_sers.evaluation import p08_serial_resources as serial
from atlas_sers.evaluation import p08_u0_admission as admission
from atlas_sers.evaluation import p08_u0_store as store
from atlas_sers.evaluation.p08_attempt_journal import replay_attempt_journal
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256

_PROPOSAL = "6639a32c1dd930612ead6ae59adf9aff5831904e883081f16c3490721af5089a"
_REAL_MANIFEST_SHA256 = "aac8523a1614610cf99cf1ab548d8970b4f8054fe8821b5250c347376e30b28a"
_MANIFEST_SCHEMA = "nato-sers-p08-attempt-manifest-v1"
_RECEIPT = "a" * 64
_GIB = 1024**3
_NANOS = 10**9

_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "measurement_started_monotonic_ns",
        "measurement_finished_monotonic_ns",
        "checked_monotonic_ns",
        "maximum_age_ns",
        "resources",
        "cuda_observed",
        "cuda_peak_allocated_bytes",
        "candidate_check",
        "proposed_serial_candidate_admissible",
        "reasons",
        "report_sha256",
    }
)

_RESOURCE_KEYS = frozenset(
    {
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
    }
)


# ---------------------------------------------------------------------------
# Invented U0 manifest/event builders (mirrors the admission leaf tests)
# ---------------------------------------------------------------------------


def _build_manifest(*, fit_count=78, cpu_fits=42, proposal_sha256=_PROPOSAL):
    jobs = []
    for index in range(1, fit_count + 1):
        worker = "cpu" if index <= cpu_fits else "gpu"
        jobs.append(
            {
                "job_id": f"fit{index:03d}",
                "stage": "source_fit",
                "worker": worker,
                "dependencies": [],
            }
        )
    for index in range(1, fit_count + 1):
        worker = "cpu" if index <= cpu_fits else "gpu"
        jobs.append(
            {
                "job_id": f"pred{index:03d}",
                "stage": "source_validation_prediction",
                "worker": worker,
                "dependencies": [f"fit{index:03d}"],
            }
        )
    payload = {
        "schema_version": _MANIFEST_SCHEMA,
        "execution_authorized": False,
        "proposal_sha256": proposal_sha256,
        "jobs": jobs,
    }
    manifest = dict(payload)
    manifest["manifest_sha256"] = canonical_sha256(payload)
    return manifest


def _event(
    seq,
    previous_sha256,
    session_id,
    event_type,
    *,
    elapsed_ns=0,
    artifact_bytes=0,
    job_id=None,
    status=None,
    receipt_sha256=None,
):
    payload = {
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
    payload["event_sha256"] = canonical_sha256(payload)
    return payload


def _chain(manifest, specs):
    events = []
    previous = manifest["manifest_sha256"]
    for index, spec in enumerate(specs, start=1):
        event = _event(index, previous, **spec)
        events.append(event)
        previous = event["event_sha256"]
    return events


def _open(session_id=1):
    return {"session_id": session_id, "event_type": "session_open"}


def _start(job_id, *, elapsed_ns=0, artifact_bytes=0, session_id=1):
    return {
        "session_id": session_id,
        "event_type": "attempt_start",
        "job_id": job_id,
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
    }


def _finish(job_id, status, *, elapsed_ns=0, artifact_bytes=0, session_id=1):
    return {
        "session_id": session_id,
        "event_type": "attempt_finish",
        "job_id": job_id,
        "status": status,
        "receipt_sha256": _RECEIPT,
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
    }


def _close(*, elapsed_ns=0, artifact_bytes=0, session_id=1):
    return {
        "session_id": session_id,
        "event_type": "session_close",
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
    }


def _progress(*, elapsed_ns=0, artifact_bytes=0, session_id=1):
    return {
        "session_id": session_id,
        "event_type": "progress",
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
    }


# ---------------------------------------------------------------------------
# Invented process/torch/filesystem doubles
# ---------------------------------------------------------------------------


class _FakeVfs:
    __slots__ = ("f_frsize", "f_bavail")

    def __init__(self, frsize, bavail):
        self.f_frsize = frsize
        self.f_bavail = bavail


class _CpuCuda:
    """Uninitialized CUDA stub where any extra method access is a failure."""

    def __init__(self):
        self.accessed = []

    def is_initialized(self):
        self.accessed.append("is_initialized")
        return False

    def __getattr__(self, name):
        accessed = self.__dict__.get("accessed")
        if accessed is not None:
            accessed.append(name)
        raise AssertionError(f"unexpected CUDA attribute: {name}")


class _GpuCuda:
    def __init__(
        self,
        *,
        initialized=True,
        device=0,
        allocated=0,
        reserved=0,
        peak=0,
        free=0,
        total=0,
        init_exc=None,
        device_exc=None,
        memory_exc=None,
        mem_override=None,
    ):
        self.initialized = initialized
        self.device = device
        self.allocated = allocated
        self.reserved = reserved
        self.peak = peak
        self.free = free
        self.total = total
        self.init_exc = init_exc
        self.device_exc = device_exc
        self.memory_exc = memory_exc
        self.mem_override = mem_override
        self.accessed = []

    def is_initialized(self):
        self.accessed.append("is_initialized")
        if self.init_exc is not None:
            raise self.init_exc
        return self.initialized

    def current_device(self):
        self.accessed.append("current_device")
        if self.device_exc is not None:
            raise self.device_exc
        return self.device

    def _memory(self, name):
        self.accessed.append(name)
        if self.memory_exc is not None:
            raise self.memory_exc

    def memory_allocated(self, device):
        self._memory("memory_allocated")
        return self.allocated

    def memory_reserved(self, device):
        self._memory("memory_reserved")
        return self.reserved

    def max_memory_allocated(self, device):
        self._memory("max_memory_allocated")
        return self.peak

    def mem_get_info(self, device):
        if self.mem_override is not None:
            return self.mem_override
        return (self.free, self.total)

    def __getattr__(self, name):
        accessed = self.__dict__.get("accessed")
        if accessed is not None:
            accessed.append(name)
        raise AssertionError(f"unexpected CUDA attribute: {name}")


class _FakeTorch:
    def __init__(self, cuda, *, num_threads=1, interop_threads=1):
        self.cuda = cuda
        self._num_threads = num_threads
        self._interop_threads = interop_threads

    def get_num_threads(self):
        return self._num_threads

    def get_num_interop_threads(self):
        return self._interop_threads


def _cpu_torch():
    return _FakeTorch(_CpuCuda())


def _gpu_torch(cuda=None, *, num_threads=1, interop_threads=1):
    if cuda is None:
        cuda = _GpuCuda()
    return _FakeTorch(cuda, num_threads=num_threads, interop_threads=interop_threads)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest(monkeypatch):
    value = _build_manifest()
    monkeypatch.setattr(admission, "U0_MANIFEST_SHA256", value["manifest_sha256"])
    return value


@pytest.fixture
def sane(monkeypatch, tmp_path):
    # Synthetic resource tests: freeze the sampling clock so real CI scheduling
    # can never trip the freshness check by accident.  Dedicated _with_clock
    # tests re-patch serial.time.monotonic_ns later.
    monkeypatch.setattr(serial.time, "monotonic_ns", lambda: 0)
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t1024 kB\n", encoding="utf-8")
    task = tmp_path / "task"
    thread = task / "1"
    thread.mkdir(parents=True)
    (thread / "children").write_text("", encoding="utf-8")
    monkeypatch.setattr(serial, "_PROC_SELF_STATUS", str(status))
    monkeypatch.setattr(serial, "_PROC_SELF_TASK", str(task))

    fs_state = {"free": 64 * _GIB}
    monkeypatch.setattr(os, "fstatvfs", lambda fd: _FakeVfs(1, fs_state["free"]))

    pool_state = {"info": [{"num_threads": 1}]}
    monkeypatch.setattr(serial.threadpoolctl, "threadpool_info", lambda: pool_state["info"])

    output = tmp_path / "output"
    output.mkdir()
    (output / "data.txt").write_text("payload", encoding="utf-8")
    return {
        "status": status,
        "task": task,
        "fs": fs_state,
        "pools": pool_state,
        "output": str(output),
        "tmp": tmp_path,
    }


@pytest.fixture
def sample_spy(monkeypatch):
    calls = []

    def fake_sample(*args, **kwargs):
        calls.append(1)
        return (
            0,
            0,
            {},
            {
                "observed": False,
                "initialized": False,
                "allocated": 0,
                "reserved": 0,
                "device_used": 0,
                "peak": 0,
            },
        )

    monkeypatch.setattr(serial, "_sample", fake_sample)
    return calls


def _run(manifest, events, *, job_id, output, torch, head=None, model_threads=1):
    if head is None:
        head = events[-1]["event_sha256"] if events else manifest["manifest_sha256"]
    return serial.check_serial_u0_candidate(
        manifest,
        events,
        job_id=job_id,
        expected_head_sha256=head,
        output_directory=output,
        torch_module=torch,
        model_threads=model_threads,
    )


def _hold_close(fds, real_close):
    for fd in fds:
        try:
            real_close(fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Public constants and happy-path snapshots
# ---------------------------------------------------------------------------


def test_registered_public_constants():
    assert admission.U0_PROPOSAL_SHA256 == _PROPOSAL
    assert admission.U0_MANIFEST_SHA256 == _REAL_MANIFEST_SHA256
    assert serial.SCHEMA_VERSION == "nato-sers-p08-serial-resource-report-v1"
    assert serial.MAXIMUM_AGE_NS == 1_000_000_000


def test_u0_cuda_ceiling_is_proposed_eight_gib():
    assert resource_guard.proposed_limits("U0")["cuda_allocated_bytes"] == 8 * _GIB


def test_replay_summary_binds_open_session(manifest):
    events = _chain(manifest, [_open()])
    summary = replay_attempt_journal(
        manifest,
        events,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_head_sha256=events[-1]["event_sha256"],
    )
    assert summary["journal_state"] == "open"
    assert summary["active_cpu_workers"] == 0
    assert summary["active_gpu_workers"] == 0


def test_happy_cpu_snapshot(manifest, sane):
    events = _chain(manifest, [_open()])
    manifest_copy = copy.deepcopy(manifest)
    events_copy = copy.deepcopy(events)
    output = pathlib.Path(sane["output"])
    listing_before = sorted(entry.name for entry in output.iterdir())
    cuda = _CpuCuda()

    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_FakeTorch(cuda),
    )

    assert set(report) == _REPORT_KEYS
    assert report["schema_version"] == serial.SCHEMA_VERSION
    assert report["execution_authorized"] is False
    assert report["maximum_age_ns"] == 1_000_000_000
    assert report["cuda_observed"] is False
    assert report["cuda_peak_allocated_bytes"] == 0
    assert report["proposed_serial_candidate_admissible"] is True
    assert report["reasons"] == []
    assert set(report["resources"]) == _RESOURCE_KEYS
    assert report["resources"] == {
        "filesystem_free_bytes": 64 * _GIB,
        "process_tree_rss_bytes": 1024 * 1024,
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
        "cuda_device_used_bytes": 0,
        "active_cpu_workers": 0,
        "active_gpu_workers": 0,
        "model_threads": 1,
        "blas_threads": 1,
        "torch_threads": 1,
    }
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    assert canonical_sha256(body) == report["report_sha256"]
    assert report["candidate_check"]["proposed_candidate_admissible"] is True
    assert cuda.accessed == ["is_initialized"]
    assert manifest == manifest_copy
    assert events == events_copy
    assert sorted(entry.name for entry in output.iterdir()) == listing_before
    assert (output / "data.txt").read_text(encoding="utf-8") == "payload"


def test_happy_gpu_snapshot(manifest, sane):
    events = _chain(manifest, [_open()])
    cuda = _GpuCuda(
        device=0,
        allocated=1 * _GIB,
        reserved=2 * _GIB,
        peak=3 * _GIB,
        free=8 * _GIB,
        total=16 * _GIB,
    )
    report = _run(
        manifest,
        events,
        job_id="fit043",
        output=sane["output"],
        torch=_gpu_torch(cuda),
    )

    assert report["proposed_serial_candidate_admissible"] is True
    assert report["reasons"] == []
    assert report["cuda_observed"] is True
    assert report["cuda_peak_allocated_bytes"] == 3 * _GIB
    resources = report["resources"]
    assert resources["cuda_allocated_bytes"] == 1 * _GIB
    assert resources["cuda_reserved_bytes"] == 2 * _GIB
    assert resources["cuda_device_used_bytes"] == 8 * _GIB
    assert resources["active_cpu_workers"] == 0
    assert resources["active_gpu_workers"] == 0
    assert "empty_cache" not in cuda.accessed
    assert "reset_peak_memory_stats" not in cuda.accessed
    assert "init" not in cuda.accessed


def test_gpu_candidate_uninitialized_cuda_blocked(manifest, sane):
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit043",
        output=sane["output"],
        torch=_gpu_torch(_GpuCuda(initialized=False)),
    )
    assert report["cuda_observed"] is False
    assert "cuda_not_initialized" in report["reasons"]
    assert report["proposed_serial_candidate_admissible"] is False


# ---------------------------------------------------------------------------
# Pre-sample rejections: no I/O, clock or pool probe may run
# ---------------------------------------------------------------------------


def test_bad_head_rejects_before_probes(manifest, sample_spy, tmp_path):
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=str(tmp_path),
            torch=_cpu_torch(),
            head="0" * 64,
        )
    assert exc.value.reason_code == "invalid_journal"
    assert sample_spy == []


def test_tampered_event_rejects_before_probes(manifest, sample_spy, tmp_path):
    events = _chain(manifest, [_open(), _progress(elapsed_ns=5)])
    events[1] = dict(events[1])
    events[1]["elapsed_ns"] = 6
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=str(tmp_path),
            torch=_cpu_torch(),
            head=events[1]["event_sha256"],
        )
    assert exc.value.reason_code == "invalid_journal"
    assert sample_spy == []


def test_unregistered_job_rejects_before_probes(manifest, sample_spy, tmp_path):
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit999",
            output=str(tmp_path),
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "unregistered_job"
    assert sample_spy == []


@pytest.mark.parametrize("bad_id", ["", "   ", None, 7, True])
def test_invalid_candidate_id_rejects_before_probes(manifest, sample_spy, tmp_path, bad_id):
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id=bad_id,
            output=str(tmp_path),
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "invalid_candidate"
    assert sample_spy == []


@pytest.mark.parametrize("threads", [0, 2, True, 1.0, "1", None])
def test_bad_model_threads_rejects_before_probes(manifest, sample_spy, tmp_path, threads):
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=str(tmp_path),
            torch=_cpu_torch(),
            model_threads=threads,
        )
    assert exc.value.reason_code == "invalid_model_threads"
    assert sample_spy == []


def test_closed_history_rejects_before_probes(manifest, sample_spy, tmp_path):
    events = _chain(manifest, [_open(), _close(elapsed_ns=1)])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=str(tmp_path),
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "session_not_open"
    assert sample_spy == []


def test_active_history_rejects_before_probes(manifest, sample_spy, tmp_path):
    events = _chain(manifest, [_open(), _start("fit001")])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit002",
            output=str(tmp_path),
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "serial_active_jobs_present"
    assert sample_spy == []


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_failed_history_rejects_before_probes(manifest, sample_spy, tmp_path, status):
    events = _chain(
        manifest,
        [_open(), _start("fit001", elapsed_ns=1), _finish("fit001", status, elapsed_ns=2)],
    )
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit002",
            output=str(tmp_path),
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "prior_failed_or_interrupted_attempt_requires_review"
    assert sample_spy == []


def test_binding_mismatch_rejects_before_probes(monkeypatch, sample_spy, tmp_path):
    bad = _build_manifest(proposal_sha256="0" * 64)
    monkeypatch.setattr(admission, "U0_MANIFEST_SHA256", bad["manifest_sha256"])
    events = _chain(bad, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            bad,
            events,
            job_id="fit001",
            output=str(tmp_path),
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "smoke_binding_mismatch"
    assert sample_spy == []


# ---------------------------------------------------------------------------
# Real replay/admission semantics flow into the report reasons
# ---------------------------------------------------------------------------


def test_missing_fit_dependency_is_reported(manifest, sane):
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="pred001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert report["proposed_serial_candidate_admissible"] is False
    assert "fit_dependency_not_succeeded" in report["reasons"]
    assert "fit_dependency_not_succeeded" in report["candidate_check"]["reasons"]


def test_already_attempted_is_reported(manifest, sane):
    events = _chain(
        manifest,
        [_open(), _start("fit001", elapsed_ns=1), _finish("fit001", "succeeded", elapsed_ns=2)],
    )
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert "job_already_attempted" in report["reasons"]
    assert report["proposed_serial_candidate_admissible"] is False
    assert report["resources"]["active_cpu_workers"] == 0
    assert report["resources"]["active_gpu_workers"] == 0


def test_full_fit_budget_blocks_retry(manifest, sane):
    specs = [_open()]
    elapsed = 0
    for index in range(1, 79):
        elapsed += 1
        specs.append(_start(f"fit{index:03d}", elapsed_ns=elapsed))
        elapsed += 1
        specs.append(_finish(f"fit{index:03d}", "succeeded", elapsed_ns=elapsed))
    events = _chain(manifest, specs)
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert "job_already_attempted" in report["reasons"]
    assert "model_fit_capacity_exhausted" in report["reasons"]
    assert report["proposed_serial_candidate_admissible"] is False


def test_wall_budget_exhaustion_is_reported(manifest, sane):
    events = _chain(manifest, [_open(), _progress(elapsed_ns=5400 * _NANOS)])
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert "resource_limits_breached" in report["reasons"]
    assert "active_wall_budget_exhausted" in report["candidate_check"]["resource_breaches"]
    assert report["proposed_serial_candidate_admissible"] is False


def test_low_filesystem_space_is_reported(manifest, sane):
    sane["fs"]["free"] = 37 * _GIB
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert "resource_limits_breached" in report["reasons"]
    assert "filesystem_reserve_insufficient" in report["candidate_check"]["resource_breaches"]


def test_rss_over_limit_is_reported(manifest, sane):
    sane["status"].write_text(f"VmRSS:\t{16 * _GIB // 1024 + 1} kB\n", encoding="utf-8")
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert "resource_limits_breached" in report["reasons"]
    assert "process_tree_memory_exceeded" in report["candidate_check"]["resource_breaches"]


def test_blas_thread_violation_is_reported(manifest, sane):
    sane["pools"]["info"] = [{"num_threads": 2}]
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert report["resources"]["blas_threads"] == 2
    assert "worker_thread_limit_violated" in report["candidate_check"]["resource_breaches"]


@pytest.mark.parametrize("num,interop", [(2, 1), (1, 2)])
def test_torch_thread_violation_is_reported(manifest, sane, num, interop):
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_FakeTorch(_CpuCuda(), num_threads=num, interop_threads=interop),
    )
    assert report["resources"]["torch_threads"] == 2
    assert "worker_thread_limit_violated" in report["candidate_check"]["resource_breaches"]


# ---------------------------------------------------------------------------
# CUDA probe errors and memory separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"initialized": 1}, "invalid_cuda_probe"),
        ({"initialized": "yes"}, "invalid_cuda_probe"),
        ({"device": -1}, "invalid_cuda_probe"),
        ({"device": True}, "invalid_cuda_probe"),
        ({"allocated": -1}, "invalid_cuda_memory"),
        ({"allocated": "1"}, "invalid_cuda_memory"),
        ({"allocated": 5, "reserved": 4}, "invalid_cuda_memory"),
        ({"allocated": 5, "peak": 4}, "invalid_cuda_memory"),
        ({"reserved": 20 * _GIB, "total": 16 * _GIB}, "invalid_cuda_memory"),
        ({"free": 20 * _GIB, "total": 16 * _GIB}, "invalid_cuda_memory"),
        ({"mem_override": (1,)}, "invalid_cuda_memory"),
        ({"mem_override": "abcd"}, "invalid_cuda_memory"),
        ({"mem_override": (1, -1)}, "invalid_cuda_memory"),
    ],
)
def test_cuda_probe_rejections(manifest, sane, kwargs, reason):
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit043",
            output=sane["output"],
            torch=_gpu_torch(_GpuCuda(**kwargs)),
        )
    assert exc.value.reason_code == reason


def test_cuda_probe_exceptions_are_normalized(manifest, sane):
    for cuda in (
        _GpuCuda(init_exc=RuntimeError("x")),
        _GpuCuda(device_exc=RuntimeError("x")),
        _GpuCuda(memory_exc=RuntimeError("x")),
    ):
        events = _chain(manifest, [_open()])
        with pytest.raises(serial.SerialResourceError) as exc:
            _run(
                manifest,
                events,
                job_id="fit043",
                output=sane["output"],
                torch=_gpu_torch(cuda),
            )
        assert exc.value.reason_code == "invalid_cuda_probe"


def test_missing_cuda_module_rejected(manifest, sane):
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit043",
            output=sane["output"],
            torch=_FakeTorch(None),
        )
    assert exc.value.reason_code == "invalid_cuda_probe"


def test_peak_at_ceiling_allowed(manifest, sane):
    cuda = _GpuCuda(
        allocated=1 * _GIB,
        reserved=2 * _GIB,
        peak=8 * _GIB,
        free=8 * _GIB,
        total=16 * _GIB,
    )
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit043",
        output=sane["output"],
        torch=_gpu_torch(cuda),
    )
    assert report["cuda_peak_allocated_bytes"] == 8 * _GIB
    assert "cuda_peak_exceeds_proposed_limit" not in report["reasons"]
    assert report["proposed_serial_candidate_admissible"] is True


def test_peak_one_byte_over_ceiling_blocked(manifest, sane):
    cuda = _GpuCuda(
        allocated=1 * _GIB,
        reserved=2 * _GIB,
        peak=8 * _GIB + 1,
        free=8 * _GIB,
        total=16 * _GIB,
    )
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit043",
        output=sane["output"],
        torch=_gpu_torch(cuda),
    )
    assert "cuda_peak_exceeds_proposed_limit" in report["reasons"]
    assert report["proposed_serial_candidate_admissible"] is False


# ---------------------------------------------------------------------------
# Monotonic measurement window
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if not self.sequence:
            raise AssertionError("unexpected clock read")
        return self.sequence.pop(0)


def _with_clock(monkeypatch, values):
    clock = _Clock(values)
    monkeypatch.setattr(serial.time, "monotonic_ns", clock)
    return clock


def test_exactly_one_second_allowed(manifest, sane, monkeypatch):
    _with_clock(monkeypatch, [100, 100, 100 + _NANOS])
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert report["measurement_started_monotonic_ns"] == 100
    assert report["checked_monotonic_ns"] == 100 + _NANOS
    assert report["proposed_serial_candidate_admissible"] is True


def test_one_nanosecond_over_ceiling_stale(manifest, sane, monkeypatch):
    _with_clock(monkeypatch, [100, 100, 100 + _NANOS + 1])
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "stale_measurement"


def test_reversed_monotonic_window_invalid(manifest, sane, monkeypatch):
    _with_clock(monkeypatch, [100, 50, 60])
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "invalid_measurement_window"


@pytest.mark.parametrize(
    "values",
    [
        [True, 0, 0],
        [-1, 0, 0],
        [1.0, 0, 0],
        [0, 0, False],
    ],
)
def test_bad_monotonic_values_invalid(manifest, sane, monkeypatch, values):
    _with_clock(monkeypatch, values)
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "invalid_measurement_window"


def test_sampling_and_candidate_delay_count_toward_total(manifest, sane, monkeypatch):
    _with_clock(monkeypatch, [0, 900_000_000, 1_000_000_000])
    events = _chain(manifest, [_open()])
    report = _run(
        manifest,
        events,
        job_id="fit001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert report["measurement_finished_monotonic_ns"] == 900_000_000
    assert report["proposed_serial_candidate_admissible"] is True

    _with_clock(monkeypatch, [0, 0, _NANOS + 1])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "stale_measurement"


# ---------------------------------------------------------------------------
# VmRSS parser fixtures
# ---------------------------------------------------------------------------


def _install_status(tmp_path, monkeypatch, text):
    path = tmp_path / "status"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(serial, "_PROC_SELF_STATUS", str(path))
    return path


def test_vm_rss_valid(tmp_path, monkeypatch):
    _install_status(
        tmp_path,
        monkeypatch,
        "Name:\tpython\nVmRSS:\t  2048 kB\nExtra:\t1\n",
    )
    assert serial._read_vm_rss_bytes() == 2048 * 1024


@pytest.mark.parametrize(
    "text",
    [
        "Name:\tpython\n",
        "VmRSS:123 2 kB\n",
        "VmRSS:\tabc kB\n",
        "VmRSS:\t-1 kB\n",
        "VmRSS:\t12 MB\n",
        "VmRSS:\t12\n",
        "VmRSS:\t12 kB\nVmRSS:\t13 kB\n",
    ],
)
def test_vm_rss_malformed(tmp_path, monkeypatch, text):
    _install_status(tmp_path, monkeypatch, text)
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._read_vm_rss_bytes()
    assert exc.value.reason_code == "invalid_proc_status"


def test_vm_rss_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(serial, "MAX_STATUS_BYTES", 8)
    _install_status(tmp_path, monkeypatch, "VmRSS:\t1 kB\n")
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._read_vm_rss_bytes()
    assert exc.value.reason_code == "invalid_proc_status"


# ---------------------------------------------------------------------------
# Children parser fixtures
# ---------------------------------------------------------------------------


def _make_task_dir(tmp_path, monkeypatch, entries):
    root = tmp_path / "task"
    root.mkdir()
    for name, children in entries.items():
        thread = root / name
        thread.mkdir()
        if children is not None:
            (thread / "children").write_text(children, encoding="utf-8")
    monkeypatch.setattr(serial, "_PROC_SELF_TASK", str(root))
    return root


def test_children_valid(tmp_path, monkeypatch):
    _make_task_dir(tmp_path, monkeypatch, {"1": "", "2": "  \n"})
    serial._assert_no_children()


def test_children_empty_task_dir_rejected(tmp_path, monkeypatch):
    root = tmp_path / "task"
    root.mkdir()
    monkeypatch.setattr(serial, "_PROC_SELF_TASK", str(root))
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._assert_no_children()
    assert exc.value.reason_code == "invalid_proc_children"


def test_children_malformed_tid_rejected(tmp_path, monkeypatch):
    _make_task_dir(tmp_path, monkeypatch, {"abc": ""})
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._assert_no_children()
    assert exc.value.reason_code == "invalid_proc_children"


def test_children_missing_file_rejected(tmp_path, monkeypatch):
    _make_task_dir(tmp_path, monkeypatch, {"1": None})
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._assert_no_children()
    assert exc.value.reason_code == "invalid_proc_children"


def test_children_unreadable_rejected(tmp_path, monkeypatch):
    root = tmp_path / "task"
    (root / "1" / "children").mkdir(parents=True)
    monkeypatch.setattr(serial, "_PROC_SELF_TASK", str(root))
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._assert_no_children()
    assert exc.value.reason_code == "invalid_proc_children"


def test_children_oversized_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(serial, "MAX_CHILDREN_BYTES", 4)
    _make_task_dir(tmp_path, monkeypatch, {"1": "12345"})
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._assert_no_children()
    assert exc.value.reason_code == "invalid_proc_children"


def test_children_present_detected(tmp_path, monkeypatch):
    _make_task_dir(tmp_path, monkeypatch, {"1": "1234\n"})
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._assert_no_children()
    assert exc.value.reason_code == "child_process_detected"


def test_children_too_many_tasks_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(serial, "MAX_TASKS", 2)
    _make_task_dir(tmp_path, monkeypatch, {"1": "", "2": "", "3": ""})
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._assert_no_children()
    assert exc.value.reason_code == "invalid_proc_children"


def test_child_at_first_check_blocks(manifest, sane):
    (sane["task"] / "1" / "children").write_text("42\n", encoding="utf-8")
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "child_process_detected"


def test_child_at_second_check_blocks(manifest, sane, monkeypatch):
    dirty = sane["tmp"] / "dirty_task"
    (dirty / "1").mkdir(parents=True)
    (dirty / "1" / "children").write_text("7\n", encoding="utf-8")
    real_measure = serial._measure_filesystem

    def swap(output_directory):
        result = real_measure(output_directory)
        monkeypatch.setattr(serial, "_PROC_SELF_TASK", str(dirty))
        return result

    monkeypatch.setattr(serial, "_measure_filesystem", swap)
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "child_process_detected"


# ---------------------------------------------------------------------------
# Bounded reader and cleanup behaviour
# ---------------------------------------------------------------------------


def test_read_bounded_success_and_exact_fd_close(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"hello world")
    recorded = {}
    real_open = os.open

    def tracking_open(pathname, flags, *args, **kwargs):
        fd = real_open(pathname, flags, *args, **kwargs)
        recorded["fd"] = fd
        return fd

    monkeypatch.setattr(os, "open", tracking_open)
    assert serial._read_bounded(str(path), 64, "invalid_proc_status") == b"hello world"
    with pytest.raises(OSError):
        os.fstat(recorded["fd"])


def test_read_bounded_missing_oversize_and_symlink(tmp_path):
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._read_bounded(str(tmp_path / "missing"), 64, "invalid_proc_status")
    assert exc.value.reason_code == "invalid_proc_status"

    big = tmp_path / "big"
    big.write_bytes(b"x" * 10)
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._read_bounded(str(big), 4, "invalid_proc_status")
    assert exc.value.reason_code == "invalid_proc_status"

    target = tmp_path / "target"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(serial.SerialResourceError) as exc:
        serial._read_bounded(str(link), 64, "invalid_proc_status")
    assert exc.value.reason_code == "invalid_proc_status"


def test_read_bounded_close_oserror_swallowed(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"ok")
    real_close = os.close
    held = []

    def fake_close(fd):
        held.append(fd)
        raise OSError(9, "injected")

    monkeypatch.setattr(os, "close", fake_close)
    try:
        assert serial._read_bounded(str(path), 64, "invalid_proc_status") == b"ok"
    finally:
        monkeypatch.setattr(os, "close", real_close)
        _hold_close(held, real_close)
    assert held


def test_read_bounded_close_interruption_propagates(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"ok")
    real_close = os.close
    held = []

    def fake_close(fd):
        held.append(fd)
        raise SystemExit(9)

    monkeypatch.setattr(os, "close", fake_close)
    try:
        with pytest.raises(SystemExit):
            serial._read_bounded(str(path), 64, "invalid_proc_status")
    finally:
        monkeypatch.setattr(os, "close", real_close)
        _hold_close(held, real_close)


def test_read_bounded_body_error_preserved_over_close_error(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"x" * 10)
    real_close = os.close
    held = []

    def fake_close(fd):
        held.append(fd)
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "close", fake_close)
    try:
        with pytest.raises(serial.SerialResourceError) as exc:
            serial._read_bounded(str(path), 4, "invalid_proc_status")
        assert exc.value.reason_code == "invalid_proc_status"
    finally:
        monkeypatch.setattr(os, "close", real_close)
        _hold_close(held, real_close)


def test_read_bounded_body_systemexit_preserved_over_close_oserror(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"ok")
    real_close = os.close
    real_read = os.read
    held = []

    def fake_read(fd, size):
        raise SystemExit(12)

    def fake_close(fd):
        held.append(fd)
        raise OSError(9, "injected")

    monkeypatch.setattr(os, "read", fake_read)
    monkeypatch.setattr(os, "close", fake_close)
    try:
        with pytest.raises(SystemExit):
            serial._read_bounded(str(path), 64, "invalid_proc_status")
    finally:
        monkeypatch.setattr(os, "read", real_read)
        monkeypatch.setattr(os, "close", real_close)
        _hold_close(held, real_close)


# ---------------------------------------------------------------------------
# Filesystem descriptor handling and cleanup
# ---------------------------------------------------------------------------


def test_measure_filesystem_success_and_exact_close(tmp_path, monkeypatch):
    fs_free = 64 * _GIB
    monkeypatch.setattr(os, "fstatvfs", lambda fd: _FakeVfs(1, fs_free))
    owned = []
    real_resolve = store._resolve_parent
    real_open_dir = store._open_dir

    def spy_resolve(root):
        fd, name = real_resolve(root)
        owned.append(fd)
        return fd, name

    def spy_open_dir(fd, name):
        result = real_open_dir(fd, name)
        owned.append(result)
        return result

    monkeypatch.setattr(store, "_resolve_parent", spy_resolve)
    monkeypatch.setattr(store, "_open_dir", spy_open_dir)

    assert serial._measure_filesystem(str(tmp_path)) == fs_free
    assert owned
    for fd in owned:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_measure_filesystem_body_error_preserved_over_cleanup(tmp_path, monkeypatch):
    def boom(_path):
        raise store.StoreError("invalid_path")

    def cleanup_boom(*_fds):
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "_resolve_parent", boom)
    monkeypatch.setattr(store, "_close_fds", cleanup_boom)
    with pytest.raises(store.StoreError) as exc:
        serial._measure_filesystem(str(tmp_path))
    assert exc.value.reason_code == "invalid_path"


def test_measure_filesystem_cleanup_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "fstatvfs", lambda fd: _FakeVfs(1, 64 * _GIB))
    real_close_fds = store._close_fds
    recorded = []

    def cleanup_boom(*fds):
        recorded.extend(fd for fd in fds if type(fd) is int and fd >= 0)
        raise SystemExit(7)

    monkeypatch.setattr(store, "_close_fds", cleanup_boom)
    try:
        with pytest.raises(SystemExit):
            serial._measure_filesystem(str(tmp_path))
    finally:
        for fd in recorded:
            try:
                real_close_fds(fd)
            except OSError:
                pass


def test_measure_filesystem_rejects_symlink(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(store.StoreError) as exc:
        serial._measure_filesystem(str(link))
    # Linux O_DIRECTORY|O_NOFOLLOW yields ENOTDIR, which the accepted guard
    # maps to invalid_layout; other platforms can yield ELOOP and map to
    # symlink_or_nonregular.  Both are valid rejections; success is never
    # accepted.
    assert exc.value.reason_code in ("symlink_or_nonregular", "invalid_layout")


def test_measure_filesystem_rejects_missing(tmp_path):
    with pytest.raises(store.StoreError) as exc:
        serial._measure_filesystem(str(tmp_path / "missing"))
    assert exc.value.reason_code == "invalid_layout"


def test_output_directory_symlink_maps_to_reason(manifest, sane, tmp_path):
    link = tmp_path / "outlink"
    link.symlink_to(sane["output"])
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=str(link),
            torch=_cpu_torch(),
        )
    # Same platform-dependent rejection set as the filesystem symlink test.
    assert exc.value.reason_code in ("symlink_or_nonregular", "invalid_layout")


def test_output_directory_missing_maps_to_reason(manifest, sane, tmp_path):
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=str(tmp_path / "nope"),
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "invalid_layout"


# ---------------------------------------------------------------------------
# Thread-pool inspection errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info",
    ["x", [1], [{"num_threads": 0}], [{"num_threads": "1"}], [{"num_threads": True}], [None]],
)
def test_blas_pool_info_invalid(manifest, sane, info):
    sane["pools"]["info"] = info
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "invalid_threadpool_info"


def test_blas_pool_exception_normalized(manifest, sane, monkeypatch):
    def boom():
        raise RuntimeError("private pool sentinel")

    monkeypatch.setattr(serial.threadpoolctl, "threadpool_info", boom)
    events = _chain(manifest, [_open()])
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            events,
            job_id="fit001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "invalid_threadpool_info"
    assert "private pool sentinel" not in str(exc.value)


@pytest.mark.parametrize("field", ["num_threads", "interop_threads"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, False, "1", None])
def test_invalid_torch_thread_counts_rejected(sane, manifest, field, value):
    other = "interop_threads" if field == "num_threads" else "num_threads"
    kwargs = {field: value, other: 1}
    torch = _FakeTorch(_CpuCuda(), **kwargs)
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            _chain(manifest, [_open()]),
            job_id="pred001",
            output=sane["output"],
            torch=torch,
        )
    assert exc.value.reason_code == "invalid_torch_threads"


def test_no_loaded_thread_pools_reports_one_blas_thread(sane, manifest):
    sane["pools"]["info"] = []
    report = _run(
        manifest,
        _chain(manifest, [_open()]),
        job_id="pred001",
        output=sane["output"],
        torch=_cpu_torch(),
    )
    assert report["resources"]["blas_threads"] == 1


def test_full_fit_journal_still_permits_pred001(sane, manifest):
    specs = [_open()]
    for index in range(1, 79):
        job_id = f"fit{index:03d}"
        specs.append(_start(job_id, session_id=1))
        specs.append(_finish(job_id, "succeeded", session_id=1))
    events = _chain(manifest, specs)
    report = _run(manifest, events, job_id="pred001", output=sane["output"], torch=_cpu_torch())
    assert report["proposed_serial_candidate_admissible"] is True
    assert report["reasons"] == []
    assert report["candidate_check"]["prospective_model_fit_attempts"] == 78


def test_cumulative_artifact_exhaustion_blocks_with_fresh_resources(sane, manifest):
    events = _chain(
        manifest,
        [_open(), _progress(elapsed_ns=17, artifact_bytes=8 * _GIB, session_id=1)],
    )
    report = _run(manifest, events, job_id="fit001", output=sane["output"], torch=_cpu_torch())
    assert report["proposed_serial_candidate_admissible"] is False
    assert report["candidate_check"]["resource_breaches"] == ["artifact_budget_exhausted"]
    assert report["resources"]["filesystem_free_bytes"] == 64 * _GIB
    usage = serial.replay_attempt_journal(
        manifest,
        events,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_head_sha256=events[-1]["event_sha256"],
    )
    assert usage["new_artifact_bytes"] == 8 * _GIB
    assert usage["active_wall_ns"] == 17


@pytest.mark.parametrize(
    "frsize,bavail",
    [(True, 1), (1.0, 1), ("1", 1), (1, "1"), (1, True), (1, 1.0), (0, 1), (-1, 1), (1, -1)],
)
def test_invalid_fstatvfs_fields_close_real_fds(sane, manifest, monkeypatch, frsize, bavail):
    monkeypatch.setattr(serial.os, "fstatvfs", lambda fd: _FakeVfs(frsize, bavail))
    real_close = os.close
    closed = []

    def recording_close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(serial.os, "close", recording_close)
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            _chain(manifest, [_open()]),
            job_id="pred001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "invalid_filesystem"
    assert closed
    with pytest.raises(OSError) as err:
        real_close(closed[0])
    assert err.value.errno == errno.EBADF


def test_read_bounded_preserves_original_systemexit_despite_close_interrupt(monkeypatch):
    original = SystemExit("body failure")
    closed = []
    monkeypatch.setattr(serial.os, "open", lambda *args, **kwargs: 4242)

    def fake_read(fd, count):
        raise original

    monkeypatch.setattr(serial.os, "read", fake_read)

    def fake_close(fd):
        closed.append(fd)
        raise KeyboardInterrupt

    monkeypatch.setattr(serial.os, "close", fake_close)
    with pytest.raises(SystemExit) as exc:
        serial._read_bounded("/fake/status", 10, "invalid_proc_status")
    assert exc.value is original
    assert closed == [4242]


def test_read_bounded_successful_read_propagates_cleanup_interrupt(monkeypatch):
    interrupt = KeyboardInterrupt()
    chunks = iter([b"ok\n", b""])
    monkeypatch.setattr(serial.os, "open", lambda *args, **kwargs: 99)
    monkeypatch.setattr(serial.os, "read", lambda fd, count: next(chunks))

    def fake_close(fd):
        raise interrupt

    monkeypatch.setattr(serial.os, "close", fake_close)
    with pytest.raises(KeyboardInterrupt) as exc:
        serial._read_bounded("/fake/status", 10, "invalid_proc_status")
    assert exc.value is interrupt


def _owned_dir_fd(path):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY)


def test_filesystem_body_systemexit_survives_close_interrupt(sane, monkeypatch):
    original = SystemExit("fstat failure")
    real_close = os.close
    parent_fd = _owned_dir_fd(sane["tmp"])
    root_fd = _owned_dir_fd(sane["tmp"])
    monkeypatch.setattr(store, "_resolve_parent", lambda directory: (parent_fd, "root"))
    monkeypatch.setattr(store, "_open_dir", lambda parent_fd, name: root_fd)

    def fake_fstatvfs(fd):
        raise original

    monkeypatch.setattr(serial.os, "fstatvfs", fake_fstatvfs)
    raised = []

    def close_then_interrupt(fd):
        real_close(fd)
        if not raised:
            raised.append(fd)
            raise KeyboardInterrupt

    monkeypatch.setattr(serial.os, "close", close_then_interrupt)
    with pytest.raises(SystemExit) as exc:
        serial._measure_filesystem(sane["output"])
    assert exc.value is original
    for owned in (parent_fd, root_fd):
        with pytest.raises(OSError) as err:
            real_close(owned)
        assert err.value.errno == errno.EBADF


def test_filesystem_cleanup_interrupt_propagates_and_closes_fds(sane, monkeypatch):
    interrupt = KeyboardInterrupt()
    real_close = os.close
    parent_fd = _owned_dir_fd(sane["tmp"])
    root_fd = _owned_dir_fd(sane["tmp"])
    monkeypatch.setattr(store, "_resolve_parent", lambda directory: (parent_fd, "root"))
    monkeypatch.setattr(store, "_open_dir", lambda parent_fd, name: root_fd)
    monkeypatch.setattr(serial.os, "fstatvfs", lambda fd: _FakeVfs(4096, 8))
    raised = []

    def close_then_interrupt(fd):
        real_close(fd)
        if not raised:
            raised.append(fd)
            raise interrupt

    monkeypatch.setattr(serial.os, "close", close_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as exc:
        serial._measure_filesystem(sane["output"])
    assert exc.value is interrupt
    for owned in (parent_fd, root_fd):
        with pytest.raises(OSError) as err:
            real_close(owned)
        assert err.value.errno == errno.EBADF


@pytest.mark.parametrize(
    "args,kwargs",
    [((), {}), ((), {"execution_authorized": True}), (({"execution_authorized": True},), {})],
)
def test_require_scientific_execution_always_refuses(args, kwargs):
    with pytest.raises(serial.SerialResourceError) as exc:
        serial.require_scientific_execution(*args, **kwargs)
    assert exc.value.reason_code == "scientific_execution_not_authorized"


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_public_api_propagates_control_flow_exceptions(sane, manifest, monkeypatch, exc_type):
    def boom(*args, **kwargs):
        raise exc_type()

    monkeypatch.setattr(serial, "_sample", boom)
    with pytest.raises(exc_type):
        _run(
            manifest,
            _chain(manifest, [_open()]),
            job_id="pred001",
            output=sane["output"],
            torch=_cpu_torch(),
        )


def test_public_api_ordinary_failure_has_static_reason(sane, manifest, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sentinel /private/path text")

    monkeypatch.setattr(serial, "_sample", boom)
    with pytest.raises(serial.SerialResourceError) as exc:
        _run(
            manifest,
            _chain(manifest, [_open()]),
            job_id="pred001",
            output=sane["output"],
            torch=_cpu_torch(),
        )
    assert exc.value.reason_code == "invalid_serial_input"
    assert "sentinel" not in str(exc.value)
    assert "/private/path" not in str(exc.value)
