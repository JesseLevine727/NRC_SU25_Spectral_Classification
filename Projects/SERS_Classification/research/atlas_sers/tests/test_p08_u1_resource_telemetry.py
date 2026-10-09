"""T314 tests: bounded resource-monitor correction over the spawn transport."""

import ctypes
import time

import pytest

from atlas_sers.evaluation import p08_u1_resource_telemetry as rt
from atlas_sers.evaluation.p08_u1_process import (
    ProcessWorker,
    ProcessWorkerError,
    StaleTelemetryError,
)

# ---------------------------------------------------------------------------
# Real spawn-transport engines/factories (module level so spawn can pickle).
# ---------------------------------------------------------------------------


class _HealthyEngine:
    def __init__(self, config):
        self.config = config

    def execute(self, job):
        return {"job_id": job["job_id"], "status": "ok", "sha": "fresh"}

    def telemetry(self):
        return {"allocated_gpu_bytes": 4096, "reserved_gpu_bytes": 8192}

    def close(self):
        pass


def healthy_factory(config, worker_id, kind):
    return _HealthyEngine(config)


def _hold_gil(seconds):
    # The benchmark runtime is Linux. PyDLL keeps this child's GIL while libc
    # sleeps; an ordinary Python busy loop would periodically release it.
    sleeper = ctypes.PyDLL(None).sleep
    sleeper.argtypes = [ctypes.c_uint]
    sleeper.restype = ctypes.c_uint
    sleeper(int(seconds))


class _GilHoldEngine:
    def __init__(self, config):
        self.config = config

    def execute(self, job):
        _hold_gil(7.0)
        return {"job_id": job["job_id"], "status": "ok", "sha": "gil"}

    def telemetry(self):
        return {"allocated_gpu_bytes": 0, "reserved_gpu_bytes": 0}

    def close(self):
        pass


def gil_hold_factory(config, worker_id, kind):
    return _GilHoldEngine(config)


def _wait_ready(worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.ready():
            return
        time.sleep(0.01)
    raise AssertionError("worker never became ready")


def _poll_until(worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("worker never produced a result")


# ---------------------------------------------------------------------------
# Bare-process telemetry unit fixtures (no spawn, deterministic).
# ---------------------------------------------------------------------------


class _NoLock:
    def acquire(self, timeout=None):
        return True

    def release(self):
        pass


class _TelemetrySlot:
    def __init__(self, stamp, alloc, reserved, health):
        self.data = [stamp, alloc, reserved, health]

    def get_lock(self):
        return _NoLock()

    def __getitem__(self, index):
        return self.data[index]


class _ProcessHandle:
    def __init__(self, pid=987654):
        self.pid = pid


def _bare_worker(stamp, alloc, reserved, health=1.0, pid=987654, kind="GPU"):
    worker = object.__new__(ProcessWorker)
    worker._worker_id = "w-unit"
    worker._kind = kind
    worker._ready = True
    worker._closed = False
    worker._busy = False
    worker._current_id = None
    worker._telemetry = _TelemetrySlot(stamp, alloc, reserved, health)
    worker._process = _ProcessHandle(pid)
    return worker


def test_telemetry_typed_stale_carries_last_evidence():
    worker = _bare_worker(time.monotonic() - 6.0, 128.0, 256.0)
    with pytest.raises(StaleTelemetryError) as excinfo:
        worker.telemetry()
    err = excinfo.value
    assert isinstance(err, ProcessWorkerError)
    assert err.worker_id == "w-unit"
    assert err.worker_kind == "GPU"
    assert err.pid == 987654
    assert err.age_seconds > 5.0
    assert err.last_allocated_gpu_bytes == 128
    assert err.last_reserved_gpu_bytes == 256
    assert str(err) == "telemetry stale or invalid"


@pytest.mark.parametrize(
    "stamp, alloc, reserved, health",
    [
        (time.monotonic() + 60.0, 0.0, 0.0, 1.0),  # future timestamp
        (float("inf"), 0.0, 0.0, 1.0),  # nonfinite timestamp
        (time.monotonic() - 6.0, float("nan"), 0.0, 1.0),  # invalid value
        (time.monotonic() - 6.0, -1.0, 0.0, 1.0),  # negative value
        (time.monotonic() - 6.0, 1.5, 0.0, 1.0),  # fractional value
        (time.monotonic() - 6.0, 0.0, 0.0, 0.0),  # unhealthy
    ],
)
def test_telemetry_non_stale_failures_are_plain_errors(stamp, alloc, reserved, health):
    worker = _bare_worker(stamp, alloc, reserved, health)
    with pytest.raises(ProcessWorkerError) as excinfo:
        worker.telemetry()
    assert not isinstance(excinfo.value, StaleTelemetryError)


# ---------------------------------------------------------------------------
# Fake helper workers.
# ---------------------------------------------------------------------------


class _StaleWorker:
    def __init__(
        self,
        worker_id="w",
        kind="CPU",
        pid=4321,
        age=6.0,
        alloc=0,
        reserved=0,
        alive=True,
        stale_pid=None,
        stale_kind=None,
    ):
        self._worker_id = worker_id
        self._kind = kind
        self._pid = pid
        self._alive = alive
        self._stale = StaleTelemetryError(
            worker_id,
            stale_kind if stale_kind is not None else kind,
            stale_pid if stale_pid is not None else pid,
            age,
            alloc,
            reserved,
        )

    worker_id = property(lambda self: self._worker_id)
    kind = property(lambda self: self._kind)
    pid = property(lambda self: self._pid)

    def alive(self):
        return self._alive

    def telemetry(self):
        raise self._stale


class _ErrorWorker:
    def __init__(self, worker_id="w", kind="CPU", pid=4321, alive=True, message="boom"):
        self._worker_id = worker_id
        self._kind = kind
        self._pid = pid
        self._alive = alive
        self._message = message

    worker_id = property(lambda self: self._worker_id)
    kind = property(lambda self: self._kind)
    pid = property(lambda self: self._pid)

    def alive(self):
        return self._alive

    def telemetry(self):
        raise ProcessWorkerError(self._message)


class _FreshWorker:
    def __init__(self, worker_id="w", kind="CPU", pid=4321, alive=True, sample=None):
        self._worker_id = worker_id
        self._kind = kind
        self._pid = pid
        self._alive = alive
        self._sample = sample

    worker_id = property(lambda self: self._worker_id)
    kind = property(lambda self: self._kind)
    pid = property(lambda self: self._pid)

    def alive(self):
        return self._alive

    def telemetry(self):
        return self._sample


def test_fresh_fake_maps_to_allocator_heartbeat():
    worker = _FreshWorker(
        sample={"timestamp": 123.0, "allocated_gpu_bytes": 7, "reserved_gpu_bytes": 9}
    )
    out = rt.sample_worker_resources(worker)
    assert out == {
        "timestamp": 123.0,
        "allocated_gpu_bytes": 7,
        "reserved_gpu_bytes": 9,
        "source": "allocator_heartbeat",
        "upper_bound": False,
        "stale_age_seconds": None,
    }


def test_cpu_stale_zero_uses_device_contract():
    worker = _StaleWorker(kind="CPU", pid=555, alloc=0, reserved=0, age=6.25)
    out = rt.sample_worker_resources(worker)
    assert out["allocated_gpu_bytes"] == 0
    assert out["reserved_gpu_bytes"] == 0
    assert out["source"] == "cpu_device_contract"
    assert out["upper_bound"] is True
    assert out["stale_age_seconds"] == 6.25
    assert out["timestamp"] <= time.monotonic()


def test_cpu_stale_nonzero_refuses():
    worker = _StaleWorker(kind="CPU", alloc=1, reserved=0)
    with pytest.raises(ProcessWorkerError):
        rt.sample_worker_resources(worker)


def test_gpu_stale_uses_device_process_upper_bound():
    worker = _StaleWorker(kind="GPU", pid=777, alloc=111, reserved=222)
    out = rt.sample_worker_resources(worker, query_gpu_process_bytes=lambda: {777: 4096})
    assert out["allocated_gpu_bytes"] == 4096
    assert out["reserved_gpu_bytes"] == 4096
    assert out["source"] == "device_process_upper_bound"
    assert out["upper_bound"] is True
    assert out["stale_age_seconds"] == 6.0
    assert out["allocated_gpu_bytes"] != 111


def test_gpu_stale_missing_target_refuses():
    worker = _StaleWorker(kind="GPU", pid=777)
    with pytest.raises(ProcessWorkerError):
        rt.sample_worker_resources(worker, query_gpu_process_bytes=lambda: {})


def test_gpu_stale_query_failure_refuses():
    def boom():
        raise RuntimeError("nvidia-smi down")

    worker = _StaleWorker(kind="GPU", pid=777)
    with pytest.raises(ProcessWorkerError):
        rt.sample_worker_resources(worker, query_gpu_process_bytes=boom)


def test_gpu_stale_invalid_bound_refuses():
    worker = _StaleWorker(kind="GPU", pid=777)
    with pytest.raises(ProcessWorkerError):
        rt.sample_worker_resources(worker, query_gpu_process_bytes=lambda: {777: -1})


def test_dead_stale_worker_refuses():
    worker = _StaleWorker(kind="CPU", pid=555, alive=False)
    with pytest.raises(ProcessWorkerError):
        rt.sample_worker_resources(worker)


def test_identity_mismatch_refuses():
    worker = _StaleWorker(kind="GPU", pid=4321, stale_pid=9999)
    with pytest.raises(ProcessWorkerError):
        rt.sample_worker_resources(worker, query_gpu_process_bytes=lambda: {9999: 4096})


def test_plain_telemetry_error_propagates_without_fallback():
    worker = _ErrorWorker(message="telemetry unhealthy")
    with pytest.raises(ProcessWorkerError) as excinfo:
        rt.sample_worker_resources(worker)
    assert not isinstance(excinfo.value, StaleTelemetryError)
    assert "unhealthy" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Real ProcessWorker integration.
# ---------------------------------------------------------------------------


def test_real_worker_fresh_is_allocator_heartbeat():
    worker = ProcessWorker("w-fresh", "CPU", factory=healthy_factory, config={})
    try:
        _wait_ready(worker)
        out = rt.sample_worker_resources(worker)
        assert out["source"] == "allocator_heartbeat"
        assert out["upper_bound"] is False
        assert out["stale_age_seconds"] is None
        assert out["allocated_gpu_bytes"] == 4096
        assert out["reserved_gpu_bytes"] == 8192
        assert isinstance(out["timestamp"], float)
    finally:
        worker.terminate()


def test_real_gil_stall_stale_cpu_returns_conservative_zero_and_result():
    worker = ProcessWorker("w-gil", "CPU", factory=gil_hold_factory, config={})
    try:
        _wait_ready(worker)
        worker.submit({"job_id": "gil-1"})
        deadline = time.monotonic() + 7.0
        while True:
            try:
                worker.telemetry()
            except StaleTelemetryError:
                break
            if time.monotonic() > deadline:
                raise AssertionError("heartbeat never became stale")
            time.sleep(0.2)
        out = rt.sample_worker_resources(worker)
        assert out["source"] == "cpu_device_contract"
        assert out["allocated_gpu_bytes"] == 0
        assert out["reserved_gpu_bytes"] == 0
        assert out["upper_bound"] is True
        assert out["stale_age_seconds"] > 5.0
        result = _poll_until(worker, timeout=10.0)
        assert result["job_id"] == "gil-1"
    finally:
        worker.terminate()


# ---------------------------------------------------------------------------
# Default nvidia-smi query parser.
# ---------------------------------------------------------------------------


class _Completed:
    def __init__(self, stdout):
        self.stdout = stdout


def test_default_query_command_and_conversion(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _Completed("1234, 16\n5678, 0\n")

    monkeypatch.setattr(rt.subprocess, "run", fake_run)
    mapping = rt.query_gpu_process_bytes()
    assert mapping == {1234: 17 * 1024 * 1024, 5678: 1024 * 1024}
    assert captured["command"] == [
        "nvidia-smi",
        "--id=0",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    assert captured["kwargs"]["timeout"] == 2
    assert captured["kwargs"]["check"] is True
    assert captured["kwargs"]["text"] is True


@pytest.mark.parametrize(
    "stdout",
    [
        "0, 16\n",
        "-3, 16\n",
        "abc, 16\n",
        "1234\n",
        "1234, 16, extra\n",
        "1234, N/A\n",
        "1234, -5\n",
        "1234, 1.5\n",
        "1234, inf\n",
        "1234, 16\n1234, 32\n",
    ],
)
def test_default_query_rejects_bad_rows(monkeypatch, stdout):
    monkeypatch.setattr(rt.subprocess, "run", lambda *a, **k: _Completed(stdout))
    with pytest.raises(ProcessWorkerError):
        rt.query_gpu_process_bytes()


def test_default_query_empty_output_is_empty_mapping(monkeypatch):
    monkeypatch.setattr(rt.subprocess, "run", lambda *a, **k: _Completed("\n"))
    assert rt.query_gpu_process_bytes() == {}
