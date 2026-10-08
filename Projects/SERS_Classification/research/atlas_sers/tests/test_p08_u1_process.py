"""T296 tests: real spawn transport exercised through tiny deterministic fakes."""

import time

import pytest

from atlas_sers.evaluation.p08_u1_process import ProcessWorker, ProcessWorkerError


class _GoodEngine:
    def __init__(self, config):
        self.config = config

    def execute(self, job):
        return {"job_id": job["job_id"], "status": "ok", "sha": "abc123"}

    def telemetry(self):
        return {"allocated_gpu_bytes": 16, "reserved_gpu_bytes": 32}

    def close(self):
        self.closed = True


def good_factory(config, worker_id, kind):
    return _GoodEngine(config)


class _BadIdEngine:
    def execute(self, job):
        return {"job_id": "other-" + str(job["job_id"]), "status": "ok", "sha": "x"}

    def telemetry(self):
        return {"allocated_gpu_bytes": 0, "reserved_gpu_bytes": 0}


def bad_id_factory(config, worker_id, kind):
    return _BadIdEngine()


class _ExplodingEngine:
    def execute(self, job):
        raise RuntimeError("execute boom")

    def telemetry(self):
        return {"allocated_gpu_bytes": 0, "reserved_gpu_bytes": 0}


def exploding_factory(config, worker_id, kind):
    return _ExplodingEngine()


def failing_factory(config, worker_id, kind):
    raise RuntimeError("factory boom")


class _BadTelemetryEngine:
    def execute(self, job):
        return {"job_id": job["job_id"], "status": "ok", "sha": "x"}

    def telemetry(self):
        raise RuntimeError("telemetry boom")


def bad_telemetry_factory(config, worker_id, kind):
    return _BadTelemetryEngine()


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


def test_success_once_no_early_submit_busy_refusal():
    worker = ProcessWorker("w-ok", "CPU", factory=good_factory, config={})
    try:
        with pytest.raises(ProcessWorkerError):
            worker.submit({"job_id": "early"})
        _wait_ready(worker)
        worker.submit({"job_id": "job-1"})
        with pytest.raises(ProcessWorkerError):
            worker.submit({"job_id": "job-2"})
        result = _poll_until(worker)
        assert result == {"job_id": "job-1", "status": "ok", "sha": "abc123"}
        assert worker.alive()
        assert worker.poll() is None
    finally:
        worker.terminate()


def test_factory_error_raises_on_ready():
    worker = ProcessWorker("w-init", "CPU", factory=failing_factory, config={})
    raised = False
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if worker.ready():
                    break
            except ProcessWorkerError:
                raised = True
                break
            time.sleep(0.01)
        assert raised
    finally:
        worker.terminate()


def test_wrong_result_id_raises():
    worker = ProcessWorker("w-badid", "CPU", factory=bad_id_factory, config={})
    try:
        _wait_ready(worker)
        worker.submit({"job_id": "job-7"})
        with pytest.raises(ProcessWorkerError):
            _poll_until(worker)
    finally:
        worker.terminate()


def test_execute_exception_stops_process():
    worker = ProcessWorker("w-boom", "CPU", factory=exploding_factory, config={})
    try:
        _wait_ready(worker)
        worker.submit({"job_id": "job-1"})
        with pytest.raises(ProcessWorkerError):
            _poll_until(worker)
        deadline = time.monotonic() + 5.0
        while worker.alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not worker.alive()
    finally:
        worker.terminate()


def test_telemetry_healthy_typed_and_fresh():
    worker = ProcessWorker("w-tel", "GPU", factory=good_factory, config={})
    try:
        _wait_ready(worker)
        sample = worker.telemetry()
        assert sample["allocated_gpu_bytes"] == 16
        assert sample["reserved_gpu_bytes"] == 32
        assert isinstance(sample["allocated_gpu_bytes"], int)
        assert isinstance(sample["reserved_gpu_bytes"], int)
        assert isinstance(sample["timestamp"], float)
        assert sample["timestamp"] <= time.monotonic()
    finally:
        worker.terminate()


def test_telemetry_error_raises():
    worker = ProcessWorker("w-telbad", "CPU", factory=bad_telemetry_factory, config={})
    try:
        with pytest.raises(ProcessWorkerError):
            _wait_ready(worker)
    finally:
        worker.terminate()


def test_terminate_idempotent_and_no_restart():
    worker = ProcessWorker("w-term", "CPU", factory=good_factory, config={})
    _wait_ready(worker)
    worker.terminate()
    assert not worker.alive()
    worker.terminate()
    worker.terminate()
    assert not worker.alive()
    with pytest.raises(ProcessWorkerError):
        worker.submit({"job_id": "after"})


def test_fresh_owned_pids_do_not_share_target():
    first = ProcessWorker("w-pid-a", "CPU", factory=good_factory, config={})
    second = ProcessWorker("w-pid-b", "CPU", factory=good_factory, config={})
    try:
        _wait_ready(first)
        _wait_ready(second)
        assert first.pid != second.pid
        first.terminate()
        assert not first.alive()
        assert second.alive()
        second.submit({"job_id": "still-1"})
        assert _poll_until(second)["job_id"] == "still-1"
    finally:
        first.terminate()
        second.terminate()
