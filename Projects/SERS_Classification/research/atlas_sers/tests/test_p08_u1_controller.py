"""T294 tests: small synthetic graphs over a real P08U1Store and fake workers."""

from __future__ import annotations

import pytest

from atlas_sers.evaluation import p08_plan
from atlas_sers.evaluation import p08_u1_controller as ctrl
from atlas_sers.evaluation import p08_u1_store as store_mod

CLASSICAL = "C-RBF-SVM"
NEURAL = "D0-M"
POLICY = "PP-U-SG"


def _job(stage, model_id, dependencies, tag, policy=POLICY):
    record = dict(
        stage=stage,
        model_id=model_id,
        policy_id=policy,
        representation_id=p08_plan.POLICY_REPRESENTATION[policy],
        array_sha256="a" * 64,
        context_id=tag,
        model_spec_sha256="a" * 64,
        unit_id="unit",
        seed="deterministic" if model_id == CLASSICAL else 20260805,
        candidate_id="candidate",
        hyperparameter_sha256="a" * 64,
        fit_uid_sha256="b" * 64,
        validation_uid_sha256="c" * 64,
        test_uid_sha256="d" * 64,
        resolution=p08_plan.FIXED_SPEC,
        evidence_status=p08_plan.EVIDENCE_FUTURE,
    )
    return p08_plan._new_job(record, dependencies)


@pytest.fixture(autouse=True)
def synthetic_reuse_boundary(monkeypatch):
    monkeypatch.setattr(store_mod, "EXPECTED_REUSE_FITS", 0)
    monkeypatch.setattr(store_mod, "EXPECTED_REUSE_PREDICTIONS", 0)


def classical_chain(tag):
    fit = _job("source_fit", CLASSICAL, [], tag + "-f")
    pred = _job("source_validation_prediction", CLASSICAL, [fit["job_id"]], tag + "-p")
    cal_fit = _job("calibration_model_fit", CLASSICAL, [pred["job_id"]], tag + "-cf")
    cal_pred = _job(
        "calibration_validation_prediction",
        CLASSICAL,
        [cal_fit["job_id"]],
        tag + "-cp",
    )
    return [fit, pred, cal_fit, cal_pred]


def neural_chain(tag):
    fit = _job("source_fit", NEURAL, [], tag + "-f")
    pred = _job("source_validation_prediction", NEURAL, [fit["job_id"]], tag + "-p")
    return [fit, pred]


def make_store(tmp_path, jobs):
    directory = tmp_path / "ledger"
    store = store_mod.P08U1Store.create(str(directory), {"run": "t294"}, jobs)
    store.seal_reuse()
    return store


class FakeWorker:
    def __init__(
        self,
        worker_id,
        kind,
        *,
        mode="auto",
        delay=0,
        fail_status=None,
        crash_on_submit=False,
        submit_error=None,
        poll_transform=None,
    ):
        self.worker_id = worker_id
        self.kind = kind
        self.mode = mode
        self.delay = delay
        self.fail_status = fail_status
        self.crash_on_submit = crash_on_submit
        self.submit_error = submit_error
        self.poll_transform = poll_transform
        self.on_submit = None
        self.current = None
        self.submitted = []
        self.terminated = 0
        self.alive_flag = True
        self.ticks = 0

    def submit(self, job):
        self.submitted.append(job["job_id"])
        if self.on_submit is not None:
            self.on_submit(job)
        if self.submit_error is not None:
            raise self.submit_error
        if self.crash_on_submit:
            self.alive_flag = False
        self.current = job

    def poll(self):
        if self.current is None:
            return None
        if self.poll_transform is not None:
            return self.poll_transform(self, self.current)
        if self.mode == "hold":
            return None
        if self.crash_on_submit:
            return None
        if self.delay:
            self.ticks += 1
            if self.ticks < self.delay:
                return None
        job = self.current
        self.current = None
        status = self.fail_status or "complete"
        return {"job_id": job["job_id"], "status": status, "receipt": {}}

    def alive(self):
        return self.alive_flag

    def terminate(self):
        self.terminated += 1
        self.alive_flag = False


class Resources:
    def __init__(self, fail_at=None):
        self.n = 0
        self.fail_at = fail_at

    def __call__(self, active_workers):
        self.n += 1
        if self.fail_at is not None and self.n >= self.fail_at:
            raise RuntimeError("resource_provider_failure")
        return {
            "active_seconds": 2000.0 + self.n,
            "artifact_bytes": 100_000_000 + self.n,
            "rss_bytes": 1_000_000,
            "gpu_bytes": 0,
            "free_disk_bytes": 256 * 2**30,
            "active_workers": int(active_workers),
        }


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def verify_ok(job, result):
    return {"sha256": ctrl.sha256_value({"job_id": job["job_id"], "status": result["status"]})}


def run(
    store,
    jobs,
    workers,
    completed=(),
    resources=None,
    verify=verify_ok,
    on_progress=None,
    poll_seconds=0.25,
    clock=None,
    sleep=None,
):
    kwargs = dict(
        store=store,
        jobs=jobs,
        workers=workers,
        completed=set(completed),
        resource_snapshot=resources if resources is not None else Resources(),
        verify_result=verify,
        poll_seconds=poll_seconds,
    )
    if on_progress is not None:
        kwargs["on_progress"] = on_progress
    if clock is not None:
        kwargs["clock"] = clock
    if sleep is not None:
        kwargs["sleep"] = sleep
    return ctrl.run_controller(**kwargs)


def attempts_by_status(store):
    return {jid: a["status"] for jid, a in store.status()["attempts"].items()}


def test_normal_multi_worker(tmp_path):
    jobs = classical_chain("a") + classical_chain("b")
    store = make_store(tmp_path, jobs)
    try:
        workers = [FakeWorker("cpu1", "CPU"), FakeWorker("cpu2", "CPU")]
        result = run(store, jobs, workers)
        assert result["state"] == "complete"
        assert result["complete"] == len(jobs)
        assert result["remaining"] == 0
        statuses = attempts_by_status(store)
        assert len(statuses) == len(jobs)
        assert set(statuses.values()) == {"complete"}
        assert sum(len(w.submitted) for w in workers) == len(jobs)
        assert all(w.terminated >= 1 for w in workers)
    finally:
        store.close()


def test_affinity_pair_stays_on_same_worker(tmp_path):
    jobs = classical_chain("a") + classical_chain("b")
    store = make_store(tmp_path, jobs)
    try:
        pred_by_fit = {}
        for job in jobs:
            if job["stage"] == "source_validation_prediction":
                pred_by_fit[job["dependencies"][0]] = job["job_id"]
        workers = [FakeWorker("cpu1", "CPU"), FakeWorker("cpu2", "CPU")]
        run(store, jobs, workers)
        for worker in workers:
            assert len(worker.submitted) >= 2
            fit_id = worker.submitted[0]
            assert fit_id in pred_by_fit
            assert worker.submitted[1] == pred_by_fit[fit_id]
    finally:
        store.close()


@pytest.mark.parametrize(
    "failure",
    [
        "failed",
        "interrupted",
        "submit",
        "crash",
        "wrong_id",
        "verification",
        "resource",
        "progress",
    ],
)
def test_failure_stops_admissions_and_retains_attempts(tmp_path, failure):
    jobs = classical_chain("a") + classical_chain("b")
    store = make_store(tmp_path, jobs)
    workers = [FakeWorker("cpu1", "CPU"), FakeWorker("cpu2", "CPU", mode="hold")]
    options = {}
    if failure in ("failed", "interrupted"):
        workers[0].fail_status = failure
    elif failure == "submit":
        workers[0].submit_error = RuntimeError("submit")
    elif failure == "crash":
        workers[0].crash_on_submit = True
    elif failure == "wrong_id":
        workers[0].poll_transform = lambda w, j: dict(job_id="wrong", status="complete", receipt={})
    elif failure == "verification":

        def invalid(job, result):
            raise ValueError("bad artifact")

        options["verify"] = invalid
    elif failure == "resource":
        options["resources"] = Resources(fail_at=2)
    elif failure == "progress":

        def invalid_progress(summary):
            raise OSError("html")

        options["on_progress"] = invalid_progress
    try:
        with pytest.raises(ctrl.ControllerError):
            run(store, jobs, workers, **options)
        statuses = attempts_by_status(store)
        assert len(statuses) <= 2
        assert "running" not in statuses.values()
        assert all(w.terminated >= 1 for w in workers)
        assert sum(len(w.submitted) for w in workers) <= 2
        assert not any(j["stage"].endswith("prediction") for j in jobs if j["job_id"] in statuses)
    finally:
        store.close()


def test_admission_is_committed_before_worker_submit(tmp_path):
    jobs = classical_chain("one")
    store = make_store(tmp_path, jobs)
    worker = FakeWorker("cpu", "CPU")

    def check(job):
        assert attempts_by_status(store)[job["job_id"]] == "running"

    worker.on_submit = check
    try:
        run(store, jobs, [worker])
        assert len(worker.submitted) == len(set(worker.submitted)) == len(jobs)
    finally:
        store.close()


def test_gpu_and_cpu_lanes(tmp_path):
    jobs = classical_chain("cpu") + neural_chain("gpu")
    store = make_store(tmp_path, jobs)
    workers = [FakeWorker("cpu", "CPU"), FakeWorker("gpu", "GPU")]
    try:
        run(store, jobs, workers)
        mapping = {j["job_id"]: j for j in jobs}
        assert all(mapping[j]["model_id"] == NEURAL for j in workers[1].submitted)
        assert all(mapping[j]["model_id"] == CLASSICAL for j in workers[0].submitted)
    finally:
        store.close()


def test_worker_panel_amended_cpu8_gpu1_limits():
    assert ctrl.MAX_CPU_WORKERS == 8
    assert ctrl.MAX_GPU_WORKERS == 1
    assert ctrl.MAX_TOTAL_WORKERS == 9
    cpu = [FakeWorker(f"cpu{index}", "CPU") for index in range(8)]
    gpu = [FakeWorker("gpu0", "GPU")]
    assert len(ctrl._validate_workers(cpu + gpu, {"CPU", "GPU"})) == 9
    with pytest.raises(ctrl.ControllerError, match="cpu_worker_limit"):
        ctrl._validate_workers(cpu + [FakeWorker("cpu8", "CPU")], {"CPU"})
    with pytest.raises(ctrl.ControllerError, match="gpu_worker_limit"):
        ctrl._validate_workers(gpu + [FakeWorker("gpu1", "GPU")], {"GPU"})


def test_orphan_completed_fit_needs_review_and_stops_workers(tmp_path):
    jobs = classical_chain("orphan")
    store = make_store(tmp_path, jobs)
    workers = [FakeWorker("cpu", "CPU")]
    try:
        with pytest.raises(ctrl.ControllerError, match="resumed_fit_without_prediction"):
            run(store, jobs, workers, completed={jobs[0]["job_id"]})
        assert not attempts_by_status(store) and workers[0].terminated
    finally:
        store.close()


@pytest.mark.parametrize("poll", [float("nan"), float("inf"), 0, -1, True])
def test_bad_poll_limit_stops_workers(tmp_path, poll):
    jobs = classical_chain("poll")
    store = make_store(tmp_path, jobs)
    worker = FakeWorker("cpu", "CPU")
    try:
        with pytest.raises(ctrl.ControllerError, match="poll_seconds_invalid"):
            run(store, jobs, [worker], poll_seconds=poll)
        assert worker.terminated and not worker.submitted
    finally:
        store.close()


def test_resource_polling_while_job_is_busy(tmp_path):
    jobs = classical_chain("busy")
    store = make_store(tmp_path, jobs)
    worker = FakeWorker("cpu", "CPU", mode="hold")
    clock = FakeClock()
    try:
        with pytest.raises(ctrl.ControllerError, match="resource_snapshot_failed"):
            run(
                store,
                jobs,
                [worker],
                resources=Resources(fail_at=3),
                clock=clock,
                sleep=clock.sleep,
            )
        assert clock.t <= 4.25 and worker.terminated
        assert set(attempts_by_status(store).values()) == {"interrupted"}
    finally:
        store.close()
