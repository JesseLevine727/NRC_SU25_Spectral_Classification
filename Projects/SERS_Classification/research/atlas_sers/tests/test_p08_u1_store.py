"""Focused synthetic tests for the P08-U1 durable ledger primitive (T275)."""

import itertools
import os
import sqlite3
import subprocess
import sys

import pytest

from atlas_sers.evaluation import p08_u1_store as ledger


def _binding(tag="synthetic"):
    return {"plan_sha256": "0" * 64, "permit_id": tag, "inputs": ["a", "b"]}


_JOB_UNITS = itertools.count()


def _job(stage, model_id="C-RBF-SVM", policy=ledger.ALLOWED_POLICIES[0], deps=(), unit=None):
    if unit is None:
        unit = next(_JOB_UNITS)
    record = {
        "policy_id": policy,
        "stage": stage,
        "model_id": model_id,
        "dependencies": list(deps),
        "unit": unit,
    }
    record["job_id"] = "P08JOB-" + ledger.job_sha256(record)
    return record


def _snapshot(
    active=2000.0,
    artifact=70_000_000,
    rss=1 * 2**30,
    gpu=0,
    free=200 * 2**30,
    workers=0,
):
    return {
        "active_seconds": active,
        "artifact_bytes": artifact,
        "rss_bytes": rss,
        "gpu_bytes": gpu,
        "free_disk_bytes": free,
        "active_workers": workers,
    }


def _patch_reuse(monkeypatch, fits=0, predictions=0):
    monkeypatch.setattr(ledger, "EXPECTED_REUSE_FITS", fits)
    monkeypatch.setattr(ledger, "EXPECTED_REUSE_PREDICTIONS", predictions)
    monkeypatch.setattr(ledger, "MAX_REUSE_FITS", fits)
    monkeypatch.setattr(ledger, "MAX_REUSE_PREDICTIONS", predictions)


def _receipt(tag="r"):
    return {"sha256": (tag * 64)[:64]}


def test_dependency_order(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    fit = _job("source_fit")
    prediction = _job("source_validation_prediction", deps=[fit["job_id"]])
    refit = _job("final_refit", deps=[prediction["job_id"]])
    store = ledger.P08U1Store.create(tmp_path / "order", _binding(), [fit, prediction, refit])
    store.seal_reuse()
    snapshot = _snapshot()

    with pytest.raises(ledger.ValidationError):
        store.start(prediction["job_id"], "CPU", snapshot)

    store.start(fit["job_id"], "CPU", snapshot)
    with pytest.raises(ledger.ValidationError):
        store.start(prediction["job_id"], "CPU", snapshot)

    store.finish(fit["job_id"], "complete", _receipt("a"))
    store.start(prediction["job_id"], "CPU", snapshot)
    store.finish(prediction["job_id"], "complete", _receipt("b"))
    store.start(refit["job_id"], "CPU", snapshot)
    store.finish(refit["job_id"], "complete", _receipt("c"))

    summary = store.verify_events()
    assert summary["ok"] is True
    store.close()


def test_reuse_import_caps_duplicates_and_parent(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch, fits=1, predictions=1)
    fit = _job("source_fit")
    prediction = _job("source_validation_prediction", deps=[fit["job_id"]])
    extra_fit = _job("source_fit")
    store = ledger.P08U1Store.create(tmp_path / "reuse", _binding(), [fit, prediction, extra_fit])
    fit_evidence = {"job_id": fit["job_id"], "job_sha256": ledger.job_sha256(fit)}
    prediction_evidence = {
        "job_id": prediction["job_id"],
        "job_sha256": ledger.job_sha256(prediction),
    }

    with pytest.raises(ledger.ValidationError):
        store.record_reuse(prediction["job_id"], prediction_evidence)
    with pytest.raises(ledger.ValidationError):
        store.record_reuse(fit["job_id"], {"job_id": "P08JOB-other", "job_sha256": "x"})

    store.record_reuse(fit["job_id"], fit_evidence)
    with pytest.raises(ledger.ValidationError):
        store.record_reuse(fit["job_id"], fit_evidence)
    with pytest.raises(ledger.ValidationError):
        store.record_reuse(
            extra_fit["job_id"],
            {"job_id": extra_fit["job_id"], "job_sha256": ledger.job_sha256(extra_fit)},
        )
    store.record_reuse(prediction["job_id"], prediction_evidence)

    with pytest.raises(ledger.ValidationError):
        sealed = ledger.P08U1Store.create(tmp_path / "short", _binding(), [fit, prediction])
        sealed.record_reuse(fit["job_id"], fit_evidence)
        sealed.seal_reuse()

    assert store.seal_reuse() == {"sealed": True, "fits": 1, "predictions": 1}
    store.close()


def test_attempt_ceilings(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    monkeypatch.setattr(ledger, "MAX_FIT_TOTAL", 7)
    monkeypatch.setattr(ledger, "MAX_UNIQUE_FIT_JOBS", 2)
    fits = [_job("source_fit") for _ in range(3)]
    store = ledger.P08U1Store.create(tmp_path / "fits", _binding(), fits)
    store.seal_reuse()
    snapshot = _snapshot()
    store.start(fits[0]["job_id"], "CPU", snapshot)
    store.start(fits[1]["job_id"], "CPU", snapshot)
    with pytest.raises(ledger.BudgetError):
        store.start(fits[2]["job_id"], "CPU", snapshot)
    with pytest.raises(ledger.ReviewRequiredError):
        store.start(fits[2]["job_id"], "CPU", snapshot)
    store.close()

    monkeypatch.setattr(ledger, "MAX_SCALAR_ATTEMPTS", 1)
    scalars = [_job("scalar_calibration") for _ in range(2)]
    store2 = ledger.P08U1Store.create(tmp_path / "scalars", _binding(), scalars)
    store2.seal_reuse()
    store2.start(scalars[0]["job_id"], "CPU", snapshot)
    with pytest.raises(ledger.BudgetError):
        store2.start(scalars[1]["job_id"], "CPU", snapshot)
    store2.close()


def test_worker_slots(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    assert ledger.MAX_CPU_WORKERS == 8
    assert ledger.MAX_GPU_WORKERS == 1
    cpu_jobs = [_job("held_prediction") for _ in range(9)]
    gpu_jobs = [_job("held_prediction", model_id="D0-M") for _ in range(2)]
    store = ledger.P08U1Store.create(tmp_path / "slots", _binding(), cpu_jobs + gpu_jobs)
    store.seal_reuse()
    snapshot = _snapshot()
    for job in cpu_jobs[:8]:
        store.start(job["job_id"], "CPU", snapshot)
    with pytest.raises(ledger.ValidationError):
        store.start(cpu_jobs[8]["job_id"], "CPU", snapshot)
    store.start(gpu_jobs[0]["job_id"], "gpu", snapshot)
    with pytest.raises(ledger.ValidationError):
        store.start(gpu_jobs[1]["job_id"], "GPU", snapshot)
    with pytest.raises(ledger.ValidationError):
        store.start(gpu_jobs[1]["job_id"], "TPU", snapshot)
    store.close()


def test_failure_blocks_new_starts_but_running_may_finish(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    jobs = [_job("held_prediction") for _ in range(3)]
    store = ledger.P08U1Store.create(tmp_path / "fail", _binding(), jobs)
    store.seal_reuse()
    snapshot = _snapshot()
    store.start(jobs[0]["job_id"], "CPU", snapshot)
    store.start(jobs[1]["job_id"], "CPU", snapshot)
    store.finish(jobs[0]["job_id"], "failed", _receipt("d"))
    with pytest.raises(ledger.ReviewRequiredError):
        store.start(jobs[2]["job_id"], "CPU", snapshot)
    store.finish(jobs[1]["job_id"], "complete", _receipt("e"))
    assert store.status()["attempts"][jobs[1]["job_id"]]["status"] == "complete"
    store.close()


def test_clean_reopen_and_binding_immutability(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "clean")
    binding = _binding()
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    store.close()

    reopened = ledger.P08U1Store.reopen(run, binding)
    assert reopened.status()["state"] == "open"
    reopened.close()

    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, _binding("different"))


def test_unclean_reopen_requires_review(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "unclean")
    binding = _binding()
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    store.start(job["job_id"], "CPU", _snapshot())
    assert store.close()["clean"] is False

    with pytest.raises(ledger.ReviewRequiredError):
        ledger.P08U1Store.reopen(run, binding)


def test_failure_state_survives_clean_reopen(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "failure")
    binding = _binding()
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    store.start(job["job_id"], "CPU", _snapshot())
    store.finish(job["job_id"], "interrupted", _receipt("f"))
    assert store.close()["clean"] is True

    reopened = ledger.P08U1Store.reopen(run, binding)
    with pytest.raises(ledger.ReviewRequiredError):
        reopened.start(job["job_id"], "CPU", _snapshot())
    reopened.close()


def test_resource_validation_regression_and_limits(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    jobs = [_job("held_prediction") for _ in range(2)]
    store = ledger.P08U1Store.create(tmp_path / "resources", _binding(), jobs)
    store.seal_reuse()

    with pytest.raises(ledger.ValidationError):
        store.start(jobs[0]["job_id"], "CPU", {"active_seconds": 1.0})
    with pytest.raises(ledger.ValidationError):
        store.start(jobs[0]["job_id"], "CPU", _snapshot(active=float("inf")))
    with pytest.raises(ledger.ValidationError):
        store.start(jobs[0]["job_id"], "CPU", _snapshot(rss="lots"))
    with pytest.raises(ledger.ValidationError):
        store.start(jobs[0]["job_id"], "CPU", _snapshot(active=True))

    store.start(jobs[0]["job_id"], "CPU", _snapshot(active=5000.0))
    with pytest.raises(ledger.ValidationError):
        store.start(jobs[1]["job_id"], "CPU", _snapshot(active=4000.0))
    store.close()

    ram_job = _job("held_prediction")
    store2 = ledger.P08U1Store.create(tmp_path / "ram", _binding(), [ram_job])
    store2.seal_reuse()
    with pytest.raises(ledger.BudgetError):
        store2.start(ram_job["job_id"], "CPU", _snapshot(rss=ledger.MAX_RAM_BYTES + 1))
    store2.close()

    disk_job = _job("held_prediction")
    store3 = ledger.P08U1Store.create(tmp_path / "disk", _binding(), [disk_job])
    store3.seal_reuse()
    with pytest.raises(ledger.BudgetError):
        store3.start(disk_job["job_id"], "CPU", _snapshot(free=1))
    store3.close()


def test_event_chain_tamper_detected(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "tamper")
    binding = _binding()
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    store.start(job["job_id"], "CPU", _snapshot())
    store.finish(job["job_id"], "complete", _receipt("a"))
    store.close()

    raw = sqlite3.connect(os.path.join(run, "ledger.sqlite3"))
    raw.execute(
        "UPDATE events SET payload_json=? WHERE seq=1",
        ('{"type":"create","job_count":0}',),
    )
    raw.commit()
    raw.close()

    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, binding)


def test_attempt_and_job_table_tamper_detected(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "tables")
    binding = _binding()
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    store.start(job["job_id"], "CPU", _snapshot())
    store.finish(job["job_id"], "complete", _receipt("a"))
    store.close()

    path = os.path.join(run, "ledger.sqlite3")
    raw = sqlite3.connect(path)
    raw.execute("UPDATE attempts SET status='running'")
    raw.commit()
    raw.close()
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, binding)

    raw = sqlite3.connect(path)
    raw.execute("UPDATE attempts SET status='complete'")
    raw.execute("UPDATE jobs SET job_json=? WHERE job_id=?", ('{"job_id":"x"}', job["job_id"]))
    raw.commit()
    raw.close()
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, binding)


def test_status_returns_copies(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    store = ledger.P08U1Store.create(tmp_path / "copy", _binding(), [job])
    store.seal_reuse()
    store.start(job["job_id"], "CPU", _snapshot())

    first = store.status()
    first["attempts"][job["job_id"]]["status"] = "tampered"
    first["jobs_by_stage"]["held_prediction"] = 999
    second = store.status()
    assert second["attempts"][job["job_id"]]["status"] == "running"
    assert second["jobs_by_stage"]["held_prediction"] == 1
    store.close()


def test_public_summary_privacy(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "private")
    store = ledger.P08U1Store.create(run, _binding(), [job])
    store.seal_reuse()
    store.start(job["job_id"], "CPU", _snapshot())
    summary = store.public_summary()
    blob = repr(summary)
    assert job["job_id"] not in blob
    assert run not in blob
    assert "receipt" not in blob.lower()
    store.close()


def test_lock_collision_refused(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    binding = _binding()
    run = str(tmp_path / "lock")
    store = ledger.P08U1Store.create(run, binding, [_job("held_prediction")])
    with pytest.raises(ledger.LockError):
        ledger.P08U1Store.reopen(run, binding)
    store.close()
    reopened = ledger.P08U1Store.reopen(run, binding)
    reopened.close()


def test_model_stage_lane_enforced(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    neural_fit = _job("source_fit", model_id="D0-M")
    classical_prediction = _job("held_prediction")
    store = ledger.P08U1Store.create(
        tmp_path / "lanes", _binding(), [neural_fit, classical_prediction]
    )
    store.seal_reuse()
    snapshot = _snapshot()
    with pytest.raises(ledger.ValidationError):
        store.start(neural_fit["job_id"], "CPU", snapshot)
    store.start(neural_fit["job_id"], "GPU", snapshot)
    with pytest.raises(ledger.ValidationError):
        store.start(classical_prediction["job_id"], "GPU", snapshot)
    store.start(classical_prediction["job_id"], "CPU", snapshot)
    store.close()


def test_reuse_sealed_refuses_further_imports(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch, fits=1, predictions=0)
    fit = _job("source_fit")
    extra = _job("source_fit")
    store = ledger.P08U1Store.create(tmp_path / "sealrefuse", _binding(), [fit, extra])
    store.record_reuse(
        fit["job_id"], {"job_id": fit["job_id"], "job_sha256": ledger.job_sha256(fit)}
    )
    store.seal_reuse()
    with pytest.raises(ledger.ValidationError):
        store.record_reuse(
            extra["job_id"],
            {"job_id": extra["job_id"], "job_sha256": ledger.job_sha256(extra)},
        )
    store.close()


def test_duplicate_job_identity_rejected(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    first = _job("held_prediction", unit="dup")
    second = _job("held_prediction", unit="dup")
    assert first["job_id"] == second["job_id"]
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.create(tmp_path / "dup", _binding(), [first, second])


def test_binding_json_tamper_detected(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "bindingtamper")
    binding = _binding()
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    store.close()
    raw = sqlite3.connect(os.path.join(run, "ledger.sqlite3"))
    raw.execute("UPDATE meta SET value=? WHERE key='binding_json'", ('{"x":1}',))
    raw.commit()
    raw.close()
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, binding)


def test_registered_job_removal_detected(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    jobs = [_job("held_prediction"), _job("held_prediction")]
    run = str(tmp_path / "jobremoved")
    binding = _binding()
    store = ledger.P08U1Store.create(run, binding, jobs)
    store.seal_reuse()
    store.close()
    raw = sqlite3.connect(os.path.join(run, "ledger.sqlite3"))
    raw.execute("DELETE FROM jobs WHERE job_id=?", (jobs[0]["job_id"],))
    raw.commit()
    raw.close()
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, binding)


def test_registered_job_identity_binding_detected(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "jobidentity")
    binding = _binding("jobidentity")
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    store.close()

    tampered = dict(job)
    tampered["unit"] = "tampered-unit"
    raw = sqlite3.connect(os.path.join(run, "ledger.sqlite3"))
    raw.execute(
        "UPDATE jobs SET job_json=?,job_sha=? WHERE job_id=?",
        (ledger._canonical(tampered), ledger.job_sha256(tampered), job["job_id"]),
    )
    raw.commit()
    raw.close()
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, binding)


def _finished_fit_run(tmp_path, monkeypatch, name):
    _patch_reuse(monkeypatch)
    job = _job("source_fit")
    run = str(tmp_path / name)
    binding = _binding(name)
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    store.start(job["job_id"], "CPU", _snapshot())
    store.finish(job["job_id"], "complete", _receipt("a"))
    store.close()
    return run, binding, job


@pytest.mark.parametrize(
    "name,sql",
    (
        ("worker", "UPDATE attempts SET worker_kind='GPU'"),
        ("receipt", 'UPDATE attempts SET receipt_json=\'{"sha256":"zzzz"}\''),
        ("resource", "UPDATE resources SET rss_bytes=12345"),
        ("counter", "UPDATE meta SET value='0' WHERE key='fit_attempt_count'"),
    ),
)
def test_ledger_field_tamper_detected(tmp_path, monkeypatch, name, sql):
    run, binding, _ = _finished_fit_run(tmp_path, monkeypatch, "field_tamper_" + name)
    raw = sqlite3.connect(os.path.join(run, "ledger.sqlite3"))
    raw.execute(sql)
    raw.commit()
    raw.close()
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, binding)


def test_reuse_evidence_tamper_detected(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch, fits=1, predictions=0)
    fit = _job("source_fit")
    run = str(tmp_path / "reusetamper")
    binding = _binding("reusetamper")
    store = ledger.P08U1Store.create(run, binding, [fit])
    store.record_reuse(
        fit["job_id"], {"job_id": fit["job_id"], "job_sha256": ledger.job_sha256(fit)}
    )
    store.seal_reuse()
    store.close()
    raw = sqlite3.connect(os.path.join(run, "ledger.sqlite3"))
    raw.execute(
        "UPDATE reuse SET evidence_json=? WHERE job_id=?", ('{"job_id":"x"}', fit["job_id"])
    )
    raw.commit()
    raw.close()
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, binding)


def test_resource_breach_recorded_and_survives_reopen(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "breach")
    binding = _binding("breach")
    store = ledger.P08U1Store.create(run, binding, [job])
    store.seal_reuse()
    breach = _snapshot(rss=ledger.MAX_RAM_BYTES + 4096, gpu=ledger.MAX_GPU_BYTES + 4096)
    with pytest.raises(ledger.BudgetError):
        store.start(job["job_id"], "CPU", breach)
    assert store.close()["clean"] is True
    reopened = ledger.P08U1Store.reopen(run, binding)
    raw = sqlite3.connect(os.path.join(run, "ledger.sqlite3"))
    row = raw.execute(
        "SELECT rss_bytes,gpu_bytes FROM resources ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    raw.close()
    assert row[0] == ledger.MAX_RAM_BYTES + 4096
    assert row[1] == ledger.MAX_GPU_BYTES + 4096
    reopened.close()


def test_ram_ceiling_amended_36gib_boundary(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    assert ledger.MAX_RAM_BYTES == 36 * 2**30
    jobs = [_job("held_prediction"), _job("held_prediction")]
    run = str(tmp_path / "ram36")
    binding = _binding("ram36")
    store = ledger.P08U1Store.create(run, binding, jobs)
    store.seal_reuse()
    # Exactly 36 GiB is admitted; one byte more is refused and latches the store.
    store.start(jobs[0]["job_id"], "CPU", _snapshot(rss=ledger.MAX_RAM_BYTES))
    store.finish(jobs[0]["job_id"], "complete", _receipt("a"))
    with pytest.raises(ledger.BudgetError):
        store.start(jobs[1]["job_id"], "CPU", _snapshot(rss=ledger.MAX_RAM_BYTES + 1))
    with pytest.raises(ledger.ReviewRequiredError):
        store.start(jobs[1]["job_id"], "CPU", _snapshot())
    assert store.close()["clean"] is True

    reopened = ledger.P08U1Store.reopen(run, binding)
    with pytest.raises(ledger.ReviewRequiredError):
        reopened.start(jobs[1]["job_id"], "CPU", _snapshot())
    reopened.close()


def test_create_failure_preserves_evidence(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    run = str(tmp_path / "crashed")
    original = ledger._append_event

    def explode(conn, payload):
        raise RuntimeError("journal_write_failure")

    monkeypatch.setattr(ledger, "_append_event", explode)
    with pytest.raises(RuntimeError):
        ledger.P08U1Store.create(run, _binding(), [job])
    monkeypatch.setattr(ledger, "_append_event", original)
    assert os.path.isdir(run)
    assert os.path.exists(os.path.join(run, "ledger.sqlite3"))
    with pytest.raises(ledger.P08U1Error) as excinfo:
        ledger.P08U1Store.reopen(run, _binding())
    assert not isinstance(excinfo.value, ledger.LockError)


def test_persistence_failure_latches_store(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    store = ledger.P08U1Store.create(tmp_path / "latch", _binding("latch"), [job])
    store.seal_reuse()
    original = ledger._append_event

    def explode(conn, payload):
        raise sqlite3.Error("journal_write_failure")

    monkeypatch.setattr(ledger, "_append_event", explode)
    with pytest.raises(sqlite3.Error):
        store.start(job["job_id"], "CPU", _snapshot())
    monkeypatch.setattr(ledger, "_append_event", original)
    with pytest.raises(ledger.ReviewRequiredError) as excinfo:
        store.start(job["job_id"], "CPU", _snapshot())
    assert "persistence_failure_latched" in str(excinfo.value)
    store.close()


def test_start_avoids_full_attempt_and_resource_scans(tmp_path, monkeypatch):
    _patch_reuse(monkeypatch)
    job = _job("held_prediction")
    store = ledger.P08U1Store.create(tmp_path / "trace", _binding(), [job])
    store.seal_reuse()
    traced = []
    store._conn.set_trace_callback(traced.append)
    try:
        store.start(job["job_id"], "CPU", _snapshot())
    finally:
        store._conn.set_trace_callback(None)
    for statement in traced:
        upper = statement.upper()
        assert " JOIN " not in upper
        assert "MAX(" not in upper
        if "FROM ATTEMPTS" in upper:
            assert "WHERE" in upper
        if "FROM RESOURCES" in upper and "INSERT" not in upper:
            assert "LIMIT 1" in upper
    store.close()


def test_lock_released_after_process_crash(tmp_path):
    lock_path = str(tmp_path / "crash.lock")
    code = (
        "import sys, time\n"
        "from atlas_sers.evaluation import p08_u1_store as ledger\n"
        "lock = ledger._Lock(sys.argv[1])\n"
        "lock.acquire()\n"
        "sys.stdout.write('locked\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    env = dict(os.environ)
    package_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(ledger.__file__)))
    )
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-c", code, lock_path],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(ledger.LockError):
            ledger._Lock(lock_path).acquire()
    finally:
        proc.kill()
        proc.wait(timeout=30)
    lock = ledger._Lock(lock_path)
    lock.acquire()
    lock.release()
