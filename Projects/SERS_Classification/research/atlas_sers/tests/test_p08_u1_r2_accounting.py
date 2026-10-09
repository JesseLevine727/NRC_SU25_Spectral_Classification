"""T308 synthetic tests for the optional R2 recovery accounting profile."""

import itertools
import json
import os

import pytest

from atlas_sers.evaluation import p08_u1_store as ledger

_R2_SMALL = {
    "R2_REPLAY_FIT_COUNT": 2,
    "R2_ADDITIONAL_FIT_COUNT": 3,
    "R2_SELECTOR_COUNT": 1,
    "R2_EXPECTED_REUSE_PAIRS": 3,
    "R2_HISTORICAL_OVERHEAD_ATTEMPTS": 14,
    "R2_MAX_FIT_TOTAL": 195216,
    "R2_MIN_ACTIVE_SECONDS": 100.0,
    "R2_MIN_ARTIFACT_BYTES": 1000,
}


def _apply_small(monkeypatch, **overrides):
    for name, value in _R2_SMALL.items():
        monkeypatch.setattr(ledger, name, overrides.get(name, value))


_UNITS = itertools.count()


def _job(stage, model_id="C-RBF-SVM", policy=None, deps=(), unit=None):
    if policy is None:
        policy = ledger.ALLOWED_POLICIES[0]
    if unit is None:
        unit = f"unit-{next(_UNITS)}"
    record = {
        "policy_id": policy,
        "stage": stage,
        "model_id": model_id,
        "dependencies": list(deps),
        "unit": unit,
    }
    record["job_id"] = "P08JOB-" + ledger.job_sha256(record)
    return record


def _evidence(record):
    return {"job_id": record["job_id"], "job_sha256": ledger.job_sha256(record)}


def _snapshot(
    active=210.0,
    artifact=2500,
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


def _mini_setup(monkeypatch, **overrides):
    _apply_small(monkeypatch, **overrides)
    fits = [_job("source_fit", unit=f"fit-{index}") for index in range(3)]
    predictions = [
        _job(
            "source_validation_prediction",
            deps=[fits[index]["job_id"]],
            unit=f"pred-{index}",
        )
        for index in range(3)
    ]
    replays = [_job("source_fit", unit=f"replay-{index}") for index in range(2)]
    selector = _job("select_refit_epochs", deps=[predictions[0]["job_id"]], unit="selector")
    jobs = fits + predictions + replays + [selector]
    profile = {
        "schema_version": "nato-sers-p08-u1-r2-accounting-v1",
        "parent_binding_sha256": "a" * 64,
        "parent_inventory_sha256": "b" * 64,
        "replay_fit_job_ids": sorted(record["job_id"] for record in replays),
        "additional_reuse_fit_job_ids": sorted(record["job_id"] for record in fits),
        "reused_epoch_selection_job_ids": [selector["job_id"]],
        "baseline_active_seconds": 200.0,
        "baseline_artifact_bytes": 2000,
    }
    binding = {"plan_sha256": "0" * 64, "permit_id": "r2", "inputs": ["a"]}
    binding["recovery_accounting"] = profile
    return binding, jobs, fits, predictions, replays, selector


def _import_all(store, fits, predictions, selector):
    for fit, prediction in zip(fits, predictions, strict=True):
        store.record_reuse(fit["job_id"], _evidence(fit))
        store.record_reuse(prediction["job_id"], _evidence(prediction))
    store.record_reuse(selector["job_id"], _evidence(selector))


def test_r2_full_profile_derivation():
    additional = sorted("P08JOB-" + f"{index:064x}" for index in range(8550))
    replay = sorted("P08JOB-" + f"{index:064x}" for index in range(8600, 8604))
    selector_id = "P08JOB-" + f"{9999:064x}"
    binding = {
        "plan_sha256": "0" * 64,
        "permit_id": "r2-full",
        "inputs": ["a"],
        "recovery_accounting": {
            "schema_version": "nato-sers-p08-u1-r2-accounting-v1",
            "parent_binding_sha256": "a" * 64,
            "parent_inventory_sha256": "b" * 64,
            "replay_fit_job_ids": replay,
            "additional_reuse_fit_job_ids": additional,
            "reused_epoch_selection_job_ids": [selector_id],
            "baseline_active_seconds": 3700.0,
            "baseline_artifact_bytes": 4285000000,
        },
    }
    accounting = ledger.execution_accounting(binding)
    assert accounting["expected_reuse_fits"] == 8550
    assert accounting["expected_reuse_predictions"] == 8550
    assert accounting["max_reuse_fits"] == 8550
    assert accounting["max_reuse_predictions"] == 8550
    assert accounting["historical_overhead_attempts"] == 14
    assert accounting["max_fit_total"] == 195216
    assert accounting["baseline_active_seconds"] == 3700.0
    assert accounting["baseline_artifact_bytes"] == 4285000000
    assert accounting["replay_fit_job_ids"] == tuple(replay)
    assert accounting["additional_reuse_fit_job_ids"] == tuple(additional)
    assert accounting["selector_job_ids"] == (selector_id,)


def test_r2_malformed_profiles(tmp_path, monkeypatch):
    binding, jobs, _, _, _, _ = _mini_setup(monkeypatch)
    profile = binding["recovery_accounting"]

    def clone():
        return json.loads(json.dumps(profile))

    bad = []
    case = clone()
    case["schema_version"] = "unsupported"
    bad.append(case)
    case = clone()
    case.pop("reused_epoch_selection_job_ids")
    bad.append(case)
    case = clone()
    case["unexpected"] = 1
    bad.append(case)
    case = clone()
    case["replay_fit_job_ids"].reverse()
    bad.append(case)
    case = clone()
    case["replay_fit_job_ids"] = case["replay_fit_job_ids"][:1]
    bad.append(case)
    case = clone()
    case["additional_reuse_fit_job_ids"] = case["additional_reuse_fit_job_ids"][:2]
    bad.append(case)
    case = clone()
    case["additional_reuse_fit_job_ids"] = case["additional_reuse_fit_job_ids"][::-1]
    bad.append(case)
    case = clone()
    case["reused_epoch_selection_job_ids"] = []
    bad.append(case)
    case = clone()
    case["reused_epoch_selection_job_ids"] = [case["additional_reuse_fit_job_ids"][0]]
    bad.append(case)
    case = clone()
    case["parent_binding_sha256"] = "Z" * 64
    bad.append(case)
    case = clone()
    case["parent_inventory_sha256"] = "a" * 63
    bad.append(case)
    case = clone()
    case["baseline_active_seconds"] = True
    bad.append(case)
    case = clone()
    case["baseline_active_seconds"] = 99.0
    bad.append(case)
    case = clone()
    case["baseline_active_seconds"] = float(ledger.MAX_WALL_SECONDS)
    bad.append(case)
    case = clone()
    case["baseline_artifact_bytes"] = True
    bad.append(case)
    case = clone()
    case["baseline_artifact_bytes"] = 999
    bad.append(case)
    case = clone()
    case["baseline_artifact_bytes"] = ledger.MAX_ARTIFACT_BYTES
    bad.append(case)

    for index, bad_profile in enumerate(bad):
        candidate = {"plan_sha256": "0" * 64, "permit_id": f"bad-{index}", "inputs": ["a"]}
        candidate["recovery_accounting"] = bad_profile
        run = tmp_path / f"bad-{index}"
        with pytest.raises(ledger.ValidationError):
            ledger.P08U1Store.create(str(run), candidate, jobs)
        assert not os.path.exists(str(run))


def test_r2_dependency_ordered_import_and_seal(tmp_path, monkeypatch):
    binding, jobs, fits, predictions, _, selector = _mini_setup(monkeypatch)
    store = ledger.P08U1Store.create(str(tmp_path / "ordered"), binding, jobs)
    _import_all(store, fits, predictions, selector)
    assert store.seal_reuse() == {"sealed": True, "fits": 3, "predictions": 3, "selectors": 1}
    assert store.verify_events()["ok"] is True
    status = store.status()
    assert status["reuse"]["fits"] == 3
    assert status["reuse"]["predictions"] == 3
    assert status["reuse"]["selectors"] == 1
    assert status["attempts_total"] == 0
    summary = store.public_summary()
    assert summary["accounted_model_fit_attempts"] == 3 + 14
    assert summary["reuse_epoch_selections"] == 1
    store.close()


def test_r2_selector_requires_dependencies_first(tmp_path, monkeypatch):
    binding, jobs, _, _, _, selector = _mini_setup(monkeypatch)
    store = ledger.P08U1Store.create(str(tmp_path / "deporder"), binding, jobs)
    with pytest.raises(ledger.ValidationError, match="selector_dependency_not_imported"):
        store.record_reuse(selector["job_id"], _evidence(selector))
    store.close()


def test_r2_nonlisted_selector_rejected(tmp_path, monkeypatch):
    binding, jobs, fits, _, _, _ = _mini_setup(monkeypatch)
    extra = _job("select_refit_epochs", deps=[fits[0]["job_id"]], unit="extra-selector")
    store = ledger.P08U1Store.create(str(tmp_path / "nonlisted"), binding, jobs + [extra])
    with pytest.raises(ledger.ValidationError, match="selector_not_permitted"):
        store.record_reuse(extra["job_id"], _evidence(extra))
    store.close()


def test_r2_replay_and_duplicate_rejected(tmp_path, monkeypatch):
    binding, jobs, fits, _, replays, _ = _mini_setup(monkeypatch)
    store = ledger.P08U1Store.create(str(tmp_path / "reject"), binding, jobs)
    with pytest.raises(ledger.ValidationError, match="replay_fit_cannot_be_reused"):
        store.record_reuse(replays[0]["job_id"], _evidence(replays[0]))
    store.record_reuse(fits[0]["job_id"], _evidence(fits[0]))
    with pytest.raises(ledger.ValidationError, match="duplicate_reuse"):
        store.record_reuse(fits[0]["job_id"], _evidence(fits[0]))
    store.close()


def test_r2_selector_identity_graph_validation(tmp_path, monkeypatch):
    binding, jobs, _, _, _, selector = _mini_setup(monkeypatch)

    missing_jobs = [record for record in jobs if record["job_id"] != selector["job_id"]]
    run = tmp_path / "missing-selector"
    with pytest.raises(ledger.ValidationError, match="recovery_accounting_selector_missing"):
        ledger.P08U1Store.create(str(run), binding, missing_jobs)
    assert not os.path.exists(str(run))

    wrongstage = _job("source_fit", unit="wrongstage-selector")
    wrong_profile = json.loads(json.dumps(binding["recovery_accounting"]))
    wrong_profile["reused_epoch_selection_job_ids"] = [wrongstage["job_id"]]
    candidate = {"plan_sha256": "0" * 64, "permit_id": "wrongstage", "inputs": ["a"]}
    candidate["recovery_accounting"] = wrong_profile
    run = tmp_path / "wrongstage"
    with pytest.raises(ledger.ValidationError, match="recovery_accounting_selector_stage_invalid"):
        ledger.P08U1Store.create(str(run), candidate, jobs + [wrongstage])
    assert not os.path.exists(str(run))

    bad_dep = _job("select_refit_epochs", unit="nested-selector")
    scoped = _job("select_refit_epochs", deps=[bad_dep["job_id"]], unit="scoped-selector")
    dep_profile = json.loads(json.dumps(binding["recovery_accounting"]))
    dep_profile["reused_epoch_selection_job_ids"] = [scoped["job_id"]]
    candidate = {"plan_sha256": "0" * 64, "permit_id": "baddep", "inputs": ["a"]}
    candidate["recovery_accounting"] = dep_profile
    run = tmp_path / "baddep"
    with pytest.raises(
        ledger.ValidationError,
        match="recovery_accounting_selector_dependency_stage_invalid",
    ):
        ledger.P08U1Store.create(str(run), candidate, jobs + [bad_dep, scoped])
    assert not os.path.exists(str(run))


def test_r2_incomplete_seal_and_start_before_seal(tmp_path, monkeypatch):
    binding, jobs, fits, predictions, replays, _ = _mini_setup(monkeypatch)
    store = ledger.P08U1Store.create(str(tmp_path / "incomplete"), binding, jobs)
    with pytest.raises(ledger.ValidationError, match="reuse_import_not_sealed"):
        store.start(replays[0]["job_id"], "CPU", _snapshot())
    store.record_reuse(fits[0]["job_id"], _evidence(fits[0]))
    with pytest.raises(ledger.ValidationError, match="reuse_import_incomplete"):
        store.seal_reuse()
    for index in (1, 2):
        store.record_reuse(fits[index]["job_id"], _evidence(fits[index]))
        store.record_reuse(predictions[index]["job_id"], _evidence(predictions[index]))
    store.record_reuse(predictions[0]["job_id"], _evidence(predictions[0]))
    with pytest.raises(ledger.ValidationError, match="recovery_selector_import_incomplete"):
        store.seal_reuse()
    store.close()


def test_r2_resource_carry_and_attempt_ceiling(tmp_path, monkeypatch):
    binding, jobs, fits, predictions, replays, selector = _mini_setup(
        monkeypatch, R2_MAX_FIT_TOTAL=18
    )
    store = ledger.P08U1Store.create(str(tmp_path / "carry"), binding, jobs)
    _import_all(store, fits, predictions, selector)
    store.seal_reuse()
    with pytest.raises(ledger.ValidationError, match="active_seconds_regressed"):
        store.start(replays[0]["job_id"], "CPU", _snapshot(active=199.0))
    with pytest.raises(ledger.ValidationError, match="artifact_bytes_regressed"):
        store.start(replays[0]["job_id"], "CPU", _snapshot(artifact=1999))
    store.start(replays[0]["job_id"], "CPU", _snapshot())
    store.finish(replays[0]["job_id"], "complete", {"sha256": "a" * 64})
    with pytest.raises(ledger.BudgetError, match="fit_attempt_ceiling"):
        store.start(replays[1]["job_id"], "CPU", _snapshot())
    store.close()


def test_r2_failure_no_auto_retry(tmp_path, monkeypatch):
    binding, jobs, fits, predictions, replays, selector = _mini_setup(monkeypatch)
    run = str(tmp_path / "failure")
    store = ledger.P08U1Store.create(run, binding, jobs)
    _import_all(store, fits, predictions, selector)
    store.seal_reuse()
    store.start(replays[0]["job_id"], "CPU", _snapshot())
    store.finish(replays[0]["job_id"], "failed", {"sha256": "b" * 64})
    assert store.close()["clean"] is True

    reopened = ledger.P08U1Store.reopen(run, json.loads(json.dumps(binding)))
    with pytest.raises(ledger.ReviewRequiredError):
        reopened.start(replays[1]["job_id"], "CPU", _snapshot())
    reopened.close()


def test_r2_clean_reopen_event_audit(tmp_path, monkeypatch):
    binding, jobs, fits, predictions, _, selector = _mini_setup(monkeypatch)
    run = str(tmp_path / "reopen")
    store = ledger.P08U1Store.create(run, binding, jobs)
    _import_all(store, fits, predictions, selector)
    store.seal_reuse()
    store.close()

    reopened = ledger.P08U1Store.reopen(run, json.loads(json.dumps(binding)))
    assert reopened.verify_events()["ok"] is True
    reopened.close()

    changed = json.loads(json.dumps(binding))
    changed["recovery_accounting"]["baseline_active_seconds"] = 201.0
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, changed)


def test_r0_r1_shape_unchanged():
    assert ledger.REUSE_STAGES == frozenset({"source_fit", "source_validation_prediction"})
    default = ledger.execution_accounting({"permit_id": "r0"})
    assert "selector_job_ids" not in default
    r1_binding = {
        "permit_id": "r1",
        "recovery_accounting": {
            "schema_version": "nato-sers-p08-u1-r1-accounting-v1",
            "parent_binding_sha256": "a" * 64,
            "parent_inventory_sha256": "b" * 64,
            "replay_fit_job_ids": sorted("P08JOB-" + f"{index:064x}" for index in range(5)),
            "additional_reuse_fit_job_ids": sorted(
                "P08JOB-" + f"{index:064x}" for index in range(10, 16)
            ),
            "baseline_active_seconds": 1600.0,
            "baseline_artifact_bytes": 1427760770,
        },
    }
    r1 = ledger.execution_accounting(r1_binding)
    assert r1["schema_version"] == "nato-sers-p08-u1-r1-accounting-v1"
    assert "selector_job_ids" not in r1
    assert r1["historical_overhead_attempts"] == ledger.RECOVERY_HISTORICAL_OVERHEAD_ATTEMPTS
    assert ledger.MAX_UNIQUE_FIT_JOBS == 195202
    assert ledger.MAX_SCALAR_ATTEMPTS == 3354
    assert ledger.MAX_RAM_BYTES == 36 * 2**30
    assert ledger.MAX_GPU_BYTES == 8 * 2**30
    assert ledger.MAX_ARTIFACT_BYTES == 80 * 2**30
    assert ledger.MAX_CPU_WORKERS == 8
    assert ledger.MAX_GPU_WORKERS == 1
    assert ledger.FREE_SPACE_FLOOR_BYTES == 30 * 2**30
