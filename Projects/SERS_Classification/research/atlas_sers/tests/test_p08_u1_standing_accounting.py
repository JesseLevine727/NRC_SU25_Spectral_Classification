"""T316 synthetic tests for the standing recovery accounting profile."""

import itertools
import json

import pytest

from atlas_sers.evaluation import p08_u1_standing_accounting as standing
from atlas_sers.evaluation import p08_u1_store as ledger


def _apply_small(monkeypatch, **overrides):
    values = {
        "STANDING_EXPECTED_PILOT_FITS": 1,
        "STANDING_MIN_ACTIVE_SECONDS": 100.0,
        "STANDING_MIN_ARTIFACT_BYTES": 1000,
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setattr(standing, name, value)


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


def _snapshot(active=210.0, artifact=2500, rss=1 * 2**30, gpu=0, free=200 * 2**30, workers=0):
    return {
        "active_seconds": active,
        "artifact_bytes": artifact,
        "rss_bytes": rss,
        "gpu_bytes": gpu,
        "free_disk_bytes": free,
        "active_workers": workers,
    }


def _completed():
    return {stage: [] for stage in standing.KNOWN_STAGES}


def _history():
    return {stage: {} for stage in standing.KNOWN_STAGES}


def _profile(completed, replays=(), history=None, generation=1, active=200.0, artifact=2000):
    return {
        "schema_version": standing.STANDING_ACCOUNTING_SCHEMA,
        "authority_sha256": "a" * 64,
        "parent_audit_sha256": "b" * 64,
        "parent_binding_sha256": "c" * 64,
        "parent_inventory_sha256": "d" * 64,
        "recovery_generation": generation,
        "replay_job_ids": sorted(record["job_id"] for record in replays),
        "replay_counts_by_stage": history if history is not None else _history(),
        "completed_job_ids_by_stage": completed,
        "baseline_active_seconds": active,
        "baseline_artifact_bytes": artifact,
    }


def _binding(profile, permit="standing"):
    return {
        "plan_sha256": "0" * 64,
        "permit_id": permit,
        "inputs": ["a"],
        "recovery_accounting": profile,
    }


def _four_kind_setup(monkeypatch, **overrides):
    _apply_small(monkeypatch, **overrides)
    fit = _job("source_fit", unit="fit")
    prediction = _job("source_validation_prediction", deps=[fit["job_id"]], unit="pred")
    selector = _job("select_refit_epochs", deps=[prediction["job_id"]], unit="selector")
    scalar = _job("scalar_calibration", deps=[prediction["job_id"]], unit="scalar")
    replay = _job("source_fit", unit="replay")
    replay2 = _job("source_fit", unit="replay2")
    completed = _completed()
    completed["source_fit"] = [fit["job_id"]]
    completed["source_validation_prediction"] = [prediction["job_id"]]
    completed["select_refit_epochs"] = [selector["job_id"]]
    completed["scalar_calibration"] = [scalar["job_id"]]
    history = _history()
    history["source_fit"] = {key: 1 for key in sorted((replay["job_id"], replay2["job_id"]))}
    binding = _binding(_profile(completed, [replay, replay2], history))
    jobs = [fit, prediction, selector, scalar, replay, replay2]
    return binding, jobs, fit, prediction, selector, scalar, replay, replay2


def _import_four(store, fit, prediction, selector, scalar):
    store.record_reuse(fit["job_id"], _evidence(fit))
    store.record_reuse(prediction["job_id"], _evidence(prediction))
    store.record_reuse(selector["job_id"], _evidence(selector))
    store.record_reuse(scalar["job_id"], _evidence(scalar))


def test_standing_derivation_from_completed_operations(monkeypatch):
    binding, _, fit, prediction, selector, scalar, replay, replay2 = _four_kind_setup(monkeypatch)
    accounting = ledger.execution_accounting(binding)
    assert accounting["standing"] is True
    assert accounting["expected_reuse_fits"] == 1
    assert accounting["expected_reuse_predictions"] == 1
    assert accounting["additional_reuse_fit_job_ids"] == (fit["job_id"],)
    assert accounting["selector_job_ids"] == (selector["job_id"],)
    assert accounting["scalar_job_ids"] == (scalar["job_id"],)
    assert accounting["operation_job_ids"] == ()
    assert accounting["all_replay_job_ids"] == tuple(sorted([replay["job_id"], replay2["job_id"]]))
    assert accounting["historical_overhead_attempts"] == 14 + 2
    assert accounting["max_fit_total"] == standing.STANDING_MAX_FIT_TOTAL
    assert accounting["baseline_active_seconds"] == 200.0
    assert accounting["baseline_artifact_bytes"] == 2000


def test_standing_import_seal_reopen_and_replay(tmp_path, monkeypatch):
    binding, jobs, fit, prediction, selector, scalar, replay, _ = _four_kind_setup(monkeypatch)
    run = str(tmp_path / "standing")
    store = ledger.P08U1Store.create(run, binding, jobs)
    with pytest.raises(ledger.ValidationError, match="reuse_import_not_sealed"):
        store.start(replay["job_id"], "CPU", _snapshot())
    _import_four(store, fit, prediction, selector, scalar)
    assert store.seal_reuse()["sealed"] is True
    assert store.verify_events()["ok"] is True
    store.start(replay["job_id"], "CPU", _snapshot())
    store.finish(replay["job_id"], "complete", {"sha256": "a" * 64})
    assert store.verify_events()["ok"] is True
    assert store.close()["clean"] is True

    reopened = ledger.P08U1Store.reopen(run, json.loads(json.dumps(binding)))
    assert reopened.verify_events()["ok"] is True
    reopened.close()


def test_standing_exact_identity_and_dependency_refusals(tmp_path, monkeypatch):
    binding, jobs, fit, prediction, selector, _, replay, _ = _four_kind_setup(monkeypatch)
    extra = _job("source_fit", unit="extra")
    store = ledger.P08U1Store.create(str(tmp_path / "refuse"), binding, jobs + [extra])
    with pytest.raises(ledger.ValidationError, match="reuse_dependency_not_imported"):
        store.record_reuse(selector["job_id"], _evidence(selector))
    with pytest.raises(ledger.ValidationError, match="replay_fit_cannot_be_reused"):
        store.record_reuse(replay["job_id"], _evidence(replay))
    with pytest.raises(ledger.ValidationError, match="reuse_job_not_declared"):
        store.record_reuse(extra["job_id"], _evidence(extra))
    with pytest.raises(ledger.ValidationError, match="reuse_evidence_job_mismatch"):
        store.record_reuse(fit["job_id"], _evidence(prediction))
    store.record_reuse(fit["job_id"], _evidence(fit))
    with pytest.raises(ledger.ValidationError, match="duplicate_reuse"):
        store.record_reuse(fit["job_id"], _evidence(fit))
    store.close()


def test_standing_bad_profiles_rejected(tmp_path, monkeypatch):
    binding, jobs, _, _, _, _, _, _ = _four_kind_setup(monkeypatch)
    good = binding["recovery_accounting"]
    ghost = _job("source_fit", unit="ghost")
    ghost2 = _job("source_fit", unit="ghost2")
    fit_id = good["completed_job_ids_by_stage"]["source_fit"][0]
    replay_id = good["replay_job_ids"][0]

    def clone():
        return json.loads(json.dumps(good))

    bad = []
    case = clone()
    case["recovery_generation"] = 0
    bad.append(case)
    case = clone()
    case["recovery_generation"] = 9
    bad.append(case)
    case = clone()
    case.pop("completed_job_ids_by_stage")
    bad.append(case)
    case = clone()
    case["unexpected"] = 1
    bad.append(case)
    case = clone()
    case["authority_sha256"] = "Z" * 64
    bad.append(case)
    case = clone()
    case["completed_job_ids_by_stage"]["source_fit"] = []
    bad.append(case)
    case = clone()
    case["completed_job_ids_by_stage"]["source_fit"] = sorted(
        [ghost["job_id"], ghost2["job_id"]], reverse=True
    )
    bad.append(case)
    case = clone()
    case["completed_job_ids_by_stage"]["source_fit"] = [ghost["job_id"]]
    case["completed_job_ids_by_stage"]["calibration_model_fit"] = [ghost["job_id"]]
    bad.append(case)
    case = clone()
    case["replay_counts_by_stage"]["source_fit"][replay_id] = 3
    bad.append(case)
    case = clone()
    case["replay_job_ids"] = ["nope"]
    bad.append(case)
    case = clone()
    case["replay_job_ids"] = [ghost["job_id"]]
    bad.append(case)
    case = clone()
    case["replay_job_ids"] = [fit_id]
    case["replay_counts_by_stage"]["source_fit"] = {fit_id: 1}
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

    for index, profile in enumerate(bad):
        candidate = _binding(profile, permit=f"bad-{index}")
        run = tmp_path / f"bad-{index}"
        with pytest.raises(ledger.ValidationError):
            ledger.P08U1Store.create(str(run), candidate, jobs)
        assert not run.exists()


def test_standing_replay_caps_enforced(tmp_path, monkeypatch):
    binding, jobs, _, _, _, _, _, _ = _four_kind_setup(monkeypatch)
    monkeypatch.setattr(standing, "STANDING_FIT_SCALAR_REPLAY_CAP", 1)
    run = tmp_path / "fit-cap"
    with pytest.raises(ledger.ValidationError, match="standing_replay_cap_exceeded"):
        ledger.P08U1Store.create(str(run), json.loads(json.dumps(binding)), jobs)
    assert not run.exists()
    monkeypatch.setattr(standing, "STANDING_FIT_SCALAR_REPLAY_CAP", 32)
    monkeypatch.setattr(standing, "STANDING_TOTAL_REPLAY_CAP", 1)
    run = tmp_path / "total-cap"
    with pytest.raises(ledger.ValidationError, match="standing_total_replay_cap_exceeded"):
        ledger.P08U1Store.create(str(run), json.loads(json.dumps(binding)), jobs)
    assert not run.exists()


def test_standing_failure_blocks_retry(tmp_path, monkeypatch):
    binding, jobs, fit, prediction, selector, scalar, replay, replay2 = _four_kind_setup(
        monkeypatch
    )
    run = str(tmp_path / "standing-failure")
    store = ledger.P08U1Store.create(run, binding, jobs)
    _import_four(store, fit, prediction, selector, scalar)
    store.seal_reuse()
    store.start(replay["job_id"], "CPU", _snapshot())
    store.finish(replay["job_id"], "failed", {"sha256": "b" * 64})
    assert store.close()["clean"] is True

    reopened = ledger.P08U1Store.reopen(run, json.loads(json.dumps(binding)))
    with pytest.raises(ledger.ReviewRequiredError):
        reopened.start(replay2["job_id"], "CPU", _snapshot())
    reopened.close()


def test_standing_all_stage_kinds_import(tmp_path, monkeypatch):
    _apply_small(monkeypatch)
    sf = _job("source_fit", unit="sf")
    svp = _job("source_validation_prediction", deps=[sf["job_id"]], unit="svp")
    shp = _job("select_hyperparameters", deps=[svp["job_id"]], unit="shp")
    cmf = _job("calibration_model_fit", deps=[shp["job_id"]], unit="cmf")
    cvp = _job("calibration_validation_prediction", deps=[cmf["job_id"]], unit="cvp")
    alias = _job("calibration_prediction_alias", deps=[cvp["job_id"]], unit="alias")
    scal = _job("scalar_calibration", deps=[cvp["job_id"]], unit="scal")
    final = _job("final_refit", deps=[scal["job_id"]], unit="final")
    held = _job("held_prediction", deps=[final["job_id"]], unit="held")
    ensemble = _job("seed_ensemble_prediction", deps=[held["job_id"]], unit="ensemble")
    sre = _job("select_refit_epochs", deps=[held["job_id"]], unit="sre")
    chain = [sf, svp, shp, cmf, cvp, alias, scal, final, held, ensemble, sre]
    completed = _completed()
    for record in chain:
        completed[record["stage"]].append(record["job_id"])
    binding = _binding(_profile(completed), permit="standing-chain")
    store = ledger.P08U1Store.create(str(tmp_path / "chain"), binding, chain)
    for record in chain:
        store.record_reuse(record["job_id"], _evidence(record))
    assert store.seal_reuse()["sealed"] is True
    accounting = ledger.execution_accounting(binding)
    assert accounting["expected_reuse_fits"] == 3
    assert accounting["expected_reuse_predictions"] == 3
    assert accounting["selector_job_ids"] == tuple(sorted([shp["job_id"], sre["job_id"]]))
    assert accounting["operation_job_ids"] == tuple(sorted([alias["job_id"], ensemble["job_id"]]))
    assert accounting["scalar_job_ids"] == (scal["job_id"],)
    status = store.status()
    assert status["reuse"]["operations"] == 2
    assert status["reuse"]["calibrations"] == 1
    summary = store.public_summary()
    assert summary["reuse_operations"] == 2
    assert summary["reuse_scalar_calibrations"] == 1
    assert store.verify_events()["ok"] is True
    store.close()


def test_standing_status_public_summary_and_idempotent_seal(tmp_path, monkeypatch):
    binding, jobs, fit, prediction, selector, scalar, _, _ = _four_kind_setup(monkeypatch)
    run = str(tmp_path / "summary")
    store = ledger.P08U1Store.create(run, binding, jobs)
    _import_four(store, fit, prediction, selector, scalar)
    sealed = store.seal_reuse()
    assert sealed["sealed"] is True
    assert (sealed["fits"], sealed["predictions"], sealed["selectors"]) == (1, 1, 1)
    assert sealed["calibrations"] == 1
    assert sealed["operations"] == 0
    assert store.seal_reuse() == sealed

    status = store.status()
    reuse = status["reuse"]
    assert set(reuse["job_ids"]) == {
        fit["job_id"],
        prediction["job_id"],
        selector["job_id"],
        scalar["job_id"],
    }
    assert (reuse["fits"], reuse["predictions"], reuse["selectors"]) == (1, 1, 1)
    assert (reuse["calibrations"], reuse["operations"]) == (1, 0)
    assert status["scalar_attempt_count"] == 0
    assert status["accounted_scalar_attempts"] == (
        reuse["calibrations"] + status["scalar_attempt_count"] + status["scalar_overhead_attempts"]
    )

    summary = store.public_summary()
    assert summary["jobs_total"] == len(jobs)
    assert summary["reuse_scalar_calibrations"] == 1
    assert summary["reuse_operations"] == 0
    assert summary["new_scalar_attempts"] == 0
    assert summary["accounted_scalar_attempts"] == (
        summary["reuse_scalar_calibrations"]
        + summary["new_scalar_attempts"]
        + summary["scalar_overhead_attempts"]
    )
    assert summary["budgets"]["max_scalar_total_attempts"] == (
        ledger.MAX_SCALAR_ATTEMPTS + summary["scalar_overhead_attempts"]
    )
    assert store.close()["clean"] is True

    reopened = ledger.P08U1Store.reopen(run, json.loads(json.dumps(binding)))
    assert reopened.verify_events()["ok"] is True
    assert reopened.status()["reuse"]["calibrations"] == 1
    reopened.close()


def test_standing_scalar_cap_and_completed_reuse_refusal(tmp_path, monkeypatch):
    binding, jobs, fit, prediction, selector, scalar, _, _ = _four_kind_setup(monkeypatch)
    fresh = _job("scalar_calibration", deps=[prediction["job_id"]], unit="fresh-scalar")
    run = str(tmp_path / "scalar-cap")
    store = ledger.P08U1Store.create(run, binding, jobs + [fresh])
    _import_four(store, fit, prediction, selector, scalar)
    store.seal_reuse()
    monkeypatch.setattr(ledger, "MAX_SCALAR_ATTEMPTS", 1)
    with pytest.raises(ledger.ValidationError, match="job_already_reused"):
        store.start(fit["job_id"], "CPU", _snapshot())
    with pytest.raises(ledger.BudgetError, match="scalar_attempt_ceiling"):
        store.start(fresh["job_id"], "CPU", _snapshot())
    store.close()


def test_standing_seal_requires_complete_import(tmp_path, monkeypatch):
    binding, jobs, fit, prediction, selector, scalar, _, _ = _four_kind_setup(monkeypatch)
    store = ledger.P08U1Store.create(str(tmp_path / "incomplete"), binding, jobs)
    store.record_reuse(fit["job_id"], _evidence(fit))
    with pytest.raises(ledger.ValidationError, match="reuse_import_incomplete"):
        store.seal_reuse()
    store.close()
