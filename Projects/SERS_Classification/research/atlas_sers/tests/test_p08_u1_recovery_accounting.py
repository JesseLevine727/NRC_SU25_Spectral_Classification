"""T298 synthetic tests for the optional R1 recovery accounting profile."""

import itertools
import json
import os

import pytest

from atlas_sers.evaluation import p08_u1_store as ledger


def _binding(tag="recovery"):
    return {"plan_sha256": "0" * 64, "permit_id": tag, "inputs": ["a", "b"]}


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


def _pair(index):
    fit = _job("source_fit", unit=f"fit-{index}")
    prediction = _job("source_validation_prediction", deps=[fit["job_id"]], unit=f"pred-{index}")
    return fit, prediction


def _snapshot(
    active=2100.0,
    artifact=2_500_000_000,
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


def _evidence(record):
    return {"job_id": record["job_id"], "job_sha256": ledger.job_sha256(record)}


def _recovery_setup(with_decoys=False):
    pairs = [_pair(index) for index in range(84)]
    replays = [_job("source_fit", unit=f"replay-{index}") for index in range(5)]
    jobs = [record for pair in pairs for record in pair] + list(replays)
    additional = [pairs[78 + index][0] for index in range(6)]
    if with_decoys:
        decoys = [_job("source_fit", unit=f"decoy-{index}") for index in range(2)]
        jobs.extend(decoys)
        additional = [pairs[index][0] for index in range(4)] + decoys
    profile = {
        "schema_version": "nato-sers-p08-u1-r1-accounting-v1",
        "parent_binding_sha256": "a" * 64,
        "parent_inventory_sha256": "b" * 64,
        "replay_fit_job_ids": sorted(record["job_id"] for record in replays),
        "additional_reuse_fit_job_ids": sorted(record["job_id"] for record in additional),
        "baseline_active_seconds": 2000.0,
        "baseline_artifact_bytes": 2_000_000_000,
    }
    binding = _binding("r1")
    binding["recovery_accounting"] = profile
    return binding, jobs, pairs, replays, additional


def _record_all_pairs(store, pairs):
    for fit, prediction in pairs:
        store.record_reuse(fit["job_id"], _evidence(fit))
        store.record_reuse(prediction["job_id"], _evidence(prediction))


def test_default_accounting_matches_constants():
    accounting = ledger.execution_accounting(_binding("default"))
    assert accounting["expected_reuse_fits"] == ledger.EXPECTED_REUSE_FITS
    assert accounting["expected_reuse_predictions"] == ledger.EXPECTED_REUSE_PREDICTIONS
    assert accounting["max_reuse_fits"] == ledger.MAX_REUSE_FITS
    assert accounting["max_reuse_predictions"] == ledger.MAX_REUSE_PREDICTIONS
    assert accounting["historical_overhead_attempts"] == ledger.HISTORICAL_OVERHEAD_ATTEMPTS
    assert accounting["max_fit_total"] == ledger.MAX_FIT_TOTAL
    assert accounting["baseline_active_seconds"] == ledger.BASELINE_ACTIVE_SECONDS
    assert accounting["baseline_artifact_bytes"] == ledger.BASELINE_ARTIFACT_BYTES
    assert accounting["replay_fit_job_ids"] == ()
    assert accounting["additional_reuse_fit_job_ids"] == ()


def test_default_accounting_honours_monkeypatched_constants(monkeypatch):
    monkeypatch.setattr(ledger, "EXPECTED_REUSE_FITS", 3)
    monkeypatch.setattr(ledger, "MAX_REUSE_FITS", 4)
    accounting = ledger.execution_accounting(_binding("patched"))
    assert accounting["expected_reuse_fits"] == 3
    assert accounting["max_reuse_fits"] == 4


def test_r1_accounting_profile_values():
    binding, _, _, replays, additional = _recovery_setup()
    accounting = ledger.execution_accounting(binding)
    assert accounting["expected_reuse_fits"] == 84
    assert accounting["expected_reuse_predictions"] == 84
    assert accounting["max_reuse_fits"] == 84
    assert accounting["max_reuse_predictions"] == 84
    assert accounting["historical_overhead_attempts"] == 10
    assert accounting["max_fit_total"] == 195212
    assert accounting["baseline_active_seconds"] == 2000.0
    assert accounting["baseline_artifact_bytes"] == 2_000_000_000
    assert accounting["replay_fit_job_ids"] == tuple(sorted(record["job_id"] for record in replays))
    assert accounting["additional_reuse_fit_job_ids"] == tuple(
        sorted(record["job_id"] for record in additional)
    )


def test_strict_profile_validation_rejects_before_creation(tmp_path):
    binding, jobs, _, _, _ = _recovery_setup()
    profile = binding["recovery_accounting"]

    def clone():
        return json.loads(json.dumps(profile))

    bad_profiles = []
    case = clone()
    case["schema_version"] = "unsupported"
    bad_profiles.append(case)
    case = clone()
    case.pop("schema_version")
    bad_profiles.append(case)
    case = clone()
    case["unexpected_key"] = 1
    bad_profiles.append(case)
    case = clone()
    case["replay_fit_job_ids"].reverse()
    bad_profiles.append(case)
    case = clone()
    case["replay_fit_job_ids"] = tuple(case["replay_fit_job_ids"])
    bad_profiles.append(case)
    case = clone()
    case["parent_binding_sha256"] = "Z" * 64
    bad_profiles.append(case)
    case = clone()
    case["parent_inventory_sha256"] = "a" * 63
    bad_profiles.append(case)
    case = clone()
    case["replay_fit_job_ids"] = case["replay_fit_job_ids"][:4]
    bad_profiles.append(case)
    case = clone()
    case["replay_fit_job_ids"] = [case["replay_fit_job_ids"][0]] * 5
    bad_profiles.append(case)
    case = clone()
    case["additional_reuse_fit_job_ids"] = case["additional_reuse_fit_job_ids"][:5] + [
        case["replay_fit_job_ids"][0]
    ]
    bad_profiles.append(case)
    case = clone()
    case["baseline_active_seconds"] = True
    bad_profiles.append(case)
    case = clone()
    case["baseline_active_seconds"] = 1515.0
    bad_profiles.append(case)
    case = clone()
    case["baseline_active_seconds"] = float(ledger.MAX_WALL_SECONDS)
    bad_profiles.append(case)
    case = clone()
    case["baseline_active_seconds"] = float("inf")
    bad_profiles.append(case)
    case = clone()
    case["baseline_artifact_bytes"] = True
    bad_profiles.append(case)
    case = clone()
    case["baseline_artifact_bytes"] = 1
    bad_profiles.append(case)
    case = clone()
    case["baseline_artifact_bytes"] = ledger.MAX_ARTIFACT_BYTES
    bad_profiles.append(case)

    for index, bad in enumerate(bad_profiles):
        candidate = _binding(f"bad-{index}")
        candidate["recovery_accounting"] = bad
        run = tmp_path / f"bad-{index}"
        with pytest.raises(ledger.ValidationError):
            ledger.P08U1Store.create(str(run), candidate, jobs)
        assert not os.path.exists(str(run))


def test_listed_jobs_must_be_registered_source_fit(tmp_path):
    binding, jobs, pairs, _, _ = _recovery_setup()

    profile = json.loads(json.dumps(binding["recovery_accounting"]))
    profile["replay_fit_job_ids"] = profile["replay_fit_job_ids"][:4] + ["P08JOB-" + "0" * 64]
    profile["replay_fit_job_ids"].sort()
    candidate = _binding("missing")
    candidate["recovery_accounting"] = profile
    run = tmp_path / "missing"
    with pytest.raises(ledger.ValidationError, match="recovery_accounting_job_missing"):
        ledger.P08U1Store.create(str(run), candidate, jobs)
    assert not os.path.exists(str(run))

    profile = json.loads(json.dumps(binding["recovery_accounting"]))
    profile["additional_reuse_fit_job_ids"] = profile["additional_reuse_fit_job_ids"][:5] + [
        pairs[0][1]["job_id"]
    ]
    profile["additional_reuse_fit_job_ids"].sort()
    candidate = _binding("wrongstage")
    candidate["recovery_accounting"] = profile
    run = tmp_path / "wrongstage"
    with pytest.raises(ledger.ValidationError, match="recovery_accounting_job_stage_invalid"):
        ledger.P08U1Store.create(str(run), candidate, jobs)
    assert not os.path.exists(str(run))


def test_r1_import_seal_and_replay_refusal(tmp_path):
    binding, jobs, pairs, replays, _ = _recovery_setup()
    store = ledger.P08U1Store.create(str(tmp_path / "r1"), binding, jobs)

    for replay in replays:
        with pytest.raises(ledger.ValidationError):
            store.record_reuse(replay["job_id"], _evidence(replay))

    _record_all_pairs(store, pairs)
    assert store.seal_reuse() == {"sealed": True, "fits": 84, "predictions": 84}
    assert store.verify_events()["ok"] is True
    summary = store.public_summary()
    assert summary["reuse_fits"] == 84
    assert summary["reuse_predictions"] == 84
    assert summary["accounted_model_fit_attempts"] == 94
    store.close()


def test_seal_requires_all_additional_fits(tmp_path):
    binding, jobs, pairs, _, _ = _recovery_setup(with_decoys=True)
    store = ledger.P08U1Store.create(str(tmp_path / "decoys"), binding, jobs)
    _record_all_pairs(store, pairs)
    with pytest.raises(ledger.ValidationError):
        store.seal_reuse()
    store.close()


def test_baseline_nonregression_and_accounting_ceiling(tmp_path):
    binding, jobs, pairs, replays, _ = _recovery_setup()
    store = ledger.P08U1Store.create(str(tmp_path / "baseline"), binding, jobs)
    assert store.status()["accounted_model_fit_attempts"] == 10

    _record_all_pairs(store, pairs)
    store.seal_reuse()
    assert store.public_summary()["accounted_model_fit_attempts"] == 94

    with pytest.raises(ledger.ValidationError):
        store.start(replays[0]["job_id"], "CPU", _snapshot(active=1900.0, artifact=2_500_000_000))
    with pytest.raises(ledger.ValidationError):
        store.start(replays[0]["job_id"], "CPU", _snapshot(active=2100.0, artifact=1_000_000_000))

    store.start(replays[0]["job_id"], "CPU", _snapshot())
    assert store.status()["accounted_model_fit_attempts"] == 95
    store.finish(replays[0]["job_id"], "complete", {"sha256": "a" * 64})
    store.close()


def test_fixed_caps_unchanged_and_breach_still_budget(tmp_path):
    assert ledger.MAX_RAM_BYTES == 24 * 2**30
    assert ledger.MAX_GPU_BYTES == 8 * 2**30
    assert ledger.MAX_WALL_SECONDS == 172800
    assert ledger.MAX_ARTIFACT_BYTES == 80 * 2**30
    assert ledger.MAX_SCALAR_ATTEMPTS == 3354
    assert ledger.MAX_UNIQUE_FIT_JOBS == 195202
    assert ledger.FREE_SPACE_FLOOR_BYTES == 30 * 2**30

    binding, jobs, pairs, replays, _ = _recovery_setup()
    store = ledger.P08U1Store.create(str(tmp_path / "caps"), binding, jobs)
    _record_all_pairs(store, pairs)
    store.seal_reuse()
    budgets = store.public_summary()["budgets"]
    assert budgets["max_fit_total"] == 195212
    assert budgets["historical_overhead_attempts"] == 10
    assert budgets["max_unique_fit_jobs"] == 195202
    assert budgets["max_scalar_attempts"] == 3354
    assert budgets["max_ram_bytes"] == 24 * 2**30
    assert budgets["max_gpu_bytes"] == 8 * 2**30
    assert budgets["max_wall_seconds"] == 172800
    assert budgets["max_artifact_bytes"] == 80 * 2**30
    with pytest.raises(ledger.BudgetError):
        store.start(replays[0]["job_id"], "CPU", _snapshot(rss=ledger.MAX_RAM_BYTES + 1))
    store.close()


def test_reopen_requires_identical_profile(tmp_path):
    binding, jobs, pairs, _, _ = _recovery_setup()
    run = str(tmp_path / "immutable")
    store = ledger.P08U1Store.create(run, binding, jobs)
    _record_all_pairs(store, pairs)
    store.seal_reuse()
    store.close()

    reopened = ledger.P08U1Store.reopen(run, json.loads(json.dumps(binding)))
    reopened.close()

    changed = json.loads(json.dumps(binding))
    changed["recovery_accounting"]["baseline_active_seconds"] = 2001.0
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, changed)

    reordered = json.loads(json.dumps(binding))
    reordered["recovery_accounting"]["replay_fit_job_ids"] = list(
        reversed(reordered["recovery_accounting"]["replay_fit_job_ids"])
    )
    with pytest.raises(ledger.ValidationError):
        ledger.P08U1Store.reopen(run, reordered)


def test_failed_state_refused_in_r1(tmp_path):
    binding, jobs, pairs, replays, _ = _recovery_setup()
    run = str(tmp_path / "failed")
    store = ledger.P08U1Store.create(run, binding, jobs)
    _record_all_pairs(store, pairs)
    store.seal_reuse()
    store.start(replays[0]["job_id"], "CPU", _snapshot())
    store.finish(replays[0]["job_id"], "failed", {"sha256": "b" * 64})
    assert store.close()["clean"] is True

    reopened = ledger.P08U1Store.reopen(run, json.loads(json.dumps(binding)))
    with pytest.raises(ledger.ReviewRequiredError):
        reopened.start(replays[1]["job_id"], "CPU", _snapshot())
    reopened.close()


def test_default_summary_overhead(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "EXPECTED_REUSE_FITS", 0)
    monkeypatch.setattr(ledger, "EXPECTED_REUSE_PREDICTIONS", 0)
    monkeypatch.setattr(ledger, "MAX_REUSE_FITS", 0)
    monkeypatch.setattr(ledger, "MAX_REUSE_PREDICTIONS", 0)
    job = _job("held_prediction", unit="default-summary")
    store = ledger.P08U1Store.create(str(tmp_path / "defaultsum"), _binding("defaultsum"), [job])
    store.seal_reuse()
    summary = store.public_summary()
    assert summary["accounted_model_fit_attempts"] == 5
    assert summary["budgets"]["max_fit_total"] == 195207
    assert summary["budgets"]["historical_overhead_attempts"] == 5
    store.close()


def test_r1_attempt_ceiling_includes_historical_overhead(tmp_path, monkeypatch):
    # Shrink only the test ceiling: 84 reused + 10 overhead allows one new fit.
    monkeypatch.setattr(ledger, "RECOVERY_MAX_FIT_TOTAL", 95)
    binding, jobs, pairs, replays, _ = _recovery_setup()
    store = ledger.P08U1Store.create(tmp_path / "attempt-cap", binding, jobs)
    _record_all_pairs(store, pairs)
    store.seal_reuse()
    store.start(replays[0]["job_id"], "CPU", _snapshot())
    store.finish(replays[0]["job_id"], "complete", {"sha256": "a" * 64})
    with pytest.raises(ledger.BudgetError, match="fit_attempt_ceiling"):
        store.start(replays[1]["job_id"], "CPU", _snapshot())
    assert store.public_summary()["accounted_model_fit_attempts"] == 95
    assert store.public_summary()["attempts_total"] == 1
    store.close()
