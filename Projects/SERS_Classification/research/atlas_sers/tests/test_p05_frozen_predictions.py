"""CPU-only read-only authentication tests for P05 frozen predictions.

The suite builds a synthetic but real-filesystem evaluation stage with genuine
prediction persistence, genuine manifests and the genuine calibration loader,
then stubs only the prerequisite refit authority.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
# ruff: noqa: E402

from atlas_sers.evaluation import classical
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as authority
from atlas_sers.evaluation import p05_frozen_predictions as frozen
from atlas_sers.evaluation import p05_prediction as prediction
from atlas_sers.evaluation import p05_prediction_io as prediction_io
from atlas_sers.evaluation import p05_refit_plan as refit_plan
from atlas_sers.evaluation import p05_selection as selection

RECIPES = ("D0-M", "D3")
CLASSES = ("A", "B", "C")
FITTING_UIDS = ("fA", "fB", "fC")
TEST_UIDS = ("001", "NA")
CONTEXT_ID = "ctx-1"
PERMIT_SHA256 = inputs.COMPREHENSIVE_PERMIT_SHA256
EPOCHS = 30
CALIBRATION_SLOTS = ("calib-1",)
ALIAS_COUNT = 9
SOURCE_OPTIMIZER_STEPS = 96
REFIT_OPTIMIZER_STEPS = 6 * EPOCHS * 4
REFIT_PRIOR_SECONDS = 4000.0
RECEIPT_ELAPSED = 3.0
SUMMARY_ELAPSED = 2.0
UNIT_ELAPSED = 0.25
UNIT_PEAK = 1024
DEADLINE = 1e12


def _h(value: str) -> str:
    return value * 64


def _write(path: Path, payload: object) -> None:
    core._atomic_write(Path(path), core._canon().canonical_json_bytes(payload))


def _patch(path: Path, field: str, value: object) -> None:
    payload = core._read_json(Path(path), "test")
    if "." in field:
        head, tail = field.split(".", 1)
        payload[head][tail] = value
    else:
        payload[field] = value
    core._atomic_write(Path(path), json.dumps(payload).encode("utf-8"))


def _build_plan() -> dict:
    source_hash = core._canon().sha256_value(list(FITTING_UIDS))
    specs = []
    for seed in selection.SEEDS[:3]:
        for recipe in RECIPES:
            identity = {
                "context_id": CONTEXT_ID,
                "fitting_role_id": "outer_fit",
                "source_uid_set_sha256": source_hash,
                "recipe_id": recipe,
                "seed": seed,
                "epochs": EPOCHS,
                "calibration_slot_ids": list(CALIBRATION_SLOTS),
                "permit_sha256": PERMIT_SHA256,
            }
            specs.append(
                {
                    **identity,
                    "refit_id": refit_plan._sha256_canonical(identity),
                    "fitting_uids": list(FITTING_UIDS),
                    "classes": list(CLASSES),
                }
            )
    unique = {spec["refit_id"]: spec for spec in specs}
    by_recipe_seed = {(spec["recipe_id"], spec["seed"]): spec["refit_id"] for spec in specs}
    aliases = [
        {
            "context_id": CONTEXT_ID,
            "strategy": strategy,
            "seed": seed,
            "refit_id": by_recipe_seed[("D3" if strategy == "D3" else "D0-M", seed)],
        }
        for seed in selection.SEEDS[:3]
        for strategy in ("D0-M", "P05-SELECTED", "D3")
    ]
    endpoint = {"context_id": CONTEXT_ID, "test_uids": list(TEST_UIDS)}
    return {
        "plan_id": "plan-1",
        "endpoints": [endpoint],
        "unique_refits": unique,
        "strategy_aliases": aliases,
    }


def _calibration() -> classical.TemperatureCalibration:
    return classical.TemperatureCalibration(
        temperature=2.0,
        class_vocabulary=CLASSES,
        observations=len(FITTING_UIDS),
        masters=len(FITTING_UIDS),
        fit_observation_uid_sha256=core._canon().sha256_value(sorted(FITTING_UIDS)),
        fit_master_uid_sha256=core._canon().sha256_value(sorted(FITTING_UIDS)),
        optimizer_success=True,
        optimizer_objective=0.5,
    )


def _frame(calibration: classical.TemperatureCalibration) -> pd.DataFrame:
    logits = np.asarray([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]], dtype=np.float64)
    probabilities = classical.apply_temperature(logits, calibration)
    data = {prediction_io.UID_COLUMN: list(TEST_UIDS)}
    for index in range(len(CLASSES)):
        data[f"logit_{index}"] = logits[:, index]
        data[f"probability_{index}"] = probabilities[:, index]
    return pd.DataFrame(data)


def _refit_digest(refit_id: str) -> str:
    return core._canon().sha256_value(["refit", refit_id])


def _audit(spec, calibration, digest) -> dict:
    return {
        "refit_id": spec["refit_id"],
        "classes": list(spec["classes"]),
        "model_state_sha256": digest,
        "calibration_state_sha256": calibration.state_sha256,
        "test_uid_set_sha256": core._canon().sha256_value(sorted(TEST_UIDS)),
        "rows": len(TEST_UIDS),
        "elapsed_seconds": UNIT_ELAPSED,
        "peak_cuda_bytes": UNIT_PEAK,
        "optimizer_steps": 0,
    }


def _write_calibration(refit_dir: Path, spec, calibration) -> None:
    state = dataclasses.asdict(calibration)
    state["class_vocabulary"] = list(state["class_vocabulary"])
    _write(
        refit_dir / "calibration.json",
        {"state_sha256": calibration.state_sha256, "state": state},
    )
    _write(
        refit_dir / "calibration_audit.json",
        {
            "calibration_state_sha256": calibration.state_sha256,
            "refit_id": spec["refit_id"],
            "context_id": spec["context_id"],
            "recipe_id": spec["recipe_id"],
            "seed": spec["seed"],
            "calibration_slot_ids": list(spec["calibration_slot_ids"]),
            "temperature": calibration.temperature,
            "optimizer_success": True,
            "optimizer_objective": calibration.optimizer_objective,
        },
    )


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): core._canon().sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _reseal(world) -> None:
    for unit in sorted(world.units.iterdir()):
        if unit.is_dir():
            core._write_manifest(unit)
    core._write_manifest(world.stage)
    payload = core._read_json(world.receipt_path, "receipt")
    payload["stage_manifest_sha256"] = core._canon().sha256_file(world.stage / frozen.MANIFEST_NAME)
    core._atomic_write(world.receipt_path, json.dumps(payload).encode("utf-8"))


def _build_world(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(prediction_io, "EXPECTED_CONTEXTS", 1)
    monkeypatch.setattr(prediction_io, "EXPECTED_ALIASES", ALIAS_COUNT)

    plan = _build_plan()
    calibration = _calibration()
    endpoint = plan["endpoints"][0]
    specs = list(plan["unique_refits"].values())

    artifact_root = tmp_path / "artifacts"
    run_root = artifact_root / frozen.COMPREHENSIVE_DIR / frozen.RUNS_DIR / PERMIT_SHA256
    stage = run_root / frozen.STAGE_NAME
    units = stage / frozen.UNITS_NAME
    refit_units = run_root / frozen.REFITS_DIR / frozen.UNITS_NAME
    units.mkdir(parents=True, exist_ok=True)
    refit_units.mkdir(parents=True, exist_ok=True)

    refit_receipt = {
        "scientific_seconds_cumulative_bound": REFIT_PRIOR_SECONDS,
        "status": "complete",
    }
    _write(run_root / frozen.REFIT_RECEIPT_NAME, refit_receipt)

    frame = _frame(calibration)
    for spec in specs:
        digest = _refit_digest(spec["refit_id"])
        unit_dir = units / spec["refit_id"]
        unit_dir.mkdir()
        _write(
            unit_dir / frozen.LEASE_NAME,
            {"selection_plan_id": plan["plan_id"], "spec": spec, "endpoint": endpoint},
        )
        prediction_io.persist_prediction(
            unit_dir, spec, endpoint, frame.copy(), _audit(spec, calibration, digest)
        )
        core._write_manifest(unit_dir)

        refit_dir = refit_units / spec["refit_id"]
        refit_dir.mkdir()
        _write(refit_dir / frozen.SUMMARY_NAME, {"terminal_state_digest": digest})
        _write_calibration(refit_dir, spec, calibration)

    counters = {
        "started": len(specs),
        "completed": len(specs),
        "failed": 0,
        "rows": len(TEST_UIDS) * len(specs),
        "contexts_completed": 1,
        "peak_cuda_bytes": UNIT_PEAK,
        "sum_prediction_elapsed_seconds": UNIT_ELAPSED * len(specs),
        "optimizer_steps": 0,
    }
    bundle = {
        "permit_sha256": PERMIT_SHA256,
        "artifact_root": artifact_root,
        "repository_root": tmp_path / "repository",
        "project_root": tmp_path / "project",
        "contract_sha256": "c" * 64,
        "core_plan_id": "core-plan-1",
        "ledger": {"ledger_id": "ledger-1"},
    }
    identity = {
        "schema_version": frozen.SCHEMA,
        "protocol_version": frozen.PROTOCOL,
        "command": frozen.COMMAND,
        "stage": frozen.STAGE_NAME,
        "permit_sha256": PERMIT_SHA256,
        "core_contract_sha256": bundle["contract_sha256"],
        "core_plan_id": bundle["core_plan_id"],
        "ledger_id": bundle["ledger"]["ledger_id"],
        "selection_plan_id": plan["plan_id"],
        "refit_receipt_sha256": core._canon().sha256_file(run_root / frozen.REFIT_RECEIPT_NAME),
        "unique_prediction_count": len(specs),
        "strategy_alias_count": len(plan["strategy_aliases"]),
        "context_count": 1,
        "fits_started": 0,
        "calibrations_started": 0,
        "optimizer_steps": 0,
        "predictions_frozen": True,
        "source_optimizer_steps": SOURCE_OPTIMIZER_STEPS,
        "refit_optimizer_steps": REFIT_OPTIMIZER_STEPS,
    }
    shared = {
        **identity,
        "predictions_complete": True,
        "all_complete": True,
        "status": "complete",
        "prior_scientific_seconds_cumulative_bound": REFIT_PRIOR_SECONDS,
        "prelaunch_audit_reserve_seconds": development.PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "maximum_total_seconds": development.MAXIMUM_TOTAL_SECONDS,
    }
    summary = {
        **shared,
        "counters": {**counters, "elapsed_seconds": SUMMARY_ELAPSED},
        "scientific_seconds_this_stage": SUMMARY_ELAPSED,
        "scientific_seconds_cumulative_bound": REFIT_PRIOR_SECONDS + SUMMARY_ELAPSED,
    }
    _write(stage / frozen.SUMMARY_NAME, summary)
    core._write_manifest(stage)
    receipt = {
        **shared,
        "counters": {**counters, "elapsed_seconds": RECEIPT_ELAPSED},
        "scientific_seconds_this_stage": RECEIPT_ELAPSED,
        "scientific_seconds_cumulative_bound": REFIT_PRIOR_SECONDS + RECEIPT_ELAPSED,
        "stage_manifest_sha256": core._canon().sha256_file(stage / frozen.MANIFEST_NAME),
    }
    _write(run_root / frozen.RECEIPT_NAME, receipt)

    calls = {"authority": 0}

    def fake_auth(bundle_arg, *, deadline):
        calls["authority"] += 1
        return {
            "plan": plan,
            "prior_seconds": REFIT_PRIOR_SECONDS,
            "source_optimizer_steps": SOURCE_OPTIMIZER_STEPS,
            "refit_optimizer_steps": REFIT_OPTIMIZER_STEPS,
            "refit_receipt": dict(refit_receipt),
        }

    monkeypatch.setattr(authority, "authenticate_refits", fake_auth)

    return SimpleNamespace(
        tmp=tmp_path,
        run_root=run_root,
        stage=stage,
        units=units,
        refit_units=refit_units,
        bundle=bundle,
        plan=plan,
        specs=specs,
        receipt_path=run_root / frozen.RECEIPT_NAME,
        summary_path=stage / frozen.SUMMARY_NAME,
        calls=calls,
        call=lambda deadline=DEADLINE: frozen.authenticate_predictions(bundle, deadline=deadline),
    )


@pytest.fixture
def world(tmp_path, monkeypatch):
    return _build_world(tmp_path, monkeypatch)


def _expect(world, code: str) -> None:
    with pytest.raises(frozen.P05FrozenPredictionsError) as info:
        world.call()
    assert info.value.reason_code == code


def test_success_is_read_only_and_returns_evaluation_cumulative(world):
    before = _snapshot(world.run_root)
    result = world.call()
    assert _snapshot(world.run_root) == before
    assert world.calls["authority"] == 1

    assert result["plan"] is world.plan
    assert result["source_optimizer_steps"] == SOURCE_OPTIMIZER_STEPS
    assert result["refit_optimizer_steps"] == REFIT_OPTIMIZER_STEPS
    assert result["prior_seconds"] == pytest.approx(REFIT_PRIOR_SECONDS + RECEIPT_ELAPSED)
    assert result["prior_seconds"] != REFIT_PRIOR_SECONDS
    assert result["evaluation_receipt"] == core._read_json(world.receipt_path, "receipt")

    expected = {spec["refit_id"] for spec in world.specs}
    assert set(result["predictions"]) == expected
    for frame in result["predictions"].values():
        assert frame[prediction_io.UID_COLUMN].tolist() == list(TEST_UIDS)
        assert all(isinstance(uid, str) for uid in frame[prediction_io.UID_COLUMN])


def test_local_verifier_does_not_repeat_checkpoint_or_array_reads(world, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("forbidden read-only call")

    monkeypatch.setattr(prediction, "_load_result", boom)
    monkeypatch.setattr(prediction, "predict_refit", boom)
    monkeypatch.setattr(np, "load", boom)

    result = world.call()
    assert result["plan"] is world.plan


def test_apply_temperature_recomputes_once_per_spec(world, monkeypatch):
    seen = {"count": 0}
    real = classical.apply_temperature

    def spy(scores, calibration):
        seen["count"] += 1
        return real(scores, calibration)

    monkeypatch.setattr(classical, "apply_temperature", spy)
    world.call()
    assert seen["count"] == len(world.specs)


def _first_refit(world) -> str:
    return world.specs[0]["refit_id"]


def _csv_frame(world, refit_id: str):
    path = world.units / refit_id / prediction_io.PREDICTIONS_FILENAME
    frame = pd.read_csv(
        path,
        dtype={prediction_io.UID_COLUMN: str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    return path, frame


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _mut_receipt_identity(world):
    _patch(world.receipt_path, "schema_version", "sentinel")


def _mut_summary_identity(world):
    _patch(world.summary_path, "command", "sentinel")
    _reseal(world)


def _mut_receipt_status(world):
    _patch(world.receipt_path, "status", "running")


def _mut_summary_status(world):
    _patch(world.summary_path, "status", "fail")
    _reseal(world)


def _mut_receipt_all_incomplete(world):
    _patch(world.receipt_path, "all_complete", False)


def _mut_summary_predictions_incomplete(world):
    _patch(world.summary_path, "predictions_complete", False)
    _reseal(world)


def _mut_counters_rows(world):
    for path in (world.receipt_path, world.summary_path):
        _patch(path, "counters.rows", 0)
    _reseal(world)


def _mut_counters_peak(world):
    for path in (world.receipt_path, world.summary_path):
        _patch(path, "counters.peak_cuda_bytes", prediction_io.MAX_PEAK_CUDA_BYTES + 1)
    _reseal(world)


def _mut_counters_sum_prediction(world):
    for path in (world.receipt_path, world.summary_path):
        _patch(path, "counters.sum_prediction_elapsed_seconds", 99.0)
    _reseal(world)


def _mut_counters_optimizer(world):
    for path in (world.receipt_path, world.summary_path):
        _patch(path, "counters.optimizer_steps", 1)
    _reseal(world)


def _mut_counters_started(world):
    for path in (world.receipt_path, world.summary_path):
        _patch(path, "counters.started", 5)
    _reseal(world)


def _mut_counters_elapsed_nan(world):
    for path in (world.receipt_path, world.summary_path):
        _patch(path, "counters.elapsed_seconds", float("nan"))
    _reseal(world)


def _mut_lease(world):
    _patch(world.units / _first_refit(world) / frozen.LEASE_NAME, "selection_plan_id", "other")
    _reseal(world)


def _mut_audit_model_digest(world):
    _patch(
        world.units / _first_refit(world) / prediction_io.AUDIT_FILENAME,
        "model_state_sha256",
        _h("0"),
    )
    _reseal(world)


def _mut_audit_calibration_digest(world):
    _patch(
        world.units / _first_refit(world) / prediction_io.AUDIT_FILENAME,
        "calibration_state_sha256",
        _h("0"),
    )
    _reseal(world)


def _mut_refit_summary_digest(world):
    _patch(
        world.refit_units / _first_refit(world) / frozen.SUMMARY_NAME,
        "terminal_state_digest",
        _h("1"),
    )


def _mut_csv_probabilities(world):
    path, frame = _csv_frame(world, _first_refit(world))
    for index in range(len(CLASSES)):
        frame[f"probability_{index}"] = 1.0 / len(CLASSES)
    _write_csv(path, frame)
    _reseal(world)


def _mut_csv_logits(world):
    path, frame = _csv_frame(world, _first_refit(world))
    frame["logit_0"] = frame["logit_0"] + 5.0
    _write_csv(path, frame)
    _reseal(world)


def _mut_missing_unit(world):
    shutil.rmtree(world.units / _first_refit(world))
    _reseal(world)


def _mut_extra_unit(world):
    extra = world.units / "extra"
    extra.mkdir()
    _write(extra / "note.json", {"note": "extra"})
    _reseal(world)


MUTATIONS = {
    "receipt_identity": (_mut_receipt_identity, "receipt_schema_version_mismatch"),
    "summary_identity": (_mut_summary_identity, "summary_command_mismatch"),
    "receipt_status": (_mut_receipt_status, "receipt_status_incomplete"),
    "summary_failed_despite_receipt_complete": (
        _mut_summary_status,
        "summary_status_incomplete",
    ),
    "receipt_all_incomplete": (_mut_receipt_all_incomplete, "receipt_all_incomplete"),
    "summary_predictions_incomplete": (
        _mut_summary_predictions_incomplete,
        "summary_predictions_incomplete",
    ),
    "counters_rows": (_mut_counters_rows, "row_count_mismatch"),
    "counters_peak": (_mut_counters_peak, "peak_cuda_exceeded"),
    "counters_sum_prediction": (
        _mut_counters_sum_prediction,
        "sum_prediction_exceeds_elapsed",
    ),
    "counters_optimizer": (_mut_counters_optimizer, "optimizer_steps_nonzero"),
    "counters_started": (_mut_counters_started, "started_mismatch"),
    "counters_elapsed_nan": (_mut_counters_elapsed_nan, "receipt_elapsed_malformed"),
    "lease": (_mut_lease, "unit_lease_mismatch"),
    "audit_model_digest": (_mut_audit_model_digest, "audit_model_digest_mismatch"),
    "audit_calibration_digest": (
        _mut_audit_calibration_digest,
        "audit_calibration_digest_mismatch",
    ),
    "refit_summary_digest": (_mut_refit_summary_digest, "audit_model_digest_mismatch"),
    "csv_probabilities": (_mut_csv_probabilities, "probability_vector_mismatch"),
    "csv_logits": (_mut_csv_logits, "probability_vector_mismatch"),
    "missing_unit": (_mut_missing_unit, "units_directory_set_mismatch"),
    "extra_unit": (_mut_extra_unit, "units_directory_set_mismatch"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_resealed_tampering_is_semantically_rejected(world, name):
    mutate, code = MUTATIONS[name]
    mutate(world)
    _expect(world, code)


def test_expired_deadline_rejected_before_authority(world):
    with pytest.raises(core.P05CoreError):
        world.call(deadline=0.0)
    assert world.calls["authority"] == 0


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("prior_scientific_seconds_cumulative_bound", 3601.0, "receipt_prior_mismatch"),
        ("scientific_seconds_cumulative_bound", 4001.0, "receipt_cumulative_mismatch"),
        ("scientific_seconds_this_stage", 4.0, "receipt_stage_seconds_inconsistent"),
        ("predictions_frozen", 1, "receipt_predictions_unfrozen"),
        ("fits_started", False, "receipt_fits_started_invalid"),
        ("calibrations_started", 1, "receipt_calibrations_started_mismatch"),
    ],
)
def test_receipt_time_and_typed_claims(world, field, value, code):
    _patch(world.receipt_path, field, value)
    _expect(world, code)


def test_changed_refit_receipt_rejected(world):
    _patch(world.run_root / frozen.REFIT_RECEIPT_NAME, "status", "fail")
    _expect(world, "refit_receipt_changed")


def test_corrupted_manifest_rejected_without_authority_side_effects(world):
    core._atomic_write(world.stage / frozen.MANIFEST_NAME, b"{not json")
    before = _snapshot(world.run_root)
    with pytest.raises(core.P05CoreError):
        world.call()
    assert world.calls["authority"] == 1
    assert _snapshot(world.run_root) == before
