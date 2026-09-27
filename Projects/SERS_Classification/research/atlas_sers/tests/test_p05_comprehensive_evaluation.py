"""CPU-only orchestration tests for the comprehensive P05 evaluation runner.

These tests never run a scientific forward pass or touch a GPU.  They drive the
runner against a synthetic canonical plan backed by real temporary filesystem
components (``StorageBudget``, manifests, prediction persistence, close-out),
while authority, outer arrays, CUDA probes, provenance, inputs preparation and
the prediction kernel are replaced by deterministic stubs.
"""

from __future__ import annotations

# The optional torch import must precede torch-dependent project modules.
# ruff: noqa: E402
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as runner
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_comprehensive_storage as storage
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as authority
from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_prediction as prediction
from atlas_sers.evaluation import p05_prediction_io as prediction_io
from atlas_sers.evaluation import p05_refit_plan as refit_plan
from atlas_sers.evaluation import p05_selection as selection

RECIPES = ("D0-M", "D3")
CLASSES = ("A", "B", "C")
FITTING_UIDS = ("fA", "fB", "fC")
TEST_UIDS = ("tA", "tB")
CONTEXT_ID = "ctx-1"
PERMIT_SHA256 = inputs.COMPREHENSIVE_PERMIT_SHA256
EPOCHS = 30
CALIBRATION_SLOTS = ("calib-1",)
ALIAS_COUNT = 9
SOURCE_OPTIMIZER_STEPS = 96
REFIT_OPTIMIZER_STEPS = 6 * EPOCHS * 4
PRIOR_SECONDS = 4000.0
PROVENANCE_BEFORE = {"marker": "before"}
PROVENANCE_AFTER = {"marker": "after"}


def _frame() -> pd.DataFrame:
    probabilities = np.asarray([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]], dtype=np.float64)
    logits = np.asarray([[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]], dtype=np.float64)
    data = {"observation_uid": list(TEST_UIDS)}
    for index in range(len(CLASSES)):
        data[f"logit_{index}"] = logits[:, index]
        data[f"probability_{index}"] = probabilities[:, index]
    return pd.DataFrame(data)


def _audit(spec) -> dict:
    return {
        "refit_id": spec["refit_id"],
        "classes": list(spec["classes"]),
        "rows": len(TEST_UIDS),
        "test_uid_set_sha256": core._canon().sha256_value(sorted(TEST_UIDS)),
        "optimizer_steps": 0,
        "elapsed_seconds": 0.0,
        "peak_cuda_bytes": 1024,
        "model_state_sha256": "a" * 64,
        "calibration_state_sha256": "b" * 64,
    }


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
        for seed in selection.SEEDS
        for strategy in ("D0-M", "P05-SELECTED", "D3")
    ]
    endpoint = {"context_id": CONTEXT_ID, "test_uids": list(TEST_UIDS)}
    return {
        "plan_id": "plan-1",
        "endpoints": [endpoint],
        "unique_refits": unique,
        "strategy_aliases": aliases,
    }


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    torch.manual_seed(0)

    artifact_root = tmp_path / "artifacts"
    project_root = tmp_path / "project"
    repository_root = tmp_path / "repository"
    contract_path = tmp_path / "contract.json"
    permit_path = tmp_path / "permit.json"
    for directory in (artifact_root, project_root, repository_root):
        directory.mkdir(parents=True, exist_ok=True)
    contract_path.write_text("{}", encoding="utf-8")
    permit_path.write_text("{}", encoding="utf-8")

    plan = _build_plan()
    refit_receipt = {"scientific_seconds_cumulative_bound": PRIOR_SECONDS}
    run_root = artifact_root / runner.COMPREHENSIVE_DIR / runner.RUNS_DIR / PERMIT_SHA256
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / runner.REFIT_RECEIPT_NAME).write_bytes(
        core._canon().canonical_json_bytes(refit_receipt)
    )

    bundle = {
        "permit_sha256": PERMIT_SHA256,
        "artifact_root": artifact_root,
        "repository_root": repository_root,
        "project_root": project_root,
        "contract_sha256": "c" * 64,
        "core_plan_id": "core-plan-1",
        "ledger": {"ledger_id": "ledger-1"},
        "contract": {"contract": 1},
        "support": {"support": 1},
    }

    record = {
        "order": [],
        "prepare": [],
        "auth": [],
        "load_contexts": [],
        "prepare_outer": [],
        "predict": [],
        "persist": [],
        "events": [],
        "reauth": [],
        "predict_handler": None,
    }

    monkeypatch.setattr(prediction_io, "EXPECTED_CONTEXTS", 1)
    monkeypatch.setattr(prediction_io, "EXPECTED_ALIASES", ALIAS_COUNT)
    monkeypatch.setattr(core, "_configure_environment", lambda: None)
    monkeypatch.setattr(core, "_capture_provenance", lambda *args: PROVENANCE_BEFORE)
    monkeypatch.setattr(pilot, "_free_cuda_bytes", lambda torch_module: 2 * 10**10)
    monkeypatch.setattr(pilot, "_enforce_cuda_cap", lambda torch_module, device: None)

    def fake_prepare(
        project_root_arg, artifact_root_arg, contract_arg, permit_arg, require_unstarted
    ):
        record["order"].append("prepare")
        record["prepare"].append(require_unstarted)
        return bundle

    monkeypatch.setattr(inputs, "prepare", fake_prepare)

    def fake_auth(bundle_arg, deadline):
        record["order"].append("auth")
        record["auth"].append(deadline)
        return {
            "prior_seconds": PRIOR_SECONDS,
            "plan": plan,
            "refit_receipt": dict(refit_receipt),
            "source_optimizer_steps": SOURCE_OPTIMIZER_STEPS,
            "refit_optimizer_steps": REFIT_OPTIMIZER_STEPS,
        }

    monkeypatch.setattr(authority, "authenticate_refits", fake_auth)

    def fake_load_contexts(bundle_arg):
        record["order"].append("load_contexts")
        record["load_contexts"].append(True)
        return [{"context_id": CONTEXT_ID}]

    monkeypatch.setattr(outer_inputs, "load_context_rows", fake_load_contexts)

    def fake_prepare_outer(bundle_arg, endpoint):
        record["order"].append("prepare_outer")
        record["prepare_outer"].append(endpoint["context_id"])
        return {
            "classes": list(CLASSES),
            "values": np.zeros((len(TEST_UIDS), 1401), dtype=np.float32),
            "observation_uids": list(TEST_UIDS),
        }

    monkeypatch.setattr(outer_inputs, "prepare_outer_inputs", fake_prepare_outer)

    def fake_predict(*, spec, unit_dir, values, observation_uids, device, deadline):
        record["order"].append("predict")
        record["predict"].append(
            {"refit_id": spec["refit_id"], "unit_dir": Path(unit_dir), "device": device}
        )
        handler = record["predict_handler"]
        if handler is not None:
            handler(spec, len(record["predict"]))
        return _frame(), _audit(spec)

    monkeypatch.setattr(prediction, "predict_refit", fake_predict)

    real_persist = prediction_io.persist_prediction

    def fake_persist(unit_dir, spec, endpoint, frame, audit):
        record["persist"].append(spec["refit_id"])
        return real_persist(unit_dir, spec, endpoint, frame, audit)

    monkeypatch.setattr(prediction_io, "persist_prediction", fake_persist)

    real_append = development._append_jsonl

    def fake_append(path, payload):
        record["events"].append((str(path), payload))
        return real_append(path, payload)

    monkeypatch.setattr(development, "_append_jsonl", fake_append)

    def fake_reauth(
        artifact_root_arg,
        contract,
        support,
        provenance_before,
        repository_root_arg,
        project_root_arg,
    ):
        record["reauth"].append(provenance_before)
        return PROVENANCE_AFTER

    monkeypatch.setattr(pilot, "_post_run_reauth", fake_reauth)

    def run_kwargs(device="cuda"):
        return {
            "project_root": project_root,
            "artifact_root": artifact_root,
            "contract_path": contract_path,
            "permit_path": permit_path,
            "device": device,
        }

    return SimpleNamespace(
        tmp=tmp_path,
        artifact_root=artifact_root,
        project_root=project_root,
        repository_root=repository_root,
        contract_path=contract_path,
        permit_path=permit_path,
        bundle=bundle,
        plan=plan,
        specs=list(plan["unique_refits"].values()),
        run_root=run_root,
        stage=run_root / runner.STAGE_NAME,
        record=record,
        run_kwargs=run_kwargs,
    )


def test_success_orchestration_authority_first_and_complete(world):
    receipt = runner.run_evaluation(**world.run_kwargs())

    assert receipt["status"] == "complete"
    assert receipt["predictions_frozen"] is True
    assert receipt["predictions_complete"] is True
    assert receipt["all_complete"] is True
    assert receipt["unique_prediction_count"] == len(world.specs) == 6
    assert receipt["strategy_alias_count"] == ALIAS_COUNT == 9
    assert receipt["context_count"] == 1
    assert receipt["source_optimizer_steps"] == SOURCE_OPTIMIZER_STEPS
    assert receipt["refit_optimizer_steps"] == REFIT_OPTIMIZER_STEPS

    counters = receipt["counters"]
    assert counters["started"] == 6
    assert counters["completed"] == 6
    assert counters["failed"] == 0
    assert counters["contexts_completed"] == 1
    assert counters["optimizer_steps"] == 0
    assert counters["rows"] == len(TEST_UIDS) * len(world.specs) == 12
    assert counters["peak_cuda_bytes"] == 1024
    assert counters["sum_prediction_elapsed_seconds"] == 0.0
    assert counters["elapsed_seconds"] > 0.0
    assert receipt["scientific_seconds_cumulative_bound"] == pytest.approx(
        PRIOR_SECONDS + receipt["scientific_seconds_this_stage"]
    )

    order = world.record["order"]
    assert order.index("auth") < order.index("load_contexts")
    assert order.index("auth") < order.index("prepare_outer")
    assert order.index("auth") < order.index("predict")
    assert len(world.record["auth"]) == 1
    assert world.record["prepare"] == [False, False]

    forwards = world.record["predict"]
    expected_ids = {spec["refit_id"] for spec in world.specs}
    assert len(forwards) == len(expected_ids)
    assert {item["refit_id"] for item in forwards} == expected_ids
    assert all(item["device"] == "cuda" for item in forwards)
    assert len(world.plan["strategy_aliases"]) > len(forwards)
    assert len(world.record["persist"]) == len(expected_ids)

    for item in forwards:
        unit_dir = item["unit_dir"]
        assert unit_dir == world.run_root / "refits" / "units" / item["refit_id"]
        assert runner.STAGE_NAME not in unit_dir.parts

    assert world.record["reauth"][0] is PROVENANCE_BEFORE
    assert core._read_json(world.stage / "provenance_before.json", "before") == PROVENANCE_BEFORE
    assert core._read_json(world.stage / "provenance_after.json", "after") == PROVENANCE_AFTER

    pilot._verify_manifest(world.stage)
    receipt_path = world.run_root / runner.RECEIPT_NAME
    assert receipt_path.is_file()
    assert core._read_json(receipt_path, "receipt") == receipt
    assert receipt["stage_manifest_sha256"] == core._canon().sha256_file(
        world.stage / "manifest.json"
    )
    for spec in world.specs:
        unit = world.stage / "units" / spec["refit_id"]
        assert (unit / prediction_io.PREDICTIONS_FILENAME).is_file()
        assert (unit / prediction_io.AUDIT_FILENAME).is_file()
        pilot._verify_manifest(unit)


def test_device_cpu_rejected_before_preparation(world):
    with pytest.raises(runner.P05ComprehensiveEvaluationError) as info:
        runner.run_evaluation(**world.run_kwargs(device="cpu"))

    assert info.value.reason_code == "device_not_cuda"
    assert world.record["order"] == []
    assert not world.stage.exists()


def test_authority_denial_writes_nothing(world, monkeypatch):
    def deny(bundle_arg, deadline):
        raise runner.P05ComprehensiveEvaluationError("authority_denied")

    monkeypatch.setattr(authority, "authenticate_refits", deny)
    with pytest.raises(runner.P05ComprehensiveEvaluationError):
        runner.run_evaluation(**world.run_kwargs())

    assert not world.stage.exists()
    assert not (world.run_root / runner.RECEIPT_NAME).exists()
    assert world.record["load_contexts"] == []
    assert world.record["predict"] == []


def test_occupied_stage_preserves_bytes(world):
    world.stage.mkdir(parents=True)
    sentinel = world.stage / "sentinel.bin"
    sentinel.write_bytes(b"keep-me")

    with pytest.raises(core.P05CoreError):
        runner.run_evaluation(**world.run_kwargs())

    assert sentinel.read_bytes() == b"keep-me"
    assert not (world.run_root / runner.RECEIPT_NAME).exists()


def test_symlink_stage_preserves_target_bytes(world, tmp_path):
    target = tmp_path / "stage_target"
    target.mkdir()
    sentinel = target / "sentinel.bin"
    sentinel.write_bytes(b"keep-me")
    world.stage.symlink_to(target)

    with pytest.raises(core.P05CoreError):
        runner.run_evaluation(**world.run_kwargs())

    assert sentinel.read_bytes() == b"keep-me"
    assert world.stage.is_symlink()


def test_second_forward_failure_keeps_first_complete(world):
    def handler(spec, count):
        if count == 2:
            raise RuntimeError("forward boom")

    world.record["predict_handler"] = handler
    with pytest.raises(RuntimeError):
        runner.run_evaluation(**world.run_kwargs())

    assert len(world.record["predict"]) == 2
    assert len(world.record["persist"]) == 1
    summary = core._read_json(world.stage / "summary.json", "summary")
    assert summary["status"] == "fail"
    assert summary["counters"]["started"] == 2
    assert summary["counters"]["completed"] == 1
    assert summary["counters"]["failed"] == 1

    first_id = world.record["predict"][0]["refit_id"]
    first_unit = world.stage / "units" / first_id
    assert (first_unit / prediction_io.PREDICTIONS_FILENAME).is_file()
    assert (first_unit / prediction_io.AUDIT_FILENAME).is_file()
    assert not (world.run_root / runner.RECEIPT_NAME).exists()


def test_persistence_failure_after_forward(world, monkeypatch):
    def boom(unit_dir, spec, endpoint, frame, audit):
        raise prediction_io.P05PredictionIOError("persist_failed")

    monkeypatch.setattr(prediction_io, "persist_prediction", boom)
    with pytest.raises(prediction_io.P05PredictionIOError):
        runner.run_evaluation(**world.run_kwargs())

    assert len(world.record["predict"]) == 1
    summary = core._read_json(world.stage / "summary.json", "summary")
    assert summary["status"] == "fail"
    assert summary["counters"]["started"] == 1
    assert summary["counters"]["completed"] == 0
    assert summary["counters"]["failed"] == 1
    assert summary["reason_code"] == "persist_failed"


def test_storage_budget_failure_records_failure(world, monkeypatch):
    def boom(self, headroom_bytes=0):
        raise storage.P05StorageError("storage_ceiling_exceeded")

    monkeypatch.setattr(freeze.StorageBudget, "check", boom)
    with pytest.raises(storage.P05StorageError):
        runner.run_evaluation(**world.run_kwargs())

    assert world.record["predict"] == []
    summary = core._read_json(world.stage / "summary.json", "summary")
    assert summary["status"] == "fail"
    assert summary["reason_code"] == "storage_ceiling_exceeded"
    assert not (world.run_root / runner.RECEIPT_NAME).exists()


def test_late_deadline_at_close_records_failure(world, monkeypatch):
    real_check = freeze._check_deadline
    real_verify = pilot._verify_manifest
    armed = {"value": False}

    def check(deadline):
        if armed["value"]:
            raise core.P05CoreError("evaluation_deadline_exceeded")
        return real_check(deadline)

    def verify(path):
        result = real_verify(path)
        if Path(path) == world.stage:
            armed["value"] = True
        return result

    monkeypatch.setattr(freeze, "_check_deadline", check)
    monkeypatch.setattr(pilot, "_verify_manifest", verify)
    with pytest.raises(core.P05CoreError):
        runner.run_evaluation(**world.run_kwargs())

    assert len(world.record["predict"]) == 6
    summary = core._read_json(world.stage / "summary.json", "summary")
    assert summary["status"] == "fail"
    assert not (world.run_root / runner.RECEIPT_NAME).exists()


def test_wrong_class_vocabulary_before_first_forward(world, monkeypatch):
    def bad_prepare_outer(bundle_arg, endpoint):
        return {
            "classes": ["A", "B", "D"],
            "values": np.zeros((len(TEST_UIDS), len(CLASSES))),
            "observation_uids": list(TEST_UIDS),
        }

    monkeypatch.setattr(outer_inputs, "prepare_outer_inputs", bad_prepare_outer)
    with pytest.raises(runner.P05ComprehensiveEvaluationError) as info:
        runner.run_evaluation(**world.run_kwargs())

    assert info.value.reason_code == "prediction_source_classes_mismatch"
    assert world.record["predict"] == []
    summary = core._read_json(world.stage / "summary.json", "summary")
    assert summary["status"] == "fail"
