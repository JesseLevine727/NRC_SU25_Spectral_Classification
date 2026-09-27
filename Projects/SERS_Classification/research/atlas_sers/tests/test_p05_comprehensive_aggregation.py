"""CPU-only orchestration tests for the comprehensive P05 aggregation stage.

These tests never read private scientific data or touch a GPU.  They drive the
real aggregation runner against the real ``p05_results.aggregate_predictions``
kernel and real parquet persistence / ``StorageBudget`` / manifest / receipt
writes, while the frozen-prediction authority, outer inputs, provenance and
``inputs.prepare`` are replaced by deterministic stubs.
"""

from __future__ import annotations

# The optional torch import must precede torch-dependent project modules.
# ruff: noqa: E402
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_comprehensive_aggregation as aggregation
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_comprehensive_storage as storage
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_frozen_predictions as frozen
from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_results as p05_results
from tests.test_p05_results import _base

REAL_AGGREGATE = p05_results.aggregate_predictions
PROVENANCE_BEFORE = {"marker": "before"}
PROVENANCE_AFTER = {"marker": "after"}
PRIOR_SECONDS = 4000.0
SOURCE_STEPS = 96
REFIT_STEPS = 6 * 30 * 4


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input_hashes(world):
    return {
        "contract": _sha(world.contract_path),
        "permit": _sha(world.permit_path),
        "receipt": _sha(world.receipt_path),
        "manifest": _sha(world.evaluation_stage / evaluation.MANIFEST_NAME),
    }


def _expected_tables(fx):
    return REAL_AGGREGATE(
        plan=fx["plan"],
        contexts=fx["contexts"],
        manifest=fx["manifest"],
        predictions=fx["predictions"],
    )


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gpu_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu_calls.append(1) or False)
    thread_calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda count: thread_calls.append(count))
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

    fx = _base()
    permit_sha256 = fx["plan"]["permit_sha256"]
    run_root = artifact_root / evaluation.COMPREHENSIVE_DIR / evaluation.RUNS_DIR / permit_sha256
    run_root.mkdir(parents=True, exist_ok=True)
    evaluation_stage = run_root / evaluation.STAGE_NAME
    evaluation_stage.mkdir()
    (evaluation_stage / "predictions.bin").write_bytes(b"evaluation")
    core._write_manifest(evaluation_stage)
    evaluation_receipt = {
        evaluation.PRIOR_FIELD: PRIOR_SECONDS,
        "stage_manifest_sha256": core._canon().sha256_file(
            evaluation_stage / evaluation.MANIFEST_NAME
        ),
    }
    receipt_path = run_root / evaluation.RECEIPT_NAME
    core._atomic_write(receipt_path, core._canon().canonical_json_bytes(evaluation_receipt))

    bundle = {
        "permit_sha256": permit_sha256,
        "artifact_root": artifact_root,
        "repository_root": repository_root,
        "project_root": project_root,
        "contract_sha256": "c" * 64,
        "core_plan_id": "core-plan-1",
        "ledger": {"ledger_id": "ledger-1"},
        "contract": {"contract": 1},
        "support": SimpleNamespace(manifest=fx["manifest"]),
    }

    record = {
        "order": [],
        "prepare": [],
        "auth": [],
        "load_contexts": [],
        "reauth": [],
        "stage_at_auth": None,
        "hook": None,
    }

    monkeypatch.setattr(core, "_configure_environment", lambda: None)
    monkeypatch.setattr(core, "_capture_provenance", lambda *args: PROVENANCE_BEFORE)

    def fake_prepare(project_arg, artifact_arg, contract_arg, permit_arg, require_unstarted):
        record["order"].append("prepare")
        record["prepare"].append(require_unstarted)
        return bundle

    monkeypatch.setattr(inputs, "prepare", fake_prepare)

    def fake_auth(bundle_arg, deadline):
        record["order"].append("auth")
        record["auth"].append(deadline)
        record["stage_at_auth"] = (run_root / aggregation.STAGE_NAME).exists()
        return {
            "prior_seconds": PRIOR_SECONDS,
            "evaluation_receipt": dict(evaluation_receipt),
            "plan": fx["plan"],
            "predictions": fx["predictions"],
            "source_optimizer_steps": SOURCE_STEPS,
            "refit_optimizer_steps": REFIT_STEPS,
        }

    monkeypatch.setattr(frozen, "authenticate_predictions", fake_auth)

    def fake_load_contexts(bundle_arg):
        record["order"].append("load_contexts")
        record["load_contexts"].append(True)
        return fx["contexts"]

    monkeypatch.setattr(outer_inputs, "load_context_rows", fake_load_contexts)

    def fake_reauth(*args):
        record["reauth"].append(args[3])
        return PROVENANCE_AFTER

    monkeypatch.setattr(pilot, "_post_run_reauth", fake_reauth)

    def recording_aggregate(*, plan, contexts, manifest, predictions):
        record["order"].append("aggregate")
        result = REAL_AGGREGATE(
            plan=plan, contexts=contexts, manifest=manifest, predictions=predictions
        )
        if record["hook"] is not None:
            return record["hook"](result)
        return result

    monkeypatch.setattr(p05_results, "aggregate_predictions", recording_aggregate)

    def run_kwargs():
        return {
            "project_root": project_root,
            "artifact_root": artifact_root,
            "contract_path": contract_path,
            "permit_path": permit_path,
        }

    return SimpleNamespace(
        tmp=tmp_path,
        artifact_root=artifact_root,
        project_root=project_root,
        repository_root=repository_root,
        contract_path=contract_path,
        permit_path=permit_path,
        bundle=bundle,
        fx=fx,
        run_root=run_root,
        stage=run_root / aggregation.STAGE_NAME,
        evaluation_stage=evaluation_stage,
        evaluation_receipt=evaluation_receipt,
        receipt_path=receipt_path,
        record=record,
        gpu_calls=gpu_calls,
        thread_calls=thread_calls,
        run_kwargs=run_kwargs,
    )


def test_success_roundtrip_authority_first(world):
    before = _input_hashes(world)
    receipt = aggregation.run_aggregation(**world.run_kwargs())

    assert receipt["status"] == "complete"
    assert receipt["aggregation_complete"] is True
    counters = receipt["counters"]
    for key in ("fits", "calibrations", "outer_predictions", "updates"):
        assert counters[key] == 0

    expected = _expected_tables(world.fx)
    for name in aggregation.TABLE_NAMES:
        assert counters[f"rows_{name}"] == len(expected[name])
        restored = pd.read_parquet(world.stage / f"{name}.parquet")
        pd.testing.assert_frame_equal(
            expected[name].reset_index(drop=True),
            restored.reset_index(drop=True),
            check_exact=True,
        )

    assert receipt["prior_scientific_seconds_cumulative_bound"] == PRIOR_SECONDS
    assert receipt["scientific_seconds_cumulative_bound"] == pytest.approx(
        PRIOR_SECONDS + receipt["scientific_seconds_this_stage"]
    )
    assert receipt["scientific_seconds_this_stage"] >= 0.0

    order = world.record["order"]
    assert order.index("auth") < order.index("load_contexts")
    assert order.index("auth") < order.index("aggregate")
    assert world.record["stage_at_auth"] is False
    assert world.record["prepare"] == [False, False]
    assert world.record["reauth"] == [PROVENANCE_BEFORE]
    assert world.thread_calls == [1]
    assert world.gpu_calls == []
    assert _input_hashes(world) == before

    assert core._read_json(world.stage / "provenance_before.json", "before") == PROVENANCE_BEFORE
    assert core._read_json(world.stage / "provenance_after.json", "after") == PROVENANCE_AFTER
    receipt_path = world.run_root / aggregation.RECEIPT_NAME
    assert core._read_json(receipt_path, "receipt") == receipt
    assert receipt["stage_manifest_sha256"] == core._canon().sha256_file(
        world.stage / aggregation.MANIFEST_NAME
    )
    pilot._verify_manifest(world.stage)


def test_occupied_stage_preserves_bytes(world):
    world.stage.mkdir(parents=True)
    sentinel = world.stage / "sentinel.bin"
    sentinel.write_bytes(b"keep-me")
    with pytest.raises(core.P05CoreError):
        aggregation.run_aggregation(**world.run_kwargs())
    assert sentinel.read_bytes() == b"keep-me"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def test_symlink_stage_preserves_target_bytes(world, tmp_path):
    target = tmp_path / "stage_target"
    target.mkdir()
    sentinel = target / "sentinel.bin"
    sentinel.write_bytes(b"keep-me")
    world.stage.symlink_to(target)
    with pytest.raises(core.P05CoreError):
        aggregation.run_aggregation(**world.run_kwargs())
    assert sentinel.read_bytes() == b"keep-me"
    assert world.stage.is_symlink()
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def test_authority_denial_writes_nothing(world, monkeypatch):
    def deny(bundle_arg, deadline):
        raise aggregation.P05ComprehensiveAggregationError("authority_denied")

    monkeypatch.setattr(frozen, "authenticate_predictions", deny)
    with pytest.raises(aggregation.P05ComprehensiveAggregationError):
        aggregation.run_aggregation(**world.run_kwargs())
    assert not world.stage.exists()
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()
    assert world.record["load_contexts"] == []


def test_aggregation_metric_failure_preserves_failure(world):
    def boom(tables):
        raise aggregation.P05ComprehensiveAggregationError("aggregation_metric_error")

    world.record["hook"] = boom
    with pytest.raises(aggregation.P05ComprehensiveAggregationError) as info:
        aggregation.run_aggregation(**world.run_kwargs())
    assert info.value.reason_code == "aggregation_metric_error"
    assert world.stage.is_dir()
    summary = core._read_json(world.stage / aggregation.SUMMARY_NAME, "summary")
    assert summary["status"] == "fail"
    assert summary["reason_code"] == "aggregation_metric_error"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def test_persistence_failure_records_failure(world, monkeypatch):
    real_atomic = core._atomic_write

    def failing(path, payload):
        if str(path).endswith(".parquet"):
            raise core.P05CoreError("persistence_failed")
        return real_atomic(path, payload)

    monkeypatch.setattr(core, "_atomic_write", failing)
    with pytest.raises(core.P05CoreError):
        aggregation.run_aggregation(**world.run_kwargs())
    summary = core._read_json(world.stage / aggregation.SUMMARY_NAME, "summary")
    assert summary["status"] == "fail"
    assert summary["reason_code"] == "persistence_failed"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def test_storage_failure_records_failure(world, monkeypatch):
    def boom(self, headroom_bytes=0):
        raise storage.P05StorageError("storage_ceiling_exceeded")

    monkeypatch.setattr(freeze.StorageBudget, "check", boom)
    with pytest.raises(storage.P05StorageError):
        aggregation.run_aggregation(**world.run_kwargs())
    summary = core._read_json(world.stage / aggregation.SUMMARY_NAME, "summary")
    assert summary["status"] == "fail"
    assert summary["reason_code"] == "storage_ceiling_exceeded"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def test_late_close_failure_records_failure(world, monkeypatch):
    real_check = freeze._check_deadline
    real_verify = pilot._verify_manifest
    armed = {"value": False}

    def check(deadline):
        if armed["value"]:
            raise core.P05CoreError("aggregation_deadline_exceeded")
        return real_check(deadline)

    def verify(path):
        result = real_verify(path)
        if Path(path) == world.stage:
            armed["value"] = True
        return result

    monkeypatch.setattr(freeze, "_check_deadline", check)
    monkeypatch.setattr(pilot, "_verify_manifest", verify)
    with pytest.raises(core.P05CoreError):
        aggregation.run_aggregation(**world.run_kwargs())
    summary = core._read_json(world.stage / aggregation.SUMMARY_NAME, "summary")
    assert summary["status"] == "fail"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def test_changed_evaluation_manifest_detected(world):
    def hook(tables):
        (world.evaluation_stage / "extra.bin").write_bytes(b"extra")
        core._write_manifest(world.evaluation_stage)
        return tables

    world.record["hook"] = hook
    with pytest.raises(aggregation.P05ComprehensiveAggregationError) as info:
        aggregation.run_aggregation(**world.run_kwargs())
    assert info.value.reason_code == "evaluation_manifest_changed"
    summary = core._read_json(world.stage / aggregation.SUMMARY_NAME, "summary")
    assert summary["status"] == "fail"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def test_changed_evaluation_receipt_detected(world):
    def hook(tables):
        changed = {**world.evaluation_receipt, "tampered": True}
        core._atomic_write(world.receipt_path, core._canon().canonical_json_bytes(changed))
        return tables

    world.record["hook"] = hook
    with pytest.raises(aggregation.P05ComprehensiveAggregationError) as info:
        aggregation.run_aggregation(**world.run_kwargs())
    assert info.value.reason_code == "evaluation_receipt_changed"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def test_coverage_row_failure_detected(world):
    def hook(tables):
        tables["coverage"] = tables["coverage"].iloc[:-1].reset_index(drop=True)
        return tables

    world.record["hook"] = hook
    with pytest.raises(aggregation.P05ComprehensiveAggregationError) as info:
        aggregation.run_aggregation(**world.run_kwargs())
    assert info.value.reason_code == "coverage_row_count_mismatch"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()


def _direct_budget(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / evaluation.COMPREHENSIVE_DIR / evaluation.RUNS_DIR / "permit"
    run_dir.mkdir(parents=True)
    budget = freeze.StorageBudget(artifact_root, run_dir, ceiling=10**9)
    return budget, run_dir


def test_write_table_detects_same_size_corruption(tmp_path, monkeypatch):
    budget, run_dir = _direct_budget(tmp_path)
    stage = run_dir / "stage"
    stage.mkdir()
    frame = pd.DataFrame({"observation_uid": ["a", "b"], "probability_0": [0.5, 0.4]})
    real_write = core._atomic_write

    def corrupted(path, data):
        return real_write(path, data[:-1] + bytes([data[-1] ^ 0xFF]))

    monkeypatch.setattr(core, "_atomic_write", corrupted)
    with pytest.raises(aggregation.P05ComprehensiveAggregationError) as info:
        aggregation._write_table(stage=stage, budget=budget, name="seed_predictions", frame=frame)
    assert info.value.reason_code == "table_bytes_mismatch"


@pytest.mark.parametrize("kind", ["existing", "symlink"])
def test_write_table_never_overwrites(tmp_path, kind):
    budget, run_dir = _direct_budget(tmp_path)
    stage = run_dir / "stage"
    stage.mkdir()
    path = stage / "seed_predictions.parquet"
    target = None
    if kind == "existing":
        path.write_bytes(b"existing-bytes")
    else:
        target = stage / "target.bin"
        target.write_bytes(b"target-bytes")
        path.symlink_to(target)
    frame = pd.DataFrame({"observation_uid": ["a"]})
    with pytest.raises((core.P05CoreError, storage.P05StorageError)):
        aggregation._write_table(stage=stage, budget=budget, name="seed_predictions", frame=frame)
    if kind == "existing":
        assert path.read_bytes() == b"existing-bytes"
    else:
        assert target.read_bytes() == b"target-bytes"
        assert path.is_symlink()
