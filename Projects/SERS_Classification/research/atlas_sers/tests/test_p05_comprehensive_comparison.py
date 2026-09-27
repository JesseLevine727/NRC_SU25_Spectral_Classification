"""CPU-only orchestration tests for the comprehensive P05 comparison stage.

These tests never read private scientific data or touch a GPU.  They drive the
real comparison runner against the real ``p05_comparison.compare_predictions``
kernel and real parquet persistence / ``StorageBudget`` / manifest / receipt
writes, while the aggregation authority, legacy reference loader, outer inputs,
provenance and ``inputs.prepare`` are replaced by deterministic stubs.
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

from atlas_sers.evaluation import p05_aggregation_authority as authority
from atlas_sers.evaluation import p05_comparison as pure_comparison
from atlas_sers.evaluation import p05_comprehensive_aggregation as aggregation
from atlas_sers.evaluation import p05_comprehensive_comparison as comparison
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_comprehensive_storage as storage
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_legacy_references as legacy_references
from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
from atlas_sers.evaluation import p05_pilot as pilot
from tests.test_p05_comparison import _fixture as _pure_fixture

REAL_COMPARE = pure_comparison.compare_predictions
PROVENANCE_BEFORE = {"marker": "before"}
PROVENANCE_AFTER = {"marker": "after"}
PRIOR_SECONDS = 4000.0
SOURCE_STEPS = 96
REFIT_STEPS = 6 * 30 * 4
RANDOM_FOREST = "C-RANDOM-FOREST"
PLAN_ID = "plan-1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input_hashes(world):
    return {
        "contract": _sha(world.contract_path),
        "permit": _sha(world.permit_path),
        "aggregation_receipt": _sha(world.aggregation_receipt_path),
        "aggregation_manifest": _sha(
            world.aggregation_stage / comparison.AGGREGATION_MANIFEST_NAME
        ),
    }


def _expected_tables(fx):
    return REAL_COMPARE(
        p05_ensemble=fx["aggregate"]["ensemble_predictions"],
        p04_ensemble=fx["p04_ensemble"],
        p03_predictions=fx["p03_predictions"],
        contexts=pd.DataFrame(fx["raw"]["contexts"]),
    )


def _summary(world):
    return core._read_json(world.stage / comparison.SUMMARY_NAME, "summary")


def _assert_consumed_failure(world, code):
    summary = _summary(world)
    assert summary["status"] == "fail"
    assert summary["comparison_complete"] is False
    assert summary["reason_code"] == code
    assert not (world.run_root / comparison.RECEIPT_NAME).exists()


def _direct_budget(tmp_path):
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / evaluation.COMPREHENSIVE_DIR / evaluation.RUNS_DIR / "permit"
    run_dir.mkdir(parents=True)
    budget = freeze.StorageBudget(artifact_root, run_dir, ceiling=10**9)
    return budget, run_dir


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

    fx = _pure_fixture()
    permit_sha256 = fx["raw"]["plan"]["permit_sha256"]
    run_root = artifact_root / evaluation.COMPREHENSIVE_DIR / evaluation.RUNS_DIR / permit_sha256
    run_root.mkdir(parents=True, exist_ok=True)

    aggregation_stage = run_root / comparison.AGGREGATION_STAGE_NAME
    aggregation_stage.mkdir()
    (aggregation_stage / "ensemble_predictions.parquet").write_bytes(b"aggregation")
    core._write_manifest(aggregation_stage)
    aggregation_receipt = {
        evaluation.PRIOR_FIELD: PRIOR_SECONDS,
        "stage_manifest_sha256": core._canon().sha256_file(
            aggregation_stage / comparison.AGGREGATION_MANIFEST_NAME
        ),
    }
    aggregation_receipt_path = run_root / comparison.AGGREGATION_RECEIPT_NAME
    core._atomic_write(
        aggregation_receipt_path,
        core._canon().canonical_json_bytes(aggregation_receipt),
    )

    bundle = {
        "permit_sha256": permit_sha256,
        "artifact_root": artifact_root,
        "repository_root": repository_root,
        "project_root": project_root,
        "contract_sha256": "c" * 64,
        "core_plan_id": "core-plan-1",
        "ledger": {"ledger_id": "ledger-1"},
        "contract": {"contract": 1},
        "support": SimpleNamespace(manifest=fx["raw"]["manifest"]),
    }

    record = {
        "order": [],
        "prepare": [],
        "auth": [],
        "load_contexts": [],
        "legacy": [],
        "reauth": [],
        "stage_at_auth": None,
        "hook": None,
        "legacy_hook": None,
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
        record["stage_at_auth"] = (run_root / comparison.STAGE_NAME).exists()
        return {
            "prior_seconds": PRIOR_SECONDS,
            "aggregation_receipt": dict(aggregation_receipt),
            "plan": {"plan_id": PLAN_ID},
            "aggregation_tables": {"ensemble_predictions": fx["aggregate"]["ensemble_predictions"]},
            "source_optimizer_steps": SOURCE_STEPS,
            "refit_optimizer_steps": REFIT_STEPS,
        }

    monkeypatch.setattr(authority, "authenticate_aggregation", fake_auth)

    def fake_load_references(bundle_arg, authenticated, deadline):
        index = len(record["legacy"])
        record["order"].append("legacy")
        record["legacy"].append(deadline)
        result = {
            "bindings": {"p04_digest": "d" * 64, "p03_digest": "e" * 64},
            "p04_ensemble": fx["p04_ensemble"],
            "p03_predictions": fx["p03_predictions"],
        }
        if record["legacy_hook"] is not None:
            result = record["legacy_hook"](index, result)
        return result

    monkeypatch.setattr(legacy_references, "load_references", fake_load_references)

    def fake_load_contexts(bundle_arg):
        record["order"].append("load_contexts")
        record["load_contexts"].append(True)
        return fx["raw"]["contexts"]

    monkeypatch.setattr(outer_inputs, "load_context_rows", fake_load_contexts)

    def fake_reauth(*args):
        record["reauth"].append(args[3])
        return PROVENANCE_AFTER

    monkeypatch.setattr(pilot, "_post_run_reauth", fake_reauth)

    def recording_compare(*, p05_ensemble, p04_ensemble, p03_predictions, contexts):
        record["order"].append("compare")
        result = REAL_COMPARE(
            p05_ensemble=p05_ensemble,
            p04_ensemble=p04_ensemble,
            p03_predictions=p03_predictions,
            contexts=contexts,
        )
        if record["hook"] is not None:
            return record["hook"](result)
        return result

    monkeypatch.setattr(pure_comparison, "compare_predictions", recording_compare)

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
        stage=run_root / comparison.STAGE_NAME,
        aggregation_stage=aggregation_stage,
        aggregation_receipt=aggregation_receipt,
        aggregation_receipt_path=aggregation_receipt_path,
        record=record,
        gpu_calls=gpu_calls,
        thread_calls=thread_calls,
        run_kwargs=run_kwargs,
    )


def test_success_roundtrip_authority_first(world):
    before = _input_hashes(world)
    receipt = comparison.run_comparison(**world.run_kwargs())

    assert receipt["status"] == "complete"
    assert receipt["comparison_complete"] is True
    assert receipt["selection_plan_id"] == PLAN_ID
    assert receipt["source_optimizer_steps"] == SOURCE_STEPS
    assert receipt["refit_optimizer_steps"] == REFIT_STEPS
    counters = receipt["counters"]
    assert counters["rows_coverage"] == 8
    assert counters["rows_endpoint_metrics"] == 16
    assert counters["rows_paired_metrics"] == 34
    assert counters["held_contexts"] == 1
    assert counters["complete_model_contexts"] == 8
    assert counters["incomplete_model_contexts"] == 0
    for key in ("fits", "calibrations", "outer_predictions", "updates"):
        assert counters[key] == 0

    expected = _expected_tables(world.fx)
    for name in comparison.TABLE_NAMES:
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
    first_legacy = order.index("legacy")
    second_legacy = order.index("legacy", first_legacy + 1)
    assert order.count("legacy") == 2
    assert order.count("auth") == 1
    assert order.index("auth") < first_legacy
    assert order.index("auth") < order.index("load_contexts")
    assert first_legacy < order.index("compare") < second_legacy
    assert world.record["stage_at_auth"] is False
    assert world.record["prepare"] == [False, False]
    assert world.record["reauth"] == [PROVENANCE_BEFORE]
    assert world.thread_calls == [1]
    assert world.gpu_calls == []
    assert _input_hashes(world) == before

    assert core._read_json(world.stage / comparison.PROVENANCE_BEFORE_NAME, "before") == (
        PROVENANCE_BEFORE
    )
    assert core._read_json(world.stage / comparison.PROVENANCE_AFTER_NAME, "after") == (
        PROVENANCE_AFTER
    )
    receipt_path = world.run_root / comparison.RECEIPT_NAME
    assert core._read_json(receipt_path, "receipt") == receipt
    assert receipt["stage_manifest_sha256"] == core._canon().sha256_file(
        world.stage / comparison.MANIFEST_NAME
    )
    pilot._verify_manifest(world.stage)


@pytest.mark.parametrize("mode", ["occupied", "symlink"])
def test_occupied_stage_rejected(world, tmp_path, mode):
    if mode == "occupied":
        world.stage.mkdir(parents=True)
        sentinel = world.stage / "sentinel.bin"
        sentinel.write_bytes(b"keep-me")
    else:
        target = tmp_path / "stage_target"
        target.mkdir()
        sentinel = target / "sentinel.bin"
        sentinel.write_bytes(b"keep-me")
        world.stage.symlink_to(target)
    with pytest.raises(core.P05CoreError):
        comparison.run_comparison(**world.run_kwargs())
    assert sentinel.read_bytes() == b"keep-me"
    if mode == "symlink":
        assert world.stage.is_symlink()
    assert not (world.run_root / comparison.RECEIPT_NAME).exists()
    assert world.record["legacy"] == []


@pytest.mark.parametrize("mode", ["occupied", "symlink"])
def test_occupied_receipt_rejected(world, tmp_path, mode):
    receipt_path = world.run_root / comparison.RECEIPT_NAME
    if mode == "occupied":
        receipt_path.write_bytes(b"keep-me")
        sentinel = receipt_path
    else:
        target = tmp_path / "receipt_target"
        target.write_bytes(b"keep-me")
        receipt_path.symlink_to(target)
        sentinel = target
    with pytest.raises(core.P05CoreError):
        comparison.run_comparison(**world.run_kwargs())
    assert sentinel.read_bytes() == b"keep-me"
    if mode == "symlink":
        assert receipt_path.is_symlink()
    assert not world.stage.exists()
    assert world.record["legacy"] == []


def _authority_variant(world, kind):
    if kind == "deny":

        def fake(bundle_arg, deadline):
            raise comparison.P05ComprehensiveComparisonError("authority_denied")

        return fake, "authority_denied"
    if kind == "prior":

        def fake(bundle_arg, deadline):
            return {
                "prior_seconds": PRIOR_SECONDS + 1.0,
                "aggregation_receipt": dict(world.aggregation_receipt),
                "plan": {"plan_id": PLAN_ID},
                "aggregation_tables": {
                    "ensemble_predictions": world.fx["aggregate"]["ensemble_predictions"]
                },
                "source_optimizer_steps": SOURCE_STEPS,
                "refit_optimizer_steps": REFIT_STEPS,
            }

        return fake, "authority_prior_mismatch"

    def fake(bundle_arg, deadline):
        changed = {**world.aggregation_receipt, "tampered": True}
        return {
            "prior_seconds": PRIOR_SECONDS,
            "aggregation_receipt": changed,
            "plan": {"plan_id": PLAN_ID},
            "aggregation_tables": {
                "ensemble_predictions": world.fx["aggregate"]["ensemble_predictions"]
            },
            "source_optimizer_steps": SOURCE_STEPS,
            "refit_optimizer_steps": REFIT_STEPS,
        }

    return fake, "aggregation_receipt_changed"


@pytest.mark.parametrize("kind", ["deny", "prior", "receipt"])
def test_authority_failures_write_nothing(world, monkeypatch, kind):
    fake, code = _authority_variant(world, kind)
    monkeypatch.setattr(authority, "authenticate_aggregation", fake)
    with pytest.raises(comparison.P05ComprehensiveComparisonError) as info:
        comparison.run_comparison(**world.run_kwargs())
    assert info.value.reason_code == code
    assert not world.stage.exists()
    assert not (world.run_root / comparison.RECEIPT_NAME).exists()
    assert world.record["legacy"] == []
    assert world.record["load_contexts"] == []


def _consumed_variant(world, kind):
    if kind == "legacy_bindings":

        def hook(index, result):
            if index == 1:
                return {
                    **result,
                    "bindings": {"p04_digest": "0" * 64, "p03_digest": "e" * 64},
                }
            return result

        world.record["legacy_hook"] = hook
        return "legacy_bindings_changed"
    if kind == "legacy_failure":

        def hook(index, result):
            raise comparison.P05ComprehensiveComparisonError("legacy_load_failed")

        world.record["legacy_hook"] = hook
        return "legacy_load_failed"
    if kind == "comparison":

        def hook(tables):
            raise comparison.P05ComprehensiveComparisonError("comparison_metric_error")

        world.record["hook"] = hook
        return "comparison_metric_error"
    if kind == "coverage_count":

        def hook(tables):
            tables["coverage"] = tables["coverage"].iloc[:-1].reset_index(drop=True)
            return tables

        world.record["hook"] = hook
        return "coverage_row_count_mismatch"
    if kind == "coverage_bool":

        def hook(tables):
            frame = tables["coverage"].copy()
            frame["complete"] = frame["complete"].astype(int)
            tables["coverage"] = frame
            return tables

        world.record["hook"] = hook
        return "coverage_complete_invalid"
    if kind == "pair_keys":

        def hook(tables):
            frame = tables["paired_metrics"].copy()
            frame.loc[frame.index[0], "reference_model_id"] = "ZZZ-NOT-A-REFERENCE"
            tables["paired_metrics"] = frame
            return tables

        world.record["hook"] = hook
        return "paired_key_mismatch"
    if kind == "bindings":

        def hook(tables):
            (world.stage / comparison.BINDINGS_NAME).write_bytes(b'{"tampered": true}')
            return tables

        world.record["hook"] = hook
        return "persisted_reference_bindings_changed"
    if kind == "aggregation_manifest":

        def hook(tables):
            (world.aggregation_stage / "extra.bin").write_bytes(b"extra")
            core._write_manifest(world.aggregation_stage)
            return tables

        world.record["hook"] = hook
        return "aggregation_manifest_changed"

    def hook(tables):
        changed = {**world.aggregation_receipt, "tampered": True}
        core._atomic_write(
            world.aggregation_receipt_path,
            core._canon().canonical_json_bytes(changed),
        )
        return tables

    world.record["hook"] = hook
    return "aggregation_receipt_changed"


@pytest.mark.parametrize(
    "kind",
    [
        "legacy_bindings",
        "legacy_failure",
        "comparison",
        "coverage_count",
        "coverage_bool",
        "pair_keys",
        "bindings",
        "aggregation_manifest",
        "aggregation_receipt",
    ],
)
def test_consumed_failures_record_and_keep_receipt_absent(world, kind):
    code = _consumed_variant(world, kind)
    with pytest.raises(comparison.P05ComprehensiveComparisonError):
        comparison.run_comparison(**world.run_kwargs())
    _assert_consumed_failure(world, code)


def test_table_write_failure_records_failure(world, monkeypatch):
    real_atomic = core._atomic_write

    def failing(path, payload):
        if str(path).endswith(".parquet"):
            raise core.P05CoreError("persistence_failed")
        return real_atomic(path, payload)

    monkeypatch.setattr(core, "_atomic_write", failing)
    with pytest.raises(core.P05CoreError):
        comparison.run_comparison(**world.run_kwargs())
    _assert_consumed_failure(world, "persistence_failed")


def test_storage_failure_records_failure(world, monkeypatch):
    def boom(self, headroom_bytes=0):
        raise storage.P05StorageError("storage_ceiling_exceeded")

    monkeypatch.setattr(freeze.StorageBudget, "check", boom)
    with pytest.raises(storage.P05StorageError):
        comparison.run_comparison(**world.run_kwargs())
    _assert_consumed_failure(world, "storage_ceiling_exceeded")


def test_deadline_failure_records_failure(world, monkeypatch):
    real_check = freeze._check_deadline

    def check(deadline):
        if world.stage.exists():
            raise core.P05CoreError("comparison_deadline_exceeded")
        return real_check(deadline)

    monkeypatch.setattr(freeze, "_check_deadline", check)
    with pytest.raises(core.P05CoreError):
        comparison.run_comparison(**world.run_kwargs())
    _assert_consumed_failure(world, "comparison_deadline_exceeded")


def test_post_reauth_failure_records_failure(world, monkeypatch):
    def boom(*args):
        raise comparison.P05ComprehensiveComparisonError("provenance_reauth_failed")

    monkeypatch.setattr(pilot, "_post_run_reauth", boom)
    with pytest.raises(comparison.P05ComprehensiveComparisonError):
        comparison.run_comparison(**world.run_kwargs())
    _assert_consumed_failure(world, "provenance_reauth_failed")


def test_changed_aggregation_file_detected(world):
    def hook(tables):
        (world.aggregation_stage / "ensemble_predictions.parquet").write_bytes(b"tampered")
        return tables

    world.record["hook"] = hook
    with pytest.raises(core.P05CoreError):
        comparison.run_comparison(**world.run_kwargs())
    summary = _summary(world)
    assert summary["status"] == "fail"
    assert not (world.run_root / comparison.RECEIPT_NAME).exists()


def test_late_failure_after_receipt_retains_failed_summary(world, monkeypatch):
    receipt_path = world.run_root / comparison.RECEIPT_NAME
    real_check = freeze._check_deadline

    def check(deadline):
        if receipt_path.exists():
            raise core.P05CoreError("comparison_deadline_exceeded")
        return real_check(deadline)

    monkeypatch.setattr(freeze, "_check_deadline", check)
    with pytest.raises(core.P05CoreError):
        comparison.run_comparison(**world.run_kwargs())
    assert receipt_path.exists()
    receipt = core._read_json(receipt_path, "receipt")
    summary = _summary(world)
    assert summary["status"] == "fail"
    assert summary["reason_code"] == "comparison_deadline_exceeded"
    assert receipt["stage_manifest_sha256"] != core._canon().sha256_file(
        world.stage / comparison.MANIFEST_NAME
    )


def test_failure_reason_is_path_free_and_no_retry(world):
    def hook(index, result):
        raise comparison.P05ComprehensiveComparisonError("legacy_load_failed")

    world.record["legacy_hook"] = hook
    with pytest.raises(comparison.P05ComprehensiveComparisonError):
        comparison.run_comparison(**world.run_kwargs())
    summary = _summary(world)
    assert summary["reason_code"] == "legacy_load_failed"
    assert "/" not in summary["reason_code"]
    assert "\\" not in summary["reason_code"]
    assert len(world.record["legacy"]) == 1
    assert not (world.run_root / comparison.RECEIPT_NAME).exists()


def test_partial_random_forest_reference(world):
    classical = world.fx["p03_predictions"]
    rf_rows = classical[classical["model_id"].eq(RANDOM_FOREST)]
    world.fx["p03_predictions"] = classical.drop(index=rf_rows.index[0]).reset_index(drop=True)

    receipt = comparison.run_comparison(**world.run_kwargs())
    counters = receipt["counters"]
    assert counters["rows_endpoint_metrics"] == 14
    assert counters["rows_paired_metrics"] == 34
    assert counters["rows_coverage"] == 8
    assert counters["complete_model_contexts"] == 7
    assert counters["incomplete_model_contexts"] == 1

    coverage = pd.read_parquet(world.stage / "coverage.parquet")
    assert int(coverage["complete"].sum()) == 7
    random_forest = coverage[coverage["model_id"].eq(RANDOM_FOREST)].iloc[0]
    assert bool(random_forest["complete"]) is False


def test_write_table_detects_same_size_corruption(tmp_path, monkeypatch):
    budget, run_dir = _direct_budget(tmp_path)
    stage = run_dir / "stage"
    stage.mkdir()
    frame = pd.DataFrame({"context_id": ["a", "b"], "complete": [True, False]})
    real_write = core._atomic_write

    def corrupted(path, data):
        return real_write(path, data[:-1] + bytes([data[-1] ^ 0xFF]))

    monkeypatch.setattr(core, "_atomic_write", corrupted)
    with pytest.raises(aggregation.P05ComprehensiveAggregationError) as info:
        aggregation._write_table(stage=stage, budget=budget, name="endpoint_metrics", frame=frame)
    assert info.value.reason_code == "table_bytes_mismatch"


@pytest.mark.parametrize("kind", ["existing", "symlink"])
def test_write_table_never_overwrites(tmp_path, kind):
    budget, run_dir = _direct_budget(tmp_path)
    stage = run_dir / "stage"
    stage.mkdir()
    path = stage / "endpoint_metrics.parquet"
    target = None
    if kind == "existing":
        path.write_bytes(b"existing-bytes")
    else:
        target = stage / "target.bin"
        target.write_bytes(b"target-bytes")
        path.symlink_to(target)
    frame = pd.DataFrame({"context_id": ["a"]})
    with pytest.raises((core.P05CoreError, storage.P05StorageError)):
        aggregation._write_table(stage=stage, budget=budget, name="endpoint_metrics", frame=frame)
    if kind == "existing":
        assert path.read_bytes() == b"existing-bytes"
    else:
        assert target.read_bytes() == b"target-bytes"
        assert path.is_symlink()
