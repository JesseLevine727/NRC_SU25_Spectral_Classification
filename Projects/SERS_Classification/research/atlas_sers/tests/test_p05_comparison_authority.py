"""CPU-only tests for the P05 comparison-authority gate.

These tests never read private scientific data, build a model, fit a
temperature or run an optimizer.  They reuse the real comprehensive-comparison
producer fixture and drive the real read-only comparison authority on top of a
completed comparison stage.  The fixture is adapted locally so the persisted
reference bindings use the real loader key/hash format and the evaluation
receipt and manifest exist on disk, because the authority checks both.
"""

from __future__ import annotations

# The optional torch import must precede torch-dependent project modules.
# ruff: noqa: E402
import time
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_aggregation_authority as aggregation_authority
from atlas_sers.evaluation import p05_comparison_authority as comparison_authority
from atlas_sers.evaluation import p05_comprehensive_comparison as comparison
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_legacy_references as legacy_references
from tests import test_p05_comprehensive_comparison as producer_fixtures

world = producer_fixtures.world
REAL_COMPARE = producer_fixtures.REAL_COMPARE
PRIOR_SECONDS = producer_fixtures.PRIOR_SECONDS
SOURCE_STEPS = producer_fixtures.SOURCE_STEPS
REFIT_STEPS = producer_fixtures.REFIT_STEPS
PLAN_ID = producer_fixtures.PLAN_ID


def _real_bindings(evaluation_receipt_sha256):
    p03_protected = legacy_references.P03_EXECUTION_ID + "0" * (
        64 - len(legacy_references.P03_EXECUTION_ID)
    )
    return {
        "p03_run_id": legacy_references.P03_RUN_ID,
        "p04_run_id": legacy_references.P04_RUN_ID,
        "p03_state_sha256": "a" * 64,
        "p04_state_sha256": "b" * 64,
        "p03_predictions_sha256": "c" * 64,
        "p04_predictions_sha256": "d" * 64,
        "p03_protected_state_sha256": p03_protected,
        "p04_shard_protected_state_sha256": legacy_references.P04_AGGREGATION_STATE_SHA256,
        "p04_execution_protected_state_sha256": legacy_references.P04_PROTECTED_STATE_SHA256,
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
    }


def _valid_auth(world):
    return {
        "prior_seconds": PRIOR_SECONDS,
        "aggregation_receipt": dict(world.aggregation_receipt),
        "evaluation_receipt": dict(world.evaluation_receipt),
        "plan": {"plan_id": PLAN_ID},
        "aggregation_tables": {
            "ensemble_predictions": world.fx["aggregate"]["ensemble_predictions"]
        },
        "source_optimizer_steps": SOURCE_STEPS,
        "refit_optimizer_steps": REFIT_STEPS,
    }


def _deadline():
    return time.perf_counter() + development.MAXIMUM_TOTAL_SECONDS


def _authenticate(world, deadline=None):
    return comparison_authority.authenticate_comparison(
        world.bundle, deadline=_deadline() if deadline is None else deadline
    )


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): core._canon().sha256_file(path)
        for path in Path(root).rglob("*")
        if path.is_file()
    }


def _expected_tables(world):
    return REAL_COMPARE(
        p05_ensemble=world.fx["aggregate"]["ensemble_predictions"],
        p04_ensemble=world.fx["p04_ensemble"],
        p03_predictions=world.fx["p03_predictions"],
        contexts=pd.DataFrame(world.fx["raw"]["contexts"]),
    )


def _reseal(world):
    core._write_manifest(world.stage)
    manifest_sha = core._canon().sha256_file(world.stage / comparison.MANIFEST_NAME)
    receipt_path = world.run_root / comparison.RECEIPT_NAME
    receipt = core._read_json(receipt_path, "receipt")
    receipt["stage_manifest_sha256"] = manifest_sha
    core._atomic_write(receipt_path, core._canon().canonical_json_bytes(receipt))


def _edit_receipt(world, mutate):
    path = world.run_root / comparison.RECEIPT_NAME
    payload = core._read_json(path, "receipt")
    mutate(payload)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))


def _edit_summary(world, mutate):
    path = world.stage / comparison.SUMMARY_NAME
    payload = core._read_json(path, "summary")
    mutate(payload)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    _reseal(world)


def _edit_table(world, name, mutate):
    path = world.stage / f"{name}.parquet"
    frame = pd.read_parquet(path)
    mutate(frame).to_parquet(path, index=False)
    _reseal(world)


def _edit_bindings(world, mutate):
    path = world.stage / comparison.BINDINGS_NAME
    payload = core._read_json(path, "bindings")
    mutate(payload)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    _reseal(world)


@pytest.fixture
def prepared(world, monkeypatch):
    record = world.record
    fx = world.fx

    evaluation_stage = world.run_root / evaluation.STAGE_NAME
    evaluation_stage.mkdir(parents=True, exist_ok=True)
    (evaluation_stage / "frozen_predictions.parquet").write_bytes(b"evaluation")
    core._write_manifest(evaluation_stage)
    evaluation_manifest_sha256 = core._canon().sha256_file(
        evaluation_stage / evaluation.MANIFEST_NAME
    )
    evaluation_receipt = {
        "status": "complete",
        "stage_manifest_sha256": evaluation_manifest_sha256,
    }
    evaluation_receipt_path = world.run_root / evaluation.RECEIPT_NAME
    core._atomic_write(
        evaluation_receipt_path,
        core._canon().canonical_json_bytes(evaluation_receipt),
    )
    evaluation_receipt_sha256 = core._canon().sha256_file(evaluation_receipt_path)
    bindings = _real_bindings(evaluation_receipt_sha256)

    def fake_auth(bundle_arg, deadline):
        record["order"].append("auth")
        record["auth"].append(deadline)
        record["stage_at_auth"] = (world.run_root / comparison.STAGE_NAME).exists()
        return {
            "prior_seconds": PRIOR_SECONDS,
            "aggregation_receipt": dict(world.aggregation_receipt),
            "evaluation_receipt": dict(evaluation_receipt),
            "plan": {"plan_id": PLAN_ID},
            "aggregation_tables": {
                "ensemble_predictions": fx["aggregate"]["ensemble_predictions"]
            },
            "source_optimizer_steps": SOURCE_STEPS,
            "refit_optimizer_steps": REFIT_STEPS,
        }

    monkeypatch.setattr(aggregation_authority, "authenticate_aggregation", fake_auth)

    def fake_load_references(bundle_arg, authenticated, deadline):
        index = len(record["legacy"])
        record["order"].append("legacy")
        record["legacy"].append(deadline)
        result = {
            "bindings": dict(bindings),
            "p04_ensemble": fx["p04_ensemble"],
            "p03_predictions": fx["p03_predictions"],
        }
        if record["legacy_hook"] is not None:
            result = record["legacy_hook"](index, result)
        return result

    monkeypatch.setattr(legacy_references, "load_references", fake_load_references)

    world.bindings = bindings
    world.evaluation_receipt = evaluation_receipt
    world.evaluation_receipt_path = evaluation_receipt_path
    world.evaluation_receipt_sha256 = evaluation_receipt_sha256
    world.evaluation_stage = evaluation_stage
    return world


@pytest.fixture
def completed(prepared):
    comparison.run_comparison(**prepared.run_kwargs())
    return prepared


def test_success_roundtrip_read_only(completed):
    before = _snapshot(completed.artifact_root)
    result = _authenticate(completed)

    assert result["comparison_receipt"]["status"] == "complete"
    assert result["aggregation_prior_seconds"] == PRIOR_SECONDS
    assert result["reference_bindings"] == completed.bindings

    expected = _expected_tables(completed)
    assert set(result["comparison_tables"]) == set(comparison.TABLE_NAMES)
    for name in comparison.TABLE_NAMES:
        pd.testing.assert_frame_equal(
            result["comparison_tables"][name].reset_index(drop=True),
            expected[name].reset_index(drop=True),
            check_exact=True,
        )
    assert _snapshot(completed.artifact_root) == before


def test_deadline_int_accepted(completed):
    deadline = int(time.perf_counter() + development.MAXIMUM_TOTAL_SECONDS)
    result = _authenticate(completed, deadline=deadline)
    assert result["comparison_receipt"]["status"] == "complete"


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("x", id="string"),
        pytest.param(True, id="bool"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_deadline_invalid_rejected(completed, bad):
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        comparison_authority.authenticate_comparison(completed.bundle, deadline=bad)
    assert info.value.reason_code == "deadline_malformed"


def test_deadline_expired_rejected(completed):
    with pytest.raises(core.P05CoreError):
        comparison_authority.authenticate_comparison(completed.bundle, deadline=0.0)


def test_upstream_refusal_before_legacy_reads(completed, monkeypatch):
    calls = []

    def deny(bundle, deadline):
        calls.append(1)
        raise aggregation_authority.P05AggregationAuthorityError("authority_denied")

    monkeypatch.setattr(aggregation_authority, "authenticate_aggregation", deny)
    before = list(completed.record["legacy"])
    with pytest.raises(aggregation_authority.P05AggregationAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == "authority_denied"
    assert calls == [1]
    assert completed.record["legacy"] == before


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda auth: None, "authority_malformed"),
        (lambda auth: {**auth, "plan": "bad"}, "authority_plan_malformed"),
        (lambda auth: {**auth, "plan": {}}, "authority_plan_id_malformed"),
        (lambda auth: {**auth, "aggregation_tables": "bad"}, "authority_tables_malformed"),
        (lambda auth: {**auth, "aggregation_tables": {}}, "authority_ensemble_missing"),
        (
            lambda auth: {**auth, "aggregation_receipt": "bad"},
            "authority_aggregation_receipt_malformed",
        ),
        (
            lambda auth: {**auth, "evaluation_receipt": "bad"},
            "authority_evaluation_receipt_malformed",
        ),
        (
            lambda auth: {**auth, "source_optimizer_steps": True},
            "authority_source_steps_malformed",
        ),
        (
            lambda auth: {**auth, "refit_optimizer_steps": True},
            "authority_refit_steps_malformed",
        ),
        (lambda auth: {**auth, "prior_seconds": True}, "authority_prior_malformed"),
        (
            lambda auth: {
                **auth,
                "prior_seconds": development.PRELAUNCH_AUDIT_RESERVE_SECONDS - 1.0,
            },
            "authority_prior_out_of_range",
        ),
        (
            lambda auth: {**auth, "prior_seconds": development.MAXIMUM_TOTAL_SECONDS + 1.0},
            "authority_prior_out_of_range",
        ),
    ],
    ids=[
        "not-mapping",
        "plan-bad",
        "plan-id-missing",
        "tables-bad",
        "ensemble-missing",
        "aggregation-receipt-bad",
        "evaluation-receipt-bad",
        "source-steps-bool",
        "refit-steps-bool",
        "prior-bool",
        "prior-below-reserve",
        "prior-above-maximum",
    ],
)
def test_authority_shape_rejected(completed, monkeypatch, mutate, code):
    monkeypatch.setattr(
        aggregation_authority,
        "authenticate_aggregation",
        lambda bundle, deadline: mutate(_valid_auth(completed)),
    )
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == code


def test_prior_comes_from_aggregation_cumulative(completed, monkeypatch):
    monkeypatch.setattr(
        aggregation_authority,
        "authenticate_aggregation",
        lambda bundle, deadline: {**_valid_auth(completed), "prior_seconds": PRIOR_SECONDS + 1.0},
    )
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == "receipt_prior_mismatch"


def _t_status_fail(payload):
    payload["status"] = "fail"


def _t_complete_false(payload):
    payload["comparison_complete"] = False


def _t_complete_truthy(payload):
    payload["comparison_complete"] = "yes"


_FLAG_CASES = [
    ("receipt", _t_status_fail, "receipt_status_incomplete"),
    ("summary", _t_status_fail, "summary_status_incomplete"),
    ("receipt", _t_complete_false, "receipt_incomplete"),
    ("receipt", _t_complete_truthy, "receipt_incomplete"),
    ("summary", _t_complete_false, "summary_incomplete"),
    ("summary", _t_complete_truthy, "summary_incomplete"),
]


@pytest.mark.parametrize(
    "target,mutate,code", _FLAG_CASES, ids=[f"{t}-{m.__name__}" for t, m, _ in _FLAG_CASES]
)
def test_status_flags_rejected(completed, target, mutate, code):
    if target == "receipt":
        _edit_receipt(completed, mutate)
    else:
        _edit_summary(completed, mutate)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == code


def _t_permit(payload):
    payload["permit_sha256"] = "0" * 64


def _t_contract(payload):
    payload["core_contract_sha256"] = "0" * 64


def _t_plan(payload):
    payload["selection_plan_id"] = "other"


def _t_ledger(payload):
    payload["ledger_id"] = "other"


def _t_schema(payload):
    payload["schema_version"] = "other"


def _t_protocol(payload):
    payload["protocol_version"] = "other"


def _t_command(payload):
    payload["command"] = "other"


def _t_stage(payload):
    payload["stage"] = "other"


def _t_agg_digest(payload):
    payload["aggregation_receipt_sha256"] = "0" * 64


def _t_binding_digest(payload):
    payload["reference_binding_digest"] = "0" * 64


def _t_source_float(payload):
    payload["source_optimizer_steps"] = float(payload["source_optimizer_steps"])


def _t_source_bool(payload):
    payload["source_optimizer_steps"] = True


def _t_source_string(payload):
    payload["source_optimizer_steps"] = str(payload["source_optimizer_steps"])


def _t_refit_float(payload):
    payload["refit_optimizer_steps"] = float(payload["refit_optimizer_steps"])


_IDENTITY_CASES = [
    ("receipt", _t_permit, "receipt_permit_sha256_mismatch"),
    ("summary", _t_permit, "summary_permit_sha256_mismatch"),
    ("receipt", _t_contract, "receipt_core_contract_sha256_mismatch"),
    ("summary", _t_ledger, "summary_ledger_id_mismatch"),
    ("receipt", _t_plan, "receipt_selection_plan_id_mismatch"),
    ("summary", _t_schema, "summary_schema_version_mismatch"),
    ("receipt", _t_protocol, "receipt_protocol_version_mismatch"),
    ("summary", _t_command, "summary_command_mismatch"),
    ("receipt", _t_stage, "receipt_stage_mismatch"),
    ("receipt", _t_agg_digest, "receipt_aggregation_receipt_sha256_mismatch"),
    ("summary", _t_binding_digest, "summary_reference_binding_digest_mismatch"),
    ("receipt", _t_source_float, "receipt_source_optimizer_steps_invalid"),
    ("summary", _t_refit_float, "summary_refit_optimizer_steps_invalid"),
    ("receipt", _t_source_bool, "receipt_source_optimizer_steps_mismatch"),
    ("summary", _t_source_string, "summary_source_optimizer_steps_mismatch"),
]


@pytest.mark.parametrize(
    "target,mutate,code",
    _IDENTITY_CASES,
    ids=[f"{t}-{m.__name__}" for t, m, _ in _IDENTITY_CASES],
)
def test_identity_tamper_rejected(completed, target, mutate, code):
    if target == "receipt":
        _edit_receipt(completed, mutate)
    else:
        _edit_summary(completed, mutate)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == code


def _t_fits_nonzero(payload):
    payload["counters"]["fits"] = 1


def _t_fits_bool(payload):
    payload["counters"]["fits"] = True


def _t_elapsed_bool(payload):
    payload["counters"]["elapsed_seconds"] = True


def _t_missing_counter(payload):
    del payload["counters"]["fits"]


def _t_extra_counter(payload):
    payload["counters"]["extra"] = 0


def _t_rows_wrong(payload):
    payload["counters"]["rows_coverage"] = int(payload["counters"]["rows_coverage"]) + 1


def _t_rows_float(payload):
    payload["counters"]["rows_coverage"] = float(payload["counters"]["rows_coverage"])


_COUNTER_CASES = [
    ("receipt", _t_fits_nonzero, "receipt_fits_nonzero"),
    ("summary", _t_fits_nonzero, "summary_fits_nonzero"),
    ("receipt", _t_fits_bool, "receipt_fits_invalid"),
    ("summary", _t_fits_bool, "summary_fits_invalid"),
    ("receipt", _t_elapsed_bool, "receipt_elapsed_malformed"),
    ("summary", _t_elapsed_bool, "summary_elapsed_malformed"),
    ("receipt", _t_missing_counter, "receipt_counter_fields_mismatch"),
    ("summary", _t_extra_counter, "summary_counter_fields_mismatch"),
    ("receipt", _t_rows_wrong, "receipt_rows_coverage_mismatch"),
    ("summary", _t_rows_wrong, "summary_rows_coverage_mismatch"),
    ("receipt", _t_rows_float, "receipt_rows_coverage_invalid"),
]


@pytest.mark.parametrize(
    "target,mutate,code", _COUNTER_CASES, ids=[f"{t}-{m.__name__}" for t, m, _ in _COUNTER_CASES]
)
def test_counter_tamper_rejected(completed, target, mutate, code):
    if target == "receipt":
        _edit_receipt(completed, mutate)
    else:
        _edit_summary(completed, mutate)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == code


def _t_prior(payload):
    payload["prior_scientific_seconds_cumulative_bound"] = PRIOR_SECONDS - 1.0


def _t_stage_seconds(payload):
    payload["scientific_seconds_this_stage"] = float(payload["counters"]["elapsed_seconds"]) + 1.0


def _t_cumulative(payload):
    payload["scientific_seconds_cumulative_bound"] = PRIOR_SECONDS + 1.0


def _t_cumulative_overrun(payload):
    overrun = development.MAXIMUM_TOTAL_SECONDS + 1.0 - PRIOR_SECONDS
    payload["scientific_seconds_this_stage"] = overrun
    payload["counters"]["elapsed_seconds"] = overrun
    payload["scientific_seconds_cumulative_bound"] = development.MAXIMUM_TOTAL_SECONDS + 1.0


def _t_reserve(payload):
    payload["prelaunch_audit_reserve_seconds"] = development.PRELAUNCH_AUDIT_RESERVE_SECONDS + 1.0


def _t_maximum(payload):
    payload["maximum_total_seconds"] = development.MAXIMUM_TOTAL_SECONDS + 1.0


def _t_summary_elapsed_over(payload):
    payload["counters"]["elapsed_seconds"] = 10**9


_ELAPSED_CASES = [
    ("receipt", _t_prior, "receipt_prior_mismatch"),
    ("summary", _t_prior, "summary_prior_mismatch"),
    ("receipt", _t_stage_seconds, "receipt_stage_seconds_inconsistent"),
    ("summary", _t_stage_seconds, "summary_stage_seconds_inconsistent"),
    ("receipt", _t_cumulative, "receipt_cumulative_mismatch"),
    ("summary", _t_cumulative, "summary_cumulative_mismatch"),
    ("receipt", _t_cumulative_overrun, "receipt_cumulative_out_of_range"),
    ("receipt", _t_reserve, "receipt_reserve_mismatch"),
    ("summary", _t_maximum, "summary_maximum_mismatch"),
    ("summary", _t_summary_elapsed_over, "summary_elapsed_exceeds_receipt"),
]


@pytest.mark.parametrize(
    "target,mutate,code", _ELAPSED_CASES, ids=[f"{t}-{m.__name__}" for t, m, _ in _ELAPSED_CASES]
)
def test_elapsed_tamper_rejected(completed, target, mutate, code):
    if target == "receipt":
        _edit_receipt(completed, mutate)
    else:
        _edit_summary(completed, mutate)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == code


def _t_float_limits(payload):
    payload["prelaunch_audit_reserve_seconds"] = float(payload["prelaunch_audit_reserve_seconds"])
    payload["maximum_total_seconds"] = float(payload["maximum_total_seconds"])


def test_float_limits_accepted(completed):
    _edit_summary(completed, _t_float_limits)
    result = _authenticate(completed)
    assert result["comparison_receipt"]["status"] == "complete"


def _mutate_numeric(frame):
    frame = frame.copy()
    target = next(
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    )
    frame.loc[0, target] = float(frame.loc[0, target]) + 1.0
    return frame


def _mutate_dtype(frame):
    frame = frame.copy()
    target = next(column for column in frame.columns if pd.api.types.is_bool_dtype(frame[column]))
    frame[target] = frame[target].astype(int)
    return frame


def _reorder_columns(frame):
    return frame[list(reversed(list(frame.columns)))].reset_index(drop=True)


def _reverse_rows(frame):
    return frame.iloc[::-1].reset_index(drop=True)


def test_missing_table_rejected(completed):
    (completed.stage / "endpoint_metrics.parquet").unlink()
    _reseal(completed)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == "comparison_inventory_mismatch"


def test_extra_file_rejected(completed):
    (completed.stage / "extra.bin").write_bytes(b"extra")
    _reseal(completed)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == "comparison_inventory_mismatch"


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("endpoint_metrics", _mutate_numeric),
        ("coverage", _mutate_dtype),
        ("paired_metrics", _reorder_columns),
        ("coverage", _reverse_rows),
    ],
    ids=["numeric", "dtype", "column-order", "row-order"],
)
def test_resealed_table_tamper_rejected(completed, name, mutate):
    _edit_table(completed, name, mutate)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == f"table_{name}_mismatch"


def _b_p03_run(payload):
    payload["p03_run_id"] = "P03-wrong"


def _b_p04_run(payload):
    payload["p04_run_id"] = "P04-wrong"


def _b_p03_protected(payload):
    payload["p03_protected_state_sha256"] = "f" * 64


def _b_p04_shard(payload):
    payload["p04_shard_protected_state_sha256"] = "f" * 64


def _b_p04_execution(payload):
    payload["p04_execution_protected_state_sha256"] = "f" * 64


def _b_p03_state_nonhex(payload):
    payload["p03_state_sha256"] = "not-hex"


def _b_eval_receipt(payload):
    payload["evaluation_receipt_sha256"] = "f" * 64


def _b_p03_state_changed(payload):
    payload["p03_state_sha256"] = "f" * 64


_BINDING_CASES = [
    (_b_p03_run, "binding_p03_run_mismatch"),
    (_b_p04_run, "binding_p04_run_mismatch"),
    (_b_p03_protected, "binding_p03_protected_mismatch"),
    (_b_p04_shard, "binding_p04_shard_protected_mismatch"),
    (_b_p04_execution, "binding_p04_execution_protected_mismatch"),
    (_b_p03_state_nonhex, "binding_p03_state_sha256_malformed"),
    (_b_eval_receipt, "binding_evaluation_receipt_mismatch"),
    (_b_p03_state_changed, "receipt_reference_binding_digest_mismatch"),
]


@pytest.mark.parametrize(
    "mutate,code", _BINDING_CASES, ids=[m.__name__ for m, _ in _BINDING_CASES]
)
def test_binding_tamper_rejected(completed, mutate, code):
    _edit_bindings(completed, mutate)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == code


def test_loader_changed_references_rejected(completed):
    def hook(index, result):
        if index == 2:
            return {
                **result,
                "bindings": {**result["bindings"], "p03_state_sha256": "f" * 64},
            }
        return result

    completed.record["legacy_hook"] = hook
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == "reference_bindings_changed"


def test_stage_symlink_rejected(completed, tmp_path):
    target = tmp_path / "stage_target"
    completed.stage.rename(target)
    completed.stage.symlink_to(target, target_is_directory=True)
    with pytest.raises(core.P05CoreError):
        _authenticate(completed)


def test_stage_symlink_file_rejected(completed):
    (completed.stage / "link.bin").symlink_to(completed.stage / comparison.SUMMARY_NAME)
    with pytest.raises(core.P05CoreError):
        _authenticate(completed)


def test_stage_subdirectory_rejected(completed):
    (completed.stage / "subdir").mkdir()
    with pytest.raises(core.P05CoreError):
        _authenticate(completed)


def test_late_producer_failure_after_receipt_rejected(prepared, monkeypatch):
    receipt_path = prepared.run_root / comparison.RECEIPT_NAME
    real_check = freeze._check_deadline

    def check(deadline):
        if receipt_path.exists():
            raise core.P05CoreError("comparison_deadline_exceeded")
        return real_check(deadline)

    monkeypatch.setattr(freeze, "_check_deadline", check)
    with pytest.raises(core.P05CoreError):
        comparison.run_comparison(**prepared.run_kwargs())
    assert receipt_path.exists()

    monkeypatch.setattr(freeze, "_check_deadline", real_check)
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(prepared)
    assert info.value.reason_code == "comparison_manifest_sha256_mismatch"


def _mut_comparison_receipt(world):
    core._atomic_write(world.run_root / comparison.RECEIPT_NAME, b'{"tampered": true}')


def _mut_comparison_manifest(world):
    core._atomic_write(world.stage / comparison.MANIFEST_NAME, b'{"tampered": true}')


def _mut_aggregation_receipt(world):
    core._atomic_write(world.aggregation_receipt_path, b'{"tampered": true}')


def _mut_aggregation_manifest(world):
    core._atomic_write(
        world.aggregation_stage / comparison.AGGREGATION_MANIFEST_NAME,
        b'{"tampered": true}',
    )


def _mut_evaluation_receipt(world):
    core._atomic_write(world.evaluation_receipt_path, b'{"tampered": true}')


def _mut_evaluation_manifest(world):
    core._atomic_write(
        world.evaluation_stage / evaluation.MANIFEST_NAME, b'{"tampered": true}'
    )


def _mut_bindings(world):
    core._atomic_write(world.stage / comparison.BINDINGS_NAME, b'{"tampered": true}')


_READ_MUTATIONS = [
    (_mut_comparison_receipt, "comparison_receipt_changed"),
    (_mut_comparison_manifest, "comparison_manifest_changed"),
    (_mut_aggregation_receipt, "aggregation_receipt_changed"),
    (_mut_aggregation_manifest, "aggregation_manifest_changed"),
    (_mut_evaluation_receipt, "evaluation_receipt_changed"),
    (_mut_evaluation_manifest, "evaluation_manifest_changed"),
    (_mut_bindings, "comparison_bindings_changed"),
]


@pytest.mark.parametrize(
    "mutate,code", _READ_MUTATIONS, ids=[m.__name__ for m, _ in _READ_MUTATIONS]
)
def test_mutation_during_reads_detected(completed, mutate, code):
    def hook(tables):
        mutate(completed)
        return tables

    completed.record["hook"] = hook
    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == code
