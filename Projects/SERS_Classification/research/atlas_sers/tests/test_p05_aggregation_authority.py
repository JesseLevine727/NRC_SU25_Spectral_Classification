"""CPU-only tests for the P05 aggregation-authority gate.

The shared ``world`` fixture drives the real aggregation producer, the real
pure-metrics kernel, parquet persistence, ``StorageBudget`` and manifests,
while the frozen prerequisite authority, outer inputs, provenance and
``inputs.prepare`` stay stubbed.  These tests add the real aggregation
authority on top and never read private data, build a model or fit anything.
"""

from __future__ import annotations

# The optional torch import must precede torch-dependent project modules.
# ruff: noqa: E402
import time

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_aggregation_authority as authority
from atlas_sers.evaluation import p05_comprehensive_aggregation as aggregation
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_frozen_predictions as frozen
from tests import test_p05_comprehensive_aggregation as aggregation_fixtures

world = aggregation_fixtures.world
PRIOR_SECONDS = aggregation_fixtures.PRIOR_SECONDS


def _deadline() -> float:
    return time.perf_counter() + development.MAXIMUM_TOTAL_SECONDS


def _authenticate(world, deadline=None):
    return authority.authenticate_aggregation(
        world.bundle, deadline=_deadline() if deadline is None else deadline
    )


def _expect_rejected(world, code):
    with pytest.raises(authority.P05AggregationAuthorityError) as info:
        _authenticate(world)
    if isinstance(code, tuple):
        assert info.value.reason_code in code
    else:
        assert info.value.reason_code == code


@pytest.fixture
def completed(world):
    aggregation.run_aggregation(**world.run_kwargs())
    return world


def _reseal(world):
    core._write_manifest(world.stage)
    manifest_sha = core._canon().sha256_file(world.stage / aggregation.MANIFEST_NAME)
    receipt_path = world.run_root / aggregation.RECEIPT_NAME
    receipt = core._read_json(receipt_path, "receipt")
    receipt["stage_manifest_sha256"] = manifest_sha
    core._atomic_write(receipt_path, core._canon().canonical_json_bytes(receipt))


def _edit_receipt(world, mutate):
    path = world.run_root / aggregation.RECEIPT_NAME
    payload = core._read_json(path, "receipt")
    mutate(payload)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))


def _edit_summary(world, mutate):
    path = world.stage / aggregation.SUMMARY_NAME
    payload = core._read_json(path, "summary")
    mutate(payload)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))
    _reseal(world)


def _edit_table(world, name, mutate):
    path = world.stage / f"{name}.parquet"
    frame = pd.read_parquet(path)
    mutate(frame).to_parquet(path, index=False)
    _reseal(world)


def _t_status_fail(payload):
    payload["status"] = "fail"


def _t_complete_false(payload):
    payload["aggregation_complete"] = False


def _t_complete_truthy(payload):
    payload["aggregation_complete"] = "yes"


def _t_permit(payload):
    payload["permit_sha256"] = "0" * 64


def _t_context_bool(payload):
    payload["context_count"] = True


def _t_source_steps(payload):
    payload["source_optimizer_steps"] = int(payload["source_optimizer_steps"]) + 1


def _t_fits_nonzero(payload):
    payload["counters"]["fits"] = 1


def _t_fits_bool(payload):
    payload["counters"]["fits"] = False


def _t_rows_float(payload):
    payload["counters"]["rows_seed_predictions"] = 1.0


def _t_prior(payload):
    payload["prior_scientific_seconds_cumulative_bound"] = PRIOR_SECONDS - 1.0


def _t_cumulative(payload):
    payload["scientific_seconds_cumulative_bound"] = 1.0


def _t_stage_seconds(payload):
    payload["scientific_seconds_this_stage"] = float(payload["counters"]["elapsed_seconds"]) + 1.0


def _t_overrun(payload):
    payload["counters"]["elapsed_seconds"] = 10**9
    payload["scientific_seconds_this_stage"] = 10**9
    payload["scientific_seconds_cumulative_bound"] = PRIOR_SECONDS + 10**9


def _t_maximum(payload):
    payload["maximum_total_seconds"] = development.MAXIMUM_TOTAL_SECONDS + 1.0


def _t_reserve(payload):
    payload["prelaunch_audit_reserve_seconds"] = development.PRELAUNCH_AUDIT_RESERVE_SECONDS + 1.0


def _t_missing_counter(payload):
    del payload["counters"]["fits"]


def _t_extra_counter(payload):
    payload["counters"]["extra"] = 0


_TAMPER_CASES = (
    ("receipt", _t_status_fail, "receipt_status_incomplete"),
    ("summary", _t_status_fail, "summary_status_incomplete"),
    ("receipt", _t_complete_truthy, "receipt_incomplete"),
    ("summary", _t_complete_false, "summary_incomplete"),
    ("receipt", _t_permit, "receipt_permit_sha256_mismatch"),
    (
        "summary",
        _t_context_bool,
        ("summary_context_count_mismatch", "summary_context_count_invalid"),
    ),
    ("receipt", _t_source_steps, "receipt_source_optimizer_steps_mismatch"),
    ("receipt", _t_fits_nonzero, "receipt_fits_nonzero"),
    ("summary", _t_fits_bool, "summary_fits_invalid"),
    ("receipt", _t_rows_float, "receipt_rows_seed_predictions_invalid"),
    ("receipt", _t_prior, "receipt_prior_mismatch"),
    ("summary", _t_cumulative, "summary_cumulative_mismatch"),
    ("receipt", _t_stage_seconds, "receipt_stage_seconds_inconsistent"),
    ("receipt", _t_overrun, "receipt_cumulative_out_of_range"),
    ("receipt", _t_maximum, "receipt_maximum_mismatch"),
    ("summary", _t_reserve, "summary_reserve_mismatch"),
    ("receipt", _t_missing_counter, "receipt_counter_fields_mismatch"),
    ("summary", _t_extra_counter, "summary_counter_fields_mismatch"),
)


def test_success_authority_roundtrip(completed):
    receipt = core._read_json(completed.run_root / aggregation.RECEIPT_NAME, "receipt")
    before = {
        path.relative_to(completed.artifact_root).as_posix(): core._canon().sha256_file(path)
        for path in completed.artifact_root.rglob("*")
        if path.is_file()
    }

    result = _authenticate(completed)

    assert result["prior_seconds"] == receipt["scientific_seconds_cumulative_bound"]
    assert result["evaluation_prior_seconds"] == PRIOR_SECONDS
    assert result["aggregation_receipt"] == receipt
    assert result["aggregation_receipt"]["status"] == "complete"

    expected = aggregation_fixtures._expected_tables(completed.fx)
    assert set(result["aggregation_tables"]) == set(aggregation.TABLE_NAMES)
    for name in aggregation.TABLE_NAMES:
        pd.testing.assert_frame_equal(
            result["aggregation_tables"][name].reset_index(drop=True),
            expected[name].reset_index(drop=True),
            check_exact=True,
        )

    assert {
        path.relative_to(completed.artifact_root).as_posix(): core._canon().sha256_file(path)
        for path in completed.artifact_root.rglob("*")
        if path.is_file()
    } == before


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("not-a-number", id="string"),
        pytest.param(True, id="bool"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_deadline_invalid_rejected(completed, bad):
    with pytest.raises(authority.P05AggregationAuthorityError) as info:
        authority.authenticate_aggregation(completed.bundle, deadline=bad)
    assert info.value.reason_code == "deadline_malformed"


def test_deadline_expired_rejected(completed):
    with pytest.raises(core.P05CoreError):
        authority.authenticate_aggregation(completed.bundle, deadline=0.0)


@pytest.mark.parametrize(
    "target,mutate,code",
    _TAMPER_CASES,
    ids=[f"{target}-{mutate.__name__}" for target, mutate, _ in _TAMPER_CASES],
)
def test_field_tamper_rejected(completed, target, mutate, code):
    if target == "receipt":
        _edit_receipt(completed, mutate)
    else:
        _edit_summary(completed, mutate)
    _expect_rejected(completed, code)


def _mutate_probabilities(frame):
    frame = frame.copy()
    probability_columns = [c for c in frame.columns if str(c).startswith("probability_")]
    if len(probability_columns) >= 2:
        first, second = probability_columns[0], probability_columns[1]
        frame.loc[0, first] = float(frame.loc[0, first]) + 0.01
        frame.loc[0, second] = float(frame.loc[0, second]) - 0.01
        return frame
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    frame.loc[0, numeric[0]] = float(frame.loc[0, numeric[0]]) + 1.0
    return frame


def _mutate_metric(frame):
    frame = frame.copy()
    target = None
    for column in frame.columns:
        text = str(column).lower()
        if "balanced" in text or "accuracy" in text:
            target = column
            break
    if target is None:
        target = next(c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]))
    frame.loc[0, target] = float(frame.loc[0, target]) + 1.0
    return frame


def _reorder_columns(frame):
    return frame[list(reversed(list(frame.columns)))].reset_index(drop=True)


def _drop_row(frame):
    return frame.iloc[:-1].reset_index(drop=True)


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("seed_predictions", _mutate_probabilities),
        ("master_metrics", _mutate_metric),
        ("ensemble_predictions", _reorder_columns),
        ("spectrum_metrics", _drop_row),
    ],
    ids=["probabilities_sum_one", "metric_ba", "reordered_columns", "dropped_row"],
)
def test_table_semantic_tamper_resealed(completed, name, mutate):
    _edit_table(completed, name, mutate)
    _expect_rejected(completed, f"table_{name}_mismatch")


def test_inventory_missing_file_rejected(completed):
    (completed.stage / aggregation.PROVENANCE_BEFORE_NAME).unlink()
    _reseal(completed)
    _expect_rejected(completed, "aggregation_inventory_mismatch")


def test_inventory_extra_file_rejected(completed):
    (completed.stage / "extra.bin").write_bytes(b"extra")
    _reseal(completed)
    _expect_rejected(completed, "aggregation_inventory_mismatch")


def test_inventory_subdir_rejected(completed):
    (completed.stage / "subdir").mkdir()
    with pytest.raises(core.P05CoreError):
        _authenticate(completed)


def test_inventory_symlink_rejected(completed):
    (completed.stage / "link.bin").symlink_to(completed.stage / aggregation.PROVENANCE_BEFORE_NAME)
    with pytest.raises(core.P05CoreError):
        _authenticate(completed)


def test_unsealed_table_edit_rejected(completed):
    path = completed.stage / "coverage.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[:-1].reset_index(drop=True).to_parquet(path, index=False)
    with pytest.raises(core.P05CoreError):
        _authenticate(completed)


def test_auth_failure_prevents_stage_read(completed, monkeypatch):
    calls = []

    def deny(bundle, deadline):
        calls.append(1)
        raise authority.P05AggregationAuthorityError("authority_denied")

    monkeypatch.setattr(frozen, "authenticate_predictions", deny)
    before = aggregation_fixtures._input_hashes(completed)
    with pytest.raises(authority.P05AggregationAuthorityError) as info:
        _authenticate(completed)
    assert info.value.reason_code == "authority_denied"
    assert calls == [1]
    assert aggregation_fixtures._input_hashes(completed) == before


def test_aggregation_failure_without_receipt_rejected(world):
    def boom(tables):
        raise aggregation.P05ComprehensiveAggregationError("late_aggregation_failure")

    world.record["hook"] = boom
    with pytest.raises(aggregation.P05ComprehensiveAggregationError):
        aggregation.run_aggregation(**world.run_kwargs())
    world.record["hook"] = None

    assert world.stage.is_dir()
    summary = core._read_json(world.stage / aggregation.SUMMARY_NAME, "summary")
    assert summary["status"] == "fail"
    assert not (world.run_root / aggregation.RECEIPT_NAME).exists()

    with pytest.raises(authority.P05AggregationAuthorityError) as info:
        _authenticate(world)
    assert info.value.reason_code == "aggregation_receipt_missing"


def test_producer_late_failure_with_retained_receipt_cannot_authenticate(world, monkeypatch):
    receipt_path = world.run_root / aggregation.RECEIPT_NAME
    original_check = aggregation.freeze.StorageBudget.check

    def fail_after_receipt(budget, **kwargs):
        original_check(budget, **kwargs)
        if receipt_path.exists():
            raise aggregation.P05ComprehensiveAggregationError("late_budget_failure")

    monkeypatch.setattr(aggregation.freeze.StorageBudget, "check", fail_after_receipt)
    with pytest.raises(core.P05CoreError):
        aggregation.run_aggregation(**world.run_kwargs())
    assert receipt_path.is_file()
    assert core._read_json(world.stage / aggregation.SUMMARY_NAME, "summary")["status"] == "fail"
    with pytest.raises(core.P05CoreError):
        _authenticate(world)


def _mutate_agg_receipt(world):
    path = world.run_root / aggregation.RECEIPT_NAME
    payload = core._read_json(path, "receipt")
    payload["status"] = "fail"
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))


def _mutate_agg_manifest(world):
    path = world.stage / aggregation.MANIFEST_NAME
    core._atomic_write(path, b'{"tampered": true}')


def _mutate_agg_table(world):
    path = world.stage / "coverage.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[:-1].reset_index(drop=True).to_parquet(path, index=False)


def _mutate_evaluation_receipt(world):
    changed = {**world.evaluation_receipt, "tampered": True}
    core._atomic_write(world.receipt_path, core._canon().canonical_json_bytes(changed))


def _mutate_evaluation_manifest(world):
    path = world.evaluation_stage / aggregation.MANIFEST_NAME
    core._atomic_write(path, b'{"tampered": true}')


@pytest.mark.parametrize(
    "mutate,code",
    [
        (_mutate_agg_receipt, "aggregation_receipt_changed"),
        (_mutate_agg_manifest, "aggregation_manifest_changed"),
        (_mutate_agg_table, None),
        (_mutate_evaluation_receipt, "evaluation_receipt_changed"),
        (_mutate_evaluation_manifest, "evaluation_manifest_changed"),
    ],
    ids=["agg_receipt", "agg_manifest", "agg_table", "eval_receipt", "eval_manifest"],
)
def test_mutation_during_recompute_rejected(completed, mutate, code):
    def hook(tables):
        mutate(completed)
        return tables

    completed.record["hook"] = hook
    if code is None:
        with pytest.raises(core.P05CoreError):
            _authenticate(completed)
    else:
        _expect_rejected(completed, code)
