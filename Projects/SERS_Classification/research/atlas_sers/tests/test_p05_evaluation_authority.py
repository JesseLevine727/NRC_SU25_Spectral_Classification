"""Bounded authority tests for the P05 comprehensive refit evaluation stage."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
# Optional torch is needed by the checkpoint loader imported by the authority.
# ruff: noqa: E402

from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as module
from atlas_sers.evaluation import p05_prediction as prediction
from atlas_sers.evaluation import p05_refit_authority as authority

EPOCHS, STEPS, ELAPSED, PEAK = 30, 120, 0.1, 0
PERMIT, CONTRACT = "a" * 64, "b" * 64
PLAN_ID, LEDGER_ID, CONTEXT, ROLE, SEED = "plan-1", "ledger-1", "ctx-0", "role-0", 0
PRIOR, SOURCE_STEPS, STAGE_ELAPSED, SUMMARY_ELAPSED, SUM_FIT = 4000.0, 96, 2.0, 1.0, 0.2
FUTURE = 1e12
RECIPES = (("D0-M", "r0"), ("D3", "r3"))


def _h(value):
    return f"{value:064x}"


SHARED_BACKBONE = _h(7)
INITIAL = {"D0-M": _h(11), "D3": _h(22)}
TERMINAL = {"D0-M": _h(33), "D3": _h(44)}


def _history():
    return [
        {
            "epoch": i,
            "optimizer_steps": 4,
            "sampling_digest": _h(i),
            "augmentation_digest": _h(i + 1000),
            "pair_digest": _h(i + 2000),
        }
        for i in range(EPOCHS)
    ]


def _spec(recipe, refit_id):
    return {
        "refit_id": refit_id,
        "context_id": CONTEXT,
        "seed": SEED,
        "recipe_id": recipe,
        "fitting_role_id": ROLE,
        "epochs": EPOCHS,
    }


def _summary(spec, history):
    recipe = spec["recipe_id"]
    return {
        "refit_id": spec["refit_id"],
        "context_id": CONTEXT,
        "seed": SEED,
        "recipe_id": recipe,
        "fitting_role_id": ROLE,
        "epochs": EPOCHS,
        "recipe": recipe,
        "role_id": ROLE,
        "status": "complete",
        "optimizer_steps": STEPS,
        "elapsed_seconds": ELAPSED,
        "peak_cuda_bytes": PEAK,
        "initial_state_digest": INITIAL[recipe],
        "initial_backbone_digest": SHARED_BACKBONE,
        "terminal_state_digest": TERMINAL[recipe],
        "paired_support": {
            "enabled": int(recipe == "D3"),
            "available_batches": 0,
            "eligible_masters": 0,
            "pairs": 0,
        },
        "history": history,
    }


def _counters(elapsed):
    return {
        "calibration_started": 2,
        "calibration_completed": 2,
        "calibration_failed": 0,
        "neural_started": 2,
        "neural_completed": 2,
        "neural_failed": 0,
        "optimizer_steps": STEPS * 2,
        "optimizer_steps_exact": True,
        "peak_cuda_bytes": PEAK,
        "elapsed_seconds": elapsed,
        "sum_fit_elapsed_seconds": SUM_FIT,
    }


def _identity():
    return {
        "schema_version": module.SCHEMA,
        "protocol_version": module.PROTOCOL_VERSION,
        "command": module.COMMAND,
        "stage": module.STAGE_NAME,
        "claim": module.CLAIM,
        "permit_sha256": PERMIT,
        "core_contract_sha256": CONTRACT,
        "core_plan_id": "core-plan",
        "ledger_id": LEDGER_ID,
        "selection_plan_id": PLAN_ID,
        "source_fit_count": module.SOURCE_FIT_COUNT,
        "reused_pilot_fit_count": module.REUSED_PILOT_FIT_COUNT,
        "unique_refit_count": 2,
        "strategy_alias_count": 3,
        "source_optimizer_steps": SOURCE_STEPS,
        "outer_predictions_started": 0,
    }


def _write(path, payload):
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))


def _read(path):
    return core._read_json(path, "test")


def _write_manifest(directory):
    core._write_manifest(directory)
    return core._canon().sha256_file(directory / module.MANIFEST_NAME)


def _boom(*args, **kwargs):
    raise AssertionError("forbidden call")


def _raise_deadline(deadline):
    raise RuntimeError("expired")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "STRATEGY_ALIAS_COUNT", 3)
    monkeypatch.setattr(module, "MAXIMUM_REFITS", 3)
    run_root = tmp_path / "p05comprehensive" / "runs" / PERMIT
    stage = run_root / module.STAGE_NAME
    units = stage / module.UNITS_NAME
    specs = {recipe: _spec(recipe, rid) for recipe, rid in RECIPES}
    plan = {
        "plan_id": PLAN_ID,
        "unique_refits": {s["refit_id"]: s for s in specs.values()},
        "strategy_aliases": [{"refit_id": "r0"}, {"refit_id": "r3"}, {"refit_id": "r0"}],
    }
    for spec in specs.values():
        unit_dir = units / spec["refit_id"]
        (unit_dir / "histories").mkdir(parents=True)
        history = _history()
        text = "".join(json.dumps(record) + "\n" for record in history)
        (unit_dir / "histories" / f"{spec['refit_id']}.jsonl").write_text(text, encoding="utf-8")
        _write(unit_dir / module.SUMMARY_NAME, _summary(spec, history))
        _write(unit_dir / module.LEASE_NAME, {"selection_plan_id": PLAN_ID, "spec": spec})
        _write_manifest(unit_dir)
    shared = dict(
        _identity(),
        refits_complete=True,
        calibrations_complete=True,
        status="complete",
        total_new_optimizer_steps=SOURCE_STEPS + STEPS * 2,
        prior_scientific_seconds_cumulative_bound=PRIOR,
        prelaunch_audit_reserve_seconds=3600,
        maximum_total_seconds=172800,
    )
    summary = dict(
        shared,
        counters=_counters(SUMMARY_ELAPSED),
        scientific_seconds_this_stage=SUMMARY_ELAPSED,
        scientific_seconds_cumulative_bound=PRIOR + SUMMARY_ELAPSED,
    )
    _write(stage / module.SUMMARY_NAME, summary)
    receipt = dict(
        shared,
        counters=_counters(STAGE_ELAPSED),
        scientific_seconds_this_stage=STAGE_ELAPSED,
        scientific_seconds_cumulative_bound=PRIOR + STAGE_ELAPSED,
        stage_manifest_sha256=_write_manifest(stage),
    )
    _write(run_root / module.RECEIPT_NAME, receipt)
    calls = {"result": 0, "calibration": 0}

    def authenticate_selection(bundle, *, deadline):
        return {
            "plan": plan,
            "prior_seconds": PRIOR,
            "source_optimizer_steps": SOURCE_STEPS,
            "selection_receipt": {"receipt_id": "selection"},
        }

    def load_result(unit_dir, spec):
        calls["result"] += 1
        return {"state": "result"}, _h(0)

    def load_calibration(unit_dir, spec):
        calls["calibration"] += 1
        return {"temperature": 1.0}, _h(0)

    monkeypatch.setattr(authority, "authenticate_selection", authenticate_selection)
    monkeypatch.setattr(prediction, "_load_result", load_result)
    monkeypatch.setattr(prediction, "_load_calibration", load_calibration)
    bundle = {
        "permit_sha256": PERMIT,
        "contract_sha256": CONTRACT,
        "core_plan_id": "core-plan",
        "ledger": {"ledger_id": LEDGER_ID},
        "artifact_root": str(tmp_path),
        "support": SimpleNamespace(
            roles=[
                {
                    "context_id": CONTEXT,
                    "role_id": ROLE,
                    "role": "outer_fit",
                    "master_sample_id": "master-0",
                    "instrument": "instrument-a",
                }
            ]
        ),
    }

    def call(deadline=FUTURE):
        return module.authenticate_refits(bundle, deadline=deadline)

    def refresh(refit_id=None):
        if refit_id is not None:
            _write_manifest(units / refit_id)
        payload = _read(run_root / module.RECEIPT_NAME)
        payload["stage_manifest_sha256"] = _write_manifest(stage)
        # Retain deliberate nonfinite receipt mutations during manifest refresh.
        core._atomic_write(run_root / module.RECEIPT_NAME, json.dumps(payload).encode())

    return SimpleNamespace(
        root=tmp_path,
        run_root=run_root,
        stage=stage,
        units=units,
        receipt_path=run_root / module.RECEIPT_NAME,
        summary_path=stage / module.SUMMARY_NAME,
        bundle=bundle,
        plan=plan,
        calls=calls,
        call=call,
        refresh=refresh,
    )


def _expect(world, code):
    with pytest.raises(core.P05CoreError) as info:
        world.call()
    assert info.value.reason_code == code


def _patch(path, field, value):
    payload = _read(path)
    if "." in field:
        head, tail = field.split(".", 1)
        payload[head][tail] = value
    else:
        payload[field] = value
    # Deliberately permit malformed nonfinite JSON in adversarial fixtures.
    core._atomic_write(path, json.dumps(payload).encode())


def _apply(world, kind, refit_id, field, value):
    unit = world.units / refit_id if refit_id else None
    if kind == "summary":
        _patch(unit / module.SUMMARY_NAME, field, value)
        world.refresh(refit_id)
    elif kind == "lease":
        _patch(unit / module.LEASE_NAME, field, value)
        world.refresh(refit_id)
    elif kind == "history":
        text = "".join(json.dumps(r) + "\n" for r in value)
        (unit / "histories" / f"{refit_id}.jsonl").write_text(text, encoding="utf-8")
        _patch(unit / module.SUMMARY_NAME, "history", value)
        world.refresh(refit_id)
    elif kind == "history_text":
        (unit / "histories" / f"{refit_id}.jsonl").write_text(value, encoding="utf-8")
        world.refresh(refit_id)
    elif kind == "remove_unit":
        shutil.rmtree(unit)
        world.refresh()
    elif kind == "add_unit":
        (world.units / "extra").mkdir()
        _write(world.units / "extra" / "note.json", {"note": "extra"})
        world.refresh()
    elif kind == "remove_lease":
        (unit / module.LEASE_NAME).unlink()
        world.refresh(refit_id)
    elif kind == "remove_history":
        (unit / "histories" / f"{refit_id}.jsonl").unlink()
        world.refresh(refit_id)


def test_success_reconciles_prior_and_steps(world):
    result = world.call()
    assert result["prior_seconds"] == 4002.0 != PRIOR
    assert result["source_optimizer_steps"] == SOURCE_STEPS
    assert result["refit_optimizer_steps"] == STEPS * 2
    assert result["plan"] is world.plan
    assert result["selection_receipt"] == {"receipt_id": "selection"}


def test_read_only_no_writes_no_training(world, monkeypatch):
    monkeypatch.setattr(core, "_atomic_write", _boom)
    monkeypatch.setattr(prediction, "predict_refit", _boom, raising=False)
    monkeypatch.setattr(prediction, "train_refit", _boom, raising=False)
    world.call()
    assert world.calls == {"result": 2, "calibration": 2}


@pytest.mark.parametrize(
    ("target", "field", "value", "code"),
    [
        ("receipt", "status", "running", "receipt_status_incomplete"),
        ("receipt", "refits_complete", False, "receipt_refits_incomplete"),
        ("receipt", "calibrations_complete", False, "receipt_calibrations_incomplete"),
        ("summary", "status", "running", "summary_status_incomplete"),
        ("summary", "refits_complete", False, "summary_refits_incomplete"),
        ("summary", "calibrations_complete", False, "summary_calibrations_incomplete"),
    ],
)
def test_completion_flags(world, target, field, value, code):
    path = world.receipt_path if target == "receipt" else world.summary_path
    _patch(path, field, value)
    if target == "summary":
        world.refresh()
    _expect(world, code)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "protocol_version",
        "command",
        "stage",
        "claim",
        "permit_sha256",
        "core_contract_sha256",
        "core_plan_id",
        "ledger_id",
        "selection_plan_id",
        "source_fit_count",
        "reused_pilot_fit_count",
        "unique_refit_count",
        "strategy_alias_count",
        "source_optimizer_steps",
        "outer_predictions_started",
    ],
)
def test_identity_mutations(world, field):
    _patch(world.receipt_path, field, "sentinel" if isinstance(_identity()[field], str) else 999)
    _expect(world, f"receipt_{field}_mismatch")


@pytest.mark.parametrize(
    ("target", "field", "value", "code"),
    [
        ("receipt", "optimizer_steps_exact", False, "optimizer_steps_inexact"),
        ("receipt", "optimizer_steps", -1, "optimizer_steps_malformed"),
        ("receipt", "optimizer_steps", True, "optimizer_steps_malformed"),
        ("receipt", "peak_cuda_bytes", -1, "peak_cuda_exceeded"),
        ("receipt", "peak_cuda_bytes", True, "peak_cuda_malformed"),
        (
            "receipt",
            "sum_fit_elapsed_seconds",
            float("nan"),
            "counters_sum_fit_elapsed_seconds_mismatch",
        ),
        ("receipt", "neural_failed", 1, "neural_failures_present"),
        ("receipt", "calibration_started", 1, "calibration_started_mismatch"),
        ("summary", "elapsed_seconds", 3.0, "summary_elapsed_exceeds_receipt"),
        ("summary", "optimizer_steps", 999, "counters_optimizer_steps_mismatch"),
        ("summary", "peak_cuda_bytes", 999, "counters_peak_cuda_bytes_mismatch"),
    ],
)
def test_counter_mutations(world, target, field, value, code):
    path = world.receipt_path if target == "receipt" else world.summary_path
    _patch(path, f"counters.{field}", value)
    if target == "receipt":
        _patch(world.summary_path, f"counters.{field}", value)
    world.refresh()
    _expect(world, code)


def test_total_and_exact_steps(world):
    for target in ("receipt", "summary"):
        path = world.receipt_path if target == "receipt" else world.summary_path
        _patch(path, "counters.optimizer_steps", STEPS * 2 + 1)
        _patch(path, "total_new_optimizer_steps", SOURCE_STEPS + STEPS * 2 + 1)
    world.refresh()
    _expect(world, "exact_optimizer_steps_mismatch")


SHORT_HISTORY = _history()[:-1]
DIVERGED_HISTORY = _history()
DIVERGED_HISTORY[5]["sampling_digest"] = _h(999999)


@pytest.mark.parametrize(
    ("kind", "refit_id", "field", "value", "code"),
    [
        ("remove_unit", "r3", None, None, "units_directory_set_mismatch"),
        ("add_unit", None, None, None, "units_directory_set_mismatch"),
        ("lease", "r0", "selection_plan_id", "other", "unit_lease_mismatch"),
        ("remove_lease", "r0", None, None, "unit_lease_missing"),
        ("remove_history", "r0", None, None, "unit_history_missing"),
        ("history_text", "r0", None, "{bad\n", "unit_history_malformed"),
        ("history", "r0", None, SHORT_HISTORY, "unit_history_length_mismatch"),
    ],
)
def test_structure_mutations(world, kind, refit_id, field, value, code):
    _apply(world, kind, refit_id, field, value)
    _expect(world, code)


@pytest.mark.parametrize(
    ("kind", "refit_id", "field", "value", "code"),
    [
        ("history", "r3", None, DIVERGED_HISTORY, "group_digest_prefix_mismatch"),
        ("summary", "r3", "initial_backbone_digest", _h(12345), "group_initial_backbone_mismatch"),
        ("summary", "r3", "paired_support.pairs", 1, "group_sparse_pairs_present"),
    ],
)
def test_group_mutations(world, kind, refit_id, field, value, code):
    _apply(world, kind, refit_id, field, value)
    _expect(world, code)


def test_manifest_and_unit_file_hash_mismatches(world):
    _patch(world.receipt_path, "stage_manifest_sha256", "0" * 64)
    _expect(world, "stage_manifest_sha256_mismatch")
    _patch(
        world.receipt_path,
        "stage_manifest_sha256",
        core._canon().sha256_file(world.stage / module.MANIFEST_NAME),
    )
    _patch(world.units / "r0" / module.SUMMARY_NAME, "elapsed_seconds", 0.05)
    with pytest.raises(core.P05CoreError):
        world.call()


def test_deadline_checked_before_selection(world, monkeypatch):
    monkeypatch.setattr(freeze, "_check_deadline", _raise_deadline)
    with pytest.raises(RuntimeError):
        world.call()
    assert world.calls == {"result": 0, "calibration": 0}


def test_deadline_expires_mid_loop(world, monkeypatch):
    state = {"n": 0}

    def check(deadline):
        state["n"] += 1
        if state["n"] >= 6:
            raise RuntimeError("expired")

    monkeypatch.setattr(freeze, "_check_deadline", check)
    with pytest.raises(RuntimeError):
        world.call()
