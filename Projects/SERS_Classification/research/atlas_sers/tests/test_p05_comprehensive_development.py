"""Synthetic boundary integration tests for the P05 comprehensive runner.

No scientific data or fits run here: ``inputs.prepare``/``import_pilot``, CUDA
capability, provenance and the per-fit acceptance helpers are stubbed, while the
real ``pilot.persist_result`` checkpoint save/reload and the real
``StorageBudget`` filesystem accounting are exercised.  Constants are scaled to
4 units / 48 slots / 12 new fits (3 reused pilot units + 1 new unit).
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from atlas_sers.evaluation import p05_comprehensive_development as dev
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation.p05_comprehensive_storage import (
    P05StorageError,
    StorageBudget,
)
from atlas_sers.evaluation.p05_core_run import P05CoreError

PERMIT = inputs.COMPREHENSIVE_PERMIT_SHA256
CONTRACT = "a" * 64
PLAN = "b" * 64
STATIONS = ("cwa", "pills", "surfaces", "extra")


def _torch():
    return pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _scaled_constants(monkeypatch):
    monkeypatch.setattr(dev, "UNIT_COUNT", 4)
    monkeypatch.setattr(dev, "INNER_SLOT_COUNT", 48)
    monkeypatch.setattr(dev, "MAXIMUM_NEW_FITS", 12)
    monkeypatch.setattr(dev, "MAXIMUM_FIT_STEPS", 9600)


@dataclasses.dataclass
class _Support:
    manifest_sha256: str = "m" * 64
    contexts_sha256: str = "c" * 64
    roles_sha256: str = "r" * 64


def _make_unit(index, station):
    return {
        "unit_id": f"unit-{index}",
        "station": station,
        "context_id": f"ctx-{index}",
        "selection_unit_id": f"sel-{index}",
        "fitting_role_id": f"fit-{index}",
        "validation_role_id": f"val-{index}",
        "fitting_uid_set_sha256": f"{index:064d}",
        "validation_uid_set_sha256": f"{index + 500:064d}",
    }


def _make_slots(unit):
    slots = []
    for recipe in pilot.PILOT_RECIPES:
        for seed in pilot.PILOT_SEEDS:
            slots.append(
                {
                    "slot_id": f"{unit['unit_id']}::{recipe}::{seed}",
                    "unit_id": unit["unit_id"],
                    "recipe_id": recipe,
                    "seed": seed,
                    "slot_kind": "development",
                    "fitting_role_id": unit["fitting_role_id"],
                    "validation_role_id": unit["validation_role_id"],
                    "excluded_by_protocol": False,
                }
            )
    return slots


def _bundle(tmp_path, *, corrupt_new_product=False):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    units = [_make_unit(i, s) for i, s in enumerate(STATIONS)]
    slots = []
    for unit in units:
        slots.extend(_make_slots(unit))
    if corrupt_new_product:
        for slot in slots:
            if slot["unit_id"] == units[3]["unit_id"] and slot["recipe_id"] == "D0-M":
                slot["recipe_id"] = "D1"
                break
    pilot_unit_ids = {unit["unit_id"] for unit in units[:3]}
    pilot_slots = [slot for slot in slots if slot["unit_id"] in pilot_unit_ids]
    return {
        "project_root": tmp_path,
        "artifact_root": artifact,
        "repository_root": tmp_path,
        "permit": {},
        "permit_sha256": PERMIT,
        "contract": {},
        "contract_sha256": CONTRACT,
        "support": _Support(),
        "p01_path": tmp_path,
        "core_plan": {},
        "core_plan_id": PLAN,
        "ledger": {
            "ledger_id": "L" * 64,
            "schema_version": "ledger-v1",
            "units": units,
            "slots": slots,
        },
        "units": units,
        "slots": slots,
        "pilot_bundle": {"units": units[:3], "slots": pilot_slots},
    }


def _run_root(bundle):
    return bundle["artifact_root"] / dev.NAMESPACE / "runs" / bundle["permit_sha256"]


def _pilot_records(bundle):
    return [
        {"slot_id": slot["slot_id"], "status": "complete"}
        for slot in bundle["pilot_bundle"]["slots"]
    ]


class _Spy:
    def __init__(self, fn=None):
        self.calls = 0
        self.fn = fn

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.fn is not None:
            return self.fn(*args, **kwargs)


def _accept_complete(result, *args, **kwargs):
    if getattr(result, "status", None) != "complete":
        raise pilot.P05PilotError("fit_not_complete")


def _build_result(module, runtime, torch, recipe, seed, role_id, *, status, epochs, steps):
    best = {"w": torch.zeros(2, dtype=torch.float32)}
    terminal = {"w": torch.ones(2, dtype=torch.float32)}
    history = [
        {
            "epoch": epoch,
            "epoch_optimizer_steps": 4,
            "total_optimizer_steps": epoch * 4,
            "train_balanced_accuracy": 0.5,
            "validation_balanced_accuracy": 0.5,
            "validation_nll": 1.0,
            "validation_macro_f1": 0.5,
            "validation_predicted_class_count": 3,
            "sampling_digest": "s" * 64,
            "augmentation_digest": "g" * 64,
            "pair_digest": "p" * 64,
        }
        for epoch in range(1, epochs + 1)
    ]
    overrides = {
        "status": status,
        "reason_code": None,
        "role_id": role_id,
        "recipe": recipe,
        "seed": seed,
        "history": history,
        "epochs_completed": epochs,
        "optimizer_steps": steps,
        "finite_gradient_batches": steps,
        "best_epoch": 1,
        "best_validation_balanced_accuracy": 0.5,
        "best_validation_nll": 1.0,
        "best_validation_macro_f1": 0.5,
        "best_validation_predicted_class_count": 3,
        "best_training_balanced_accuracy": 0.5,
        "parameter_count": 1,
        "elapsed_seconds": 0.25,
        "peak_cuda_bytes": 0,
        "validation_logits": np.zeros((2, 3), dtype=np.float64),
        "validation_uids": ("v1", "v2"),
        "classes": ("A", "B", "C"),
        "state_dict": best,
        "best_state_dict": best,
        "terminal_state_dict": terminal,
        "best_state_digest": runtime._state_hash(best),
        "terminal_state_digest": runtime._state_hash(terminal),
        "initial_state_digest": runtime._state_hash(best),
        "initial_backbone_digest": "i" * 64,
        "paired_support": {"available_batches": 0, "eligible_masters": 0, "pairs": 0},
        "sampling_digest": "s" * 64,
        "augmentation_digest": "g" * 64,
        "pair_digest": "p" * 64,
    }
    values = {
        field.name: overrides.get(field.name)
        for field in dataclasses.fields(module.DevelopmentFitResult)
    }
    return module.DevelopmentFitResult(**values)


def _install(
    monkeypatch,
    bundle,
    *,
    status="complete",
    throw_after_epoch=False,
    pilot_records=None,
    epochs=2,
    steps=8,
):
    torch = _torch()
    module = importlib.import_module("atlas_sers.evaluation.p05_development")
    runtime = importlib.import_module("atlas_sers.evaluation.p04_runtime")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 0)
    monkeypatch.setattr(inputs, "prepare", lambda *a, **k: bundle)
    monkeypatch.setattr(
        inputs,
        "import_pilot",
        lambda b, device=None: (
            pilot_records if pilot_records is not None else _pilot_records(bundle)
        ),
    )
    monkeypatch.setattr(
        pilot,
        "prepare_role_inputs",
        lambda b: {str(unit["unit_id"]): {"ok": True} for unit in b["units"]},
    )
    monkeypatch.setattr(pilot, "_free_cuda_bytes", lambda _torch: 10**12)
    monkeypatch.setattr(pilot, "_enforce_cuda_cap", lambda *a, **k: None)
    monkeypatch.setattr(pilot, "run", lambda *a, **k: pytest.fail("pilot.run called"))
    monkeypatch.setattr(pilot, "preflight", lambda *a, **k: pytest.fail("preflight called"))
    monkeypatch.setattr(
        core, "_capture_provenance", lambda *a, **k: {"protected_environment_sha256": "fixed"}
    )
    monkeypatch.setattr(
        pilot, "_post_run_reauth", lambda *a, **k: {"protected_environment_sha256": "fixed"}
    )
    accept = _Spy(_accept_complete)
    shared = _Spy()
    sparse = _Spy()
    monkeypatch.setattr(pilot, "check_completed_result", accept)
    monkeypatch.setattr(pilot, "check_sparse_support", _Spy())
    monkeypatch.setattr(pilot, "check_shared_prefixes", shared)
    monkeypatch.setattr(pilot, "check_sparse_equivalences", sparse)
    probes = {"ledger_before_first_fit": None}
    run_root = _run_root(bundle)

    def train_fit(unit_inputs, unit, slot, device, deadline, guarded):
        if probes["ledger_before_first_fit"] is None:
            probes["ledger_before_first_fit"] = (run_root / "develop" / "ledger.json").is_file()
        if throw_after_epoch:
            guarded(
                {
                    "epoch": 1,
                    "epoch_optimizer_steps": 4,
                    "total_optimizer_steps": 4,
                    "train_balanced_accuracy": 0.5,
                    "validation_balanced_accuracy": 0.5,
                    "validation_nll": 1.0,
                    "validation_macro_f1": 0.5,
                    "validation_predicted_class_count": 3,
                    "sampling_digest": "s" * 64,
                    "augmentation_digest": "g" * 64,
                    "pair_digest": "p" * 64,
                }
            )
            raise RuntimeError("kernel exploded")
        result = _build_result(
            module,
            runtime,
            torch,
            slot["recipe_id"],
            slot["seed"],
            unit["fitting_role_id"],
            status=status,
            epochs=epochs,
            steps=steps,
        )
        for record in result.history:
            guarded(dict(record))
        return result

    monkeypatch.setattr(pilot, "train_fit", train_fit)
    deadlines = []
    real_check = dev._check_deadline

    def check_deadline(deadline):
        deadlines.append(deadline)
        return real_check(deadline)

    monkeypatch.setattr(dev, "_check_deadline", check_deadline)
    return {
        "accept": accept,
        "shared": shared,
        "sparse": sparse,
        "probes": probes,
        "deadlines": deadlines,
    }


def _invoke(bundle, tmp_path):
    return dev.run_development(
        project_root=tmp_path,
        artifact_root=bundle["artifact_root"],
        contract_path=tmp_path / "c.json",
        permit_path=tmp_path / "p.json",
        device="cuda",
    )


def _failure_summary(bundle):
    path = _run_root(bundle) / "develop" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_completes_exact_new_and_reuse(monkeypatch, tmp_path):
    _torch()
    bundle = _bundle(tmp_path)
    spies = _install(monkeypatch, bundle)
    start = time.perf_counter()
    summary = _invoke(bundle, tmp_path)
    end = time.perf_counter()

    assert summary["status"] == "complete"
    assert summary["started"] == 12
    assert summary["completed"] == 12
    assert summary["failed"] == 0
    assert summary["new_completions"] == 12
    assert summary["selector_records"] == 48
    assert summary["reused_pilot_slots"] == 36
    assert summary["optimizer_steps"] == 12 * 8
    assert summary["selection_authorized"] is False
    assert summary["refit_authorized"] is False
    assert summary["calibration_authorized"] is False
    assert summary["outer_evaluation_authorized"] is False

    assert spies["accept"].calls == 12
    assert spies["shared"].calls == 1
    assert spies["sparse"].calls == 1
    assert spies["probes"]["ledger_before_first_fit"] is True

    run_root = _run_root(bundle)
    stage = run_root / "develop"
    assert (stage / "ledger.json").read_bytes() == core._canon().canonical_json_bytes(
        bundle["ledger"]
    )
    assert (stage / "source_ledger.json").is_file()

    lease_root = bundle["artifact_root"] / "p05development" / "slot_leases"
    leases = sorted(lease_root.rglob("lease.json"))
    assert len(leases) == 12
    for path in leases:
        lease = json.loads(path.read_text(encoding="utf-8"))
        assert lease["permit_sha256"] == PERMIT
        assert lease["contract_sha256"] == CONTRACT
        assert lease["core_plan_id"] == PLAN

    receipt = json.loads((run_root / "development_receipt.json").read_text(encoding="utf-8"))
    assert receipt["stage_manifest_sha256"] == core._canon().sha256_file(stage / "manifest.json")
    assert receipt["prelaunch_audit_reserve_seconds"] == dev.PRELAUNCH_AUDIT_RESERVE_SECONDS
    assert receipt["scientific_seconds_cumulative_bound"] == pytest.approx(
        receipt["scientific_seconds_this_stage"] + dev.PRELAUNCH_AUDIT_RESERVE_SECONDS
    )
    assert "scientific_seconds_cumulative_measured" not in receipt

    reserve = dev.MAXIMUM_TOTAL_SECONDS - dev.PRELAUNCH_AUDIT_RESERVE_SECONDS
    assert start + reserve <= min(spies["deadlines"]) <= end + reserve


def test_occupied_run_prevents_repeat(monkeypatch, tmp_path):
    _torch()
    bundle = _bundle(tmp_path)
    _install(monkeypatch, bundle)
    _invoke(bundle, tmp_path)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "comprehensive_run_exists"


def test_preexisting_shared_slot_never_overwritten(monkeypatch, tmp_path):
    _torch()
    bundle = _bundle(tmp_path)
    _install(monkeypatch, bundle)
    new_unit_id = bundle["ledger"]["units"][3]["unit_id"]
    slot = min(
        (s for s in bundle["ledger"]["slots"] if s["unit_id"] == new_unit_id),
        key=lambda s: (s["recipe_id"], s["seed"]),
    )
    directory = pilot._slot_lease_dir(bundle["artifact_root"], CONTRACT, PLAN, slot["slot_id"])
    directory.mkdir(parents=True)
    sentinel = directory / "lease.json"
    sentinel.write_bytes(b"sentinel")
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "slot_lease_exists"
    assert sentinel.read_bytes() == b"sentinel"


@pytest.mark.parametrize("status", ["finite", "collapse"])
def test_noncomplete_status_rejected(monkeypatch, tmp_path, status):
    _torch()
    bundle = _bundle(tmp_path)
    _install(monkeypatch, bundle, status=status)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "fit_not_complete"
    summary = _failure_summary(bundle)
    assert summary["status"] == "fail"
    assert summary["started"] == 1
    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["optimizer_steps"] == 8
    assert summary["optimizer_steps_exact"] is True


def test_persistence_error_stops_first_fit(monkeypatch, tmp_path):
    _torch()
    bundle = _bundle(tmp_path)
    _install(monkeypatch, bundle)
    real_save = core._save_state

    def failing_save(torch_arg, state, path, *args, **kwargs):
        if "executions" in Path(path).parts:
            raise RuntimeError("serializer boom")
        return real_save(torch_arg, state, path, *args, **kwargs)

    monkeypatch.setattr(core, "_save_state", failing_save)
    with pytest.raises(dev.P05ComprehensiveDevelopmentError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "comprehensive_execution_failed"

    summary = _failure_summary(bundle)
    assert summary["started"] == 1
    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["optimizer_steps"] == 8
    assert summary["optimizer_steps_exact"] is True
    run_root = _run_root(bundle)
    lease_root = bundle["artifact_root"] / "p05development" / "slot_leases"
    assert len(list(lease_root.rglob("lease.json"))) == 1
    assert len(list((run_root / "develop" / "units").rglob("*.jsonl"))) == 1


def test_kernel_throw_after_epoch_reports_lower_bound(monkeypatch, tmp_path):
    _torch()
    bundle = _bundle(tmp_path)
    _install(monkeypatch, bundle, throw_after_epoch=True)
    with pytest.raises(dev.P05ComprehensiveDevelopmentError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "comprehensive_execution_failed"
    summary = _failure_summary(bundle)
    assert summary["started"] == 1
    assert summary["failed"] == 1
    assert summary["optimizer_steps"] == pilot.BATCH_DRAWS_PER_EPOCH
    assert summary["optimizer_steps_exact"] is False


def test_duplicate_pilot_record_rejected(monkeypatch, tmp_path):
    _torch()
    bundle = _bundle(tmp_path)
    records = _pilot_records(bundle)
    records[1] = dict(records[0])
    _install(monkeypatch, bundle, pilot_records=records)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "pilot_selector_identity_mismatch"
    assert not _run_root(bundle).exists()


def test_bad_recipe_product_rejected(monkeypatch, tmp_path):
    _torch()
    bundle = _bundle(tmp_path, corrupt_new_product=True)
    _install(monkeypatch, bundle)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "unit_slot_product_mismatch"
    assert not _run_root(bundle).exists()


def test_storage_budget_cap_and_headroom(tmp_path):
    artifact = tmp_path / "art"
    run_root = artifact / "p05comprehensive" / "runs" / "r"
    run_root.mkdir(parents=True)
    budget = StorageBudget(artifact, run_root, ceiling=1000)
    grow = run_root / "grow.jsonl"
    budget.register_growing(grow)
    grow.write_bytes(b"x" * 100)
    live = budget.check()
    assert isinstance(live, int) and live >= 100
    with pytest.raises(P05StorageError) as exc:
        budget.check(headroom_bytes=10**6)
    assert exc.value.reason_code == "storage_ceiling_exceeded"
    grow.write_bytes(b"x" * 2000)
    with pytest.raises(P05StorageError) as exc:
        budget.check()
    assert exc.value.reason_code == "storage_ceiling_exceeded"


def test_reserve_constants_bound_not_measured():
    assert dev.MAXIMUM_TOTAL_SECONDS == 172800.0
    assert dev.PRELAUNCH_AUDIT_RESERVE_SECONDS == 3600.0
    assert dev.PRELAUNCH_AUDIT_RESERVE_SECONDS < dev.MAXIMUM_TOTAL_SECONDS
