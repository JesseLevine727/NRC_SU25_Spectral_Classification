"""P05-T016 approved 36-fit source-validation pilot boundary tests.

The suite exercises the current ``atlas_sers.evaluation.p05_pilot`` API only:
metadata selection, permit/plan authentication, artifact and lease refusal,
orchestration through the real checkpoint saver, failure stopping and the real
small synthetic acceptance check.  Neural/serialization tests skip when torch
is unavailable; the metadata tests import and run without torch.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from atlas_sers.evaluation import p05_pilot
from atlas_sers.evaluation.p05_core_run import P05CoreError

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = _REPO_ROOT / "plan" / "contracts" / "p05_core_contract.json"
_PERMIT_PATHS = sorted(_REPO_ROOT.rglob("p05_development_pilot.json"))


def _torch():
    return pytest.importorskip("torch")


def _runtime():
    return pytest.importorskip("atlas_sers.evaluation.p04_runtime")


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Metadata-only unit and slot selection
# --------------------------------------------------------------------------- #


def _unit(
    unit_id,
    station,
    mode,
    context_id,
    selection_unit_id,
    *,
    kind="inherited",
    phase="held_evaluation",
    excluded=False,
):
    return {
        "unit_id": unit_id,
        "unit_kind": kind,
        "phase_gate": phase,
        "station": station,
        "selection_mode": mode,
        "context_id": context_id,
        "selection_unit_id": selection_unit_id,
        "fitting_role_id": f"fit-{unit_id}",
        "validation_role_id": f"val-{unit_id}",
        "excluded_by_protocol": excluded,
        "fitting_uid_set_sha256": "f" * 64,
        "validation_uid_set_sha256": "v" * 64,
        "fitting_uids": ["fa", "fb", "fc"],
        "validation_uids": ["va", "vb", "vc"],
    }


def _ledger(units, slots=None):
    return {
        "units": list(units),
        "slots": list(slots or []),
        "ledger_id": "L" * 64,
        "schema_version": "nato-sers-p05-development-ledger-v1",
    }


def _full_slots(units):
    slots = []
    for unit in units:
        for recipe in p05_pilot.PILOT_RECIPES:
            for seed in p05_pilot.PILOT_SEEDS:
                slots.append(
                    {
                        "slot_id": f"{unit['unit_id']}::{recipe}::{seed}",
                        "unit_id": unit["unit_id"],
                        "recipe_id": recipe,
                        "seed": seed,
                        "fitting_role_id": unit["fitting_role_id"],
                        "validation_role_id": unit["validation_role_id"],
                        "excluded_by_protocol": False,
                    }
                )
    return slots


def test_select_units_prefers_pseudo_then_master_cv_and_breaks_ties():
    units = [
        _unit("u-cwa-pseudo", "cwa", "pseudo_domain", "ctx-z", "unit-z"),
        _unit("u-cwa-mcv", "cwa", "master_cv", "ctx-a", "unit-a"),
        _unit("u-pills-b", "pills", "pseudo_domain", "ctx-b", "unit-b"),
        _unit("u-pills-a", "pills", "pseudo_domain", "ctx-a", "unit-a"),
        _unit("u-surfaces-mcv", "surfaces", "master_cv", "ctx-m", "unit-m"),
        _unit("u-cwa-excluded", "cwa", "pseudo_domain", "ctx-a", "unit-0", excluded=True),
        _unit("u-cwa-dev", "cwa", "pseudo_domain", "ctx-a", "unit-1", phase="development"),
    ]
    chosen = p05_pilot._select_units(_ledger(units))
    assert [unit["unit_id"] for unit in chosen] == [
        "u-cwa-pseudo",
        "u-pills-a",
        "u-surfaces-mcv",
    ]


def test_select_slots_returns_exact_36_in_station_recipe_seed_order():
    units = [
        _unit("u-surfaces", "surfaces", "pseudo_domain", "ctx-s", "unit-s"),
        _unit("u-cwa", "cwa", "pseudo_domain", "ctx-c", "unit-c"),
        _unit("u-pills", "pills", "pseudo_domain", "ctx-p", "unit-p"),
    ]
    slots = p05_pilot._select_slots(_ledger(units, _full_slots(units)), units)
    station_of = {unit["unit_id"]: unit["station"] for unit in units}
    assert len(slots) == 36
    assert [(station_of[s["unit_id"]], s["recipe_id"], s["seed"]) for s in slots] == [
        (station, recipe, seed)
        for station in ("cwa", "pills", "surfaces")
        for recipe in p05_pilot.PILOT_RECIPES
        for seed in p05_pilot.PILOT_SEEDS
    ]


def test_select_slots_rejects_incomplete_or_wrong_product():
    units = [
        _unit("u-cwa", "cwa", "pseudo_domain", "ctx-c", "unit-c"),
        _unit("u-pills", "pills", "pseudo_domain", "ctx-p", "unit-p"),
        _unit("u-surfaces", "surfaces", "pseudo_domain", "ctx-s", "unit-s"),
    ]
    slots = _full_slots(units)
    slots.pop()
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._select_slots(_ledger(units, slots), units)
    assert exc.value.reason_code == "pilot_unit_slot_count_mismatch"


# --------------------------------------------------------------------------- #
# Permit, stored plan and artifact location
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _PERMIT_PATHS, reason="public pilot permit is not present")
def test_load_permit_accepts_public_and_rejects_modified(tmp_path):
    permit_path = _PERMIT_PATHS[0]
    permit, digest = p05_pilot._load_permit(permit_path, p05_pilot.PILOT_PERMIT_SHA256)
    assert digest == p05_pilot.PILOT_PERMIT_SHA256
    assert p05_pilot._canon().sha256_value(permit) == p05_pilot.PILOT_PERMIT_SHA256

    modified = copy.deepcopy(permit)
    modified["unapproved_override"] = "tamper"
    target = tmp_path / "modified.json"
    target.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._load_permit(target, p05_pilot.PILOT_PERMIT_SHA256)
    assert exc.value.reason_code == "permit_digest_mismatch"

    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._load_permit(permit_path, "0" * 64)
    assert exc.value.reason_code == "permit_pin_not_approved"


def test_store_pilot_plan_rejects_tampered_content_and_symlink(tmp_path):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    payload = p05_pilot._canon().canonical_json_bytes({"a": 1})
    plan_id = p05_pilot._canon().sha256_value({"a": 1})
    p05_pilot._store_pilot_plan(artifact, payload, plan_id)
    p05_pilot._store_pilot_plan(artifact, payload, plan_id)
    stored = p05_pilot._pilot_plan_dir(artifact, plan_id) / "plan.json"
    stored.write_bytes(b'{"a": 2}')
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._store_pilot_plan(artifact, b'{"a": 2}', plan_id)
    assert exc.value.reason_code == "pilot_plan_content_hash_mismatch"

    link_id = "b" * 64
    link = p05_pilot._pilot_plan_dir(artifact, link_id)
    link.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "elsewhere"
    target.mkdir()
    link.symlink_to(target)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._store_pilot_plan(artifact, b"{}", link_id)
    assert exc.value.reason_code == "symlink_path_rejected"


def test_resolve_paths_rejects_artifact_inside_repository():
    inside = _REPO_ROOT / "artifacts-pilot-should-not-exist"
    with pytest.raises(P05CoreError):
        p05_pilot._resolve_paths(_REPO_ROOT, inside)


def test_symlinked_artifact_path_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(P05CoreError):
        p05_pilot.core._reject_symlink_chain(link)


# --------------------------------------------------------------------------- #
# Stopping, shared prefix and slot coverage units
# --------------------------------------------------------------------------- #


def _stop_history(length, *, improve_until):
    records = []
    for epoch in range(1, length + 1):
        ceiling = min(epoch, improve_until)
        records.append(
            {
                "epoch": epoch,
                "validation_balanced_accuracy": 0.5 + 0.001 * ceiling,
                "validation_nll": 1.0,
            }
        )
    return records


def test_check_stopping_accepts_full_ladder_and_legal_stop_at_200():
    p05_pilot._check_stopping(_stop_history(200, improve_until=200))
    p05_pilot._check_stopping(_stop_history(200, improve_until=180))


def test_check_stopping_rejects_out_of_range_and_premature_stop():
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._check_stopping(_stop_history(201, improve_until=201))
    assert exc.value.reason_code == "history_length_out_of_range"

    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._check_stopping(_stop_history(31, improve_until=1))
    assert exc.value.reason_code == "history_stopping_rule_mismatch"

    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._check_stopping(_stop_history(29, improve_until=1))
    assert exc.value.reason_code == "history_stopped_before_minimum"


@dataclasses.dataclass
class _StubResult:
    history: list
    initial_backbone_digest: str


def _stub_result(digest="s" * 64):
    history = [
        {"sampling_digest": digest, "augmentation_digest": digest, "pair_digest": digest}
        for _ in range(30)
    ]
    return _StubResult(history=history, initial_backbone_digest="i" * 64)


def _prefix_item(unit_id, recipe, seed, result):
    return {
        "unit_id": unit_id,
        "seed": seed,
        "slot": {"recipe_id": recipe, "seed": seed},
        "result": result,
    }


def test_check_shared_prefixes_requires_full_recipe_coverage():
    items = [_prefix_item("u", recipe, 1, _stub_result()) for recipe in ("D0-M", "D1", "D2")]
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot.check_shared_prefixes(items)
    assert exc.value.reason_code == "pilot_unit_recipe_coverage_mismatch"


def test_check_shared_prefixes_detects_shared_prefix_digest_tamper():
    items = [_prefix_item("u", recipe, 1, _stub_result()) for recipe in p05_pilot.PILOT_RECIPES]
    items[-1]["result"].history[0]["sampling_digest"] = "t" * 64
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot.check_shared_prefixes(items)
    assert exc.value.reason_code == "shared_prefix_digest_mismatch"


def test_check_expected_slots_requires_exact_coverage():
    units = [
        _unit("u-cwa", "cwa", "pseudo_domain", "ctx-c", "unit-c"),
        _unit("u-pills", "pills", "pseudo_domain", "ctx-p", "unit-p"),
        _unit("u-surfaces", "surfaces", "pseudo_domain", "ctx-s", "unit-s"),
    ]
    slots = _full_slots(units)
    results = [{"unit_id": slot["unit_id"], "slot": slot} for slot in slots]
    p05_pilot._check_expected_slots(results, slots)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._check_expected_slots(results[:-1], slots)
    assert exc.value.reason_code == "execution_slot_count_mismatch"


# --------------------------------------------------------------------------- #
# Run orchestration fixtures
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _Support:
    manifest_sha256: str = "m" * 64
    contexts_sha256: str = "c" * 64
    roles_sha256: str = "r" * 64


def _fake_units():
    units = []
    for station in p05_pilot.PILOT_STATIONS:
        units.append(
            {
                "unit_id": f"unit-{station}",
                "station": station,
                "context_id": f"ctx-{station}",
                "selection_unit_id": f"sel-{station}",
                "selection_mode": "pseudo_domain",
                "unit_kind": "inherited",
                "phase_gate": "held_evaluation",
                "excluded_by_protocol": False,
                "fitting_role_id": f"fit-{station}",
                "validation_role_id": f"val-{station}",
                "fitting_uids": ["f-a", "f-b", "f-c"],
                "validation_uids": ["v-a", "v-b", "v-c"],
                "fitting_uid_set_sha256": "f" * 64,
                "validation_uid_set_sha256": "v" * 64,
                "validation_classes": ["A", "B", "C"],
                "auxiliary_support": {
                    "cross_instrument_master_pairs": 0,
                    "same_chemical_positive_pairs": 0,
                },
            }
        )
    return units


def _fake_slots(units):
    slots = []
    for unit in units:
        for recipe in p05_pilot.PILOT_RECIPES:
            for seed in p05_pilot.PILOT_SEEDS:
                slots.append(
                    {
                        "slot_id": f"{unit['unit_id']}::{recipe}::{seed}",
                        "unit_id": unit["unit_id"],
                        "recipe_id": recipe,
                        "seed": seed,
                    }
                )
    return slots


def _fake_bundle(tmp_path, contract):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    units = _fake_units()
    slots = _fake_slots(units)
    return {
        "project_root": tmp_path,
        "artifact_root": artifact,
        "repository_root": tmp_path,
        "permit": {},
        "permit_sha256": p05_pilot.PILOT_PERMIT_SHA256,
        "contract": contract,
        "contract_sha256": "a" * 64,
        "support": _Support(),
        "p01_path": tmp_path,
        "core_plan": {},
        "core_plan_id": "d" * 64,
        "ledger": {"ledger_id": "e" * 64, "schema_version": "ledger-v1"},
        "units": units,
        "slots": slots,
        "pilot_plan": {"schema_version": "pilot-v1", "slots": slots},
        "pilot_plan_id": p05_pilot._canon().sha256_value(
            {"schema_version": "pilot-v1", "slots": slots}
        ),
    }


def _fake_inputs(units):
    return {
        unit["unit_id"]: {
            "fitting_values": np.zeros((3, 1401), dtype=np.float32),
            "fitting_observations": [],
            "noise": None,
            "validation_values": np.zeros((3, 1401), dtype=np.float32),
            "validation_observations": [],
        }
        for unit in units
    }


@dataclasses.dataclass
class _FakeResult:
    status: str
    reason_code: str | None
    history: list
    epochs_completed: int
    parameter_count: int
    optimizer_steps: int
    best_epoch: int | None
    best_validation_balanced_accuracy: float | None
    best_validation_nll: float | None
    best_validation_macro_f1: float | None
    best_validation_predicted_class_count: int | None
    best_training_balanced_accuracy: float | None
    initial_state_digest: str | None
    best_state_digest: str | None
    terminal_state_digest: str | None
    initial_backbone_digest: str | None
    best_state_dict: dict | None
    terminal_state_dict: dict | None
    classes: tuple
    validation_uids: tuple
    validation_logits: object
    sampling_digest: str | None
    augmentation_digest: str | None
    pair_digest: str | None
    finite_gradient_batches: int
    paired_support: dict
    role_id: str | None
    recipe: str | None
    seed: int | None
    elapsed_seconds: float
    peak_cuda_bytes: int


def _fake_result(
    runtime, torch, recipe, seed, role_id, *, status="complete", steps=120, epochs=30, reason=None
):
    best = {"backbone.0.weight": torch.zeros(2, dtype=torch.float32)}
    terminal = {"backbone.0.weight": torch.ones(2, dtype=torch.float32)}
    initial = {"backbone.0.weight": torch.full((2,), 0.25, dtype=torch.float32)}
    history = [
        {
            "epoch": epoch,
            "epoch_optimizer_steps": 4,
            "total_optimizer_steps": epoch * 4,
            "train_balanced_accuracy": 0.6,
            "validation_balanced_accuracy": 0.5,
            "validation_nll": 1.0,
            "validation_macro_f1": 0.5,
            "validation_predicted_class_count": 3,
            "sampling_digest": "s" * 64,
            "augmentation_digest": "a" * 64,
            "pair_digest": "p" * 64,
        }
        for epoch in range(1, epochs + 1)
    ]
    return _FakeResult(
        status=status,
        reason_code=reason,
        history=history,
        epochs_completed=epochs,
        parameter_count=208691,
        optimizer_steps=steps,
        best_epoch=1,
        best_validation_balanced_accuracy=0.5,
        best_validation_nll=1.0,
        best_validation_macro_f1=0.5,
        best_validation_predicted_class_count=3,
        best_training_balanced_accuracy=0.6,
        initial_state_digest=runtime._state_hash(initial),
        best_state_digest=runtime._state_hash(best),
        terminal_state_digest=runtime._state_hash(terminal),
        initial_backbone_digest="i" * 64,
        best_state_dict=best,
        terminal_state_dict=terminal,
        classes=("A", "B", "C"),
        validation_uids=("v-a", "v-b", "v-c"),
        validation_logits=np.zeros((3, 3), dtype=np.float64),
        sampling_digest="s" * 64,
        augmentation_digest="a" * 64,
        pair_digest="p" * 64,
        finite_gradient_batches=steps,
        paired_support={"enabled": 1, "available_batches": 0, "eligible_masters": 0, "pairs": 0},
        role_id=role_id,
        recipe=recipe,
        seed=seed,
        elapsed_seconds=0.5,
        peak_cuda_bytes=0,
    )


class _FakeKernel:
    def __init__(self, runtime, torch, *, status="complete", steps=120, epochs=30, reason=None):
        self._runtime = runtime
        self._torch = torch
        self._status = status
        self._steps = steps
        self._epochs = epochs
        self._reason = reason

    def train_development_fit(self, **kwargs):
        result = _fake_result(
            self._runtime,
            self._torch,
            kwargs["recipe"],
            kwargs["seed"],
            kwargs["role_id"],
            status=self._status,
            steps=self._steps,
            epochs=self._epochs,
            reason=self._reason,
        )
        on_epoch = kwargs.get("on_epoch")
        if on_epoch is not None:
            for record in result.history:
                on_epoch(dict(record))
        return result


def _install_run_mocks(monkeypatch, bundle, kernel, inputs):
    monkeypatch.setattr(p05_pilot, "prepare", lambda *a, **k: bundle)
    monkeypatch.setattr(p05_pilot, "prepare_role_inputs", lambda b: inputs)
    monkeypatch.setattr(p05_pilot, "_development_kernel", lambda: kernel)
    monkeypatch.setattr(p05_pilot, "check_completed_result", lambda *a, **k: None)
    monkeypatch.setattr(
        p05_pilot.core,
        "_capture_provenance",
        lambda *a, **k: {"protected_environment_sha256": "fixed"},
    )
    monkeypatch.setattr(
        p05_pilot.core, "_authenticate", lambda *a, **k: (bundle["support"], None, None)
    )


def _invoke_run(bundle, tmp_path, device="cpu"):
    return p05_pilot.run(
        project_root=tmp_path,
        artifact_root=bundle["artifact_root"],
        contract_path=tmp_path / "contract.json",
        permit_path=tmp_path / "permit.json",
        pilot_plan_id=bundle["pilot_plan_id"],
        device=device,
    )


def _run_dir(bundle):
    return p05_pilot._pilot_run_dir(bundle["artifact_root"], bundle["permit_sha256"])


def _read_summary(run_dir):
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _events(run_dir):
    return [
        json.loads(line)["event"]
        for line in (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _slot_lease_files(bundle):
    root = bundle["artifact_root"] / p05_pilot.P05DEVELOPMENT_NAMESPACE / "slot_leases"
    return sorted(root.rglob("lease.json")) if root.exists() else []


def test_run_orchestrates_36_through_real_serialization(monkeypatch, tmp_path, contract):
    torch = _torch()
    runtime = _runtime()
    bundle = _fake_bundle(tmp_path, contract)
    kernel = _FakeKernel(runtime, torch)
    _install_run_mocks(monkeypatch, bundle, kernel, _fake_inputs(bundle["units"]))

    summary = _invoke_run(bundle, tmp_path)
    assert summary["status"] == "complete"
    assert summary["started"] == 36
    assert summary["completed"] == 36
    assert summary["failed"] == 0
    assert summary["unstarted"] == 0
    assert summary["optimizer_steps"] == 36 * 120
    assert summary["execution_authorized"] is False
    assert summary["outer_evaluation_authorized"] is False

    artifact = bundle["artifact_root"]
    run_dir = _run_dir(bundle)
    best = sorted((run_dir / "executions").rglob("best.pt"))
    terminal = sorted((run_dir / "executions").rglob("terminal.pt"))
    assert len(best) == 36
    assert len(terminal) == 36
    assert len(list((run_dir / "executions").rglob("*.pt"))) == 72
    for path in best + terminal:
        loaded = torch.load(path, weights_only=True, map_location="cpu")
        assert "state_dict" in loaded

    assert len(_slot_lease_files(bundle)) == 36
    events = _events(run_dir)
    assert events.count("planned") == 36
    assert events.count("started") == 36
    assert events.count("completed") == 36
    assert "failed" not in events

    assert {entry.name for entry in artifact.iterdir()} == {p05_pilot.P05DEVELOPMENT_NAMESPACE}
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _invoke_run(bundle, tmp_path)
    assert exc.value.reason_code == "pilot_lease_exists"


def test_run_stops_after_partial_kernel_failure(monkeypatch, tmp_path, contract):
    torch = _torch()
    runtime = _runtime()
    bundle = _fake_bundle(tmp_path, contract)
    kernel = _FakeKernel(
        runtime, torch, status="resource_failure", steps=2, epochs=1, reason="deadline_exceeded"
    )
    _install_run_mocks(monkeypatch, bundle, kernel, _fake_inputs(bundle["units"]))

    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _invoke_run(bundle, tmp_path)
    assert exc.value.reason_code == "fit_failed"

    run_dir = _run_dir(bundle)
    summary = _read_summary(run_dir)
    assert summary["status"] == "fail"
    assert summary["started"] == 1
    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["optimizer_steps"] == 2
    assert summary["optimizer_steps_exact"] is True

    execution_dirs = [path for path in (run_dir / "executions").iterdir() if path.is_dir()]
    assert len(execution_dirs) == 1
    assert (execution_dirs[0] / "summary.json").is_file()
    assert (execution_dirs[0] / "terminal.pt").is_file()
    assert len(_slot_lease_files(bundle)) == 1
    events = _events(run_dir)
    assert events.count("started") == 1
    assert "completed" not in events
    assert events.count("failed") == 1


def test_run_retains_exact_steps_when_serializer_raises(monkeypatch, tmp_path, contract):
    torch = _torch()
    runtime = _runtime()
    bundle = _fake_bundle(tmp_path, contract)
    kernel = _FakeKernel(runtime, torch)
    _install_run_mocks(monkeypatch, bundle, kernel, _fake_inputs(bundle["units"]))

    real_save = p05_pilot.core._save_state

    def failing_save(torch_arg, state, path, *args, **kwargs):
        if "executions" in Path(path).parts:
            raise RuntimeError("serializer boom")
        return real_save(torch_arg, state, path, *args, **kwargs)

    monkeypatch.setattr(p05_pilot.core, "_save_state", failing_save)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _invoke_run(bundle, tmp_path)
    assert exc.value.reason_code == "pilot_execution_failed"

    run_dir = _run_dir(bundle)
    summary = _read_summary(run_dir)
    assert summary["status"] == "fail"
    assert summary["started"] == 1
    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["optimizer_steps"] == 120
    assert summary["optimizer_steps_exact"] is True


def test_run_zero_start_when_input_load_fails_after_lease(monkeypatch, tmp_path, contract):
    torch = _torch()
    runtime = _runtime()
    bundle = _fake_bundle(tmp_path, contract)
    kernel = _FakeKernel(runtime, torch)
    _install_run_mocks(monkeypatch, bundle, kernel, _fake_inputs(bundle["units"]))

    def boom(_bundle):
        raise p05_pilot.P05PilotError("unit_uid_missing_representation")

    monkeypatch.setattr(p05_pilot, "prepare_role_inputs", boom)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _invoke_run(bundle, tmp_path)
    assert exc.value.reason_code == "unit_uid_missing_representation"

    run_dir = _run_dir(bundle)
    summary = _read_summary(run_dir)
    assert summary["status"] == "fail"
    assert summary["started"] == 0
    assert summary["completed"] == 0
    assert summary["failed"] == 0
    assert summary["optimizer_steps"] == 0
    executions = run_dir / "executions"
    assert not executions.exists() or not any(executions.iterdir())
    assert _slot_lease_files(bundle) == []


def test_run_rejects_changed_protected_provenance(monkeypatch, tmp_path, contract):
    torch = _torch()
    runtime = _runtime()
    bundle = _fake_bundle(tmp_path, contract)
    kernel = _FakeKernel(runtime, torch)
    _install_run_mocks(monkeypatch, bundle, kernel, _fake_inputs(bundle["units"]))

    calls = {"count": 0}

    def drifting(*args, **kwargs):
        calls["count"] += 1
        return {"protected_environment_sha256": ("a" * 64 if calls["count"] == 1 else "b" * 64)}

    monkeypatch.setattr(p05_pilot.core, "_capture_provenance", drifting)
    with pytest.raises(P05CoreError):
        _invoke_run(bundle, tmp_path)

    summary = _read_summary(_run_dir(bundle))
    assert summary["status"] == "fail"
    assert summary["started"] == 36
    assert summary["completed"] == 36


# --------------------------------------------------------------------------- #
# Occupied leases, resource refusal and guarded caps
# --------------------------------------------------------------------------- #


def _stub_preflight(monkeypatch, bundle):
    monkeypatch.setattr(p05_pilot, "prepare", lambda *a, **k: bundle)
    monkeypatch.setattr(
        p05_pilot,
        "_check_resources",
        lambda *a, **k: {"device": "cpu", "free_disk_bytes": 0, "free_cuda_bytes": 0},
    )
    monkeypatch.setattr(p05_pilot, "_checkpoint_preflight", lambda *a, **k: None)
    monkeypatch.setattr(p05_pilot, "_store_pilot_plan", lambda *a, **k: None)
    monkeypatch.setattr(
        p05_pilot, "prepare_role_inputs", lambda *a, **k: pytest.fail("arrays loaded")
    )
    monkeypatch.setattr(p05_pilot, "_development_kernel", lambda: pytest.fail("kernel loaded"))


def _call_preflight(bundle, tmp_path):
    return p05_pilot.preflight(
        tmp_path, bundle["artifact_root"], tmp_path / "c.json", tmp_path / "p.json"
    )


def test_preflight_refuses_occupied_permit_lease_before_arrays(monkeypatch, tmp_path, contract):
    bundle = _fake_bundle(tmp_path, contract)
    _stub_preflight(monkeypatch, bundle)
    p05_pilot._pilot_lease_dir(bundle["artifact_root"], bundle["permit_sha256"]).mkdir(parents=True)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _call_preflight(bundle, tmp_path)
    assert exc.value.reason_code == "pilot_lease_exists"


def test_preflight_refuses_occupied_run_directory(monkeypatch, tmp_path, contract):
    bundle = _fake_bundle(tmp_path, contract)
    _stub_preflight(monkeypatch, bundle)
    p05_pilot._pilot_run_dir(bundle["artifact_root"], bundle["permit_sha256"]).mkdir(parents=True)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _call_preflight(bundle, tmp_path)
    assert exc.value.reason_code == "pilot_run_exists"


def test_preflight_refuses_shared_slot_lease_under_changed_permit(monkeypatch, tmp_path, contract):
    bundle = _fake_bundle(tmp_path, contract)
    changed = dict(bundle)
    changed["permit_sha256"] = "9" * 64
    changed["pilot_plan_id"] = "8" * 64
    _stub_preflight(monkeypatch, changed)
    slot = changed["slots"][0]
    p05_pilot._slot_lease_dir(
        changed["artifact_root"],
        changed["contract_sha256"],
        changed["core_plan_id"],
        slot["slot_id"],
    ).mkdir(parents=True)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _call_preflight(changed, tmp_path)
    assert exc.value.reason_code == "slot_lease_exists"


def test_run_refuses_low_disk_before_lease_or_fit(monkeypatch, tmp_path, contract):
    torch = _torch()
    runtime = _runtime()
    bundle = _fake_bundle(tmp_path, contract)
    kernel = _FakeKernel(runtime, torch)
    _install_run_mocks(monkeypatch, bundle, kernel, _fake_inputs(bundle["units"]))
    monkeypatch.setattr(p05_pilot, "_free_disk_bytes", lambda path: 0)
    monkeypatch.setattr(
        p05_pilot, "prepare_role_inputs", lambda *a, **k: pytest.fail("arrays loaded")
    )

    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _invoke_run(bundle, tmp_path)
    assert exc.value.reason_code == "insufficient_free_disk"
    assert not p05_pilot._pilot_lease_dir(bundle["artifact_root"], bundle["permit_sha256"]).exists()
    assert not _run_dir(bundle).exists()
    assert _slot_lease_files(bundle) == []


def test_guarded_recorder_runs_real_storage_cap(monkeypatch):
    written = []
    closed = {"done": False}

    class _Recorder:
        def __call__(self, record):
            written.append(dict(record))

        def close(self):
            closed["done"] = True

    monkeypatch.setattr(
        p05_pilot,
        "_directory_size_bytes",
        lambda path: p05_pilot.PRIVATE_STORAGE_CEILING_BYTES + 1,
    )
    guarded = p05_pilot._GuardedRecorder(
        _Recorder(), (lambda: p05_pilot._enforce_storage_cap(Path("/tmp")),)
    )
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        guarded({"epoch": 1})
    assert exc.value.reason_code == "private_storage_ceiling_exceeded"
    assert written == [{"epoch": 1}]
    guarded.close()
    assert closed["done"] is True


def test_enforce_cuda_cap_reads_allocated_memory(monkeypatch):
    torch = _torch()
    p05_pilot._enforce_cuda_cap(torch, "cpu")
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda *a, **k: p05_pilot.MAXIMUM_CUDA_ALLOCATED_BYTES + 1,
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot._enforce_cuda_cap(torch, "cuda")
    assert exc.value.reason_code == "fit_cuda_exceeded"


# --------------------------------------------------------------------------- #
# Unexpected kernel throw keeps lower-bound counts without retries
# --------------------------------------------------------------------------- #


class _ThrowingKernel:
    def train_development_fit(self, **kwargs):
        on_epoch = kwargs.get("on_epoch")
        if on_epoch is not None:
            on_epoch({"epoch": 1, "total_optimizer_steps": 4})
        raise RuntimeError("kernel exploded")


def test_run_unexpected_kernel_throw_keeps_lower_bound_steps(monkeypatch, tmp_path, contract):
    _torch()
    bundle = _fake_bundle(tmp_path, contract)
    _install_run_mocks(monkeypatch, bundle, _ThrowingKernel(), _fake_inputs(bundle["units"]))

    with pytest.raises(p05_pilot.P05PilotError) as exc:
        _invoke_run(bundle, tmp_path)
    assert exc.value.reason_code == "pilot_execution_failed"

    run_dir = _run_dir(bundle)
    summary = _read_summary(run_dir)
    assert summary["status"] == "fail"
    assert summary["started"] == 1
    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["optimizer_steps"] == p05_pilot.BATCH_DRAWS_PER_EPOCH
    assert summary["optimizer_steps_exact"] is False
    assert len(_slot_lease_files(bundle)) == 1
    events = _events(run_dir)
    assert events.count("started") == 1
    assert "completed" not in events


# --------------------------------------------------------------------------- #
# Real synthetic CPU fit through persistence and acceptance
# --------------------------------------------------------------------------- #


def _development_test_module():
    path = _REPO_ROOT / "tests" / "test_p05_development.py"
    spec = importlib.util.spec_from_file_location("_p05_development_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_small_fit_accepts_then_corrupt_checkpoint_rejects(tmp_path, contract):
    torch = _torch()
    helpers = _development_test_module()
    fit, validation = helpers._small_roles()
    fit_values, fit_observations, fit_metadata = fit
    validation_values, validation_observations, _ = validation

    result = p05_pilot.train_fit(
        {
            "fitting_values": fit_values,
            "fitting_observations": fit_observations,
            "noise": fit_metadata,
            "validation_values": validation_values,
            "validation_observations": validation_observations,
        },
        {"fitting_role_id": "role-1", "validation_role_id": "role-2"},
        {"recipe_id": "D0-M", "seed": 20260925},
        "cpu",
        time.perf_counter() + 3600.0,
        lambda record: None,
    )
    assert result.status == "complete"
    assert result.validation_logits is not None

    unit = {
        "unit_id": "unit-cwa",
        "station": "cwa",
        "fitting_role_id": "role-1",
        "validation_role_id": "role-2",
        "validation_uids": sorted(str(row.uid) for row in validation_observations),
        "validation_classes": sorted({str(row.target) for row in validation_observations}),
    }
    slot = {"slot_id": "slot-1", "recipe_id": "D0-M", "seed": 20260925}
    unit_inputs = {
        "validation_values": validation_values,
        "validation_observations": validation_observations,
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    p05_pilot.persist_result(torch, run_dir, unit, slot, result)
    p05_pilot.check_completed_result(
        result, run_dir, unit, slot, contract, unit_inputs, torch, "cpu"
    )

    best_path = run_dir / "executions" / p05_pilot.execution_id(unit, slot) / "best.pt"
    assert best_path.is_file()
    corrupted = {name: value + 10.0 for name, value in result.best_state_dict.items()}
    p05_pilot.core._save_state(torch, corrupted, best_path)
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot.check_completed_result(
            result, run_dir, unit, slot, contract, unit_inputs, torch, "cpu"
        )
    assert exc.value.reason_code in {
        "restored_best_logits_mismatch",
        "restored_best_metric_mismatch",
    }


@pytest.mark.parametrize("violation", [None, "duplicate", "outside", "master", "test_master"])
def test_source_boundaries_are_rechecked_before_fitting(violation):
    unit = {"context_id": "ctx", "fitting_uids": ["f"], "validation_uids": ["v"]}
    roles = {"ctx": {"outer_fit": {"f", "v"}, "outer_test": {"t"}}}
    manifest = {uid: {"master": uid} for uid in ("f", "v", "t")}
    reasons = {
        "duplicate": "unit_uid_duplicate",
        "outside": "unit_uid_outside_outer_fit",
        "master": "unit_master_overlap",
        "test_master": "outer_test_master_in_unit",
    }
    if violation == "duplicate":
        unit["fitting_uids"].append("f")
    elif violation == "outside":
        roles["ctx"]["outer_fit"].remove("v")
    elif violation == "master":
        manifest["v"]["master"] = "f"
    elif violation == "test_master":
        manifest["t"]["master"] = "f"
    if violation is None:
        p05_pilot._validate_unit_sources(unit, roles, manifest)
    else:
        with pytest.raises(p05_pilot.P05PilotError) as exc:
            p05_pilot._validate_unit_sources(unit, roles, manifest)
        assert exc.value.reason_code == reasons[violation]
