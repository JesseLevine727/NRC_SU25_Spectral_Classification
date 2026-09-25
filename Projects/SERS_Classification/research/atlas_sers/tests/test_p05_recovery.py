"""Bounded tests for the P05-T012 checkpoint-recovery surface.

The fixtures here are tiny synthetic JSON and real CPU torch checkpoints.  No
model is ever trained: the orchestration tests substitute ``core._train_one``
with a controlled result factory while keeping the real schedule, persistence,
checkpoint reload, fit check, replay comparison, acceptance and original
evidence recheck paths intact.  Numerical imports are guarded with
``importorskip``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_recovery as recovery
from tests.test_p05_core_run import (
    RECIPES,
    ROLES,
    SEEDS,
    _contract,
    _digest,
    _history,
    _plan,
    _result,
)

PLAN_ID = "b" * 64
CONTRACT_SHA = "a" * 64
PROTECTED_ENV = "9" * 64


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canon_bytes(payload) -> bytes:
    return core._canon().canonical_json_bytes(payload)


def _write(path: Path, payload) -> None:
    path.write_bytes(_canon_bytes(payload))


def _group(role_label: str, recipe_id: str, sparse_variant: bool = False) -> str:
    if not role_label.endswith("_sparse") or sparse_variant:
        return recipe_id
    return "A" if recipe_id in ("D0-M", "D2") else "B"


def _expected_digest(torch, p04_runtime, role_label="cwa_dense", recipe_id="D0-M") -> str:
    group = _group(role_label, recipe_id)
    state = {"weight": torch.tensor([float(sum(group.encode("utf-8")))])}
    return p04_runtime._state_hash(state)


def _started_id(plan) -> str:
    return plan["smoke_fits"][0]["execution_id"]


def _synthetic_world(
    tmp_path: Path,
    started_id: str,
    *,
    original_final_digest: str | None = None,
    history=None,
) -> SimpleNamespace:
    project_root = tmp_path / "project"
    artifact_root = tmp_path / "artifacts"
    repository_root = tmp_path / "repo"
    for directory in (project_root, artifact_root, repository_root):
        directory.mkdir()

    source_rel = "src/atlas_sers/evaluation/p05_smoke.py"
    source_path = project_root / source_rel
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"# synthetic numerical source\n")

    original_dir = artifact_root / recovery.P05CORE_NAMESPACE / "runs" / CONTRACT_SHA
    execution_dir = original_dir / "executions" / started_id
    execution_dir.mkdir(parents=True)
    lease_dir = artifact_root / recovery.P05CORE_NAMESPACE / "leases" / CONTRACT_SHA
    lease_dir.mkdir(parents=True)
    lease_path = lease_dir / "lease.json"
    _write(lease_path, {"state": "failed", "reason_code": "RuntimeError"})

    original = _result("cwa_dense", "D0-M", SEEDS[0])
    record = {key: value for key, value in vars(original).items() if key != "history"}
    record["execution_id"] = started_id
    if original_final_digest is not None:
        record["final_state_digest"] = original_final_digest
    result_path = execution_dir / "result.json"
    _write(result_path, record)
    history_path = execution_dir / "history.json"
    _write(history_path, history if history is not None else original.history)

    ledger_path = original_dir / "ledger.jsonl"
    planned = [
        _canon_bytes({"event": "planned", "execution_id": f"planned-{index:02d}"})
        for index in range(34)
    ]
    started = _canon_bytes({"event": "started", "execution_id": started_id})
    ledger_path.write_bytes(b"\n".join(planned + [started]) + b"\n")

    provenance_path = original_dir / "provenance.json"
    _write(
        provenance_path,
        {
            "runtime": {"python": "3.11"},
            "compute": {"device": "cpu"},
            "dependencies": {"torch": "synthetic"},
            "dependency_lock_sha256": "d" * 64,
            "protected_environment_sha256": PROTECTED_ENV,
            "storage": {
                "artifact_filesystem_total_bytes": 1,
                "artifact_filesystem_free_bytes": 1,
            },
        },
    )
    state_path = original_dir / "P01_STATE.json"
    _write(state_path, {"execution_status": "complete"})
    notes_path = original_dir / "notes.txt"
    notes_path.write_bytes(b"synthetic original evidence\n")

    relative_files = {
        "provenance.json": provenance_path,
        "ledger.jsonl": ledger_path,
        f"executions/{started_id}/result.json": result_path,
        f"executions/{started_id}/history.json": history_path,
        "P01_STATE.json": state_path,
        "notes.txt": notes_path,
    }
    files = {
        relative: {"sha256": _sha_bytes(path.read_bytes()), "size_bytes": path.stat().st_size}
        for relative, path in relative_files.items()
    }
    manifest_path = original_dir / "manifest.json"
    _write(manifest_path, {"files": files})

    permit = {
        "schema_version": "nato-sers-p05-checkpoint-recovery-synthetic-v1",
        "original_contract_sha256": CONTRACT_SHA,
        "original_plan_id": PLAN_ID,
        "original_execution_id": started_id,
        "original_manifest_sha256": _sha_bytes(manifest_path.read_bytes()),
        "original_lease_sha256": _sha_bytes(lease_path.read_bytes()),
        "original_result_sha256": _sha_bytes(result_path.read_bytes()),
        "original_history_sha256": _sha_bytes(history_path.read_bytes()),
        "original_final_state_digest": record["final_state_digest"],
        "numerical_source_sha256": {source_rel: _sha_bytes(source_path.read_bytes())},
        "budget": {
            "prior_executions": 1,
            "new_executions": 34,
            "total_executions": 35,
            "prior_optimizer_steps": 32,
            "new_optimizer_steps": 1088,
            "total_optimizer_steps": 1120,
            "automatic_retries": 0,
        },
    }
    return SimpleNamespace(
        permit=permit,
        permit_sha256=_sha_bytes(_canon_bytes(permit)),
        project_root=project_root,
        artifact_root=artifact_root,
        repository_root=repository_root,
        original_run_dir=original_dir,
        original_execution_id=started_id,
        source_path=source_path,
        lease_path=lease_path,
        manifest_path=manifest_path,
        result_path=result_path,
        history_path=history_path,
        files=files,
        original_result=original,
    )


def _run_dir(world) -> Path:
    return (
        world.artifact_root
        / recovery.P05CORE_NAMESPACE
        / recovery.RECOVERY_NAMESPACE
        / world.permit_sha256
        / "run"
    )


def _orchestration(
    tmp_path: Path,
    monkeypatch,
    world,
    *,
    sparse_variant: bool = False,
    bad_state: bool = False,
) -> SimpleNamespace:
    torch = pytest.importorskip("torch")
    p04_runtime = pytest.importorskip("atlas_sers.evaluation.p04_runtime")
    contract = _contract()
    plan = _plan(contract)
    started = _started_id(plan)
    monkeypatch.setattr(recovery, "RECOVERY_PERMIT_SHA256", world.permit_sha256)
    schedule, recovered = recovery._build_recovery_schedule(plan, started, world.permit)
    original_result = recovery._load_original_result(world.original_run_dir, started, world.permit)

    def state_for(fit):
        if bad_state:
            return {"weight": torch.tensor([1.0]), "broken": lambda: None}
        group = _group(fit["role_label"], fit["recipe_id"], sparse_variant)
        return {"weight": torch.tensor([float(sum(group.encode("utf-8")))])}

    def result_for(fit):
        result = _result(fit["role_label"], fit["recipe_id"], fit["seed"])
        state = state_for(fit)
        result.state_dict = state
        result.final_state_digest = (
            _digest("broken-state") if bad_state else p04_runtime._state_hash(state)
        )
        return result

    provenance_before = {
        "runtime": {"python": "3.11"},
        "compute": {"device": "cpu"},
        "dependencies": {"torch": "synthetic"},
        "dependency_lock_sha256": "d" * 64,
        "protected_environment_sha256": PROTECTED_ENV,
        "storage": {"artifact_filesystem_total_bytes": 1, "artifact_filesystem_free_bytes": 1},
    }
    payload = {
        "permit": world.permit,
        "project_root": world.project_root,
        "artifact_root": world.artifact_root,
        "repository_root": world.repository_root,
        "contract": contract,
        "contract_sha256": CONTRACT_SHA,
        "plan": plan,
        "plan_id": PLAN_ID,
        "support": {},
        "schedule": schedule,
        "recovered_fit": recovered,
        "original_result": original_result,
        "original_run_dir": world.original_run_dir,
        "original_execution_id": started,
        "role_inputs": {role: (object(), object(), object()) for role in ROLES},
        "smoke": SimpleNamespace(),
        "torch": torch,
        "device": "cpu",
        "p04_runtime": p04_runtime,
        "recipes": {recipe["recipe_id"]: recipe for recipe in contract["recipes"]},
        "model": contract["model"],
        "provenance_before": provenance_before,
    }

    def train_one(fit, values, observations, noise, smoke, device, deadline, history_path):
        return result_for(fit)

    return SimpleNamespace(
        torch=torch,
        p04_runtime=p04_runtime,
        contract=contract,
        plan=plan,
        schedule=schedule,
        recovered=recovered,
        payload=payload,
        provenance_before=provenance_before,
        permit_sha256=world.permit_sha256,
        train_one=train_one,
        result_for=result_for,
    )


def _install(monkeypatch, orchestration) -> None:
    monkeypatch.setattr(recovery, "RECOVERY_PERMIT_SHA256", orchestration.permit_sha256)
    monkeypatch.setattr(recovery, "_prepare", lambda **kwargs: orchestration.payload)
    monkeypatch.setattr(core, "_authenticate", lambda *args, **kwargs: (None, None, None))
    monkeypatch.setattr(
        core,
        "_capture_provenance",
        lambda *args, **kwargs: dict(orchestration.provenance_before),
    )
    monkeypatch.setattr(core, "_train_one", orchestration.train_one)


def _invoke_recovery(world):
    return recovery.run_recovery(
        project_root=world.project_root,
        artifact_root=world.artifact_root,
        contract_path=world.project_root / "contract.json",
        plan_id=PLAN_ID,
        contract_sha256=CONTRACT_SHA,
        permit_path=world.project_root / "permit.json",
    )


def _failure_reason(run_dir: Path) -> str | None:
    for name in ("summary.json", "failure.json"):
        path = run_dir / name
        if path.is_file():
            payload = json.loads(path.read_bytes())
            return payload.get("reason_code") or payload.get("reason")
    return None


def test_public_permit_matches_constant_and_budget():
    canonical = core._canon()
    permit_path = (
        Path(__file__).resolve().parents[1] / "plan" / "contracts" / "p05_checkpoint_recovery.json"
    )
    permit = json.loads(permit_path.read_bytes().decode("utf-8"))
    digest = canonical.sha256_bytes(canonical.canonical_json_bytes(permit))
    assert digest == recovery.RECOVERY_PERMIT_SHA256
    assert permit["budget"]["prior_executions"] == 1
    assert permit["budget"]["new_executions"] == 34
    assert permit["budget"]["total_executions"] == 35
    assert permit["budget"]["prior_optimizer_steps"] == 32
    assert permit["budget"]["new_optimizer_steps"] == 1088
    assert permit["budget"]["total_optimizer_steps"] == 1120
    assert "src/atlas_sers/evaluation/p05_smoke.py" in permit["numerical_source_sha256"]


def test_permit_roundtrip_and_tamper(tmp_path, monkeypatch):
    canonical = core._canon()
    permit = {"original_plan_id": PLAN_ID}
    permit_path = tmp_path / "permit.json"
    _write(permit_path, permit)
    monkeypatch.setattr(
        recovery, "RECOVERY_PERMIT_SHA256", canonical.sha256_bytes(_canon_bytes(permit))
    )
    assert recovery._load_permit(permit_path)["original_plan_id"] == PLAN_ID

    _write(permit_path, {"original_plan_id": "c" * 64})
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._load_permit(permit_path)
    assert caught.value.reason_code == "permit_digest_mismatch"

    permit_path.write_bytes(b"not-json")
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._load_permit(permit_path)
    assert caught.value.reason_code == "permit_malformed"


def test_numerical_source_tamper_detected(tmp_path, monkeypatch):
    plan = _plan(_contract())
    world = _synthetic_world(tmp_path, _started_id(plan))
    monkeypatch.setattr(recovery, "RECOVERY_PERMIT_SHA256", world.permit_sha256)
    recovery._verify_numerical_sources(world.project_root, world.permit)

    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_numerical_sources(world.project_root, {})
    assert caught.value.reason_code == "permit_source_pins_missing"

    world.source_path.write_bytes(b"tampered source\n")
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_numerical_sources(world.project_root, world.permit)
    assert caught.value.reason_code == "numerical_source_hash_mismatch"


def test_original_manifest_tamper_detected(tmp_path):
    plan = _plan(_contract())
    world = _synthetic_world(tmp_path, _started_id(plan))
    pristine = world.manifest_path.read_bytes()
    recovery._verify_original_manifest(world.original_run_dir, world.permit)

    world.manifest_path.write_bytes(b'{"files": {}}')
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_original_manifest(world.original_run_dir, world.permit)
    assert caught.value.reason_code == "original_manifest_hash_mismatch"

    world.manifest_path.write_bytes(pristine)
    manifest = json.loads(pristine)
    manifest["files"] = dict(list(manifest["files"].items())[:5])
    world.manifest_path.write_bytes(_canon_bytes(manifest))
    permit = dict(world.permit)
    permit["original_manifest_sha256"] = _sha_bytes(world.manifest_path.read_bytes())
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_original_manifest(world.original_run_dir, permit)
    assert caught.value.reason_code == "original_inventory_count_mismatch"


def test_original_inventory_tamper_detected(tmp_path):
    plan = _plan(_contract())
    world = _synthetic_world(tmp_path, _started_id(plan))
    recovery._verify_original_inventory(world.original_run_dir, world.files)

    notes = world.original_run_dir / "notes.txt"
    pristine = notes.read_bytes()
    notes.write_bytes(b"tampered evidence\n")
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_original_manifest(world.original_run_dir, world.permit)
    assert caught.value.reason_code == "original_inventory_hash_mismatch"

    notes.write_bytes(pristine)
    manifest = json.loads(world.manifest_path.read_bytes())
    relative = next(iter(manifest["files"]))
    manifest["files"][relative]["size_bytes"] += 1
    world.manifest_path.write_bytes(_canon_bytes(manifest))
    permit = dict(world.permit)
    permit["original_manifest_sha256"] = _sha_bytes(world.manifest_path.read_bytes())
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_original_manifest(world.original_run_dir, permit)
    assert caught.value.reason_code == "original_inventory_size_mismatch"

    (world.original_run_dir / "extra.bin").write_bytes(b"extra")
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_original_inventory(world.original_run_dir, world.files)
    assert caught.value.reason_code == "original_inventory_set_mismatch"


def test_original_checkpoint_present_rejected(tmp_path):
    plan = _plan(_contract())
    world = _synthetic_world(tmp_path, _started_id(plan))
    recovery._verify_original_checkpoint_absent(world.original_run_dir)
    checkpoint = world.original_run_dir / "executions" / world.original_execution_id / "state.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_original_checkpoint_absent(world.original_run_dir)
    assert caught.value.reason_code == "original_checkpoint_present"


def test_original_lease_tamper_and_state(tmp_path):
    plan = _plan(_contract())
    world = _synthetic_world(tmp_path, _started_id(plan))
    recovery._verify_original_lease(world.artifact_root, world.permit)

    world.lease_path.write_bytes(b'{"state": "failed", "reason_code": "RuntimeError"}')
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_original_lease(world.artifact_root, world.permit)
    assert caught.value.reason_code == "original_lease_hash_mismatch"

    _write(world.lease_path, {"state": "running", "reason_code": None})
    permit = dict(world.permit)
    permit["original_lease_sha256"] = _sha_bytes(world.lease_path.read_bytes())
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_original_lease(world.artifact_root, permit)
    assert caught.value.reason_code == "original_lease_not_failed"


def test_original_ledger_shape(tmp_path):
    plan = _plan(_contract())
    world = _synthetic_world(tmp_path, _started_id(plan))
    assert recovery._read_original_ledger(world.original_run_dir) == world.original_execution_id

    ledger = world.original_run_dir / "ledger.jsonl"
    lines = ledger.read_text().splitlines()
    ledger.write_text("\n".join(lines[:33]) + "\n")
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._read_original_ledger(world.original_run_dir)
    assert caught.value.reason_code == "original_ledger_event_mismatch"


def test_original_result_loaded_and_preserved(tmp_path):
    plan = _plan(_contract())
    world = _synthetic_world(tmp_path, _started_id(plan))
    loaded = recovery._load_original_result(
        world.original_run_dir, world.original_execution_id, world.permit
    )
    assert loaded.final_state_digest == world.permit["original_final_state_digest"]
    assert len(loaded.history) == core.EPOCHS
    assert not hasattr(loaded, "result_hash")

    before = {
        path: path.read_bytes() for path in world.original_run_dir.rglob("*") if path.is_file()
    }
    recovery._recheck_original(world.artifact_root, world.original_run_dir, world.permit)
    after = {
        path: path.read_bytes() for path in world.original_run_dir.rglob("*") if path.is_file()
    }
    assert before == after
    assert not list(world.original_run_dir.rglob("*.pt"))


def test_recovery_schedule_rewrites_first_only(monkeypatch):
    plan = _plan(_contract())
    original = plan["smoke_fits"][0]
    started = original["execution_id"]
    monkeypatch.setattr(recovery, "RECOVERY_PERMIT_SHA256", "e" * 64)
    schedule, recovered = recovery._build_recovery_schedule(
        plan, started, {"original_execution_id": started}
    )
    assert len(schedule) == 34
    assert recovered["recovery_of"] == started
    assert recovered["execution_id"].startswith("P05REC-")
    assert recovered["execution_id"] != started
    assert recovered["execution_kind"] == "primary"
    for key in ("fit_id", "p05_role_id", "recipe_id", "seed"):
        assert recovered[key] == original[key]
    assert schedule[0] == recovered
    assert schedule[1:] == plan["smoke_fits"][1:]


def test_recovery_schedule_rejects_wrong_original(monkeypatch):
    plan = _plan(_contract())
    first = plan["smoke_fits"][0]["execution_id"]
    wrong = plan["smoke_fits"][1]["execution_id"]
    monkeypatch.setattr(recovery, "RECOVERY_PERMIT_SHA256", "e" * 64)
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._build_recovery_schedule(plan, wrong, {"original_execution_id": first})
    assert caught.value.reason_code == "original_execution_id_mismatch"
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._build_recovery_schedule(plan, wrong, {"original_execution_id": wrong})
    assert caught.value.reason_code == "original_fit_mismatch"


def test_provenance_match_detects_differences_and_ignores_storage_free_bytes():
    base = {
        "runtime": {"python": "3.11"},
        "compute": {"device": "cpu"},
        "dependencies": {"torch": "synthetic"},
        "dependency_lock_sha256": "f" * 64,
        "storage": {"artifact_filesystem_total_bytes": 100, "artifact_filesystem_free_bytes": 1},
    }
    recovery._verify_provenance_match(base, base)
    recovery._verify_provenance_match(
        base,
        {
            **base,
            "storage": {
                "artifact_filesystem_total_bytes": 100,
                "artifact_filesystem_free_bytes": 99,
            },
        },
    )
    comparable = recovery._provenance_comparable(base)
    assert list(comparable) == list(recovery.PROVENANCE_KEYS) + ["storage"]
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._verify_provenance_match(base, {**base, "runtime": {"python": "3.12"}})
    assert caught.value.reason_code == "recovery_provenance_mismatch"


def test_reload_and_verify_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    p04_runtime = pytest.importorskip("atlas_sers.evaluation.p04_runtime")
    directory = tmp_path / "execution"
    directory.mkdir()
    state = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    core._save_state(torch, state, directory / "state.pt")
    result = SimpleNamespace(final_state_digest=p04_runtime._state_hash(state))
    recovery._reload_and_verify(torch, p04_runtime, directory, result)

    result.final_state_digest = _digest("wrong")
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._reload_and_verify(torch, p04_runtime, directory, result)
    assert caught.value.reason_code == "checkpoint_digest_mismatch"

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(recovery.P05RecoveryError) as caught:
        recovery._reload_and_verify(torch, p04_runtime, empty, result)
    assert caught.value.reason_code == "checkpoint_reload_failed"


def test_run_recovery_full_budget_real_checkpoints_and_evidence_unchanged(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    p04_runtime = pytest.importorskip("atlas_sers.evaluation.p04_runtime")
    plan = _plan(_contract())
    started = _started_id(plan)
    world = _synthetic_world(
        tmp_path, started, original_final_digest=_expected_digest(torch, p04_runtime)
    )
    original_before = {
        path: path.read_bytes() for path in world.original_run_dir.rglob("*") if path.is_file()
    }
    lease_before = world.lease_path.read_bytes()
    orchestration = _orchestration(tmp_path, monkeypatch, world)
    _install(monkeypatch, orchestration)

    summary = _invoke_recovery(world)
    run_dir = _run_dir(world)
    assert summary["status"] == "complete"
    assert summary["executions"] == 34
    assert summary["primary_fits"] == 32 and summary["replays"] == 2
    assert summary["optimizer_steps"] == 1088
    assert summary["total_executions"] == 35
    assert summary["total_optimizer_steps"] == 1120
    checkpoints = sorted(run_dir.glob("executions/*/state.pt"))
    assert len(checkpoints) == 34
    for checkpoint in checkpoints:
        record = json.loads((checkpoint.parent / "result.json").read_bytes())
        loaded = torch.load(checkpoint, weights_only=True, map_location="cpu")
        assert p04_runtime._state_hash(loaded["state_dict"]) == record["final_state_digest"]

    payload = json.loads((run_dir / "summary.json").read_bytes())
    assert payload["original_replay_comparison"] == "pass"
    assert payload["checkpoint_reload_count"] == 34
    assert payload["original_evidence_unchanged"] == "pass"
    assert payload["executions"] == 34 and payload["optimizer_steps"] == 1088
    assert payload["prior_executions"] == 1 and payload["prior_optimizer_steps"] == 32

    assert (run_dir / "provenance.json").read_bytes() == _canon_bytes(
        orchestration.provenance_before
    )
    assert (
        json.loads((run_dir / "provenance_after.json").read_bytes())
        == orchestration.provenance_before
    )
    lease = json.loads(
        (
            world.artifact_root
            / recovery.P05CORE_NAMESPACE
            / recovery.RECOVERY_NAMESPACE
            / world.permit_sha256
            / "lease"
            / "lease.json"
        ).read_bytes()
    )
    assert lease["state"] == "complete"
    original_after = {
        path: path.read_bytes() for path in world.original_run_dir.rglob("*") if path.is_file()
    }
    assert original_before == original_after
    assert world.lease_path.read_bytes() == lease_before


def test_run_recovery_refuses_second_attempt(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    p04_runtime = pytest.importorskip("atlas_sers.evaluation.p04_runtime")
    plan = _plan(_contract())
    started = _started_id(plan)
    world = _synthetic_world(
        tmp_path, started, original_final_digest=_expected_digest(torch, p04_runtime)
    )
    orchestration = _orchestration(tmp_path, monkeypatch, world)
    _install(monkeypatch, orchestration)

    _invoke_recovery(world)
    ledger = _run_dir(world) / "ledger.jsonl"
    ledger_before = ledger.read_bytes()
    with pytest.raises(core.P05CoreError) as caught:
        _invoke_recovery(world)
    assert caught.value.reason_code == "recovery_lease_exists"
    assert ledger.read_bytes() == ledger_before


@pytest.mark.parametrize(
    "mode,expected_reason",
    [
        ("state", "replay_terminal_state_mismatch"),
        ("history", "replay_history_mismatch"),
    ],
)
def test_run_recovery_replay_mismatch_stops_after_one(tmp_path, monkeypatch, mode, expected_reason):
    torch = pytest.importorskip("torch")
    p04_runtime = pytest.importorskip("atlas_sers.evaluation.p04_runtime")
    plan = _plan(_contract())
    started = _started_id(plan)
    history = None
    if mode == "state":
        mismatch_digest = _digest("original-replay-state-mismatch")
    else:
        mismatch_digest = _expected_digest(torch, p04_runtime)
        history = [dict(entry) for entry in _history(RECIPES[0], False)]
        history[0]["chemical_ce"] = history[0]["chemical_ce"] + 5.0
    world = _synthetic_world(
        tmp_path, started, original_final_digest=mismatch_digest, history=history
    )
    orchestration = _orchestration(tmp_path, monkeypatch, world)
    _install(monkeypatch, orchestration)

    with pytest.raises(recovery.P05RecoveryError) as caught:
        _invoke_recovery(world)
    assert caught.value.reason_code == "recovery_execution_failed"
    run_dir = _run_dir(world)
    events = [
        json.loads(line)
        for line in (run_dir / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert sum(1 for event in events if event["event"] == "started") == 1
    assert any(event["event"] == "acceptance_failed" for event in events)
    assert not any(event["event"] == "completed" for event in events)
    errors = list(run_dir.glob("executions/*/error.json"))
    assert len(errors) == 1
    assert _failure_reason(run_dir) == expected_reason


def test_run_recovery_saving_failure_retains_diagnostic(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    p04_runtime = pytest.importorskip("atlas_sers.evaluation.p04_runtime")
    plan = _plan(_contract())
    started = _started_id(plan)
    world = _synthetic_world(
        tmp_path, started, original_final_digest=_expected_digest(torch, p04_runtime)
    )
    orchestration = _orchestration(tmp_path, monkeypatch, world, bad_state=True)
    _install(monkeypatch, orchestration)

    with pytest.raises(recovery.P05RecoveryError) as caught:
        _invoke_recovery(world)
    assert caught.value.reason_code == "recovery_execution_failed"
    run_dir = _run_dir(world)
    errors = list(run_dir.glob("executions/*/error.json"))
    assert len(errors) == 1
    payload = json.loads(errors[0].read_bytes())
    assert payload["status"] == "fail"
    assert core._is_hex64(payload["traceback_sha256"])
    events = [
        json.loads(line)
        for line in (run_dir / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert sum(1 for event in events if event["event"] == "started") == 1
    assert any(event["event"] == "persistence_failed" for event in events)
    assert not any(event["event"] == "completed" for event in events)


def test_run_recovery_sparse_mismatch_detected(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    p04_runtime = pytest.importorskip("atlas_sers.evaluation.p04_runtime")
    plan = _plan(_contract())
    started = _started_id(plan)
    world = _synthetic_world(
        tmp_path, started, original_final_digest=_expected_digest(torch, p04_runtime)
    )
    orchestration = _orchestration(tmp_path, monkeypatch, world, sparse_variant=True)
    _install(monkeypatch, orchestration)

    with pytest.raises(recovery.P05RecoveryError) as caught:
        _invoke_recovery(world)
    assert caught.value.reason_code == "recovery_execution_failed"
    run_dir = _run_dir(world)
    assert len(list(run_dir.glob("executions/*/state.pt"))) == 34
    assert _failure_reason(run_dir) == "sparse_equivalence_state_mismatch"
