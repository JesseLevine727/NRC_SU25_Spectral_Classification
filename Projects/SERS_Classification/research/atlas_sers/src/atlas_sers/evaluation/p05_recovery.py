"""P05-T012 bounded checkpoint recovery.

Replays the single completed-but-unpersisted numerical fit from the original
failed P05-T009 attempt with corrected real-torch serialization, then runs the
remaining 33 plan records.  Original evidence and the failed lease stay
immutable; every new result is verified by reloading its real checkpoint.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

RECOVERY_PERMIT_SHA256 = "01e0835d8a6ece2ee98cee654e9f707e894643788dc9daabc438527c1c19c058"
P05CORE_NAMESPACE = "p05core"
RECOVERY_NAMESPACE = "recovery"
ORIGINAL_FILE_COUNT = 6
LEDGER_PLANNED = 34
LEDGER_STARTED = 1
EXPECTED_RECOVERY_EXECUTIONS = 34
PROVENANCE_KEYS = ("runtime", "compute", "dependencies", "dependency_lock_sha256")


class P05RecoveryError(RuntimeError):
    """Stable recovery failure carrying a path-free reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _core() -> Any:
    return importlib.import_module("atlas_sers.evaluation.p05_core_run")


def _recovery_execution_id(original_execution_id: str) -> str:
    digest = (
        _core()
        ._canon()
        .sha256_value(
            {
                "permit_sha256": RECOVERY_PERMIT_SHA256,
                "recovery_of": original_execution_id,
            }
        )
    )
    return f"P05REC-{digest[:24]}"


def _recovery_root(artifact_root: Path) -> Path:
    return artifact_root / P05CORE_NAMESPACE / RECOVERY_NAMESPACE / RECOVERY_PERMIT_SHA256


def _load_permit(permit_path: Path) -> dict[str, Any]:
    core = _core()
    canonical = core._canon()
    raw = core._read_bytes(Path(permit_path), "recovery_permit")
    try:
        permit = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P05RecoveryError("permit_malformed") from error
    if not isinstance(permit, Mapping):
        raise P05RecoveryError("permit_malformed")
    digest = canonical.sha256_bytes(canonical.canonical_json_bytes(dict(permit)))
    if digest != RECOVERY_PERMIT_SHA256:
        raise P05RecoveryError("permit_digest_mismatch")
    return dict(permit)


def _verify_numerical_sources(project_root: Path, permit: Mapping[str, Any]) -> None:
    core = _core()
    canonical = core._canon()
    sources = permit.get("numerical_source_sha256")
    if not isinstance(sources, Mapping) or not sources:
        raise P05RecoveryError("permit_source_pins_missing")
    for relative, digest in sources.items():
        path = project_root / relative
        core._reject_symlink_chain(path)
        if not path.is_file() or canonical.sha256_file(path) != digest:
            raise P05RecoveryError("numerical_source_hash_mismatch")


def _verify_original_manifest(original_run_dir: Path, permit: Mapping[str, Any]) -> dict[str, Any]:
    core = _core()
    canonical = core._canon()
    manifest_path = original_run_dir / "manifest.json"
    core._reject_symlink_chain(manifest_path)
    if canonical.sha256_file(manifest_path) != permit["original_manifest_sha256"]:
        raise P05RecoveryError("original_manifest_hash_mismatch")
    manifest = core._read_json(manifest_path, "original_manifest")
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(files, Mapping) or len(files) != ORIGINAL_FILE_COUNT:
        raise P05RecoveryError("original_inventory_count_mismatch")
    for relative, record in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise P05RecoveryError("original_inventory_path_invalid")
        if not isinstance(record, Mapping) or set(record) != {"sha256", "size_bytes"}:
            raise P05RecoveryError("original_inventory_record_malformed")
        target = original_run_dir / relative
        core._reject_symlink_chain(target)
        if not target.is_file():
            raise P05RecoveryError("original_inventory_file_missing")
        if canonical.sha256_file(target) != record["sha256"]:
            raise P05RecoveryError("original_inventory_hash_mismatch")
        if target.stat().st_size != record["size_bytes"]:
            raise P05RecoveryError("original_inventory_size_mismatch")
    return dict(files)


def _verify_original_inventory(original_run_dir: Path, files: Mapping[str, Any]) -> None:
    present: set[str] = set()
    for path in original_run_dir.rglob("*"):
        if path.is_symlink():
            raise P05RecoveryError("symlink_path_rejected")
        if path.is_file():
            present.add(path.relative_to(original_run_dir).as_posix())
    if present != set(files) | {"manifest.json"}:
        raise P05RecoveryError("original_inventory_set_mismatch")


def _verify_original_checkpoint_absent(original_run_dir: Path) -> None:
    for path in original_run_dir.rglob("*.pt"):
        if path.is_symlink():
            raise P05RecoveryError("symlink_path_rejected")
        raise P05RecoveryError("original_checkpoint_present")


def _verify_original_lease(artifact_root: Path, permit: Mapping[str, Any]) -> Path:
    core = _core()
    canonical = core._canon()
    lease_path = (
        artifact_root
        / P05CORE_NAMESPACE
        / "leases"
        / permit["original_contract_sha256"]
        / "lease.json"
    )
    core._reject_symlink_chain(lease_path)
    if canonical.sha256_file(lease_path) != permit["original_lease_sha256"]:
        raise P05RecoveryError("original_lease_hash_mismatch")
    lease = core._read_json(lease_path, "original_lease")
    if not isinstance(lease, Mapping):
        raise P05RecoveryError("original_lease_malformed")
    if lease.get("state") != "failed" or lease.get("reason_code") != "RuntimeError":
        raise P05RecoveryError("original_lease_not_failed")
    return lease_path


def _read_original_ledger(original_run_dir: Path) -> str:
    core = _core()
    ledger_path = original_run_dir / "ledger.jsonl"
    core._reject_symlink_chain(ledger_path)
    events: list[Any] = []
    for line in core._read_bytes(ledger_path, "original_ledger").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise P05RecoveryError("original_ledger_malformed") from error
    planned = [event for event in events if event.get("event") == "planned"]
    started = [event for event in events if event.get("event") == "started"]
    if len(planned) != LEDGER_PLANNED or len(started) != LEDGER_STARTED:
        raise P05RecoveryError("original_ledger_event_mismatch")
    if len(events) != LEDGER_PLANNED + LEDGER_STARTED:
        raise P05RecoveryError("original_ledger_event_mismatch")
    if any(event.get("event") != "planned" for event in events[:LEDGER_PLANNED]):
        raise P05RecoveryError("original_ledger_order_mismatch")
    if events[LEDGER_PLANNED].get("event") != "started":
        raise P05RecoveryError("original_ledger_order_mismatch")
    execution_id = started[0].get("execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        raise P05RecoveryError("original_ledger_started_malformed")
    return execution_id


def _load_original_result(
    original_run_dir: Path, execution_id: str, permit: Mapping[str, Any]
) -> Any:
    core = _core()
    canonical = core._canon()
    directory = original_run_dir / "executions" / execution_id
    result_path = directory / "result.json"
    history_path = directory / "history.json"
    core._reject_symlink_chain(result_path)
    core._reject_symlink_chain(history_path)
    if canonical.sha256_file(result_path) != permit["original_result_sha256"]:
        raise P05RecoveryError("original_result_hash_mismatch")
    if canonical.sha256_file(history_path) != permit["original_history_sha256"]:
        raise P05RecoveryError("original_history_hash_mismatch")
    record = core._read_json(result_path, "original_result")
    history = core._read_json(history_path, "original_history")
    if not isinstance(record, Mapping) or "history" in record:
        raise P05RecoveryError("original_result_malformed")
    if record.get("execution_id") != execution_id:
        raise P05RecoveryError("original_result_execution_mismatch")
    if record.get("status") != "complete":
        raise P05RecoveryError("original_result_not_complete")
    if int(record.get("optimizer_steps", -1)) != core.OPTIMIZER_STEPS:
        raise P05RecoveryError("original_result_steps_mismatch")
    if not isinstance(history, list) or len(history) != core.EPOCHS:
        raise P05RecoveryError("original_history_malformed")
    if record.get("final_state_digest") != permit["original_final_state_digest"]:
        raise P05RecoveryError("original_state_digest_mismatch")
    joined = SimpleNamespace(**dict(record))
    joined.history = [dict(item) for item in history]
    return joined


def _build_recovery_schedule(
    plan: Mapping[str, Any], started_execution_id: str, permit: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fits = [dict(fit) for fit in plan["smoke_fits"]]
    if len(fits) != EXPECTED_RECOVERY_EXECUTIONS:
        raise P05RecoveryError("recovery_schedule_mismatch")
    if started_execution_id != permit["original_execution_id"]:
        raise P05RecoveryError("original_execution_id_mismatch")
    first = fits[0]
    if first["execution_id"] != started_execution_id or first["execution_kind"] != "primary":
        raise P05RecoveryError("original_fit_mismatch")
    recovered = dict(first)
    recovered["execution_id"] = _recovery_execution_id(first["execution_id"])
    recovered["recovery_of"] = first["execution_id"]
    schedule = [recovered] + fits[1:]
    return schedule, recovered


def _provenance_comparable(provenance: Mapping[str, Any]) -> dict[str, Any]:
    comparable: dict[str, Any] = {key: provenance.get(key) for key in PROVENANCE_KEYS}
    storage = provenance.get("storage")
    comparable["storage"] = {
        "artifact_filesystem_total_bytes": (
            storage.get("artifact_filesystem_total_bytes") if isinstance(storage, Mapping) else None
        )
    }
    return comparable


def _verify_provenance_match(
    original_provenance: Mapping[str, Any], current_provenance: Mapping[str, Any]
) -> None:
    if _provenance_comparable(original_provenance) != _provenance_comparable(current_provenance):
        raise P05RecoveryError("recovery_provenance_mismatch")


def _prepare(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    plan_id: str,
    contract_sha256: str,
    permit_path: Path | str,
) -> dict[str, Any]:
    core = _core()
    core._configure_environment()
    canonical = core._canon()
    permit = _load_permit(Path(permit_path))
    project_root = Path(project_root)
    artifact_root = Path(artifact_root)
    if not core._is_hex64(plan_id) or not core._is_hex64(contract_sha256):
        raise P05RecoveryError("identifier_malformed")
    repository_root = core._repository_root(project_root)
    core._assert_artifact_location(project_root, repository_root, artifact_root)
    _verify_numerical_sources(project_root, permit)
    contract, observed_contract = core._load_contract(Path(contract_path), contract_sha256)
    if observed_contract != permit["original_contract_sha256"]:
        raise P05RecoveryError("original_contract_mismatch")
    recipes = {recipe["recipe_id"]: recipe for recipe in contract["recipes"]}
    model = contract["model"]
    support, p01_run, _p04_run = core._authenticate(artifact_root, contract)
    expected_rows = int(contract["population"]["rows"])
    manifest_uids = core._manifest_uids(support, expected_rows)
    plan_content = core._read_bytes(core._plan_dir(artifact_root, plan_id) / "plan.json", "plan")
    if canonical.sha256_bytes(plan_content) != plan_id:
        raise P05RecoveryError("plan_content_hash_mismatch")
    plan = core._build_plan(support, contract, project_root)
    core._minimal_plan_checks(plan, contract)
    if canonical.sha256_bytes(canonical.canonical_json_bytes(plan)) != plan_id:
        raise P05RecoveryError("plan_authority_mismatch")
    if plan_id != permit["original_plan_id"]:
        raise P05RecoveryError("plan_id_mismatch")
    original_run_dir = (
        artifact_root / P05CORE_NAMESPACE / "runs" / permit["original_contract_sha256"]
    )
    _verify_original_lease(artifact_root, permit)
    files = _verify_original_manifest(original_run_dir, permit)
    _verify_original_inventory(original_run_dir, files)
    _verify_original_checkpoint_absent(original_run_dir)
    started_id = _read_original_ledger(original_run_dir)
    if started_id != permit["original_execution_id"]:
        raise P05RecoveryError("original_execution_id_mismatch")
    original_result = _load_original_result(original_run_dir, started_id, permit)
    core._check_fit(dict(plan["smoke_fits"][0]), original_result, recipes, model)
    original_provenance = core._read_json(
        original_run_dir / "provenance.json", "original_provenance"
    )
    schedule, recovered_fit = _build_recovery_schedule(plan, started_id, permit)
    recovery_root = _recovery_root(artifact_root)
    lease_dir = recovery_root / "lease"
    run_dir = recovery_root / "run"
    if lease_dir.exists() or lease_dir.is_symlink() or run_dir.exists() or run_dir.is_symlink():
        raise P05RecoveryError("recovery_attempt_exists")
    numpy, pandas, torch, smoke = core._import_stack()
    device = core._select_device(torch)
    expected_device = "cuda" if int(original_result.peak_cuda_bytes) > 0 else "cpu"
    if device != expected_device:
        raise P05RecoveryError("recovery_device_mismatch")
    observation_type = importlib.import_module("atlas_sers.evaluation.p05_sampling").Observation
    intensity, labels = core._load_representation(
        p01_run / core.REPRESENTATION_REL,
        contract["input_pins"]["representation_sha256"],
        manifest_uids,
        expected_rows,
    )
    uid_index = {uid: index for index, uid in enumerate(labels)}
    core._verify_role_membership(plan, uid_index, set(manifest_uids))
    role_inputs = core._build_role_inputs(
        plan, p01_run, contract, intensity, uid_index, pandas, observation_type
    )
    core._validate_source_inputs(smoke, role_inputs)
    core._validate_sample_capacity(smoke, role_inputs, contract)
    core._checkpoint_preflight(torch, artifact_root)
    provenance_before = core._capture_provenance(repository_root, project_root, artifact_root)
    _verify_provenance_match(original_provenance, provenance_before)
    return {
        "permit": permit,
        "project_root": project_root,
        "artifact_root": artifact_root,
        "repository_root": repository_root,
        "contract": contract,
        "contract_sha256": observed_contract,
        "plan": plan,
        "plan_id": plan_id,
        "schedule": schedule,
        "recovered_fit": recovered_fit,
        "original_result": original_result,
        "original_run_dir": original_run_dir,
        "original_execution_id": started_id,
        "role_inputs": role_inputs,
        "smoke": smoke,
        "torch": torch,
        "device": device,
        "p04_runtime": importlib.import_module("atlas_sers.evaluation.p04_runtime"),
        "recipes": recipes,
        "model": model,
        "provenance_before": provenance_before,
    }


def _write_recovery_link(run_dir: Path, fit: Mapping[str, Any]) -> None:
    core = _core()
    directory = run_dir / "executions" / str(fit["execution_id"])
    core._atomic_write(
        directory / "recovery_link.json",
        core._canon().canonical_json_bytes(
            {
                "recovery_of": fit["recovery_of"],
                "execution_id": fit["execution_id"],
                "permit_sha256": RECOVERY_PERMIT_SHA256,
            }
        ),
    )


def _reload_and_verify(torch: Any, p04_runtime: Any, directory: Path, result: Any) -> None:
    core = _core()
    state_path = directory / "state.pt"
    core._reject_symlink_chain(state_path)
    try:
        loaded = torch.load(state_path, weights_only=True, map_location="cpu")
    except Exception as error:
        raise P05RecoveryError("checkpoint_reload_failed") from error
    payload = loaded.get("state_dict") if isinstance(loaded, Mapping) else None
    if not isinstance(payload, Mapping):
        raise P05RecoveryError("checkpoint_reload_malformed")
    state_hash = getattr(p04_runtime, "_state_hash", None)
    if state_hash is None:
        raise P05RecoveryError("checkpoint_state_hash_missing")
    if state_hash(payload) != result.final_state_digest:
        raise P05RecoveryError("checkpoint_digest_mismatch")


def _recheck_original(
    artifact_root: Path, original_run_dir: Path, permit: Mapping[str, Any]
) -> None:
    _verify_original_lease(artifact_root, permit)
    files = _verify_original_manifest(original_run_dir, permit)
    _verify_original_inventory(original_run_dir, files)
    _verify_original_checkpoint_absent(original_run_dir)
    execution_id = _read_original_ledger(original_run_dir)
    _load_original_result(original_run_dir, execution_id, permit)


def preflight_recovery(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    plan_id: str,
    contract_sha256: str,
    permit_path: Path | str,
) -> dict[str, Any]:
    context = _prepare(
        project_root=project_root,
        artifact_root=artifact_root,
        contract_path=contract_path,
        plan_id=plan_id,
        contract_sha256=contract_sha256,
        permit_path=permit_path,
    )
    return {
        "status": "ok",
        "command": "preflight",
        "permit_sha256": RECOVERY_PERMIT_SHA256,
        "contract_sha256": context["contract_sha256"],
        "plan_id": context["plan_id"],
        "original_execution_id": context["original_execution_id"],
        "recovery_execution_id": context["recovered_fit"]["execution_id"],
        "scheduled_executions": len(context["schedule"]),
        "device": context["device"],
        "checkpoint_preflight": "pass",
    }


def run_recovery(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    plan_id: str,
    contract_sha256: str,
    permit_path: Path | str,
) -> dict[str, Any]:
    core = _core()
    core._configure_environment()
    canonical = core._canon()
    context = _prepare(
        project_root=project_root,
        artifact_root=artifact_root,
        contract_path=contract_path,
        plan_id=plan_id,
        contract_sha256=contract_sha256,
        permit_path=permit_path,
    )
    artifact_root = context["artifact_root"]
    contract = context["contract"]
    contract_sha256 = context["contract_sha256"]
    plan_id = context["plan_id"]
    schedule = context["schedule"]
    recovered_fit = context["recovered_fit"]
    original_result = context["original_result"]
    role_inputs = context["role_inputs"]
    smoke = context["smoke"]
    torch = context["torch"]
    device = context["device"]
    p04_runtime = context["p04_runtime"]
    recipes = context["recipes"]
    model = context["model"]
    permit = context["permit"]
    provenance_before = context["provenance_before"]
    recovery_root = _recovery_root(artifact_root)
    lease_dir = recovery_root / "lease"
    run_dir = recovery_root / "run"
    core._mkdir_exclusive(lease_dir, "recovery_lease_exists")
    core._atomic_write(
        lease_dir / "lease.json",
        canonical.canonical_json_bytes(
            {
                "state": "running",
                "permit_sha256": RECOVERY_PERMIT_SHA256,
                "contract_sha256": contract_sha256,
                "plan_id": plan_id,
            }
        ),
    )
    core._mkdir_exclusive(run_dir, "recovery_run_exists")
    core._atomic_write(
        run_dir / "provenance.json", canonical.canonical_json_bytes(provenance_before)
    )
    ledger_path = run_dir / "ledger.jsonl"

    def _ledger(entry: Mapping[str, Any]) -> None:
        with ledger_path.open("ab") as stream:
            stream.write(canonical.canonical_json_bytes(dict(entry)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    for fit in schedule:
        _ledger(
            {
                "event": "planned",
                "execution_id": fit["execution_id"],
                "execution_kind": fit["execution_kind"],
            }
        )
    global_deadline = time.perf_counter() + core.MAX_TOTAL_SECONDS
    wall_start = time.monotonic()
    executions: list[dict[str, Any]] = []
    reload_count = 0
    try:
        for fit in schedule:
            if time.perf_counter() >= global_deadline:
                raise P05RecoveryError("global_deadline_exceeded")
            if time.monotonic() - wall_start > core.MAX_TOTAL_SECONDS:
                raise P05RecoveryError("total_wall_exceeded")
            core._reserve_row(lease_dir / "rows", fit)
            _ledger({"event": "started", "execution_id": fit["execution_id"]})
            values, observations, noise = role_inputs[fit["role_label"]]
            try:
                result = core._train_one(
                    fit,
                    values,
                    observations,
                    noise,
                    smoke,
                    device,
                    global_deadline,
                    run_dir / "histories" / f"{fit['execution_id']}.jsonl",
                )
            except BaseException as error:
                core._persist_error(run_dir, fit, error)
                _ledger({"event": "failed", "execution_id": fit["execution_id"]})
                raise
            try:
                core._persist_execution(run_dir, fit, result, torch)
                if fit is recovered_fit:
                    _write_recovery_link(run_dir, fit)
                _reload_and_verify(
                    torch, p04_runtime, run_dir / "executions" / fit["execution_id"], result
                )
                reload_count += 1
            except BaseException as error:
                core._persist_error(run_dir, fit, error)
                _ledger({"event": "persistence_failed", "execution_id": fit["execution_id"]})
                raise
            try:
                core._check_fit(fit, result, recipes, model)
                if fit is recovered_fit:
                    core._compare_replay(original_result, result)
            except BaseException as error:
                core._persist_error(run_dir, fit, error)
                _ledger({"event": "acceptance_failed", "execution_id": fit["execution_id"]})
                raise
            _ledger(
                {
                    "event": "completed",
                    "execution_id": fit["execution_id"],
                    "status": result.status,
                }
            )
            executions.append({"fit": fit, "result": result})
        core._authenticate(artifact_root, contract)
        _recheck_original(artifact_root, context["original_run_dir"], permit)
        core._check_acceptance(contract, executions)
        provenance_after = core._capture_provenance(
            context["repository_root"], context["project_root"], artifact_root
        )
        core._assert_protected_identity(provenance_before, provenance_after)
        wall_seconds = time.monotonic() - wall_start
        if wall_seconds > core.MAX_TOTAL_SECONDS:
            raise P05RecoveryError("total_wall_exceeded")
        budget = permit["budget"]
        optimizer_steps = sum(int(item["result"].optimizer_steps) for item in executions)
        if len(executions) != int(budget["new_executions"]):
            raise P05RecoveryError("budget_execution_mismatch")
        if optimizer_steps != int(budget["new_optimizer_steps"]):
            raise P05RecoveryError("budget_step_mismatch")
        total_executions = int(budget["prior_executions"]) + len(executions)
        total_steps = int(budget["prior_optimizer_steps"]) + optimizer_steps
        if total_executions != int(budget["total_executions"]):
            raise P05RecoveryError("budget_total_execution_mismatch")
        if total_steps != int(budget["total_optimizer_steps"]):
            raise P05RecoveryError("budget_total_step_mismatch")
        summary = {
            "status": "complete",
            "permit_sha256": RECOVERY_PERMIT_SHA256,
            "contract_sha256": contract_sha256,
            "plan_id": plan_id,
            "device": device,
            "recovered_execution_id": recovered_fit["execution_id"],
            "recovery_of": recovered_fit["recovery_of"],
            "executions": len(executions),
            "primary_fits": sum(
                1 for item in executions if item["fit"]["execution_kind"] == "primary"
            ),
            "replays": sum(1 for item in executions if item["fit"]["execution_kind"] == "replay"),
            "optimizer_steps": optimizer_steps,
            "prior_executions": int(budget["prior_executions"]),
            "prior_optimizer_steps": int(budget["prior_optimizer_steps"]),
            "total_executions": total_executions,
            "total_optimizer_steps": total_steps,
            "checkpoint_reload_count": reload_count,
            "original_replay_comparison": "pass",
            "original_evidence_unchanged": "pass",
            "sum_elapsed_seconds": sum(
                float(item["result"].elapsed_seconds) for item in executions
            ),
            "maximum_peak_cuda_bytes": max(
                int(item["result"].peak_cuda_bytes) for item in executions
            ),
            "wall_seconds": wall_seconds,
            "checkpoint_preflight": "pass",
            "checks": {"status": "pass"},
        }
        core._atomic_write(
            run_dir / "provenance_after.json", canonical.canonical_json_bytes(provenance_after)
        )
        core._atomic_write(run_dir / "summary.json", canonical.canonical_json_bytes(summary))
        core._atomic_write(
            lease_dir / "lease.json",
            canonical.canonical_json_bytes(
                {
                    "state": "complete",
                    "permit_sha256": RECOVERY_PERMIT_SHA256,
                    "contract_sha256": contract_sha256,
                    "plan_id": plan_id,
                }
            ),
        )
        core._write_manifest(run_dir)
    except BaseException as error:
        try:
            core._atomic_write(
                lease_dir / "lease.json",
                canonical.canonical_json_bytes(
                    {
                        "state": "failed",
                        "permit_sha256": RECOVERY_PERMIT_SHA256,
                        "contract_sha256": contract_sha256,
                        "plan_id": plan_id,
                        "reason_code": getattr(error, "reason_code", type(error).__name__),
                    }
                ),
            )
        except Exception:
            pass
        core._write_failure(run_dir, contract_sha256, plan_id, error)
        if isinstance(error, (KeyboardInterrupt, SystemExit, P05RecoveryError)):
            raise
        raise P05RecoveryError("recovery_execution_failed") from error
    return {
        "status": "complete",
        "command": "recover",
        "permit_sha256": RECOVERY_PERMIT_SHA256,
        "plan_id": plan_id,
        "contract_sha256": contract_sha256,
        "executions": summary["executions"],
        "primary_fits": summary["primary_fits"],
        "replays": summary["replays"],
        "optimizer_steps": summary["optimizer_steps"],
        "total_executions": summary["total_executions"],
        "total_optimizer_steps": summary["total_optimizer_steps"],
    }
