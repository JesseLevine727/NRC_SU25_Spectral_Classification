"""P05-T009 governed private plan and smoke boundary.

Two operations only: an immutable metadata-only plan and a bounded numerical
smoke run guarded by a contract-level lease.  The reviewed planner
(:mod:`atlas_sers.evaluation.p05_core_plan`) and numerical kernel
(:mod:`atlas_sers.evaluation.p05_smoke`) are reused unchanged.  No future
development, outer evaluation, model selection, calibration or public export is
reachable here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

P05CORE_NAMESPACE = "p05core"

PINNED_REPORTS = {
    "p01": "a9b3502adca7dad5edc48a5ef160fa19f81854cd5163c5c49340c7ccab266c49",
    "p04plan": "e979b3e804d9f4826ee6354bf3ee4a65e0486c0e94e1a69ef51bd4d04dc252c3",
}
REPORT_NAMES = {
    "p01": "P01_VALIDATION_REPORT.json",
    "p04plan": "P04_PLAN_VALIDATION_REPORT.json",
}
REPRESENTATION_REL = "representations/R_MIN_400_1800.npz"
REPRESENTATION_FEATURES = 1401
REPRESENTATION_AXIS_START = 400
REPRESENTATION_AXIS_STOP = 1801
MANIFEST_UID_COLUMN = "observation_uid"
QC_COLUMNS = ("observation_uid", "first_difference_noise_mad", "intensity_range")

EPOCHS = 8
DRAWS_PER_EPOCH = 4
OPTIMIZER_STEPS = EPOCHS * DRAWS_PER_EPOCH
PLANNED_EXECUTIONS = 34
PLANNED_PRIMARY = 32
PLANNED_REPLAYS = 2
MAX_FIT_SECONDS = 120.0
MAX_TOTAL_SECONDS = 900.0
MAX_CUDA_ALLOCATED_BYTES = 4294967296
MIN_FREE_CUDA_BYTES = 5368709120
EXPECTED_FINITE_GRADIENT_BATCHES = 32
EXPECTED_SUPCON_BATCHES = EPOCHS * DRAWS_PER_EPOCH
SPARSE_SUPCON_ELIGIBLE_ANCHORS = 64
SPARSE_SUPCON_ZERO_POSITIVE = 64

REQUIRED_HISTORY_FIELDS = frozenset(
    {
        "epoch",
        "chemical_ce",
        "total_loss",
        "supcon_loss",
        "paired_loss",
        "supcon_enabled",
        "paired_enabled",
        "supcon_available_batches",
        "paired_available_batches",
        "eligible_anchor_count",
        "zero_positive_anchor_count",
        "paired_master_count",
        "gradient_norm_mean",
        "gradient_norm_max",
        "head_gradient_norm_mean",
        "backbone_gradient_norm_mean",
        "clipped_fraction",
        "embedding_variance",
        "embedding_norm_mean",
        "train_ba",
        "train_nll",
        "train_predicted_class_count",
        "optimizer_steps",
    }
)
BOOL_HISTORY_FIELDS = frozenset({"supcon_enabled", "paired_enabled"})
STATE_DIGEST_FIELDS = (
    "initial_state_digest",
    "final_state_digest",
    "initial_backbone_digest",
    "final_backbone_digest",
    "initial_head_digest",
    "final_head_digest",
)
STREAM_DIGEST_FIELDS = (
    "initial_backbone_digest",
    "augmentation_digest",
    "sampling_digest",
    "pair_digest",
)


class P05CoreError(RuntimeError):
    """Stable, stage-specific failure with a path-free reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _planner() -> Any:
    return importlib.import_module("atlas_sers.evaluation.p05_core_plan")


def _canon() -> Any:
    return importlib.import_module("atlas_sers.governance.canonical")


def _is_hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _require_digest(value: Any, code: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not _is_hex64(value):
        raise P05CoreError(code)


def _reject_symlink_chain(path: Path) -> None:
    candidate = Path(path)
    while True:
        if candidate.is_symlink():
            raise P05CoreError("symlink_path_rejected")
        if candidate.parent == candidate:
            return
        candidate = candidate.parent


def _read_bytes(path: Path, label: str) -> bytes:
    _reject_symlink_chain(path)
    try:
        return path.read_bytes()
    except OSError as error:
        raise P05CoreError(f"{label}_unreadable") from error


def _read_json(path: Path, label: str) -> Any:
    raw = _read_bytes(path, label)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P05CoreError(f"{label}_malformed") from error


def _atomic_write(path: Path, content: bytes) -> None:
    _reject_symlink_chain(path.parent)
    if path.is_symlink():
        raise P05CoreError("output_symlink_rejected")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".p05-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _mkdir_exclusive(path: Path, code: str) -> None:
    _reject_symlink_chain(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise P05CoreError("symlink_path_rejected")
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise P05CoreError(code) from error


def _repository_root(project_root: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise P05CoreError("repository_root_unavailable") from error
    location = completed.stdout.strip()
    if not location:
        raise P05CoreError("repository_root_unavailable")
    return Path(location).resolve()


def _assert_artifact_location(
    project_root: Path, repository_root: Path, artifact_root: Path
) -> None:
    _reject_symlink_chain(project_root)
    _reject_symlink_chain(artifact_root)
    artifact = artifact_root.resolve()
    repository = repository_root.resolve()
    package = project_root.resolve()
    if artifact == repository or repository in artifact.parents:
        raise P05CoreError("artifact_inside_repository")
    if artifact == package or package in artifact.parents:
        raise P05CoreError("artifact_inside_package")


def _load_contract(contract_path: Path, contract_sha256: str) -> tuple[dict[str, Any], str]:
    if not _is_hex64(contract_sha256):
        raise P05CoreError("contract_pin_malformed")
    contract = _read_json(contract_path, "contract")
    if not isinstance(contract, Mapping):
        raise P05CoreError("contract_malformed")
    _planner().validate_core_contract(contract)
    observed = _canon().sha256_value(contract)
    if observed != contract_sha256.lower():
        raise P05CoreError("contract_digest_mismatch")
    return dict(contract), observed


def _authenticate_phase(
    phase_root: Path,
    run_dir: Path,
    run_id: str,
    report_name: str,
    report_sha256: str,
    required_files: Mapping[str, str],
) -> None:
    _reject_symlink_chain(phase_root)
    latest = _read_json(phase_root / "LATEST.json", "latest")
    state = _read_json(run_dir / "_STATE.json", "state")
    if not isinstance(latest, Mapping) or not isinstance(state, Mapping):
        raise P05CoreError("phase_records_malformed")
    if latest.get("status") != "pass":
        raise P05CoreError("latest_not_passed")
    if latest.get("run_id") != run_id or state.get("run_id") != run_id:
        raise P05CoreError("run_id_mismatch")
    if state.get("execution_status") != "complete" or state.get("scientific_status") != "pass":
        raise P05CoreError("state_not_successful")
    if latest.get("protected_state_sha256") != state.get("protected_state_sha256"):
        raise P05CoreError("protected_state_mismatch")
    files = state.get("files")
    if not isinstance(files, Mapping):
        raise P05CoreError("state_files_missing")
    if latest.get("report_sha256") != report_sha256 or files.get(report_name) != report_sha256:
        raise P05CoreError("report_pin_mismatch")
    report_path = run_dir / report_name
    _reject_symlink_chain(report_path)
    if _canon().sha256_file(report_path) != report_sha256:
        raise P05CoreError("report_hash_mismatch")
    for relative, digest in required_files.items():
        if files.get(relative) != digest:
            raise P05CoreError("state_file_pin_mismatch")
        path = run_dir / relative
        _reject_symlink_chain(path)
        if _canon().sha256_file(path) != digest:
            raise P05CoreError("input_hash_mismatch")


def _authenticate(artifact_root: Path, contract: Mapping[str, Any]) -> tuple[Any, Path, Path]:
    pins = contract["input_pins"]
    p01_run = artifact_root / "p01" / "runs" / pins["p01_run_id"]
    p04_run = artifact_root / "p04plan" / "runs" / pins["p04plan_run_id"]
    _authenticate_phase(
        artifact_root / "p01",
        p01_run,
        pins["p01_run_id"],
        REPORT_NAMES["p01"],
        PINNED_REPORTS["p01"],
        {
            "primary_manifest.csv": pins["manifest_sha256"],
            REPRESENTATION_REL: pins["representation_sha256"],
        },
    )
    _authenticate_phase(
        artifact_root / "p04plan",
        p04_run,
        pins["p04plan_run_id"],
        REPORT_NAMES["p04plan"],
        PINNED_REPORTS["p04plan"],
        {
            "context_registry.csv": pins["contexts_sha256"],
            "role_registry.csv": pins["roles_sha256"],
        },
    )
    support_module = importlib.import_module("atlas_sers.evaluation.p05_support")
    try:
        support = support_module.load_support_inputs(
            manifest_path=p01_run / "primary_manifest.csv",
            manifest_sha256=pins["manifest_sha256"],
            contexts_path=p04_run / "context_registry.csv",
            contexts_sha256=pins["contexts_sha256"],
            roles_path=p04_run / "role_registry.csv",
            roles_sha256=pins["roles_sha256"],
        )
    except support_module.SupportAuditError as error:
        raise P05CoreError("support_audit_failed") from error
    return support, p01_run, p04_run


def _manifest_uids(support: Any, expected_rows: int) -> list[str]:
    uids: list[str] = []
    for row in support.manifest:
        value = row.get(MANIFEST_UID_COLUMN)
        if not isinstance(value, str) or not value:
            raise P05CoreError("manifest_uid_missing")
        uids.append(value)
    if len(set(uids)) != len(uids):
        raise P05CoreError("manifest_uid_duplicate")
    if len(uids) != expected_rows:
        raise P05CoreError("manifest_population_mismatch")
    return uids


def _load_representation(
    path: Path, expected_sha256: str, manifest_uids: Sequence[str], expected_rows: int
) -> tuple[Any, list[str]]:
    _reject_symlink_chain(path)
    if _canon().sha256_file(path) != expected_sha256:
        raise P05CoreError("representation_hash_mismatch")
    numpy = importlib.import_module("numpy")
    try:
        with numpy.load(path, allow_pickle=False) as archive:
            if not {"axis_cm1", "intensity", "observation_uid"}.issubset(set(archive.files)):
                raise P05CoreError("representation_members_missing")
            axis = numpy.asarray(archive["axis_cm1"])
            intensity = numpy.asarray(archive["intensity"])
            uids = numpy.asarray(archive["observation_uid"])
    except P05CoreError:
        raise
    except Exception as error:
        raise P05CoreError("representation_unreadable") from error
    expected_axis = numpy.arange(REPRESENTATION_AXIS_START, REPRESENTATION_AXIS_STOP)
    if axis.shape != expected_axis.shape or not bool(numpy.all(axis == expected_axis)):
        raise P05CoreError("representation_axis_mismatch")
    if intensity.ndim != 2 or intensity.shape[1] != REPRESENTATION_FEATURES:
        raise P05CoreError("representation_shape_mismatch")
    if uids.ndim != 1:
        raise P05CoreError("representation_uid_shape_mismatch")
    labels = [str(value) for value in uids.tolist()]
    if len(set(labels)) != len(labels):
        raise P05CoreError("representation_uid_duplicate")
    if intensity.shape[0] != len(labels):
        raise P05CoreError("representation_population_mismatch")
    if len(labels) != expected_rows or len(manifest_uids) != expected_rows:
        raise P05CoreError("representation_population_mismatch")
    if set(labels) != set(manifest_uids):
        raise P05CoreError("representation_population_mismatch")
    return intensity.astype("float32", copy=False), labels


def _verify_role_membership(
    plan: Mapping[str, Any], uid_index: Mapping[str, int], population: set[str]
) -> None:
    observations: dict[str, list[Mapping[str, Any]]] = {}
    for row in plan["smoke_observations"]:
        observations.setdefault(row["role_label"], []).append(row)
    for role in plan["smoke_roles"]:
        fitting = [str(uid) for uid in role["fitting_uids"]]
        if fitting != sorted(fitting) or len(set(fitting)) != len(fitting):
            raise P05CoreError("role_fitting_uids_malformed")
        if not set(fitting).issubset(population):
            raise P05CoreError("role_uid_outside_population")
        if not set(fitting).issubset(set(uid_index)):
            raise P05CoreError("role_uid_missing_representation")
        declared = sorted(str(row["uid"]) for row in observations.get(role["role_label"], []))
        if declared != fitting:
            raise P05CoreError("role_observation_mismatch")
        if _canon().sha256_value(fitting) != role["uid_set_sha256"]:
            raise P05CoreError("role_uid_set_digest_mismatch")


def _check_fit_budget(fit: Mapping[str, Any]) -> None:
    if int(fit["epochs"]) != EPOCHS:
        raise P05CoreError("plan_epoch_budget_altered")
    if int(fit["batches_per_epoch"]) != DRAWS_PER_EPOCH:
        raise P05CoreError("plan_batch_budget_altered")
    if int(fit["optimizer_steps"]) != OPTIMIZER_STEPS:
        raise P05CoreError("plan_step_budget_altered")


def _validate_source_inputs(
    smoke: Any, role_inputs: Mapping[str, tuple[Any, list[Any], Any]]
) -> None:
    validator = getattr(smoke, "_prepare_inputs", None)
    if validator is None:
        raise P05CoreError("source_input_validator_missing")
    for values, observations, noise in role_inputs.values():
        try:
            validator(
                values,
                list(observations),
                noise_metadata=noise,
                epochs=EPOCHS,
                batches_per_epoch=DRAWS_PER_EPOCH,
            )
        except Exception as error:
            raise P05CoreError("source_inputs_invalid") from error


def _validate_sample_capacity(
    smoke: Any,
    role_inputs: Mapping[str, tuple[Any, list[Any], Any]],
    contract: Mapping[str, Any],
) -> None:
    sampler = getattr(smoke, "sample_master_views", None)
    if sampler is None:
        raise P05CoreError("source_capacity_validator_missing")
    ceiling = int(contract["sampler"]["batch_size_ceiling"])
    for _values, observations, _noise in role_inputs.values():
        try:
            sampler(
                list(observations),
                role_id="p05-source-capacity-preflight",
                seed=0,
                epoch=1,
                batch_ordinal=0,
                max_batch_size=ceiling,
            )
        except Exception as error:
            raise P05CoreError("source_capacity_invalid") from error


def _minimal_plan_checks(plan: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    for key in (
        "smoke_roles",
        "smoke_observations",
        "smoke_fits",
        "development_slots",
        "guard_roles",
    ):
        if key not in plan:
            raise P05CoreError("plan_key_missing")
    smoke = contract["smoke"]
    later = contract["later_core_plan"]
    if len(plan["smoke_roles"]) != len(smoke["role_labels"]):
        raise P05CoreError("plan_role_count_mismatch")
    if len(plan["development_slots"]) != int(later["inner_fit_slot_ceiling"]):
        raise P05CoreError("plan_development_slots_mismatch")
    if len(plan["guard_roles"]) != int(later["extra_guard_unit_slots"]):
        raise P05CoreError("plan_guard_roles_mismatch")
    labels = set(smoke["role_labels"])
    recipes = {recipe["recipe_id"] for recipe in contract["recipes"]}
    seeds = {int(seed) for seed in smoke["seeds"]}
    expected_primary = {
        (role, recipe, seed) for role in labels for recipe in recipes for seed in seeds
    }
    planned_replays = {(r["role_label"], r["recipe_id"], int(r["seed"])) for r in smoke["replays"]}
    executions: set[str] = set()
    primary_ids: set[str] = set()
    primary_keys: set[tuple[str, str, int]] = set()
    replay_keys: set[tuple[str, str, int]] = set()
    replay_targets: list[str] = []
    for fit in plan["smoke_fits"]:
        execution_id = fit["execution_id"]
        if execution_id in executions:
            raise P05CoreError("plan_execution_duplicate")
        executions.add(execution_id)
        if fit["role_label"] not in labels or fit["recipe_id"] not in recipes:
            raise P05CoreError("plan_reference_unknown")
        _check_fit_budget(fit)
        key = (fit["role_label"], fit["recipe_id"], int(fit["seed"]))
        if fit["execution_kind"] == "primary":
            primary_keys.add(key)
            primary_ids.add(execution_id)
        elif fit["execution_kind"] == "replay":
            replay_keys.add(key)
            replay_targets.append(fit["replay_of"])
        else:
            raise P05CoreError("plan_execution_kind_unknown")
    if primary_keys != expected_primary:
        raise P05CoreError("plan_primary_set_mismatch")
    if replay_keys != planned_replays:
        raise P05CoreError("plan_replay_set_mismatch")
    for target in replay_targets:
        if target not in primary_ids:
            raise P05CoreError("plan_replay_target_missing")
    if len(executions) != PLANNED_EXECUTIONS:
        raise P05CoreError("plan_fit_count_mismatch")


def _build_plan(support: Any, contract: Mapping[str, Any], project_root: Path) -> Mapping[str, Any]:
    planner = _planner()
    readiness_module = importlib.import_module("atlas_sers.evaluation.p05_readiness")
    readiness = readiness_module.build_readiness_report(project_root=project_root)
    return planner.build_core_plan(support, contract=contract, readiness=readiness)


def _plan_dir(artifact_root: Path, plan_id: str) -> Path:
    return artifact_root / P05CORE_NAMESPACE / "plans" / plan_id


def _lease_dir(artifact_root: Path, contract_sha256: str) -> Path:
    return artifact_root / P05CORE_NAMESPACE / "leases" / contract_sha256


def _run_dir(artifact_root: Path, contract_sha256: str) -> Path:
    return artifact_root / P05CORE_NAMESPACE / "runs" / contract_sha256


def _store_plan(artifact_root: Path, content: bytes, plan_id: str) -> None:
    directory = _plan_dir(artifact_root, plan_id)
    if directory.exists():
        if directory.is_symlink():
            raise P05CoreError("symlink_path_rejected")
        existing = _read_bytes(directory / "plan.json", "plan")
        if _canon().sha256_bytes(existing) != plan_id:
            raise P05CoreError("plan_content_hash_mismatch")
        return
    _mkdir_exclusive(directory, "plan_directory_exists")
    _atomic_write(directory / "plan.json", content)


def run_plan(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    contract_sha256: str,
) -> dict[str, Any]:
    project_root = Path(project_root)
    artifact_root = Path(artifact_root)
    repository_root = _repository_root(project_root)
    _assert_artifact_location(project_root, repository_root, artifact_root)
    contract, contract_sha256 = _load_contract(Path(contract_path), contract_sha256)
    support, _p01_run, _p04_run = _authenticate(artifact_root, contract)
    _manifest_uids(support, int(contract["population"]["rows"]))
    plan = _build_plan(support, contract, project_root)
    _minimal_plan_checks(plan, contract)
    content = _canon().canonical_json_bytes(plan)
    plan_id = _canon().sha256_bytes(content)
    _store_plan(artifact_root, content, plan_id)
    return {
        "status": "ok",
        "command": "plan",
        "plan_id": plan_id,
        "contract_sha256": contract_sha256,
        "smoke_roles": len(plan["smoke_roles"]),
        "smoke_observations": len(plan["smoke_observations"]),
        "smoke_fits": len(plan["smoke_fits"]),
        "development_slots": len(plan["development_slots"]),
        "guard_roles": len(plan["guard_roles"]),
    }


def _configure_environment() -> None:
    os.environ.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )


def _import_stack() -> tuple[Any, Any, Any, Any]:
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    numpy = importlib.import_module("numpy")
    pandas = importlib.import_module("pandas")
    smoke = importlib.import_module("atlas_sers.evaluation.p05_smoke")
    return numpy, pandas, torch, smoke


def _select_device(torch: Any) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free_bytes, _total = torch.cuda.mem_get_info()
    return "cuda" if int(free_bytes) >= MIN_FREE_CUDA_BYTES else "cpu"


def _capture_provenance(repository_root: Path, project_root: Path, artifact_root: Path) -> Any:
    module = importlib.import_module("atlas_sers.governance.provenance")
    return module.capture_provenance(
        repository_root=repository_root,
        project_root=project_root,
        artifact_root=artifact_root,
    )


class _Recorder:
    __slots__ = ("_stream",)

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("ab")

    def __call__(self, record: Any) -> None:
        if isinstance(record, Mapping):
            self._stream.write(_canon().canonical_json_bytes(dict(record)) + b"\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


def _noise_frame(path: Path, manifest_sha256: str, fitting: Sequence[str], pandas: Any) -> Any:
    _reject_symlink_chain(path)
    if _canon().sha256_file(path) != manifest_sha256:
        raise P05CoreError("manifest_hash_mismatch")
    frame = pandas.read_csv(path, usecols=list(QC_COLUMNS), keep_default_na=False)
    if _canon().sha256_file(path) != manifest_sha256:
        raise P05CoreError("manifest_hash_changed")
    frame = frame.reindex(columns=list(QC_COLUMNS)).set_index("observation_uid")
    try:
        selected = frame.loc[list(fitting)]
    except KeyError as error:
        raise P05CoreError("qc_uid_missing") from error
    return selected.reset_index()


def _build_role_inputs(
    plan: Mapping[str, Any],
    p01_run: Path,
    contract: Mapping[str, Any],
    intensity: Any,
    uid_index: Mapping[str, int],
    pandas: Any,
    observation_type: Any,
) -> dict[str, tuple[Any, list[Any], Any]]:
    rows_by_role: dict[str, list[Mapping[str, Any]]] = {}
    for row in plan["smoke_observations"]:
        rows_by_role.setdefault(row["role_label"], []).append(row)
    manifest_sha256 = contract["input_pins"]["manifest_sha256"]
    manifest_path = p01_run / "primary_manifest.csv"
    result: dict[str, tuple[Any, list[Any], Any]] = {}
    for role in plan["smoke_roles"]:
        label = role["role_label"]
        fitting = sorted(str(uid) for uid in role["fitting_uids"])
        by_uid = {str(row["uid"]): row for row in rows_by_role[label]}
        observations = [
            observation_type(
                uid=uid,
                master=str(by_uid[uid]["master"]),
                station=str(by_uid[uid]["station"]),
                target=str(by_uid[uid]["target"]),
                instrument=str(by_uid[uid]["instrument"]),
                substrate=str(by_uid[uid]["substrate"]),
            )
            for uid in fitting
        ]
        values = intensity[[uid_index[uid] for uid in fitting]]
        noise = _noise_frame(manifest_path, manifest_sha256, fitting, pandas)
        result[label] = (values, observations, noise)
    return result


def _reserve_row(rows_dir: Path, fit: Mapping[str, Any]) -> None:
    target = rows_dir / f"{fit['execution_id']}.json"
    _reject_symlink_chain(target.parent)
    rows_dir.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise P05CoreError("symlink_path_rejected")
    payload = _canon().canonical_json_bytes(
        {
            "execution_id": fit["execution_id"],
            "fit_id": fit["fit_id"],
            "execution_kind": fit["execution_kind"],
            "recipe_id": fit["recipe_id"],
            "seed": int(fit["seed"]),
            "optimizer_steps": int(fit["optimizer_steps"]),
        }
    )
    if target.exists():
        raise P05CoreError("execution_reservation_exists")
    handle = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(handle, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _save_state(torch: Any, state: Any, path: Path) -> None:
    handle, temporary = tempfile.mkstemp(prefix=".ckpt-", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        torch.save({"state_dict": state}, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _persist_execution(run_dir: Path, fit: Mapping[str, Any], result: Any, torch: Any) -> None:
    directory = run_dir / "executions" / str(fit["execution_id"])
    _mkdir_exclusive(directory, "execution_directory_exists")
    history = [dict(record) for record in (result.history or [])]
    _atomic_write(directory / "history.json", _canon().canonical_json_bytes(history))
    record = {
        "execution_id": fit["execution_id"],
        "fit_id": fit["fit_id"],
        "execution_kind": fit["execution_kind"],
        "replay_of": fit.get("replay_of"),
        "role_label": fit["role_label"],
        "recipe_id": fit["recipe_id"],
        "seed": int(fit["seed"]),
        "status": result.status,
        "reason_code": result.reason_code,
        "parameter_count": int(result.parameter_count),
        "optimizer_steps": int(result.optimizer_steps),
        "elapsed_seconds": float(result.elapsed_seconds),
        "peak_cuda_bytes": int(result.peak_cuda_bytes),
        "finite_gradient_batches": int(result.finite_gradient_batches),
        "nonzero_gradient_elements": int(result.nonzero_gradient_elements),
        "traceback_digest": getattr(result, "traceback_digest", None),
        "augmentation_digest": result.augmentation_digest,
        "sampling_digest": result.sampling_digest,
        "pair_digest": result.pair_digest,
        "supcon_support": dict(result.supcon_support or {}),
        "paired_support": dict(result.paired_support or {}),
    }
    for field in STATE_DIGEST_FIELDS:
        record[field] = getattr(result, field)
    _atomic_write(directory / "result.json", _canon().canonical_json_bytes(record))
    if result.state_dict is not None:
        _save_state(torch, result.state_dict, directory / "state.pt")


def _persist_error(run_dir: Path, fit: Mapping[str, Any], error: BaseException) -> None:
    directory = run_dir / "executions" / str(fit["execution_id"])
    if not directory.exists():
        _mkdir_exclusive(directory, "execution_directory_exists")
    detail = traceback.format_exc().encode("utf-8")
    _atomic_write(
        directory / "error.json",
        _canon().canonical_json_bytes(
            {
                "execution_id": fit["execution_id"],
                "status": "fail",
                "reason_code": getattr(error, "reason_code", type(error).__name__),
                "traceback_sha256": _canon().sha256_bytes(detail),
            }
        ),
    )


def _train_one(
    fit: Mapping[str, Any],
    values: Any,
    observations: Sequence[Any],
    noise: Any,
    smoke: Any,
    device: str,
    global_deadline: float,
    history_path: Path,
) -> Any:
    recorder = _Recorder(history_path)
    try:
        return smoke.train_smoke_fit(
            values=values,
            observations=list(observations),
            noise_metadata=noise,
            role_id=str(fit["p05_role_id"]),
            recipe=str(fit["recipe_id"]),
            seed=int(fit["seed"]),
            device=device,
            epochs=EPOCHS,
            batches_per_epoch=DRAWS_PER_EPOCH,
            maximum_fit_seconds=MAX_FIT_SECONDS,
            global_deadline=global_deadline,
            maximum_cuda_allocated_bytes=MAX_CUDA_ALLOCATED_BYTES,
            on_epoch=recorder,
        )
    finally:
        recorder.close()


def _has_positive(history: Sequence[Mapping[str, Any]], field: str) -> bool:
    for record in history:
        value = record.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)) and float(value) > 0.0:
                return True
    return False


def _check_result_format(result: Any) -> None:
    for field in STATE_DIGEST_FIELDS:
        _require_digest(getattr(result, field), "state_digest_invalid", allow_none="head" in field)
    for field in ("augmentation_digest", "sampling_digest", "pair_digest"):
        _require_digest(getattr(result, field), "stream_digest_invalid")
    for field in ("elapsed_seconds", "peak_cuda_bytes"):
        value = getattr(result, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise P05CoreError("cost_format_invalid")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise P05CoreError("cost_format_invalid")


def _check_history(
    fit: Mapping[str, Any], result: Any, recipe: Mapping[str, Any], sparse: bool
) -> None:
    history = result.history
    if (
        not isinstance(history, Sequence)
        or isinstance(history, (str, bytes))
        or len(history) != EPOCHS
    ):
        raise P05CoreError("history_length_mismatch")
    supcon_on = float(recipe["lambda_supcon"]) > 0.0
    pair_on = float(recipe["lambda_pair"]) > 0.0
    for index, record in enumerate(history, start=1):
        if not isinstance(record, Mapping):
            raise P05CoreError("history_record_malformed")
        if REQUIRED_HISTORY_FIELDS - set(record):
            raise P05CoreError("history_fields_missing")
        for field in BOOL_HISTORY_FIELDS:
            if not isinstance(record[field], bool):
                raise P05CoreError("history_flag_not_bool")
        for field in REQUIRED_HISTORY_FIELDS:
            if field in BOOL_HISTORY_FIELDS:
                continue
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise P05CoreError("history_numeric_type_invalid")
            if not math.isfinite(float(value)):
                raise P05CoreError("history_non_finite")
        if int(record["epoch"]) != index:
            raise P05CoreError("history_epoch_mismatch")
        if int(record["optimizer_steps"]) != index * DRAWS_PER_EPOCH:
            raise P05CoreError("history_steps_mismatch")
        if record["supcon_enabled"] is not supcon_on:
            raise P05CoreError("history_supcon_flag_mismatch")
        if record["paired_enabled"] is not pair_on:
            raise P05CoreError("history_paired_flag_mismatch")
        if supcon_on and int(record["supcon_available_batches"]) != DRAWS_PER_EPOCH:
            raise P05CoreError("history_supcon_batches_mismatch")
        if pair_on and not sparse and int(record["paired_available_batches"]) != DRAWS_PER_EPOCH:
            raise P05CoreError("history_paired_batches_mismatch")
        if pair_on and sparse and int(record["paired_available_batches"]) != 0:
            raise P05CoreError("history_sparse_pair_not_absent")
    if not _has_positive(history, "backbone_gradient_norm_mean"):
        raise P05CoreError("backbone_gradient_missing")
    if bool(recipe["projection"]) and not _has_positive(history, "head_gradient_norm_mean"):
        raise P05CoreError("head_gradient_missing")


def _check_support(recipe: Mapping[str, Any], result: Any, sparse: bool) -> None:
    supcon = dict(result.supcon_support or {})
    paired = dict(result.paired_support or {})
    if float(recipe["lambda_supcon"]) > 0.0:
        if int(supcon.get("available_batches", 0)) != EXPECTED_SUPCON_BATCHES:
            raise P05CoreError("supcon_batches_mismatch")
        if sparse:
            if int(supcon.get("eligible_anchors", 0)) != SPARSE_SUPCON_ELIGIBLE_ANCHORS:
                raise P05CoreError("sparse_supcon_anchor_mismatch")
            if int(supcon.get("zero_positive_anchors", 0)) != SPARSE_SUPCON_ZERO_POSITIVE:
                raise P05CoreError("sparse_supcon_zero_positive_mismatch")
        elif int(supcon.get("eligible_anchors", 0)) <= 0:
            raise P05CoreError("dense_supcon_anchor_mismatch")
    if float(recipe["lambda_pair"]) > 0.0:
        if sparse:
            for field in ("available_batches", "eligible_masters", "pairs"):
                if int(paired.get(field, 0)) != 0:
                    raise P05CoreError("sparse_paired_not_absent")
        else:
            if int(paired.get("available_batches", 0)) != EXPECTED_SUPCON_BATCHES:
                raise P05CoreError("paired_batches_mismatch")
            if int(paired.get("eligible_masters", 0)) <= 0:
                raise P05CoreError("paired_masters_missing")
            if int(paired.get("pairs", 0)) <= 0:
                raise P05CoreError("paired_pairs_missing")


def _check_fit(
    fit: Mapping[str, Any], result: Any, recipes: Mapping[str, Any], model: Mapping[str, Any]
) -> None:
    if result.status != "complete":
        raise P05CoreError("fit_not_complete")
    _check_result_format(result)
    if int(result.optimizer_steps) != OPTIMIZER_STEPS:
        raise P05CoreError("fit_steps_mismatch")
    if int(result.finite_gradient_batches) != EXPECTED_FINITE_GRADIENT_BATCHES:
        raise P05CoreError("fit_finite_gradient_mismatch")
    if int(result.nonzero_gradient_elements) <= 0:
        raise P05CoreError("fit_gradient_zero")
    if float(result.elapsed_seconds) > MAX_FIT_SECONDS:
        raise P05CoreError("fit_time_exceeded")
    if int(result.peak_cuda_bytes) > MAX_CUDA_ALLOCATED_BYTES:
        raise P05CoreError("fit_cuda_exceeded")
    recipe = recipes[fit["recipe_id"]]
    expected = (
        model["projection_model_parameters"] if recipe["projection"] else model["base_parameters"]
    )
    if int(result.parameter_count) != int(expected):
        raise P05CoreError("fit_parameter_mismatch")
    if result.initial_state_digest == result.final_state_digest:
        raise P05CoreError("fit_state_unchanged")
    if result.initial_backbone_digest == result.final_backbone_digest:
        raise P05CoreError("fit_backbone_unchanged")
    if bool(recipe["projection"]) and result.initial_head_digest == result.final_head_digest:
        raise P05CoreError("fit_head_unchanged")
    sparse = str(fit["role_label"]).endswith("_sparse")
    _check_history(fit, result, recipe, sparse)
    _check_support(recipe, result, sparse)


def _numeric_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for record in history
    ]


def _compare_sparse_controls(left: Any, right: Any) -> None:
    for field in STATE_DIGEST_FIELDS:
        if getattr(left, field) != getattr(right, field):
            raise P05CoreError("sparse_equivalence_state_mismatch")
    if _numeric_history(left.history) != _numeric_history(right.history):
        raise P05CoreError("sparse_equivalence_history_mismatch")


def _compare_replay(primary: Any, replay: Any) -> None:
    for field in STATE_DIGEST_FIELDS + ("augmentation_digest", "sampling_digest", "pair_digest"):
        if getattr(primary, field) != getattr(replay, field):
            raise P05CoreError("replay_terminal_state_mismatch")
    if [dict(record) for record in primary.history] != [dict(record) for record in replay.history]:
        raise P05CoreError("replay_history_mismatch")


def _check_acceptance(contract: Mapping[str, Any], executions: Sequence[Mapping[str, Any]]) -> None:
    if len(executions) != PLANNED_EXECUTIONS:
        raise P05CoreError("execution_count_mismatch")
    recipes = {recipe["recipe_id"]: recipe for recipe in contract["recipes"]}
    model = contract["model"]
    by_id: dict[str, Mapping[str, Any]] = {}
    primaries: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for item in executions:
        fit = item["fit"]
        execution_id = fit["execution_id"]
        if execution_id in by_id:
            raise P05CoreError("execution_duplicate")
        by_id[execution_id] = item
        _check_fit(fit, item["result"], recipes, model)
        if fit["execution_kind"] == "primary":
            key = (fit["role_label"], fit["recipe_id"], int(fit["seed"]))
            if key in primaries:
                raise P05CoreError("primary_duplicate")
            primaries[key] = item
    if len(by_id) != PLANNED_EXECUTIONS:
        raise P05CoreError("execution_duplicate")
    if sum(float(item["result"].elapsed_seconds) for item in executions) > MAX_TOTAL_SECONDS:
        raise P05CoreError("total_time_exceeded")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for (role_label, _recipe_id, seed), item in primaries.items():
        grouped.setdefault((role_label, seed), []).append(item)
    for members in grouped.values():
        for field in STREAM_DIGEST_FIELDS:
            if len({getattr(member["result"], field) for member in members}) != 1:
                raise P05CoreError("common_stream_mismatch")
    smoke = contract["smoke"]
    sparse_roles = [label for label in smoke["role_labels"] if label.endswith("_sparse")]
    for pair in smoke["expected_sparse_equivalences"]:
        for role in sparse_roles:
            for seed in smoke["seeds"]:
                left = primaries.get((role, pair[0], int(seed)))
                right = primaries.get((role, pair[1], int(seed)))
                if left is None or right is None:
                    raise P05CoreError("sparse_control_missing")
                _compare_sparse_controls(left["result"], right["result"])
    for replay in smoke["replays"]:
        primary = primaries.get((replay["role_label"], replay["recipe_id"], int(replay["seed"])))
        if primary is None:
            raise P05CoreError("replay_primary_missing")
        matches = [
            item
            for item in executions
            if item["fit"]["execution_kind"] == "replay"
            and item["fit"]["replay_of"] == primary["fit"]["execution_id"]
        ]
        if len(matches) != 1:
            raise P05CoreError("replay_missing")
        _compare_replay(primary["result"], matches[0]["result"])


def _assert_protected_identity(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if before.get("protected_environment_sha256") != after.get("protected_environment_sha256"):
        raise P05CoreError("protected_environment_changed")


def _write_manifest(run_dir: Path) -> None:
    base = run_dir.resolve()
    paths = [path for path in base.rglob("*") if path.is_file() and path.name != "manifest.json"]
    _atomic_write(
        run_dir / "manifest.json",
        _canon().canonical_json_bytes({"files": _canon().hash_relative_files(base, paths)}),
    )


def _write_failure(run_dir: Path, contract_sha256: str, plan_id: str, error: BaseException) -> None:
    try:
        _atomic_write(
            run_dir / "summary.json",
            _canon().canonical_json_bytes(
                {
                    "status": "fail",
                    "contract_sha256": contract_sha256,
                    "plan_id": plan_id,
                    "reason_code": getattr(error, "reason_code", type(error).__name__),
                }
            ),
        )
        _write_manifest(run_dir)
    except Exception:
        pass


def run_smoke(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    plan_id: str,
    contract_sha256: str,
) -> dict[str, Any]:
    _configure_environment()
    project_root = Path(project_root)
    artifact_root = Path(artifact_root)
    if not _is_hex64(plan_id) or not _is_hex64(contract_sha256):
        raise P05CoreError("identifier_malformed")
    contract_sha256 = contract_sha256.lower()
    repository_root = _repository_root(project_root)
    _assert_artifact_location(project_root, repository_root, artifact_root)
    plan_content = _read_bytes(_plan_dir(artifact_root, plan_id) / "plan.json", "plan")
    if _canon().sha256_bytes(plan_content) != plan_id:
        raise P05CoreError("plan_content_hash_mismatch")
    contract, actual_contract_sha256 = _load_contract(Path(contract_path), contract_sha256)
    if actual_contract_sha256 != contract_sha256:
        raise P05CoreError("contract_digest_mismatch")
    lease = _lease_dir(artifact_root, contract_sha256)
    if lease.exists() or lease.is_symlink():
        raise P05CoreError("lease_exists")
    support, p01_run, _p04_run = _authenticate(artifact_root, contract)
    expected_rows = int(contract["population"]["rows"])
    manifest_uids = _manifest_uids(support, expected_rows)
    plan = _build_plan(support, contract, project_root)
    _minimal_plan_checks(plan, contract)
    if _canon().sha256_bytes(_canon().canonical_json_bytes(plan)) != plan_id:
        raise P05CoreError("plan_authority_mismatch")
    numpy, pandas, torch, smoke = _import_stack()
    device = _select_device(torch)
    observation_type = importlib.import_module("atlas_sers.evaluation.p05_sampling").Observation
    provenance_before = _capture_provenance(repository_root, project_root, artifact_root)
    intensity, labels = _load_representation(
        p01_run / REPRESENTATION_REL,
        contract["input_pins"]["representation_sha256"],
        manifest_uids,
        expected_rows,
    )
    uid_index = {uid: index for index, uid in enumerate(labels)}
    _verify_role_membership(plan, uid_index, set(manifest_uids))
    role_inputs = _build_role_inputs(
        plan, p01_run, contract, intensity, uid_index, pandas, observation_type
    )
    _validate_source_inputs(smoke, role_inputs)
    _validate_sample_capacity(smoke, role_inputs, contract)
    _mkdir_exclusive(lease, "lease_exists")
    _atomic_write(
        lease / "lease.json",
        _canon().canonical_json_bytes(
            {"state": "running", "contract_sha256": contract_sha256, "plan_id": plan_id}
        ),
    )
    run_dir = _run_dir(artifact_root, contract_sha256)
    _mkdir_exclusive(run_dir, "run_exists")
    _atomic_write(run_dir / "provenance.json", _canon().canonical_json_bytes(provenance_before))
    ledger_path = run_dir / "ledger.jsonl"

    def _ledger(entry: Mapping[str, Any]) -> None:
        with ledger_path.open("ab") as stream:
            stream.write(_canon().canonical_json_bytes(dict(entry)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    for fit in plan["smoke_fits"]:
        _ledger(
            {
                "event": "planned",
                "execution_id": fit["execution_id"],
                "execution_kind": fit["execution_kind"],
            }
        )
    wall_start = time.monotonic()
    global_deadline = time.perf_counter() + MAX_TOTAL_SECONDS
    executions: list[dict[str, Any]] = []
    try:
        for fit in plan["smoke_fits"]:
            if time.perf_counter() >= global_deadline:
                raise P05CoreError("global_deadline_exceeded")
            if time.monotonic() - wall_start > MAX_TOTAL_SECONDS:
                raise P05CoreError("total_wall_exceeded")
            _reserve_row(lease / "rows", fit)
            _ledger({"event": "started", "execution_id": fit["execution_id"]})
            values, observations, noise = role_inputs[fit["role_label"]]
            try:
                result = _train_one(
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
                _persist_error(run_dir, fit, error)
                _ledger({"event": "failed", "execution_id": fit["execution_id"]})
                raise
            _persist_execution(run_dir, fit, result, torch)
            _ledger(
                {"event": "completed", "execution_id": fit["execution_id"], "status": result.status}
            )
            executions.append({"fit": fit, "result": result})
            if result.status != "complete":
                raise P05CoreError("fit_failed")
        support_after, _, _ = _authenticate(artifact_root, contract)
        if (
            support_after.manifest_sha256 != support.manifest_sha256
            or support_after.contexts_sha256 != support.contexts_sha256
            or support_after.roles_sha256 != support.roles_sha256
        ):
            raise P05CoreError("input_changed_during_execution")
        _check_acceptance(contract, executions)
        provenance_after = _capture_provenance(repository_root, project_root, artifact_root)
        _assert_protected_identity(provenance_before, provenance_after)
        wall_seconds = time.monotonic() - wall_start
        if wall_seconds > MAX_TOTAL_SECONDS:
            raise P05CoreError("total_wall_exceeded")
        summary = {
            "status": "complete",
            "contract_sha256": contract_sha256,
            "plan_id": plan_id,
            "device": device,
            "executions": len(executions),
            "primary_fits": sum(
                1 for item in executions if item["fit"]["execution_kind"] == "primary"
            ),
            "replays": sum(1 for item in executions if item["fit"]["execution_kind"] == "replay"),
            "optimizer_steps": sum(int(item["result"].optimizer_steps) for item in executions),
            "sum_elapsed_seconds": sum(
                float(item["result"].elapsed_seconds) for item in executions
            ),
            "maximum_peak_cuda_bytes": max(
                int(item["result"].peak_cuda_bytes) for item in executions
            ),
            "wall_seconds": wall_seconds,
            "checks": {"status": "pass"},
        }
        _atomic_write(
            run_dir / "provenance_after.json", _canon().canonical_json_bytes(provenance_after)
        )
        _atomic_write(run_dir / "summary.json", _canon().canonical_json_bytes(summary))
        _atomic_write(
            lease / "lease.json",
            _canon().canonical_json_bytes(
                {"state": "complete", "contract_sha256": contract_sha256, "plan_id": plan_id}
            ),
        )
        _write_manifest(run_dir)
    except BaseException as error:
        try:
            _atomic_write(
                lease / "lease.json",
                _canon().canonical_json_bytes(
                    {
                        "state": "failed",
                        "contract_sha256": contract_sha256,
                        "plan_id": plan_id,
                        "reason_code": getattr(error, "reason_code", type(error).__name__),
                    }
                ),
            )
        except Exception:
            pass
        _write_failure(run_dir, contract_sha256, plan_id, error)
        if isinstance(error, (KeyboardInterrupt, SystemExit, P05CoreError)):
            raise
        raise P05CoreError("smoke_execution_failed") from error
    return {
        "status": "complete",
        "command": "smoke",
        "plan_id": plan_id,
        "contract_sha256": contract_sha256,
        "executions": summary["executions"],
        "primary_fits": summary["primary_fits"],
        "replays": summary["replays"],
        "optimizer_steps": summary["optimizer_steps"],
    }


def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(_canon().canonical_json_bytes(dict(payload)).decode("utf-8") + "\n")


def cli_main(argv: Sequence[str] | None = None) -> int:
    _configure_environment()
    parser = argparse.ArgumentParser(prog="run_p05_core")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--plan-id", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "plan":
            report = run_plan(
                project_root=arguments.project_root,
                artifact_root=arguments.artifact_root,
                contract_path=arguments.contract,
                contract_sha256=arguments.contract_sha256,
            )
        else:
            report = run_smoke(
                project_root=arguments.project_root,
                artifact_root=arguments.artifact_root,
                contract_path=arguments.contract,
                plan_id=arguments.plan_id,
                contract_sha256=arguments.contract_sha256,
            )
    except P05CoreError as error:
        _emit({"status": "fail", "command": arguments.command, "reason_code": error.reason_code})
        return 1
    _emit(report)
    return 0
