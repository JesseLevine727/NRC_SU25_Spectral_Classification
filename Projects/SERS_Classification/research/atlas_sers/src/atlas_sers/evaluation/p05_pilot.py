"""P05-T016 approved 36-fit source-validation pilot boundary.

``prepare``/``preflight`` authenticate the independent pilot permit by
canonical digest, authenticate the frozen P05 core contract/plan through the
reused core helpers, rebuild the T015 development ledger, select one
nonexcluded inherited unit per station from held-evaluation contexts, bind the
36 registered development slots and write a private metadata pilot plan.

It also owns the one-shot ``run`` operation: the private input-preparation,
kernel-invocation, checkpoint, result-persistence and acceptance helpers, the
exclusive permit lease, the independent contract/plan slot leases, the guarded
per-epoch resource callback and the strict aggregate public summary.  It
performs real scientific writes into the private ``p05development`` namespace
(histories, checkpoints, logits, summaries) but no public export, and it never
changes the frozen core false authorization flags.  The separate pilot permit
is the sole execution authority.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import io
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation.p05_core_run import P05CoreError
from atlas_sers.evaluation.p05_development_plan import (
    DevelopmentLedgerError,
    build_development_ledger,
)

P05DEVELOPMENT_NAMESPACE = "p05development"

PILOT_SCHEMA_VERSION = "nato-sers-p05-development-pilot-v1"
PILOT_PROTOCOL_VERSION = "nato-sers-p05-development-pilot-20260925-v1"
PILOT_PERMIT_SHA256 = "652f5c07a1076a907778a9dd80394203ded9084298a95ce791cb5ee2814e576d"
PILOT_CORE_CONTRACT_SHA256 = "60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae"
PILOT_CORE_PLAN_ID = "a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37"

PILOT_STATIONS = ("cwa", "pills", "surfaces")
PILOT_RECIPES = ("D0-M", "D1", "D2", "D3")
PILOT_SEEDS = (20260805, 20260817, 20260829)

MAXIMUM_FIT_EXECUTIONS = 36
MINIMUM_EPOCHS = 30
MAXIMUM_EPOCHS = 200
PATIENCE = 20
BATCH_DRAWS_PER_EPOCH = 4
MAXIMUM_OPTIMIZER_STEPS = 28800
MAXIMUM_FIT_OPTIMIZER_STEPS = MAXIMUM_EPOCHS * BATCH_DRAWS_PER_EPOCH
MAXIMUM_FIT_SECONDS = 120.0
MAXIMUM_TOTAL_SECONDS = 5400.0
MAXIMUM_CUDA_ALLOCATED_BYTES = 4294967296
MINIMUM_FREE_CUDA_BYTES = 5368709120
PRIVATE_STORAGE_CEILING_BYTES = 2147483648

SLOTS_PER_UNIT = len(PILOT_RECIPES) * len(PILOT_SEEDS)

PILOT_CLAIM = "bounded_source_validation_pilot_not_definitive_generalization_or_recipe_selection"


class P05PilotError(P05CoreError):
    """Stable pilot failure with a path-free reason code."""


def _canon() -> Any:
    return core._canon()


def _development_kernel() -> Any:
    return importlib.import_module("atlas_sers.evaluation.p05_development")


def _pilot_plan_dir(artifact_root: Path, plan_id: str) -> Path:
    return artifact_root / P05DEVELOPMENT_NAMESPACE / "plans" / plan_id


def _pilot_preflight_dir(artifact_root: Path) -> Path:
    return artifact_root / P05DEVELOPMENT_NAMESPACE / "preflight"


def _existing_ancestor(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists():
        if candidate.parent == candidate:
            raise P05PilotError("artifact_root_unavailable")
        candidate = candidate.parent
    return candidate


def _free_disk_bytes(path: Path) -> int:
    try:
        return int(shutil.disk_usage(_existing_ancestor(path)).free)
    except OSError as error:
        raise P05PilotError("artifact_root_unavailable") from error


def _free_cuda_bytes(torch: Any) -> int:
    try:
        if not bool(torch.cuda.is_available()):
            return 0
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        return 0
    return int(free_bytes)


def _resolve_paths(project_root: Path | str, artifact_root: Path | str) -> tuple[Path, Path, Path]:
    project = Path(project_root)
    artifact = Path(artifact_root)
    repository_root = core._repository_root(project)
    core._assert_artifact_location(project, repository_root, artifact)
    return project, artifact, repository_root


def _load_permit(permit_path: Path | str, permit_sha256: str) -> tuple[dict[str, Any], str]:
    if not core._is_hex64(permit_sha256):
        raise P05PilotError("permit_pin_malformed")
    if permit_sha256.lower() != PILOT_PERMIT_SHA256:
        raise P05PilotError("permit_pin_not_approved")
    permit = core._read_json(Path(permit_path), "permit")
    if not isinstance(permit, Mapping):
        raise P05PilotError("permit_malformed")
    observed = _canon().sha256_value(permit)
    if observed != PILOT_PERMIT_SHA256:
        raise P05PilotError("permit_digest_mismatch")
    return dict(permit), observed


def _build_ledger(
    plan: Mapping[str, Any], support: Any, contract: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return build_development_ledger(plan=plan, support=support, contract=contract)
    except DevelopmentLedgerError as error:
        raise P05PilotError(f"development_ledger_{error.reason_code}") from error


def _check_ledger_bounds(ledger: Mapping[str, Any]) -> None:
    if ledger.get("execution_authorized") is not False:
        raise P05PilotError("development_ledger_authorization_invalid")
    if ledger.get("arrays_loaded") is not False:
        raise P05PilotError("development_ledger_arrays_invalid")
    if int(ledger.get("fits_started", -1)) != 0:
        raise P05PilotError("development_ledger_fits_invalid")
    summary = ledger.get("summary")
    if not isinstance(summary, Mapping):
        raise P05PilotError("development_ledger_summary_malformed")
    if int(summary.get("recipe_count", -1)) != len(PILOT_RECIPES):
        raise P05PilotError("development_ledger_recipe_mismatch")
    if int(summary.get("seed_count", -1)) != len(PILOT_SEEDS):
        raise P05PilotError("development_ledger_seed_mismatch")
    if int(summary.get("minimum_optimizer_updates_per_fit", -1)) != (
        MINIMUM_EPOCHS * BATCH_DRAWS_PER_EPOCH
    ):
        raise P05PilotError("development_ledger_minimum_schedule_mismatch")
    if int(summary.get("maximum_optimizer_updates_per_fit", -1)) != (MAXIMUM_FIT_OPTIMIZER_STEPS):
        raise P05PilotError("development_ledger_maximum_schedule_mismatch")


def _select_units(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        dict(unit)
        for unit in ledger["units"]
        if unit.get("unit_kind") == "inherited"
        and unit.get("phase_gate") == "held_evaluation"
        and not bool(unit.get("excluded_by_protocol"))
    ]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for station in PILOT_STATIONS:
        pool = [unit for unit in candidates if unit.get("station") == station]
        pseudo = [unit for unit in pool if unit.get("selection_mode") == "pseudo_domain"]
        if pseudo:
            pool = pseudo
        else:
            pool = [unit for unit in pool if unit.get("selection_mode") == "master_cv"]
        if not pool:
            raise P05PilotError("pilot_unit_unavailable")
        pool.sort(key=lambda unit: (str(unit["context_id"]), str(unit["selection_unit_id"])))
        choice = pool[0]
        if choice["unit_id"] in seen:
            raise P05PilotError("pilot_unit_duplicate")
        seen.add(choice["unit_id"])
        chosen.append(choice)
    if len(chosen) != len(PILOT_STATIONS):
        raise P05PilotError("pilot_unit_count_mismatch")
    return chosen


def _select_slots(
    ledger: Mapping[str, Any], units: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_unit = {str(unit["unit_id"]): unit for unit in units}
    slots = [dict(slot) for slot in ledger["slots"] if str(slot.get("unit_id")) in by_unit]
    expected_product = {(recipe, seed) for recipe in PILOT_RECIPES for seed in PILOT_SEEDS}
    for unit in units:
        unit_id = str(unit["unit_id"])
        group = [slot for slot in slots if str(slot["unit_id"]) == unit_id]
        if len(group) != SLOTS_PER_UNIT:
            raise P05PilotError("pilot_unit_slot_count_mismatch")
        if {(str(slot["recipe_id"]), int(slot["seed"])) for slot in group} != expected_product:
            raise P05PilotError("pilot_unit_slot_product_mismatch")
        for slot in group:
            if bool(slot.get("excluded_by_protocol")):
                raise P05PilotError("pilot_slot_excluded")
            if str(slot.get("fitting_role_id")) != str(unit["fitting_role_id"]):
                raise P05PilotError("pilot_slot_fitting_role_mismatch")
            if str(slot.get("validation_role_id")) != str(unit["validation_role_id"]):
                raise P05PilotError("pilot_slot_validation_role_mismatch")
    ordered = sorted(
        slots,
        key=lambda slot: (
            str(by_unit[str(slot["unit_id"])]["station"]),
            str(slot["recipe_id"]),
            int(slot["seed"]),
        ),
    )
    if len(ordered) != MAXIMUM_FIT_EXECUTIONS:
        raise P05PilotError("pilot_slot_count_mismatch")
    return ordered


def _build_pilot_plan(
    contract_sha256: str,
    core_plan_id: str,
    ledger: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    slot_payload: list[dict[str, Any]] = []
    for order, slot in enumerate(slots):
        item = dict(slot)
        item["order"] = order
        slot_payload.append(item)
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "protocol_version": PILOT_PROTOCOL_VERSION,
        "permit_sha256": PILOT_PERMIT_SHA256,
        "core_contract_sha256": str(contract_sha256),
        "core_plan_id": str(core_plan_id),
        "ledger_id": str(ledger["ledger_id"]),
        "ledger_schema_version": str(ledger["schema_version"]),
        "stations": list(PILOT_STATIONS),
        "recipes": list(PILOT_RECIPES),
        "seeds": list(PILOT_SEEDS),
        "units": [dict(unit) for unit in units],
        "slots": slot_payload,
        "bounds": {
            "maximum_fit_executions": MAXIMUM_FIT_EXECUTIONS,
            "minimum_epochs": MINIMUM_EPOCHS,
            "maximum_epochs": MAXIMUM_EPOCHS,
            "patience": PATIENCE,
            "batch_draws_per_epoch": BATCH_DRAWS_PER_EPOCH,
            "maximum_optimizer_steps": MAXIMUM_OPTIMIZER_STEPS,
            "maximum_fit_seconds": MAXIMUM_FIT_SECONDS,
            "maximum_total_seconds": MAXIMUM_TOTAL_SECONDS,
            "maximum_cuda_allocated_bytes": MAXIMUM_CUDA_ALLOCATED_BYTES,
            "minimum_free_cuda_bytes_before_launch": MINIMUM_FREE_CUDA_BYTES,
            "private_storage_ceiling_bytes": PRIVATE_STORAGE_CEILING_BYTES,
            "automatic_retries": 0,
        },
        "execution_authorized": False,
        "arrays_loaded": False,
        "fits_started": 0,
        "outer_evaluation_authorized": False,
    }


def prepare(
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    permit_path: Path | str,
    permit_sha256: str = PILOT_PERMIT_SHA256,
) -> dict[str, Any]:
    """Authenticate metadata and bind the 36-slot pilot plan (no lease/fits)."""

    project, artifact, repository_root = _resolve_paths(project_root, artifact_root)
    permit, permit_digest = _load_permit(permit_path, permit_sha256)
    contract, contract_sha256 = core._load_contract(Path(contract_path), PILOT_CORE_CONTRACT_SHA256)
    support, p01_run, _p04_run = core._authenticate(artifact, contract)
    core._manifest_uids(support, int(contract["population"]["rows"]))
    core_plan = core._build_plan(support, contract, project)
    core._minimal_plan_checks(core_plan, contract)
    core_plan_id = _canon().sha256_bytes(_canon().canonical_json_bytes(core_plan))
    if core_plan_id != PILOT_CORE_PLAN_ID:
        raise P05PilotError("core_plan_identity_mismatch")
    ledger = _build_ledger(core_plan, support, contract)
    _check_ledger_bounds(ledger)
    units = _select_units(ledger)
    slots = _select_slots(ledger, units)
    pilot_plan = _build_pilot_plan(contract_sha256, core_plan_id, ledger, units, slots)
    pilot_plan_id = _canon().sha256_bytes(_canon().canonical_json_bytes(pilot_plan))
    return {
        "project_root": project,
        "artifact_root": artifact,
        "repository_root": repository_root,
        "permit": permit,
        "permit_sha256": permit_digest,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "support": support,
        "p01_path": p01_run,
        "core_plan": core_plan,
        "core_plan_id": core_plan_id,
        "ledger": ledger,
        "units": units,
        "slots": slots,
        "pilot_plan": pilot_plan,
        "pilot_plan_id": pilot_plan_id,
    }


def _check_resources(artifact_root: Path, torch: Any) -> dict[str, Any]:
    free_disk = _free_disk_bytes(artifact_root)
    if free_disk < PRIVATE_STORAGE_CEILING_BYTES:
        raise P05PilotError("insufficient_free_disk")
    device = core._select_device(torch)
    free_cuda = _free_cuda_bytes(torch)
    if device == "cuda" and free_cuda < MINIMUM_FREE_CUDA_BYTES:
        raise P05PilotError("insufficient_free_cuda")
    return {
        "free_disk_bytes": free_disk,
        "free_cuda_bytes": free_cuda,
        "device": device,
    }


def _checkpoint_preflight(torch: Any, artifact_root: Path) -> None:
    directory = _pilot_preflight_dir(artifact_root)
    core._reject_symlink_chain(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preflight-", dir=directory) as temporary:
        scratch = Path(temporary)
        state = {"probe": torch.zeros(4, dtype=torch.float32)}
        target = scratch / "probe.pt"
        try:
            core._save_state(torch, state, target)
            loaded = torch.load(target, weights_only=True, map_location="cpu")
        except P05CoreError:
            raise
        except Exception as error:
            raise P05PilotError("checkpoint_preflight_failed") from error
        payload = loaded.get("state_dict") if isinstance(loaded, Mapping) else None
        if not isinstance(payload, Mapping) or set(payload) != set(state):
            raise P05PilotError("checkpoint_preflight_mismatch")
        for key, tensor in state.items():
            if not bool(torch.equal(payload[key], tensor)):
                raise P05PilotError("checkpoint_preflight_mismatch")


def _store_pilot_plan(artifact_root: Path, content: bytes, plan_id: str) -> None:
    directory = _pilot_plan_dir(artifact_root, plan_id)
    if directory.exists():
        if directory.is_symlink():
            raise P05PilotError("symlink_path_rejected")
        existing = core._read_bytes(directory / "plan.json", "pilot_plan")
        if _canon().sha256_bytes(existing) != plan_id:
            raise P05PilotError("pilot_plan_content_hash_mismatch")
        return
    core._mkdir_exclusive(directory, "pilot_plan_directory_exists")
    core._atomic_write(directory / "plan.json", content)


def _public_summary(bundle: Mapping[str, Any], resources: Mapping[str, Any]) -> dict[str, Any]:
    units = bundle["units"]
    return {
        "status": "ok",
        "command": "preflight",
        "permit_sha256": bundle["permit_sha256"],
        "core_contract_sha256": bundle["contract_sha256"],
        "core_plan_id": bundle["core_plan_id"],
        "ledger_id": bundle["ledger"]["ledger_id"],
        "pilot_plan_id": bundle["pilot_plan_id"],
        "stations": [str(unit["station"]) for unit in units],
        "recipes": list(PILOT_RECIPES),
        "seeds": list(PILOT_SEEDS),
        "unit_count": len(units),
        "slot_count": len(bundle["slots"]),
        "maximum_fit_executions": MAXIMUM_FIT_EXECUTIONS,
        "device": resources["device"],
        "free_disk_bytes": resources["free_disk_bytes"],
        "free_cuda_bytes": resources["free_cuda_bytes"],
        "checkpoint_preflight": "pass",
        "claim": PILOT_CLAIM,
        "execution_authorized": False,
        "arrays_loaded": False,
        "fits_started": 0,
        "outer_evaluation_authorized": False,
    }


def preflight(
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    permit_path: Path | str,
    permit_sha256: str = PILOT_PERMIT_SHA256,
) -> dict[str, Any]:
    """Run metadata/hash/resource/serialization checks and store the pilot plan."""

    bundle = prepare(project_root, artifact_root, contract_path, permit_path, permit_sha256)
    artifact = bundle["artifact_root"]
    _assert_no_prior_leases(
        artifact,
        bundle["permit_sha256"],
        bundle["contract_sha256"],
        bundle["core_plan_id"],
        bundle["slots"],
    )
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    resources = _check_resources(artifact, torch)
    _checkpoint_preflight(torch, artifact)
    _store_pilot_plan(
        artifact,
        _canon().canonical_json_bytes(bundle["pilot_plan"]),
        bundle["pilot_plan_id"],
    )
    summary = _public_summary(bundle, resources)
    core._atomic_write(
        _pilot_preflight_dir(artifact) / "summary.json",
        _canon().canonical_json_bytes(summary),
    )
    return summary


# --------------------------------------------------------------------------- #
# Private input preparation
# --------------------------------------------------------------------------- #


def _manifest_rows(support: Any) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in support.manifest:
        uid = str(row["observation_uid"])
        index[uid] = {
            "uid": uid,
            "master": str(row["master_sample_id"]),
            "station": str(row["station"]),
            "target": str(row["target_analyte"]),
            "instrument": str(row["instrument"]),
            "substrate": str(row["sensor_family"]),
        }
    return index


def _observation(manifest: Mapping[str, Mapping[str, str]], uid: str, observation_type: Any) -> Any:
    row = manifest.get(uid)
    if row is None:
        raise P05PilotError("manifest_uid_missing")
    return observation_type(
        uid=uid,
        master=row["master"],
        station=row["station"],
        target=row["target"],
        instrument=row["instrument"],
        substrate=row["substrate"],
    )


def _reassert_unit_boundaries(
    unit: Mapping[str, Any], fitting_uids: Sequence[str], validation_uids: Sequence[str]
) -> None:
    support = importlib.import_module("atlas_sers.evaluation.p05_support")
    if support.uid_set_hash(fitting_uids) != str(unit["fitting_uid_set_sha256"]):
        raise P05PilotError("fitting_uid_set_mismatch")
    if support.uid_set_hash(validation_uids) != str(unit["validation_uid_set_sha256"]):
        raise P05PilotError("validation_uid_set_mismatch")
    if set(fitting_uids) & set(validation_uids):
        raise P05PilotError("unit_uid_overlap")


def _support_role_sets(support: Any) -> dict[str, dict[str, set[str]]]:
    roles: dict[str, dict[str, set[str]]] = {}
    for row in support.roles:
        context_id = str(row["context_id"])
        role = str(row["role"])
        uid = str(row["observation_uid"])
        roles.setdefault(context_id, {}).setdefault(role, set()).add(uid)
    return roles


def _validate_unit_sources(
    unit: Mapping[str, Any],
    roles: Mapping[str, Mapping[str, set[str]]],
    manifest: Mapping[str, Mapping[str, str]],
) -> None:
    """Reassert UID/master uniqueness and outer fit/test boundaries from support."""

    context_id = str(unit["context_id"])
    outer = roles.get(context_id, {})
    fit_outer = set(outer.get("outer_fit", set()))
    test_outer = set(outer.get("outer_test", set()))
    fitting = [str(uid) for uid in unit["fitting_uids"]]
    validation = [str(uid) for uid in unit["validation_uids"]]
    if len(set(fitting)) != len(fitting) or len(set(validation)) != len(validation):
        raise P05PilotError("unit_uid_duplicate")
    if set(fitting) & set(validation):
        raise P05PilotError("unit_uid_overlap")
    if not set(fitting) <= fit_outer or not set(validation) <= fit_outer:
        raise P05PilotError("unit_uid_outside_outer_fit")
    if (set(fitting) | set(validation)) & test_outer:
        raise P05PilotError("outer_test_uid_in_unit")
    for uid in fitting + validation:
        if uid not in manifest:
            raise P05PilotError("unit_uid_unknown")
    fitting_masters = {str(manifest[uid]["master"]) for uid in fitting}
    validation_masters = {str(manifest[uid]["master"]) for uid in validation}
    if fitting_masters & validation_masters:
        raise P05PilotError("unit_master_overlap")
    test_masters = {str(manifest[uid]["master"]) for uid in test_outer if uid in manifest}
    if (fitting_masters | validation_masters) & test_masters:
        raise P05PilotError("outer_test_master_in_unit")


def prepare_role_inputs(bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Authenticate the frozen container; expose only selected fit/validation rows.

    The pinned NPZ container is read in full for integrity checks. Only the
    selected source rows are passed to training or validation; no outer-test
    prediction or metric is computed.
    """

    contract = bundle["contract"]
    support = bundle["support"]
    p01_run = bundle["p01_path"]
    pandas = importlib.import_module("pandas")
    observation_type = importlib.import_module("atlas_sers.evaluation.p05_sampling").Observation
    expected_rows = int(contract["population"]["rows"])
    manifest_uids = core._manifest_uids(support, expected_rows)
    intensity, labels = core._load_representation(
        p01_run / core.REPRESENTATION_REL,
        contract["input_pins"]["representation_sha256"],
        manifest_uids,
        expected_rows,
    )
    uid_index = {uid: index for index, uid in enumerate(labels)}
    manifest = _manifest_rows(support)
    roles = _support_role_sets(support)
    manifest_sha256 = contract["input_pins"]["manifest_sha256"]
    manifest_path = p01_run / "primary_manifest.csv"
    inputs: dict[str, dict[str, Any]] = {}
    for unit in bundle["units"]:
        unit_id = str(unit["unit_id"])
        fitting_uids = [str(uid) for uid in unit["fitting_uids"]]
        validation_uids = [str(uid) for uid in unit["validation_uids"]]
        if fitting_uids != sorted(fitting_uids) or validation_uids != sorted(validation_uids):
            raise P05PilotError("unit_uid_order_invalid")
        _reassert_unit_boundaries(unit, fitting_uids, validation_uids)
        _validate_unit_sources(unit, roles, manifest)
        try:
            fitting_values = intensity[[uid_index[uid] for uid in fitting_uids]]
            validation_values = intensity[[uid_index[uid] for uid in validation_uids]]
        except KeyError as error:
            raise P05PilotError("unit_uid_missing_representation") from error
        inputs[unit_id] = {
            "fitting_values": fitting_values,
            "fitting_observations": [
                _observation(manifest, uid, observation_type) for uid in fitting_uids
            ],
            "validation_values": validation_values,
            "validation_observations": [
                _observation(manifest, uid, observation_type) for uid in validation_uids
            ],
            "noise": core._noise_frame(manifest_path, manifest_sha256, fitting_uids, pandas),
        }
    return inputs


def write_input_manifest(run_dir: Path, bundle: Mapping[str, Any]) -> None:
    """Persist a canonical private input-order manifest before any fit."""

    payload = {
        "ledger_id": str(bundle["ledger"]["ledger_id"]),
        "pilot_plan_id": str(bundle["pilot_plan_id"]),
        "units": [
            {
                "unit_id": str(unit["unit_id"]),
                "fitting_role_id": str(unit["fitting_role_id"]),
                "validation_role_id": str(unit["validation_role_id"]),
                "fitting_uids": [str(uid) for uid in unit["fitting_uids"]],
                "validation_uids": [str(uid) for uid in unit["validation_uids"]],
                "fitting_uid_set_sha256": str(unit["fitting_uid_set_sha256"]),
                "validation_uid_set_sha256": str(unit["validation_uid_set_sha256"]),
            }
            for unit in bundle["units"]
        ],
    }
    core._atomic_write(run_dir / "input_manifest.json", _canon().canonical_json_bytes(payload))


# --------------------------------------------------------------------------- #
# Kernel invocation and persistence
# --------------------------------------------------------------------------- #


def execution_id(unit: Mapping[str, Any], slot: Mapping[str, Any]) -> str:
    return f"{unit['station']}-{slot['recipe_id']}-{int(slot['seed'])}-{str(slot['slot_id'])[:16]}"


def open_history_recorder(run_dir: Path, identifier: str) -> Any:
    return core._Recorder(run_dir / "histories" / f"{identifier}.jsonl")


def train_fit(
    unit_inputs: Mapping[str, Any],
    unit: Mapping[str, Any],
    slot: Mapping[str, Any],
    device: str,
    global_deadline: float,
    history_recorder: Any,
) -> Any:
    """Invoke the frozen development kernel once with the registered fitting role."""

    kernel = _development_kernel()
    return kernel.train_development_fit(
        values=unit_inputs["fitting_values"],
        observations=list(unit_inputs["fitting_observations"]),
        noise_metadata=unit_inputs["noise"],
        validation_values=unit_inputs["validation_values"],
        validation_observations=list(unit_inputs["validation_observations"]),
        role_id=str(unit["fitting_role_id"]),
        recipe=str(slot["recipe_id"]),
        seed=int(slot["seed"]),
        device=device,
        maximum_fit_seconds=MAXIMUM_FIT_SECONDS,
        global_deadline=global_deadline,
        maximum_cuda_allocated_bytes=MAXIMUM_CUDA_ALLOCATED_BYTES,
        on_epoch=history_recorder,
    )


_PRIVATE_RESULT_EXCLUDE = frozenset(
    {
        "state_dict",
        "best_state_dict",
        "terminal_state_dict",
        "validation_logits",
        "validation_uids",
        "classes",
    }
)


def _result_private_summary(result: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in dataclasses.fields(result):
        if field.name in _PRIVATE_RESULT_EXCLUDE:
            continue
        value = getattr(result, field.name)
        summary[field.name] = list(value) if isinstance(value, tuple) else value
    return summary


def _save_logits(path: Path, result: Any) -> None:
    numpy = importlib.import_module("numpy")
    buffer = io.BytesIO()
    numpy.savez_compressed(
        buffer,
        logits=numpy.asarray(result.validation_logits, dtype=numpy.float64),
        classes=numpy.asarray(list(result.classes), dtype=numpy.str_),
        uids=numpy.asarray(list(result.validation_uids), dtype=numpy.str_),
    )
    core._atomic_write(path, buffer.getvalue())


def persist_checkpoints(torch: Any, directory: Path, result: Any) -> dict[str, Any]:
    """Save, reload and re-hash every best and terminal checkpoint."""

    runtime = importlib.import_module("atlas_sers.evaluation.p04_runtime")
    outputs: dict[str, Any] = {}
    for label, state, expected in (
        ("best", result.best_state_dict, result.best_state_digest),
        ("terminal", result.terminal_state_dict, result.terminal_state_digest),
    ):
        if state is None:
            outputs[label] = None
            continue
        path = directory / f"{label}.pt"
        core._save_state(torch, state, path)
        loaded = torch.load(path, weights_only=True, map_location="cpu")
        payload = loaded.get("state_dict") if isinstance(loaded, Mapping) else None
        if not isinstance(payload, Mapping) or set(payload) != set(state):
            raise P05PilotError("checkpoint_reload_mismatch")
        observed = runtime._state_hash(payload)
        if expected is not None and observed != expected:
            raise P05PilotError("checkpoint_digest_mismatch")
        outputs[label] = {
            "file_sha256": core._canon().sha256_file(path),
            "state_sha256": observed,
        }
    return outputs


def persist_result(
    torch: Any, run_dir: Path, unit: Mapping[str, Any], slot: Mapping[str, Any], result: Any
) -> dict[str, Any]:
    """Persist diagnostics for any terminal result, success or failure."""

    identifier = execution_id(unit, slot)
    directory = run_dir / "executions" / identifier
    core._mkdir_exclusive(directory, "execution_directory_exists")
    summary = _result_private_summary(result)
    summary["execution_id"] = identifier
    summary["unit_id"] = str(unit["unit_id"])
    summary["slot_id"] = str(slot["slot_id"])
    summary["recipe_id"] = str(slot["recipe_id"])
    summary["seed"] = int(slot["seed"])
    core._atomic_write(directory / "summary.json", _canon().canonical_json_bytes(summary))
    if result.validation_logits is not None:
        _save_logits(directory / "validation_logits.npz", result)
    checkpoints = persist_checkpoints(torch, directory, result)
    return {
        "execution_id": identifier,
        "unit_id": str(unit["unit_id"]),
        "seed": int(slot["seed"]),
        "recipe_id": str(slot["recipe_id"]),
        "status": str(result.status),
        "reason_code": result.reason_code,
        "optimizer_steps": int(result.optimizer_steps),
        "epochs_completed": int(result.epochs_completed),
        "checkpoints": checkpoints,
    }


def write_run_manifest(run_dir: Path) -> None:
    core._write_manifest(run_dir)


# --------------------------------------------------------------------------- #
# Acceptance checks
# --------------------------------------------------------------------------- #


def _check_history_finite(result: Any) -> list[dict[str, Any]]:
    history = result.history
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise P05PilotError("history_malformed")
    if not (MINIMUM_EPOCHS <= len(history) <= MAXIMUM_EPOCHS):
        raise P05PilotError("history_length_out_of_range")
    required = (
        "epoch",
        "epoch_optimizer_steps",
        "total_optimizer_steps",
        "train_balanced_accuracy",
        "validation_balanced_accuracy",
        "validation_nll",
        "validation_macro_f1",
        "validation_predicted_class_count",
        "sampling_digest",
        "augmentation_digest",
        "pair_digest",
    )
    for index, record in enumerate(history, start=1):
        if not isinstance(record, Mapping):
            raise P05PilotError("history_record_malformed")
        if any(name not in record for name in required):
            raise P05PilotError("history_fields_missing")
        if int(record.get("epoch", -1)) != index:
            raise P05PilotError("history_epoch_mismatch")
        if int(record.get("total_optimizer_steps", -1)) != index * BATCH_DRAWS_PER_EPOCH:
            raise P05PilotError("history_steps_mismatch")
        if int(record.get("epoch_optimizer_steps", -1)) != BATCH_DRAWS_PER_EPOCH:
            raise P05PilotError("history_epoch_steps_mismatch")
        for value in record.values():
            if isinstance(value, bool) or value is None or isinstance(value, str):
                continue
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    raise P05PilotError("history_non_finite")
            else:
                raise P05PilotError("history_field_type_invalid")
    return list(history)


def _reload_checkpoint(torch: Any, path: Path) -> dict[str, Any]:
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    payload = loaded.get("state_dict") if isinstance(loaded, Mapping) else None
    if not isinstance(payload, Mapping):
        raise P05PilotError("checkpoint_reload_mismatch")
    return dict(payload)


def _history_key(record: Mapping[str, Any]) -> tuple[float, float]:
    return (
        -float(record["validation_balanced_accuracy"]),
        float(record["validation_nll"]),
    )


def _expected_best_epoch(history: Sequence[Mapping[str, Any]]) -> int | None:
    best_epoch: int | None = None
    best_key: tuple[float, float] | None = None
    for record in history:
        key = _history_key(record)
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = int(record["epoch"])
    return best_epoch


def _nonimproving_count(history: Sequence[Mapping[str, Any]], upto: int) -> int:
    best_key: tuple[float, float] | None = None
    nonimproving = 0
    for record in history[:upto]:
        key = _history_key(record)
        if best_key is None or key < best_key:
            best_key = key
            nonimproving = 0
        else:
            nonimproving += 1
    return nonimproving


def _earliest_stop(history: Sequence[Mapping[str, Any]]) -> int | None:
    for epoch in range(MINIMUM_EPOCHS, len(history) + 1):
        if _nonimproving_count(history, epoch) >= PATIENCE:
            return epoch
    return None


def _check_stopping(history: Sequence[Mapping[str, Any]]) -> None:
    length = len(history)
    if length < MINIMUM_EPOCHS:
        raise P05PilotError("history_stopped_before_minimum")
    if length > MAXIMUM_EPOCHS:
        raise P05PilotError("history_length_out_of_range")
    earliest = _earliest_stop(history)
    if earliest is not None:
        if length != earliest:
            raise P05PilotError("history_stopping_rule_mismatch")
        return
    if length != MAXIMUM_EPOCHS:
        raise P05PilotError("history_stopped_early")


def check_completed_result(
    result: Any,
    run_dir: Path,
    unit: Mapping[str, Any],
    slot: Mapping[str, Any],
    contract: Mapping[str, Any],
    unit_inputs: Mapping[str, Any],
    torch: Any,
    device: str,
) -> None:
    """Verify one completed fit against the reloaded best checkpoint on disk."""

    if result.status != "complete":
        raise P05PilotError("fit_not_complete")
    if str(result.role_id) != str(unit["fitting_role_id"]):
        raise P05PilotError("result_role_mismatch")
    if str(result.recipe) != str(slot["recipe_id"]):
        raise P05PilotError("result_recipe_mismatch")
    if int(result.seed) != int(slot["seed"]):
        raise P05PilotError("result_seed_mismatch")
    history = _check_history_finite(result)
    _check_stopping(history)
    expected_best = _expected_best_epoch(history)
    if expected_best is None or int(result.best_epoch) != expected_best:
        raise P05PilotError("best_epoch_mismatch")
    best_record = history[expected_best - 1]
    if float(best_record["train_balanced_accuracy"]) != float(
        result.best_training_balanced_accuracy
    ):
        raise P05PilotError("best_training_metric_mismatch")
    for record_field, result_field in (
        ("validation_balanced_accuracy", "best_validation_balanced_accuracy"),
        ("validation_nll", "best_validation_nll"),
        ("validation_macro_f1", "best_validation_macro_f1"),
    ):
        if float(best_record[record_field]) != float(getattr(result, result_field)):
            raise P05PilotError("best_validation_metric_mismatch")
    if int(best_record["validation_predicted_class_count"]) != int(
        result.best_validation_predicted_class_count
    ):
        raise P05PilotError("best_validation_metric_mismatch")
    if int(result.epochs_completed) != len(history):
        raise P05PilotError("epochs_completed_mismatch")
    if int(result.optimizer_steps) != len(history) * BATCH_DRAWS_PER_EPOCH:
        raise P05PilotError("optimizer_steps_mismatch")
    if int(result.optimizer_steps) > MAXIMUM_OPTIMIZER_STEPS:
        raise P05PilotError("optimizer_steps_exceeded")
    if int(result.finite_gradient_batches) != len(history) * BATCH_DRAWS_PER_EPOCH:
        raise P05PilotError("finite_gradient_batches_mismatch")
    elapsed = result.elapsed_seconds
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise P05PilotError("elapsed_format_invalid")
    if not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
        raise P05PilotError("elapsed_format_invalid")
    if float(elapsed) > MAXIMUM_FIT_SECONDS:
        raise P05PilotError("fit_time_exceeded")
    peak = result.peak_cuda_bytes
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise P05PilotError("cuda_format_invalid")
    if int(peak) > MAXIMUM_CUDA_ALLOCATED_BYTES:
        raise P05PilotError("fit_cuda_exceeded")
    if tuple(str(uid) for uid in result.validation_uids) != tuple(
        str(uid) for uid in unit["validation_uids"]
    ):
        raise P05PilotError("validation_uid_order_mismatch")
    if tuple(str(label) for label in result.classes) != tuple(
        str(label) for label in unit["validation_classes"]
    ):
        raise P05PilotError("class_vocabulary_mismatch")
    if result.terminal_state_dict is None:
        raise P05PilotError("terminal_state_missing")
    recipes = {recipe["recipe_id"]: recipe for recipe in contract["recipes"]}
    model_contract = contract["model"]
    recipe = recipes[str(slot["recipe_id"])]
    expected = (
        int(model_contract["projection_model_parameters"])
        if bool(recipe["projection"])
        else int(model_contract["base_parameters"])
    )
    if int(result.parameter_count) != expected:
        raise P05PilotError("parameter_count_mismatch")
    numpy = importlib.import_module("numpy")
    kernel = _development_kernel()
    acquisition = importlib.import_module("atlas_sers.models.acquisition")
    smoke = importlib.import_module("atlas_sers.evaluation.p05_smoke")
    runtime = importlib.import_module("atlas_sers.evaluation.p04_runtime")
    _supcon, _pair, use_projection = smoke.RECIPE_SPECIFICATIONS[str(slot["recipe_id"])]
    torch_device = torch.device(device)
    best_path = run_dir / "executions" / execution_id(unit, slot) / "best.pt"
    state = _reload_checkpoint(torch, best_path)
    model = acquisition.AcquisitionClassifier(
        class_count=len(result.classes), use_projection=use_projection
    )
    model.load_state_dict(state)
    model.to(torch_device)
    values = torch.from_numpy(numpy.ascontiguousarray(unit_inputs["validation_values"][:, None, :]))
    recomputed = kernel._predict_logits(model, values, torch_device)
    stored = numpy.asarray(result.validation_logits, dtype=numpy.float64)
    if recomputed.shape != stored.shape or not bool(numpy.array_equal(recomputed, stored)):
        raise P05PilotError("restored_best_logits_mismatch")
    lookup = {label: index for index, label in enumerate(result.classes)}
    labels = numpy.asarray(
        [lookup[observation.target] for observation in unit_inputs["validation_observations"]],
        dtype=numpy.int64,
    )
    metrics = runtime._metric_values(labels, recomputed, tuple(result.classes))
    if float(metrics["balanced_accuracy"]) != float(result.best_validation_balanced_accuracy):
        raise P05PilotError("restored_best_metric_mismatch")
    if float(metrics["negative_log_likelihood"]) != float(result.best_validation_nll):
        raise P05PilotError("restored_best_metric_mismatch")
    if float(metrics["macro_f1"]) != float(result.best_validation_macro_f1):
        raise P05PilotError("restored_best_metric_mismatch")
    if int(metrics["predicted_class_count"]) != int(result.best_validation_predicted_class_count):
        raise P05PilotError("restored_best_metric_mismatch")


def check_shared_prefixes(items: Sequence[Mapping[str, Any]]) -> None:
    """Require shared RNG prefixes per unit/seed across all recipes and 36 slots."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for item in items:
        grouped.setdefault((str(item["unit_id"]), int(item["seed"])), []).append(item)
    for members in grouped.values():
        recipes = {str(member["slot"]["recipe_id"]) for member in members}
        if recipes != set(PILOT_RECIPES):
            raise P05PilotError("pilot_unit_recipe_coverage_mismatch")
        prefix = min(len(member["result"].history) for member in members)
        for field in ("sampling_digest", "augmentation_digest", "pair_digest"):
            for index in range(prefix):
                digests = {str(member["result"].history[index][field]) for member in members}
                if len(digests) != 1:
                    raise P05PilotError("shared_prefix_digest_mismatch")
        initials = {str(member["result"].initial_backbone_digest) for member in members}
        if len(initials) != 1 or "None" in initials:
            raise P05PilotError("initial_backbone_prefix_mismatch")


def _numeric_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for record in history
    ]


def _check_equivalent_results(left: Any, right: Any) -> None:
    for field in ("initial_state_digest", "best_state_digest", "terminal_state_digest"):
        if getattr(left, field) != getattr(right, field):
            raise P05PilotError("sparse_equivalence_state_mismatch")
    numpy = importlib.import_module("numpy")
    left_logits = numpy.asarray(left.validation_logits, dtype=numpy.float64)
    right_logits = numpy.asarray(right.validation_logits, dtype=numpy.float64)
    if left_logits.shape != right_logits.shape or not bool(
        numpy.array_equal(left_logits, right_logits)
    ):
        raise P05PilotError("sparse_equivalence_logits_mismatch")
    if _numeric_history(left.history) != _numeric_history(right.history):
        raise P05PilotError("sparse_equivalence_history_mismatch")


def check_sparse_equivalences(
    items: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    """Require pair-free units to reproduce D0-M/D2 and D1/D3 exactly."""

    by_unit = {str(unit["unit_id"]): unit for unit in units}
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for item in items:
        grouped.setdefault((str(item["unit_id"]), int(item["seed"])), {})[
            str(item["slot"]["recipe_id"])
        ] = item
    for (unit_id, _seed), members in grouped.items():
        unit = by_unit.get(unit_id)
        if unit is None:
            raise P05PilotError("unit_unknown")
        support = unit.get("auxiliary_support") or {}
        if int(support.get("cross_instrument_master_pairs", 0)) != 0:
            continue
        for left_id, right_id in (("D0-M", "D2"), ("D1", "D3")):
            left = members.get(left_id)
            right = members.get(right_id)
            if left is None or right is None:
                raise P05PilotError("sparse_equivalence_missing")
            _check_equivalent_results(left["result"], right["result"])


def check_sparse_support(
    unit: Mapping[str, Any], slot: Mapping[str, Any], contract: Mapping[str, Any], result: Any
) -> None:
    """A pair-enabled recipe on a pair-free unit must report absent pairs exactly."""

    recipes = {recipe["recipe_id"]: recipe for recipe in contract["recipes"]}
    recipe = recipes[str(slot["recipe_id"])]
    if float(recipe["lambda_pair"]) <= 0.0:
        return
    support = unit.get("auxiliary_support") or {}
    if int(support.get("cross_instrument_master_pairs", 0)) != 0:
        return
    paired = dict(result.paired_support or {})
    for field in ("available_batches", "eligible_masters", "pairs"):
        if int(paired.get(field, 0)) != 0:
            raise P05PilotError("sparse_paired_not_absent")


# --------------------------------------------------------------------------- #
# Run boundary
# --------------------------------------------------------------------------- #


def _pilot_lease_dir(artifact_root: Path, permit_sha256: str) -> Path:
    return artifact_root / P05DEVELOPMENT_NAMESPACE / "leases" / permit_sha256


def _pilot_run_dir(artifact_root: Path, permit_sha256: str) -> Path:
    return artifact_root / P05DEVELOPMENT_NAMESPACE / "runs" / permit_sha256


def _slot_lease_dir(
    artifact_root: Path, contract_sha256: str, core_plan_id: str, slot_id: str
) -> Path:
    return (
        artifact_root
        / P05DEVELOPMENT_NAMESPACE
        / "slot_leases"
        / str(contract_sha256)
        / str(core_plan_id)
        / str(slot_id)
    )


def _assert_no_prior_leases(
    artifact_root: Path,
    permit_sha256: str,
    contract_sha256: str,
    core_plan_id: str,
    slots: Sequence[Mapping[str, Any]],
) -> None:
    for path, code in (
        (_pilot_lease_dir(artifact_root, permit_sha256), "pilot_lease_exists"),
        (_pilot_run_dir(artifact_root, permit_sha256), "pilot_run_exists"),
    ):
        if path.exists() or path.is_symlink():
            raise P05PilotError(code)
    for slot in slots:
        path = _slot_lease_dir(artifact_root, contract_sha256, core_plan_id, str(slot["slot_id"]))
        if path.exists() or path.is_symlink():
            raise P05PilotError("slot_lease_exists")


def _reserve_slot_lease(
    artifact_root: Path,
    contract_sha256: str,
    core_plan_id: str,
    slot: Mapping[str, Any],
) -> None:
    directory = _slot_lease_dir(artifact_root, contract_sha256, core_plan_id, str(slot["slot_id"]))
    core._mkdir_exclusive(directory, "slot_lease_exists")
    core._atomic_write(
        directory / "lease.json",
        _canon().canonical_json_bytes(
            {
                "slot_id": str(slot["slot_id"]),
                "unit_id": str(slot["unit_id"]),
                "recipe_id": str(slot["recipe_id"]),
                "seed": int(slot["seed"]),
                "contract_sha256": str(contract_sha256),
                "core_plan_id": str(core_plan_id),
                "permit_sha256": PILOT_PERMIT_SHA256,
            }
        ),
    )


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _enforce_storage_cap(artifact_root: Path) -> None:
    if _directory_size_bytes(artifact_root / P05DEVELOPMENT_NAMESPACE) > (
        PRIVATE_STORAGE_CEILING_BYTES
    ):
        raise P05PilotError("private_storage_ceiling_exceeded")


def _enforce_cuda_cap(torch: Any, device: str) -> None:
    if str(device) != "cuda":
        return
    try:
        current = int(torch.cuda.memory_allocated())
        peak = int(torch.cuda.max_memory_allocated())
    except Exception:
        raise P05PilotError("cuda_memory_unreadable") from None
    if current > MAXIMUM_CUDA_ALLOCATED_BYTES or peak > MAXIMUM_CUDA_ALLOCATED_BYTES:
        raise P05PilotError("fit_cuda_exceeded")


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() > float(deadline):
        raise P05PilotError("global_deadline_exceeded")


def _journal(path: Path) -> Any:
    def _write(entry: Mapping[str, Any]) -> None:
        with path.open("ab") as stream:
            stream.write(_canon().canonical_json_bytes(dict(entry)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    return _write


def _lower_bound_updates(run_dir: Path, identifier: str) -> int:
    path = run_dir / "histories" / f"{identifier}.jsonl"
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count * BATCH_DRAWS_PER_EPOCH


def _check_expected_slots(
    results: Sequence[Mapping[str, Any]], slots: Sequence[Mapping[str, Any]]
) -> None:
    if len(slots) != MAXIMUM_FIT_EXECUTIONS or len(results) != MAXIMUM_FIT_EXECUTIONS:
        raise P05PilotError("execution_slot_count_mismatch")
    expected_ids = [str(slot["slot_id"]) for slot in slots]
    observed_ids = [str(item["slot"]["slot_id"]) for item in results]
    if observed_ids != expected_ids:
        raise P05PilotError("execution_slot_order_mismatch")


class _GuardedRecorder:
    """Persist each epoch, then enforce global deadline/storage/CUDA bounds."""

    __slots__ = ("_recorder", "_checks")

    def __init__(self, recorder: Any, checks: Sequence[Any]) -> None:
        self._recorder = recorder
        self._checks = tuple(checks)

    def __call__(self, record: Any) -> None:
        self._recorder(record)
        for check in self._checks:
            check()

    def close(self) -> None:
        self._recorder.close()


def _verify_manifest(run_dir: Path) -> None:
    base = run_dir.resolve()
    manifest = core._read_json(run_dir / "manifest.json", "manifest")
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(files, Mapping):
        raise P05PilotError("manifest_malformed")
    paths = [path for path in base.rglob("*") if path.is_file() and path.name != "manifest.json"]
    if dict(files) != core._canon().hash_relative_files(base, paths):
        raise P05PilotError("manifest_integrity_mismatch")


def _run_public_summary(
    bundle: Mapping[str, Any], device: str, state: Mapping[str, Any], wall_seconds: float
) -> dict[str, Any]:
    return {
        "status": "complete",
        "command": "run",
        "permit_sha256": bundle["permit_sha256"],
        "core_contract_sha256": bundle["contract_sha256"],
        "core_plan_id": bundle["core_plan_id"],
        "ledger_id": bundle["ledger"]["ledger_id"],
        "pilot_plan_id": bundle["pilot_plan_id"],
        "device": device,
        "stations": [str(unit["station"]) for unit in bundle["units"]],
        "recipes": list(PILOT_RECIPES),
        "seeds": list(PILOT_SEEDS),
        "started": int(state["started"]),
        "completed": int(state["completed"]),
        "failed": int(state["failed"]),
        "unstarted": MAXIMUM_FIT_EXECUTIONS - int(state["started"]),
        "optimizer_steps": int(state["steps"]),
        "sum_elapsed_seconds": float(state["elapsed"]),
        "maximum_peak_cuda_bytes": int(state["peak"]),
        "wall_seconds": float(wall_seconds),
        "checkpoint_preflight": "pass",
        "claim": PILOT_CLAIM,
        "execution_authorized": False,
        "outer_evaluation_authorized": False,
    }


def _write_run_failure(
    run_dir: Path, bundle: Mapping[str, Any], error: BaseException, state: Mapping[str, Any]
) -> None:
    try:
        core._atomic_write(
            run_dir / "summary.json",
            _canon().canonical_json_bytes(
                {
                    "status": "fail",
                    "command": "run",
                    "permit_sha256": bundle["permit_sha256"],
                    "core_contract_sha256": bundle["contract_sha256"],
                    "core_plan_id": bundle["core_plan_id"],
                    "pilot_plan_id": bundle["pilot_plan_id"],
                    "reason_code": getattr(error, "reason_code", type(error).__name__),
                    "started": int(state["started"]),
                    "completed": int(state["completed"]),
                    "failed": int(state["failed"]),
                    "unstarted": MAXIMUM_FIT_EXECUTIONS - int(state["started"]),
                    "optimizer_steps": int(state["steps"]),
                    "optimizer_steps_exact": bool(state.get("steps_exact", True)),
                    "claim": PILOT_CLAIM,
                    "execution_authorized": False,
                    "outer_evaluation_authorized": False,
                }
            ),
        )
    except Exception:
        pass


def _post_run_reauth(
    artifact_root: Path,
    contract: Mapping[str, Any],
    support: Any,
    provenance_before: Any,
    repository_root: Path,
    project_root: Path,
) -> Any:
    support_after, _p01, _p04 = core._authenticate(artifact_root, contract)
    if (
        support_after.manifest_sha256 != support.manifest_sha256
        or support_after.contexts_sha256 != support.contexts_sha256
        or support_after.roles_sha256 != support.roles_sha256
    ):
        raise P05PilotError("input_changed_during_execution")
    provenance_after = core._capture_provenance(repository_root, project_root, artifact_root)
    core._assert_protected_identity(provenance_before, provenance_after)
    return provenance_after


def run(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    permit_path: Path | str,
    pilot_plan_id: str,
    device: str,
    permit_sha256: str = PILOT_PERMIT_SHA256,
) -> dict[str, Any]:
    """Execute exactly the 36 reviewed development slots once, or fail closed."""

    core._configure_environment()
    deadline = time.perf_counter() + MAXIMUM_TOTAL_SECONDS
    if device not in ("cpu", "cuda"):
        raise P05PilotError("device_invalid")
    if not core._is_hex64(pilot_plan_id):
        raise P05PilotError("pilot_plan_id_malformed")
    bundle = prepare(project_root, artifact_root, contract_path, permit_path, permit_sha256)
    if pilot_plan_id != bundle["pilot_plan_id"]:
        raise P05PilotError("pilot_plan_authority_mismatch")
    artifact = bundle["artifact_root"]
    contract = bundle["contract"]
    support = bundle["support"]
    slots = bundle["slots"]
    units = bundle["units"]
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    contract_sha256 = bundle["contract_sha256"]
    core_plan_id = bundle["core_plan_id"]
    permit_digest = bundle["permit_sha256"]
    _assert_no_prior_leases(artifact, permit_digest, contract_sha256, core_plan_id, slots)
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    _development_kernel()
    if _free_disk_bytes(artifact) < PRIVATE_STORAGE_CEILING_BYTES:
        raise P05PilotError("insufficient_free_disk")
    if device == "cuda":
        if not bool(torch.cuda.is_available()):
            raise P05PilotError("cuda_unavailable")
        if _free_cuda_bytes(torch) < MINIMUM_FREE_CUDA_BYTES:
            raise P05PilotError("insufficient_free_cuda")
    _enforce_storage_cap(artifact)
    _enforce_cuda_cap(torch, device)
    _checkpoint_preflight(torch, artifact)
    _check_deadline(deadline)
    provenance_before = core._capture_provenance(
        bundle["repository_root"], bundle["project_root"], artifact
    )
    _store_pilot_plan(artifact, _canon().canonical_json_bytes(bundle["pilot_plan"]), pilot_plan_id)
    lease = _pilot_lease_dir(artifact, permit_digest)
    run_dir = _pilot_run_dir(artifact, permit_digest)
    state: dict[str, Any] = {
        "started": 0,
        "completed": 0,
        "failed": 0,
        "steps": 0,
        "steps_exact": True,
        "elapsed": 0.0,
        "peak": 0,
    }
    results: list[dict[str, Any]] = []
    journal: Any = None
    lease_created = False
    run_dir_created = False
    current_identifier: str | None = None
    result_available = False
    in_flight = False
    try:
        core._mkdir_exclusive(lease, "pilot_lease_exists")
        lease_created = True
        core._atomic_write(
            lease / "lease.json",
            _canon().canonical_json_bytes(
                {
                    "state": "running",
                    "permit_sha256": permit_digest,
                    "pilot_plan_id": pilot_plan_id,
                    "core_contract_sha256": contract_sha256,
                    "core_plan_id": core_plan_id,
                }
            ),
        )
        core._mkdir_exclusive(run_dir, "pilot_run_exists")
        run_dir_created = True
        _enforce_storage_cap(artifact)
        core._atomic_write(
            run_dir / "provenance.json", _canon().canonical_json_bytes(provenance_before)
        )
        write_input_manifest(run_dir, bundle)
        unit_inputs = prepare_role_inputs(bundle)
        journal = _journal(run_dir / "ledger.jsonl")
        for order, slot in enumerate(slots):
            journal(
                {
                    "event": "planned",
                    "order": order,
                    "slot_id": str(slot["slot_id"]),
                    "unit_id": str(slot["unit_id"]),
                    "recipe_id": str(slot["recipe_id"]),
                    "seed": int(slot["seed"]),
                }
            )
        for slot in slots:
            _check_deadline(deadline)
            _enforce_storage_cap(artifact)
            _enforce_cuda_cap(torch, device)
            if state["started"] >= MAXIMUM_FIT_EXECUTIONS:
                raise P05PilotError("execution_ceiling_exceeded")
            if state["steps"] > MAXIMUM_OPTIMIZER_STEPS:
                raise P05PilotError("optimizer_steps_exceeded")
            unit = unit_by_id[str(slot["unit_id"])]
            identifier = execution_id(unit, slot)
            _reserve_slot_lease(artifact, contract_sha256, core_plan_id, slot)
            current_identifier = identifier
            result_available = False
            in_flight = True
            state["started"] += 1
            journal(
                {
                    "event": "started",
                    "execution_id": identifier,
                    "slot_id": str(slot["slot_id"]),
                    "unit_id": str(unit["unit_id"]),
                    "recipe_id": str(slot["recipe_id"]),
                    "seed": int(slot["seed"]),
                    "used_fit_count": state["started"],
                }
            )
            recorder = open_history_recorder(run_dir, identifier)
            guarded = _GuardedRecorder(
                recorder,
                (
                    lambda: _check_deadline(deadline),
                    lambda: _enforce_storage_cap(artifact),
                    lambda: _enforce_cuda_cap(torch, device),
                ),
            )
            try:
                result = train_fit(
                    unit_inputs[str(unit["unit_id"])],
                    unit,
                    slot,
                    device,
                    deadline,
                    guarded,
                )
                result_available = True
                state["steps"] += int(result.optimizer_steps)
                state["elapsed"] += float(result.elapsed_seconds)
                state["peak"] = max(state["peak"], int(result.peak_cuda_bytes))
            finally:
                guarded.close()
            persist_result(torch, run_dir, unit, slot, result)
            if result.status != "complete":
                state["failed"] += 1
                in_flight = False
                current_identifier = None
                journal(
                    {
                        "event": "failed",
                        "execution_id": identifier,
                        "slot_id": str(slot["slot_id"]),
                        "status": str(result.status),
                        "reason_code": result.reason_code,
                    }
                )
                raise P05PilotError("fit_failed")
            check_completed_result(
                result,
                run_dir,
                unit,
                slot,
                contract,
                unit_inputs[str(unit["unit_id"])],
                torch,
                device,
            )
            check_sparse_support(unit, slot, contract, result)
            _check_deadline(deadline)
            _enforce_cuda_cap(torch, device)
            if device == "cuda":
                state["peak"] = max(state["peak"], int(torch.cuda.max_memory_allocated()))
            _enforce_storage_cap(artifact)
            results.append(
                {
                    "slot": slot,
                    "unit": unit,
                    "unit_id": str(unit["unit_id"]),
                    "seed": int(slot["seed"]),
                    "result": result,
                }
            )
            state["completed"] += 1
            in_flight = False
            current_identifier = None
            journal(
                {
                    "event": "completed",
                    "execution_id": identifier,
                    "slot_id": str(slot["slot_id"]),
                    "status": str(result.status),
                    "optimizer_steps": int(result.optimizer_steps),
                }
            )
        _check_deadline(deadline)
        if state["started"] > MAXIMUM_FIT_EXECUTIONS:
            raise P05PilotError("execution_ceiling_exceeded")
        if state["steps"] > MAXIMUM_OPTIMIZER_STEPS:
            raise P05PilotError("optimizer_steps_exceeded")
        _check_expected_slots(results, slots)
        check_shared_prefixes(results)
        check_sparse_equivalences(results, units, contract)
        provenance_after = _post_run_reauth(
            artifact,
            contract,
            support,
            provenance_before,
            bundle["repository_root"],
            bundle["project_root"],
        )
        wall_seconds = time.perf_counter() - (deadline - MAXIMUM_TOTAL_SECONDS)
        if wall_seconds > MAXIMUM_TOTAL_SECONDS:
            raise P05PilotError("total_wall_exceeded")
        _check_deadline(deadline)
        _enforce_storage_cap(artifact)
        _enforce_cuda_cap(torch, device)
        summary = _run_public_summary(bundle, device, state, wall_seconds)
        core._atomic_write(
            run_dir / "provenance_after.json",
            _canon().canonical_json_bytes(provenance_after),
        )
        core._atomic_write(run_dir / "summary.json", _canon().canonical_json_bytes(summary))
        core._atomic_write(
            lease / "lease.json",
            _canon().canonical_json_bytes(
                {
                    "state": "complete",
                    "permit_sha256": permit_digest,
                    "pilot_plan_id": pilot_plan_id,
                    "core_contract_sha256": contract_sha256,
                    "core_plan_id": core_plan_id,
                }
            ),
        )
        core._write_manifest(run_dir)
        _check_deadline(deadline)
        _enforce_storage_cap(artifact)
        _enforce_cuda_cap(torch, device)
        _verify_manifest(run_dir)
        _check_deadline(deadline)
        return summary
    except BaseException as error:
        if in_flight:
            state["failed"] += 1
            if journal is not None:
                try:
                    journal(
                        {
                            "event": "failed",
                            "execution_id": current_identifier,
                            "reason_code": getattr(error, "reason_code", type(error).__name__),
                        }
                    )
                except Exception:
                    pass
        if not result_available and current_identifier is not None:
            state["steps"] += _lower_bound_updates(run_dir, current_identifier)
            state["steps_exact"] = False
        if run_dir_created:
            _write_run_failure(run_dir, bundle, error, state)
            try:
                provenance_after = _post_run_reauth(
                    artifact,
                    contract,
                    support,
                    provenance_before,
                    bundle["repository_root"],
                    bundle["project_root"],
                )
            except Exception as reauth_error:
                try:
                    core._atomic_write(
                        run_dir / "provenance_after.json",
                        _canon().canonical_json_bytes(
                            {
                                "status": "fail",
                                "reason_code": getattr(
                                    reauth_error, "reason_code", type(reauth_error).__name__
                                ),
                            }
                        ),
                    )
                except Exception:
                    pass
            else:
                try:
                    core._atomic_write(
                        run_dir / "provenance_after.json",
                        _canon().canonical_json_bytes(provenance_after),
                    )
                except Exception:
                    pass
            try:
                core._write_manifest(run_dir)
            except Exception:
                pass
        if lease_created:
            try:
                core._atomic_write(
                    lease / "lease.json",
                    _canon().canonical_json_bytes(
                        {
                            "state": "failed",
                            "permit_sha256": permit_digest,
                            "pilot_plan_id": pilot_plan_id,
                            "reason_code": getattr(error, "reason_code", type(error).__name__),
                        }
                    ),
                )
            except Exception:
                pass
        if isinstance(error, (KeyboardInterrupt, SystemExit, P05CoreError)):
            raise
        raise P05PilotError("pilot_execution_failed") from error


def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(_canon().canonical_json_bytes(dict(payload)).decode("utf-8") + "\n")


class _PilotArgumentParser(argparse.ArgumentParser):
    """Argument parser that never echoes private argument values."""

    def error(self, message: str) -> Any:
        raise P05PilotError("arguments_invalid")


def cli_main(argv: Sequence[str] | None = None) -> int:
    core._configure_environment()
    parser = _PilotArgumentParser(prog="run_p05_pilot")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--permit", required=True)
    parser.add_argument("--permit-sha256", default=PILOT_PERMIT_SHA256)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--pilot-plan-id", required=True)
    run_parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    try:
        arguments = parser.parse_args(argv)
    except P05CoreError as error:
        _emit({"status": "fail", "command": "cli", "reason_code": error.reason_code})
        return 1
    try:
        if arguments.command == "preflight":
            report = preflight(
                arguments.project_root,
                arguments.artifact_root,
                arguments.contract,
                arguments.permit,
                arguments.permit_sha256,
            )
        else:
            report = run(
                project_root=arguments.project_root,
                artifact_root=arguments.artifact_root,
                contract_path=arguments.contract,
                permit_path=arguments.permit,
                pilot_plan_id=arguments.pilot_plan_id,
                device=arguments.device,
                permit_sha256=arguments.permit_sha256,
            )
    except P05CoreError as error:
        _emit(
            {
                "status": "fail",
                "command": arguments.command,
                "reason_code": error.reason_code,
            }
        )
        return 1
    except Exception:
        _emit(
            {
                "status": "fail",
                "command": arguments.command,
                "reason_code": "internal_error",
            }
        )
        return 1
    _emit(report)
    return 0


__all__ = [
    "BATCH_DRAWS_PER_EPOCH",
    "MAXIMUM_CUDA_ALLOCATED_BYTES",
    "MAXIMUM_EPOCHS",
    "MAXIMUM_FIT_EXECUTIONS",
    "MAXIMUM_FIT_SECONDS",
    "MAXIMUM_OPTIMIZER_STEPS",
    "MAXIMUM_TOTAL_SECONDS",
    "MINIMUM_EPOCHS",
    "MINIMUM_FREE_CUDA_BYTES",
    "P05DEVELOPMENT_NAMESPACE",
    "P05PilotError",
    "PATIENCE",
    "PILOT_CLAIM",
    "PILOT_CORE_CONTRACT_SHA256",
    "PILOT_CORE_PLAN_ID",
    "PILOT_PERMIT_SHA256",
    "PILOT_PROTOCOL_VERSION",
    "PILOT_RECIPES",
    "PILOT_SCHEMA_VERSION",
    "PILOT_SEEDS",
    "PILOT_STATIONS",
    "PRIVATE_STORAGE_CEILING_BYTES",
    "check_completed_result",
    "check_shared_prefixes",
    "check_sparse_equivalences",
    "check_sparse_support",
    "cli_main",
    "execution_id",
    "open_history_recorder",
    "persist_checkpoints",
    "persist_result",
    "prepare",
    "prepare_role_inputs",
    "preflight",
    "run",
    "train_fit",
    "write_input_manifest",
    "write_run_manifest",
]
