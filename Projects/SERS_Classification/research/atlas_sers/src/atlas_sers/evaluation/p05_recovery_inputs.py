"""Read-only recovery input authentication for the P05 interruption prefix.

The boundary authenticates the original interrupted comprehensive development
run and its private evidence in place.  It never trains, instantiates a model,
loads a checkpoint or logits payload, copies files, writes files, creates a
stage or lease, performs selection or exports public data.  Checkpoint and
logits artifacts are treated as opaque bytes and only hashed.  It imports only
the standard library and the existing stdlib-only P05 boundary modules; torch
and numpy are never imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan

__all__ = [
    "RecoveryInputsError",
    "authenticate_original_stage",
    "prepare_recovery",
]

RECOVERY_INPUTS_SCHEMA_VERSION = "nato-sers-p05-recovery-inputs-v1"
INTERRUPTION_ANCHOR_SCHEMA_VERSION = "nato-sers-p05-interruption-anchor-v1"
COMPREHENSIVE_NAMESPACE = "p05comprehensive"
DEVELOP_STAGE_NAME = "develop"

ORIGINAL_TOP_FILES = (
    "ledger.json",
    "source_ledger.json",
    "events.jsonl",
    "selector.jsonl",
    "progress.json",
    "input_manifest.json",
    "provenance_before.json",
)
ORIGINAL_ALLOWED_ROOT_ENTRIES = frozenset({DEVELOP_STAGE_NAME})
SLOT_EXECUTION_FILES = (
    "best.pt",
    "terminal.pt",
    "summary.json",
    "validation_logits.npz",
)
SEALED_UNIT_MANIFEST_NAME = "manifest.json"

_MAX_JSON_BYTES = 128 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024


class RecoveryInputsError(ValueError):
    """Stable, path-free recovery-input failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


def _strict_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryInputsError(code)
    return value


def _check_deadline(deadline: Any) -> None:
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise RecoveryInputsError("deadline_invalid")
    try:
        value = float(deadline)
    except (ValueError, OverflowError) as error:
        raise RecoveryInputsError("deadline_invalid") from error
    if not math.isfinite(value):
        raise RecoveryInputsError("deadline_invalid")
    if time.perf_counter() > value:
        raise RecoveryInputsError("deadline_exceeded")


def _check_component(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise RecoveryInputsError("path_component_rejected")
    if name in (".", "..") or "/" in name or "\\" in name:
        raise RecoveryInputsError("path_component_rejected")
    for character in name:
        code = ord(character)
        if code < 0x20 or 0x7F <= code <= 0x9F:
            raise RecoveryInputsError("path_component_rejected")
    try:
        name.encode("utf-8")
    except UnicodeError as error:
        raise RecoveryInputsError("path_component_rejected") from error
    return name


def _reject_symlink_chain(path: Path) -> None:
    try:
        raw = os.fspath(path)
        if not isinstance(raw, str) or not raw or "\0" in raw:
            raise RecoveryInputsError("path_rejected")
        raw.encode("utf-8")
        for component in Path(raw).parts:
            if component != os.path.sep:
                _check_component(component)
        candidate = Path(os.path.abspath(raw))
    except (TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, RecoveryInputsError):
            raise
        raise RecoveryInputsError("path_rejected") from error
    while True:
        if os.path.islink(candidate):
            raise RecoveryInputsError("symlink_path_rejected")
        parent = candidate.parent
        if parent == candidate:
            return
        candidate = parent


def _open_regular(path: Path, code: str) -> Any:
    _reject_symlink_chain(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, ValueError, TypeError) as error:
        raise RecoveryInputsError(code) from error
    try:
        info = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise RecoveryInputsError(code) from error
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise RecoveryInputsError(code)
    try:
        return os.fdopen(descriptor, "rb")
    except (OSError, ValueError) as error:
        os.close(descriptor)
        raise RecoveryInputsError(code) from error


def _hash_stream(handle: Any, deadline: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        _check_deadline(deadline)
        try:
            chunk = handle.read(_HASH_CHUNK_BYTES)
        except OSError as error:
            raise RecoveryInputsError("file_unreadable") from error
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _hash_file_record(path: Path, deadline: Any) -> dict[str, Any]:
    with _open_regular(path, "file_unreadable") as handle:
        digest, size = _hash_stream(handle, deadline)
    return {"sha256": digest, "size_bytes": size}


def _read_bytes_bounded(path: Path, limit: int, code: str, deadline: Any) -> bytes:
    with _open_regular(path, code) as handle:
        chunks: list[bytes] = []
        total = 0
        while True:
            _check_deadline(deadline)
            try:
                chunk = handle.read(min(_HASH_CHUNK_BYTES, limit - total + 1))
            except OSError as error:
                raise RecoveryInputsError(code) from error
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise RecoveryInputsError(f"{code}_too_large")
            chunks.append(chunk)
    return b"".join(chunks)


def _reject_json_constant(_token: str) -> Any:
    raise RecoveryInputsError("json_nonfinite_constant")


def _object_pairs_no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryInputsError("json_duplicate_key")
        result[key] = value
    return result


def _decode_utf8(raw: bytes, code: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecoveryInputsError(f"{code}_not_utf8") from error


def _parse_json_text(text: str, code: str) -> Any:
    try:
        result = json.loads(
            text,
            object_pairs_hook=_object_pairs_no_duplicates,
            parse_constant=_reject_json_constant,
        )
        stack = [(result, 0)]
        while stack:
            value, depth = stack.pop()
            if depth > 128:
                raise RecoveryInputsError("json_nesting_exceeded")
            if isinstance(value, str):
                value.encode("utf-8")
            elif isinstance(value, float) and not math.isfinite(value):
                raise RecoveryInputsError("json_nonfinite_number")
            elif isinstance(value, dict):
                for key, item in value.items():
                    key.encode("utf-8")
                    stack.append((item, depth + 1))
            elif isinstance(value, list):
                stack.extend((item, depth + 1) for item in value)
        return result
    except RecoveryInputsError:
        raise
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError) as error:
        raise RecoveryInputsError(f"{code}_malformed") from error


def _parse_json(raw: bytes, code: str) -> Any:
    return _parse_json_text(_decode_utf8(raw, code), code)


def _parse_jsonl(raw: bytes, code: str) -> list[dict[str, Any]]:
    text = _decode_utf8(raw, code)
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = _parse_json_text(line, code)
        if not isinstance(value, Mapping):
            raise RecoveryInputsError(f"{code}_malformed")
        records.append(dict(value))
    return records


def _read_json_mapping(path: Path, code: str, deadline: Any) -> dict[str, Any]:
    raw = _read_bytes_bounded(path, _MAX_JSON_BYTES, code, deadline)
    value = _parse_json(raw, code)
    if not isinstance(value, Mapping):
        raise RecoveryInputsError(f"{code}_malformed")
    return dict(value)


def _read_jsonl(path: Path, code: str, deadline: Any) -> list[dict[str, Any]]:
    raw = _read_bytes_bounded(path, _MAX_JSON_BYTES, code, deadline)
    return _parse_jsonl(raw, code)


def _list_entries(directory: Path, code: str) -> dict[str, str]:
    _reject_symlink_chain(directory)
    try:
        scanned = list(os.scandir(directory))
    except OSError as error:
        raise RecoveryInputsError(code) from error
    result: dict[str, str] = {}
    for entry in scanned:
        name = _check_component(entry.name)
        try:
            if entry.is_symlink():
                raise RecoveryInputsError("symlink_path_rejected")
            if entry.is_dir(follow_symlinks=False):
                result[name] = "dir"
            elif entry.is_file(follow_symlinks=False):
                result[name] = "file"
            else:
                raise RecoveryInputsError("entry_type_rejected")
        except OSError as error:
            raise RecoveryInputsError(code) from error
    return result


def _collect_tree(root: Path, deadline: Any) -> tuple[dict[str, Path], set[str]]:
    files: dict[str, Path] = {}
    directories: set[str] = set()
    stack: list[tuple[Path, str]] = [(root, "")]
    while stack:
        _check_deadline(deadline)
        directory, prefix = stack.pop()
        entries = _list_entries(directory, "directory_unreadable")
        for name, kind in entries.items():
            relative = f"{prefix}/{name}" if prefix else name
            if kind == "dir":
                directories.add(relative)
                stack.append((directory / name, relative))
            else:
                files[relative] = directory / name
    return files, directories


def _record_equal(entry: Any, record: Mapping[str, Any]) -> bool:
    if isinstance(entry, Mapping) and set(entry) == {"sha256", "size_bytes"}:
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int):
            return False
        return entry.get("sha256") == record["sha256"] and size == record["size_bytes"]
    return False


def _slot_files(unit: Mapping[str, Any], slot: Mapping[str, Any]) -> list[str]:
    identifier = _check_component(pilot.execution_id(unit, slot))
    base = f"executions/{identifier}"
    files = [f"{base}/{name}" for name in SLOT_EXECUTION_FILES]
    files.append(f"histories/{identifier}.jsonl")
    return files


def _original_run_root(artifact_root: Path | str) -> Path:
    return (
        Path(artifact_root)
        / COMPREHENSIVE_NAMESPACE
        / "runs"
        / authority.BASECOMPREHENSIVE_PERMIT_SHA256
    )


def _assert_root_exclusive(run_root: Path) -> None:
    entries = _list_entries(run_root, "original_run_root_unreadable")
    if set(entries) != set(ORIGINAL_ALLOWED_ROOT_ENTRIES):
        raise RecoveryInputsError("original_run_root_not_exclusive")
    if entries.get(DEVELOP_STAGE_NAME) != "dir":
        raise RecoveryInputsError("original_run_root_not_exclusive")


def _expected_source_ledger(bundle: Mapping[str, Any]) -> dict[str, Any]:
    from atlas_sers.evaluation.p05_comprehensive_freeze import _expected_source_ledger

    return _expected_source_ledger(bundle)


def _verify_pilot_inventory(run_dir: Path, deadline: Any) -> None:
    """Strict inventory, including nested manifests and unexpected empty dirs."""
    raw = _read_bytes_bounded(
        run_dir / "manifest.json", _MAX_JSON_BYTES, "pilot_manifest", deadline
    )
    if hashlib.sha256(raw).hexdigest() != base_inputs.PILOT_MANIFEST_SHA256:
        raise RecoveryInputsError("pilot_manifest_digest_mismatch")
    manifest = _parse_json(raw, "pilot_manifest")
    expected = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(expected, Mapping):
        raise RecoveryInputsError("pilot_manifest_malformed")
    expected_dirs: set[str] = set()
    for relative in expected:
        parts = relative.split("/")
        for component in parts:
            _check_component(component)
        expected_dirs.update("/".join(parts[:n]) for n in range(1, len(parts)))
    observed, directories = _collect_tree(run_dir, deadline)
    if set(observed) != set(expected) | {"manifest.json"} or directories != expected_dirs:
        raise RecoveryInputsError("pilot_inventory_mismatch")
    for relative, record in expected.items():
        if not _record_equal(record, _hash_file_record(observed[relative], deadline)):
            raise RecoveryInputsError("pilot_inventory_digest_mismatch")
    if _hash_file_record(run_dir / "manifest.json", deadline) != {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }:
        raise RecoveryInputsError("pilot_manifest_changed")


def _authenticate_pilot(bundle: Mapping[str, Any], deadline: Any) -> None:
    run_dir = base_inputs._pilot_run_dir(bundle["artifact_root"])
    _check_deadline(deadline)
    _verify_pilot_inventory(run_dir, deadline)
    base_inputs._check_pilot_run_manifest(run_dir, bundle["permit"])
    _check_deadline(deadline)
    base_inputs._check_pilot_run_summary(run_dir)
    _check_deadline(deadline)
    base_inputs._check_pilot_slot_leases(bundle)
    _check_deadline(deadline)


def _read_original_leases(
    lease_root: Path, deadline: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = _list_entries(lease_root, "original_lease_root_unreadable")
    directories = sorted(name for name, kind in entries.items() if kind == "dir")
    if len(directories) != recovery_plan.LEASE_COUNT or len(directories) != len(entries):
        raise RecoveryInputsError("original_lease_count_mismatch")
    records: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {}
    for name in directories:
        _check_deadline(deadline)
        slot_id = _check_component(name)
        directory = lease_root / slot_id
        inner = _list_entries(directory, "original_lease_directory_unreadable")
        if set(inner) != {"lease.json"} or inner.get("lease.json") != "file":
            raise RecoveryInputsError("original_lease_directory_malformed")
        lease_path = directory / "lease.json"
        raw = _read_bytes_bounded(lease_path, _MAX_JSON_BYTES, "original_slot_lease", deadline)
        inventory[f"{slot_id}/lease.json"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        value = _parse_json(raw, "original_slot_lease")
        if not isinstance(value, Mapping):
            raise RecoveryInputsError("original_slot_lease_malformed")
        record = dict(value)
        if str(record.get("slot_id")) != slot_id:
            raise RecoveryInputsError("original_lease_identity_mismatch")
        records.append(record)
    return records, inventory


def _check_plan_against_permit(plan: Mapping[str, Any], permit: Mapping[str, Any]) -> None:
    counts = plan["counts"]
    updates = plan["optimizer_updates"]
    checks = (
        (counts, "source_slots", permit, "final_source_evidence_slots"),
        (counts, "pilot_reused", permit, "reused_pilot_slots"),
        (counts, "original_started", permit, "original_new_attempts_started"),
        (counts, "original_completed", permit, "original_new_fits_completed"),
        (counts, "original_interrupted", permit, "original_interrupted_attempts"),
        (counts, "unstarted", permit, "original_unstarted_slots"),
        (counts, "recovery_fits", permit, "maximum_recovery_fit_executions"),
        (
            counts,
            "final_new_source_successes",
            permit,
            "final_successful_new_source_slots",
        ),
        (counts, "final_new_source_attempts", permit, "final_new_source_attempts"),
        (
            updates,
            "original_completed_exact",
            permit,
            "original_completed_optimizer_updates",
        ),
        (
            updates,
            "interrupted_observed_lower_bound",
            permit,
            "original_interrupted_observed_update_lower_bound",
        ),
        (
            updates,
            "interrupted_charged_upper_bound",
            permit,
            "original_interrupted_charged_update_upper_bound",
        ),
    )
    for source, key, permit_source, permit_key in checks:
        observed = _strict_int(source.get(key), "plan_count_invalid")
        expected = _strict_int(permit_source.get(permit_key), "permit_count_invalid")
        if observed != expected:
            raise RecoveryInputsError("plan_permit_count_mismatch")
    sealed = len(list(plan["sealed_unit_ids"]))
    if sealed != _strict_int(permit.get("original_sealed_units"), "permit_count_invalid"):
        raise RecoveryInputsError("plan_permit_count_mismatch")


def _expected_layout(
    bundle: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[set[str], set[str], set[str], dict[str, set[str]], set[str]]:
    ledger = bundle["ledger"]
    unit_by_id = {str(unit["unit_id"]): unit for unit in ledger["units"]}
    slots_by_unit: dict[str, list[Mapping[str, Any]]] = {}
    for slot in ledger["slots"]:
        slots_by_unit.setdefault(str(slot["unit_id"]), []).append(slot)

    completed = {str(slot_id) for slot_id in plan["reused_original_slot_ids"]}
    interrupted_slot_id = str(plan["interrupted_slot_id"])
    incomplete_unit_id = str(plan["incomplete_unit_id"])
    sealed_unit_ids = [str(unit_id) for unit_id in plan["sealed_unit_ids"]]

    files: set[str] = set(ORIGINAL_TOP_FILES)
    directories: set[str] = {"units"}
    sealed_manifests: set[str] = set()
    sealed_unit_files: dict[str, set[str]] = {}
    unsealed_files: set[str] = set()

    for unit_id in sealed_unit_ids:
        unit = unit_by_id[unit_id]
        group = sorted(
            slots_by_unit[unit_id],
            key=lambda slot: (str(slot["recipe_id"]), int(slot["seed"])),
        )
        if len(group) != recovery_plan.SLOTS_PER_UNIT:
            raise RecoveryInputsError("original_unit_slot_count_mismatch")
        unit_dir = f"units/{unit_id}"
        directories.update({unit_dir, f"{unit_dir}/executions", f"{unit_dir}/histories"})
        manifest = f"{unit_dir}/{SEALED_UNIT_MANIFEST_NAME}"
        files.add(manifest)
        sealed_manifests.add(manifest)
        unit_files: set[str] = set()
        for slot in group:
            identifier = pilot.execution_id(unit, slot)
            directories.add(f"{unit_dir}/executions/{identifier}")
            for relative in _slot_files(unit, slot):
                path = f"{unit_dir}/{relative}"
                files.add(path)
                unit_files.add(path)
        sealed_unit_files[unit_id] = unit_files

    unit = unit_by_id[incomplete_unit_id]
    group = sorted(
        slots_by_unit[incomplete_unit_id],
        key=lambda slot: (str(slot["recipe_id"]), int(slot["seed"])),
    )
    if len(group) != recovery_plan.SLOTS_PER_UNIT:
        raise RecoveryInputsError("original_unit_slot_count_mismatch")
    unit_dir = f"units/{incomplete_unit_id}"
    directories.update({unit_dir, f"{unit_dir}/executions", f"{unit_dir}/histories"})
    completed_in_unit = 0
    for slot in group:
        slot_id = str(slot["slot_id"])
        if slot_id in completed:
            completed_in_unit += 1
            identifier = pilot.execution_id(unit, slot)
            directories.add(f"{unit_dir}/executions/{identifier}")
            for relative in _slot_files(unit, slot):
                path = f"{unit_dir}/{relative}"
                files.add(path)
                unsealed_files.add(path)
        elif slot_id == interrupted_slot_id:
            identifier = pilot.execution_id(unit, slot)
            path = f"{unit_dir}/histories/{identifier}.jsonl"
            files.add(path)
            unsealed_files.add(path)
    if completed_in_unit != recovery_plan.PARTIAL_COMPLETED:
        raise RecoveryInputsError("original_partial_completion_mismatch")

    return files, directories, sealed_manifests, sealed_unit_files, unsealed_files


def authenticate_original_stage(
    bundle: Mapping[str, Any], permit: Mapping[str, Any], deadline: Any
) -> dict[str, Any]:
    """Authenticate the original develop stage and slot leases in place."""

    start = time.perf_counter()
    _check_deadline(deadline)
    try:
        permit = authority.validate_recovery_permit(permit)
    except authority.RecoveryAuthorityError as error:
        raise RecoveryInputsError("recovery_permit_rejected") from error
    if bundle.get("permit_sha256") != authority.BASECOMPREHENSIVE_PERMIT_SHA256:
        raise RecoveryInputsError("base_permit_hash_mismatch")
    artifact_root = Path(bundle["artifact_root"])
    run_root = _original_run_root(artifact_root)
    stage = run_root / DEVELOP_STAGE_NAME
    _reject_symlink_chain(stage)
    _check_deadline(deadline)

    _authenticate_pilot(bundle, deadline)
    _check_deadline(deadline)

    top_raw: dict[str, bytes] = {}
    for name in ORIGINAL_TOP_FILES:
        top_raw[name] = _read_bytes_bounded(
            stage / name, _MAX_JSON_BYTES, f"original_{name}", deadline
        )
        _check_deadline(deadline)

    ledger_raw = top_raw["ledger.json"]
    _parse_json(ledger_raw, "original_ledger")
    if ledger_raw != core._canon().canonical_json_bytes(bundle["ledger"]):
        raise RecoveryInputsError("original_ledger_mismatch")

    source_raw = top_raw["source_ledger.json"]
    _parse_json(source_raw, "original_source_ledger")
    if source_raw != core._canon().canonical_json_bytes(_expected_source_ledger(bundle)):
        raise RecoveryInputsError("original_source_ledger_mismatch")

    events_raw = top_raw["events.jsonl"]
    events = _parse_jsonl(events_raw, "original_events")
    if hashlib.sha256(events_raw).hexdigest() != str(permit["original_events_sha256"]):
        raise RecoveryInputsError("original_events_digest_mismatch")

    selector_raw = top_raw["selector.jsonl"]
    selector_records = _parse_jsonl(selector_raw, "original_selector")
    if hashlib.sha256(selector_raw).hexdigest() != str(permit["original_selector_sha256"]):
        raise RecoveryInputsError("original_selector_digest_mismatch")

    for name in ("progress.json", "input_manifest.json", "provenance_before.json"):
        _parse_json(top_raw[name], f"original_{name}")

    unit_by_id = {str(unit["unit_id"]): unit for unit in bundle["ledger"]["units"]}
    slot_by_id = {str(slot["slot_id"]): slot for slot in bundle["ledger"]["slots"]}

    if not events:
        raise RecoveryInputsError("original_events_empty")
    last = events[-1]
    if last.get("event") != "started":
        raise RecoveryInputsError("original_interruption_missing")
    interrupted_unit_id = str(last.get("unit_id"))
    interrupted_slot_id = str(last.get("slot_id"))
    interrupted_unit = unit_by_id.get(interrupted_unit_id)
    interrupted_slot = slot_by_id.get(interrupted_slot_id)
    if (
        interrupted_unit is None
        or interrupted_slot is None
        or str(interrupted_slot["unit_id"]) != interrupted_unit_id
    ):
        raise RecoveryInputsError("original_interruption_identity_mismatch")
    _check_component(interrupted_unit_id)
    _check_component(interrupted_slot_id)
    interrupted_execution_id = _check_component(
        pilot.execution_id(interrupted_unit, interrupted_slot)
    )
    if str(last.get("execution_id")) != interrupted_execution_id:
        raise RecoveryInputsError("original_interruption_execution_mismatch")
    interrupted_history_path = (
        stage / "units" / interrupted_unit_id / "histories" / f"{interrupted_execution_id}.jsonl"
    )
    interrupted_raw = _read_bytes_bounded(
        interrupted_history_path, _MAX_JSON_BYTES, "original_interrupted_history", deadline
    )
    interrupted_history = _parse_jsonl(interrupted_raw, "original_interrupted_history")

    lease_root = base_inputs._slot_lease_root(artifact_root)
    lease_records, lease_inventory = _read_original_leases(lease_root, deadline)

    try:
        plan = recovery_plan.build_recovery_plan(
            ledger=bundle["ledger"],
            pilot_slots=bundle["pilot_bundle"]["slots"],
            events=events,
            selector_records=selector_records,
            leases=lease_records,
            interrupted_history=interrupted_history,
        )
    except recovery_plan.RecoveryPlanError as error:
        raise RecoveryInputsError(f"recovery_plan_{error.reason_code}") from error
    _check_deadline(deadline)

    _check_plan_against_permit(plan, permit)
    if plan.get("execution_authorized") is not False:
        raise RecoveryInputsError("plan_authorization_invalid")
    if _strict_int(plan.get("fits_started"), "plan_fits_invalid") != 0:
        raise RecoveryInputsError("plan_authorization_invalid")

    (
        expected_files,
        expected_dirs,
        sealed_manifests,
        sealed_unit_files,
        unsealed_files,
    ) = _expected_layout(bundle, plan)

    observed_files, observed_dirs = _collect_tree(stage, deadline)
    if set(observed_files) != expected_files:
        raise RecoveryInputsError("original_stage_inventory_mismatch")
    if observed_dirs != expected_dirs:
        raise RecoveryInputsError("original_stage_directory_mismatch")

    records: dict[str, dict[str, Any]] = {}
    for relative in sorted(observed_files):
        _check_deadline(deadline)
        records[relative] = _hash_file_record(observed_files[relative], deadline)

    parsed_bytes = {
        **top_raw,
        interrupted_history_path.relative_to(stage).as_posix(): interrupted_raw,
    }
    for relative, raw in parsed_bytes.items():
        if records[relative] != {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}:
            raise RecoveryInputsError("original_evidence_changed")

    for manifest_rel in sorted(sealed_manifests):
        _check_deadline(deadline)
        unit_id = manifest_rel.split("/")[1]
        manifest_raw = _read_bytes_bounded(
            observed_files[manifest_rel], _MAX_JSON_BYTES, "original_unit_manifest", deadline
        )
        if records[manifest_rel] != {
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "size_bytes": len(manifest_raw),
        }:
            raise RecoveryInputsError("original_evidence_changed")
        manifest = _parse_json(manifest_raw, "original_unit_manifest")
        manifest_files = manifest.get("files") if isinstance(manifest, Mapping) else None
        if not isinstance(manifest_files, Mapping):
            raise RecoveryInputsError("original_unit_manifest_malformed")
        expected_relative = {path.split("/", 2)[2] for path in sealed_unit_files[unit_id]}
        if set(manifest_files) != expected_relative:
            raise RecoveryInputsError("original_unit_manifest_inventory_mismatch")
        for relative in expected_relative:
            if not _record_equal(manifest_files[relative], records[f"units/{unit_id}/{relative}"]):
                raise RecoveryInputsError("original_unit_manifest_digest_mismatch")

    anchor_entries: dict[str, Any] = {}
    for name in ORIGINAL_TOP_FILES:
        anchor_entries[name] = records[name]
    for manifest_rel in sorted(sealed_manifests):
        anchor_entries[manifest_rel] = records[manifest_rel]
    for relative in sorted(unsealed_files):
        anchor_entries[relative] = records[relative]

    expected_anchor_count = len(ORIGINAL_TOP_FILES) + len(sealed_manifests) + len(unsealed_files)
    if len(anchor_entries) != expected_anchor_count:
        raise RecoveryInputsError("original_anchor_count_mismatch")
    if expected_anchor_count != _strict_int(
        permit.get("original_evidence_anchor_files"), "permit_count_invalid"
    ):
        raise RecoveryInputsError("original_anchor_count_mismatch")

    anchor = {
        "schema_version": INTERRUPTION_ANCHOR_SCHEMA_VERSION,
        "files": anchor_entries,
    }
    anchor_sha256 = core._canon().sha256_value(anchor)
    if anchor_sha256 != authority.ORIGINAL_EVIDENCE_ANCHOR_SHA256:
        raise RecoveryInputsError("original_anchor_digest_mismatch")
    if anchor_sha256 != str(permit.get("original_evidence_anchor_sha256")):
        raise RecoveryInputsError("original_anchor_permit_mismatch")

    _check_deadline(deadline)
    for relative in sorted(records):
        _check_deadline(deadline)
        if _hash_file_record(observed_files[relative], deadline) != records[relative]:
            raise RecoveryInputsError("original_evidence_changed")
    for relative, record in lease_inventory.items():
        _check_deadline(deadline)
        if _hash_file_record(lease_root / relative, deadline) != record:
            raise RecoveryInputsError("original_lease_changed")
    final_files, final_dirs = _collect_tree(stage, deadline)
    if set(final_files) != set(records) or final_dirs != expected_dirs:
        raise RecoveryInputsError("original_evidence_inventory_changed")
    lease_files, lease_dirs = _collect_tree(lease_root, deadline)
    if set(lease_files) != set(lease_inventory) or lease_dirs != {
        relative.split("/")[0] for relative in lease_inventory
    }:
        raise RecoveryInputsError("original_lease_inventory_changed")
    _authenticate_pilot(bundle, deadline)

    inventory_total_bytes = sum(int(record["size_bytes"]) for record in records.values())
    _check_deadline(deadline)
    return {
        "original_run_root": run_root,
        "original_stage": stage,
        "plan": plan,
        "original_inventory": records,
        "original_anchor": anchor,
        "original_anchor_sha256": anchor_sha256,
        "original_lease_inventory": lease_inventory,
        "inventory_total_bytes": inventory_total_bytes,
        "elapsed_seconds": time.perf_counter() - start,
    }


def prepare_recovery(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    base_permit_path: Path | str,
    recovery_permit_path: Path | str,
    deadline: float,
) -> dict[str, Any]:
    """Authenticate the recovery permit and the original interruption prefix."""

    start = time.perf_counter()
    _check_deadline(deadline)
    for path in (
        project_root,
        artifact_root,
        contract_path,
        base_permit_path,
        recovery_permit_path,
    ):
        _reject_symlink_chain(path)
    try:
        recovery_permit = authority.load_recovery_permit(recovery_permit_path)
    except authority.RecoveryAuthorityError as error:
        raise RecoveryInputsError("recovery_permit_rejected") from error
    _check_deadline(deadline)
    try:
        bundle = base_inputs.prepare(
            project_root=Path(project_root),
            artifact_root=Path(artifact_root),
            contract_path=Path(contract_path),
            permit_path=Path(base_permit_path),
            require_unstarted=False,
        )
    except Exception as error:
        raise RecoveryInputsError("base_inputs_rejected") from error
    _check_deadline(deadline)

    if bundle["permit_sha256"] != authority.BASECOMPREHENSIVE_PERMIT_SHA256:
        raise RecoveryInputsError("base_permit_hash_mismatch")

    original_run_root = _original_run_root(bundle["artifact_root"])
    _assert_root_exclusive(original_run_root)
    _check_deadline(deadline)

    try:
        authentication = authenticate_original_stage(bundle, recovery_permit, deadline)
    except RecoveryInputsError:
        raise
    except Exception as exc:
        raise RecoveryInputsError("original_stage_authentication_failed") from exc

    _check_deadline(deadline)
    if Path(authentication["original_run_root"]) != original_run_root:
        raise RecoveryInputsError("original_run_root_mismatch")
    _assert_root_exclusive(original_run_root)
    _check_deadline(deadline)

    return {
        **authentication,
        "base_bundle": bundle,
        "recovery_permit": recovery_permit,
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "schema_version": RECOVERY_INPUTS_SCHEMA_VERSION,
        "fits_started": 0,
        "files_written": 0,
        "scientific_model_reauthentication_complete": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
