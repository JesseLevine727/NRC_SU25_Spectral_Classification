"""Read-only per-unit scientific authentication for the P05 recovery prefix.

This boundary verifies exactly one original ``develop`` unit subtree in place.
It re-hashes the unit evidence against the authenticated original inventory,
restores the accepted completed fits through the existing development-result
boundary, reproduces the saved best-validation logits and re-runs the pilot
per-unit acceptance checks. Models are instantiated only to restore and check
saved weights; no new weights are trained. It never
creates a lease, writes, copies, selects or exports and never reads outer-test
data.  It imports only the standard library and the existing stdlib-only P05
modules; torch is injected by the caller and numpy is imported lazily by the
reused boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_inputs as inputs
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan

__all__ = ["RecoveryEvidenceError", "load_verified_original_unit"]

_DIGEST_FIELDS = ("sampling_digest", "augmentation_digest", "pair_digest")
_EQUIVALENT_RECIPE_PAIRS = (("D0-M", "D2"), ("D1", "D3"))


class RecoveryEvidenceError(ValueError):
    """Stable, path-free recovery-evidence failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


def _call(code: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except RecoveryEvidenceError:
        raise
    except Exception as error:
        detail = getattr(error, "reason_code", None)
        if isinstance(detail, str) and detail:
            raise RecoveryEvidenceError(code, detail) from error
        raise RecoveryEvidenceError(code) from error


def _deadline(deadline: Any) -> None:
    try:
        inputs._check_deadline(deadline)
    except inputs.RecoveryInputsError as error:
        raise RecoveryEvidenceError(error.reason_code) from error


def _strict_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryEvidenceError(code)
    return value


def _guard_resources(torch: Any) -> None:
    try:
        authority.check_resources(torch, phase="fit")
    except RecoveryEvidenceError:
        raise
    except authority.RecoveryAuthorityError as error:
        raise RecoveryEvidenceError(f"resources_{error.reason_code}") from error
    except Exception as error:
        raise RecoveryEvidenceError("resources_check_failed") from error


def _base_bundle(recovery_bundle: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(recovery_bundle, Mapping):
        raise RecoveryEvidenceError("recovery_bundle_malformed")
    if recovery_bundle.get("recovery_permit_sha256") != authority.RECOVERY_PERMIT_SHA256:
        raise RecoveryEvidenceError("recovery_permit_identity_mismatch")
    try:
        authority.validate_recovery_permit(recovery_bundle.get("recovery_permit"))
    except authority.RecoveryAuthorityError as error:
        raise RecoveryEvidenceError("recovery_permit_rejected") from error
    base = recovery_bundle.get("base_bundle")
    if not isinstance(base, Mapping):
        raise RecoveryEvidenceError("base_bundle_malformed")
    if base.get("permit_sha256") != authority.BASECOMPREHENSIVE_PERMIT_SHA256:
        raise RecoveryEvidenceError("base_permit_identity_mismatch")
    if recovery_bundle.get("original_anchor_sha256") != authority.ORIGINAL_EVIDENCE_ANCHOR_SHA256:
        raise RecoveryEvidenceError("original_anchor_identity_mismatch")
    anchor = recovery_bundle.get("original_anchor")
    if not isinstance(anchor, Mapping):
        raise RecoveryEvidenceError("original_anchor_malformed")
    try:
        observed_anchor = core._canon().sha256_value(dict(anchor))
    except Exception as error:
        raise RecoveryEvidenceError("original_anchor_malformed") from error
    if observed_anchor != authority.ORIGINAL_EVIDENCE_ANCHOR_SHA256:
        raise RecoveryEvidenceError("original_anchor_digest_mismatch")
    return base


def _stage_root(recovery_bundle: Mapping[str, Any], base: Mapping[str, Any]) -> Path:
    artifact_root = base.get("artifact_root")
    if not isinstance(artifact_root, (str, Path)):
        raise RecoveryEvidenceError("base_bundle_malformed")
    expected = (
        _call("stage_unreadable", inputs._original_run_root, artifact_root)
        / inputs.DEVELOP_STAGE_NAME
    )
    stage = recovery_bundle.get("original_stage")
    if not isinstance(stage, (str, Path)):
        raise RecoveryEvidenceError("original_stage_missing")
    try:
        observed = Path(stage)
    except (TypeError, ValueError) as error:
        raise RecoveryEvidenceError("original_stage_missing") from error
    if observed != expected:
        raise RecoveryEvidenceError("original_stage_mismatch")
    _call("stage_unreadable", inputs._reject_symlink_chain, observed)
    return expected


def _resolve_unit(
    unit: Mapping[str, Any], ledger: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[str, bool]:
    if not isinstance(unit, Mapping):
        raise RecoveryEvidenceError("unit_malformed")
    unit_id = str(unit.get("unit_id"))
    _call("unit_identity_invalid", inputs._check_component, unit_id)
    ledger_units = ledger.get("units")
    if not isinstance(ledger_units, Sequence) or isinstance(ledger_units, (str, bytes)):
        raise RecoveryEvidenceError("ledger_units_malformed")
    unit_by_id: dict[str, dict[str, Any]] = {}
    for entry in ledger_units:
        if not isinstance(entry, Mapping):
            raise RecoveryEvidenceError("ledger_units_malformed")
        key = str(entry.get("unit_id"))
        if key in unit_by_id:
            raise RecoveryEvidenceError("ledger_unit_duplicate")
        unit_by_id[key] = dict(entry)
    ledger_unit = unit_by_id.get(unit_id)
    if ledger_unit is None:
        raise RecoveryEvidenceError("unit_unknown")
    if _call("unit_malformed", core._canon().canonical_json_bytes, dict(unit)) != _call(
        "unit_malformed", core._canon().canonical_json_bytes, ledger_unit
    ):
        raise RecoveryEvidenceError("unit_identity_mismatch")
    sealed = [str(item) for item in plan.get("sealed_unit_ids", ())]
    incomplete = str(plan.get("incomplete_unit_id"))
    if unit_id in sealed:
        return unit_id, True
    if unit_id == incomplete:
        return unit_id, False
    raise RecoveryEvidenceError("unit_not_authenticated")


def _unit_slots(ledger: Mapping[str, Any], unit_id: str) -> list[dict[str, Any]]:
    slots = ledger.get("slots")
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise RecoveryEvidenceError("ledger_slots_malformed")
    if any(not isinstance(slot, Mapping) for slot in slots):
        raise RecoveryEvidenceError("ledger_slots_malformed")
    group = [dict(slot) for slot in slots if str(slot.get("unit_id")) == unit_id]
    try:
        for slot in group:
            _strict_int(slot.get("seed"), "unit_slot_seed_invalid")
            _call("unit_slot_identity_invalid", inputs._check_component, slot.get("slot_id"))
        group.sort(key=lambda slot: (str(slot["recipe_id"]), int(slot["seed"])))
        product = {(str(slot["recipe_id"]), int(slot["seed"])) for slot in group}
    except (KeyError, TypeError, ValueError) as error:
        raise RecoveryEvidenceError("unit_slot_malformed") from error
    expected_product = {
        (recipe, seed) for recipe in recovery_plan.RECIPES for seed in recovery_plan.SEEDS
    }
    if len(group) != recovery_plan.SLOTS_PER_UNIT or product != expected_product:
        raise RecoveryEvidenceError("unit_slot_count_mismatch")
    return group


def _completed_slots(
    group: Sequence[Mapping[str, Any]], plan: Mapping[str, Any], full: bool
) -> list[dict[str, Any]]:
    reused = {str(item) for item in plan.get("reused_original_slot_ids", ())}
    interrupted = str(plan.get("interrupted_slot_id"))
    if full:
        completed = [dict(slot) for slot in group]
    else:
        completed = [dict(slot) for slot in group if str(slot["slot_id"]) in reused]
        if len(completed) != recovery_plan.PARTIAL_COMPLETED:
            raise RecoveryEvidenceError("unit_partial_completion_mismatch")
    completed_ids = {str(slot["slot_id"]) for slot in completed}
    if not completed_ids <= reused:
        raise RecoveryEvidenceError("unit_slot_not_completed")
    if interrupted in completed_ids:
        raise RecoveryEvidenceError("interrupted_slot_not_completed")
    if len(completed_ids) != len(completed):
        raise RecoveryEvidenceError("unit_slot_duplicate")
    return completed


def _inventory_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {"sha256", "size_bytes"}:
        raise RecoveryEvidenceError("original_inventory_malformed")
    digest = record.get("sha256")
    size = record.get("size_bytes")
    if not isinstance(digest, str) or not core._is_hex64(digest):
        raise RecoveryEvidenceError("original_inventory_malformed")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RecoveryEvidenceError("original_inventory_malformed")
    return {"sha256": digest, "size_bytes": size}


def _expected_unit_files(inventory: Any, unit_id: str) -> dict[str, dict[str, Any]]:
    if not isinstance(inventory, Mapping):
        raise RecoveryEvidenceError("original_inventory_malformed")
    prefix = f"units/{unit_id}/"
    expected: dict[str, dict[str, Any]] = {}
    for relative, record in inventory.items():
        if not isinstance(relative, str):
            raise RecoveryEvidenceError("original_inventory_malformed")
        if relative.startswith(prefix):
            expected[relative[len(prefix) :]] = _inventory_record(record)
    if not expected:
        raise RecoveryEvidenceError("unit_inventory_missing")
    return expected


def _expected_dirs(expected_files: Mapping[str, Any]) -> set[str]:
    directories: set[str] = set()
    for relative in expected_files:
        parts = relative.split("/")
        for part in parts:
            _call("unit_inventory_malformed", inputs._check_component, part)
        for depth in range(1, len(parts)):
            directories.add("/".join(parts[:depth]))
    return directories


def _check_inventory(unit_dir: Path, expected_files: Mapping[str, Any], deadline: Any) -> None:
    observed_files, observed_dirs = _call(
        "unit_inventory_unreadable", inputs._collect_tree, unit_dir, deadline
    )
    if set(observed_files) != set(expected_files):
        raise RecoveryEvidenceError("unit_inventory_mismatch")
    if observed_dirs != _expected_dirs(expected_files):
        raise RecoveryEvidenceError("unit_directory_mismatch")


def _hash_unit(
    unit_dir: Path, expected_files: Mapping[str, Any], deadline: Any
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in sorted(expected_files):
        observed = _call(
            "unit_file_unreadable", inputs._hash_file_record, unit_dir / relative, deadline
        )
        if not inputs._record_equal(observed, expected_files[relative]):
            raise RecoveryEvidenceError("unit_file_digest_mismatch")
        records[relative] = observed
    return records


def _rehash_unit(
    unit_dir: Path,
    expected_files: Mapping[str, Any],
    previous: Mapping[str, Any],
    deadline: Any,
) -> None:
    for relative in sorted(expected_files):
        observed = _call(
            "unit_file_unreadable", inputs._hash_file_record, unit_dir / relative, deadline
        )
        if observed != previous.get(relative):
            raise RecoveryEvidenceError("unit_evidence_changed")
    final_files, final_dirs = _call(
        "unit_inventory_unreadable", inputs._collect_tree, unit_dir, deadline
    )
    if set(final_files) != set(expected_files) or final_dirs != _expected_dirs(expected_files):
        raise RecoveryEvidenceError("unit_inventory_changed")


def _bind_unit_inventory(
    unit_dir: Path,
    unit_id: str,
    full: bool,
    anchor: Mapping[str, Any],
    expected_files: Mapping[str, Any],
    deadline: Any,
) -> None:
    """Tie a supplied inventory back to the immutable interruption anchor."""
    anchored = anchor.get("files")
    if not isinstance(anchored, Mapping):
        raise RecoveryEvidenceError("original_anchor_malformed")
    prefix = f"units/{unit_id}/"
    if full:
        manifest_record = anchored.get(prefix + "manifest.json")
        if not isinstance(manifest_record, Mapping) or not inputs._record_equal(
            expected_files.get("manifest.json"), manifest_record
        ):
            raise RecoveryEvidenceError("unit_inventory_anchor_mismatch")
        manifest = _call(
            "unit_manifest_unreadable",
            inputs._read_json_mapping,
            unit_dir / "manifest.json",
            "unit_manifest",
            deadline,
        )
        original_files = manifest.get("files")
        supplied = {key: record for key, record in expected_files.items() if key != "manifest.json"}
    else:
        original_files = {
            key[len(prefix) :]: record
            for key, record in anchored.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        supplied = dict(expected_files)
    if not isinstance(original_files, Mapping) or _call(
        "unit_inventory_anchor_mismatch", core._canon().canonical_json_bytes, dict(original_files)
    ) != _call("unit_inventory_anchor_mismatch", core._canon().canonical_json_bytes, supplied):
        raise RecoveryEvidenceError("unit_inventory_anchor_mismatch")


def _check_selector(
    expected_selectors: Mapping[str, Mapping[str, Any]],
    unit: Mapping[str, Any],
    slot: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    slot_id = str(slot["slot_id"])
    expected = expected_selectors.get(slot_id)
    if not isinstance(expected, Mapping):
        raise RecoveryEvidenceError("selector_record_missing")
    expected = dict(expected)
    for field, value in (
        ("slot_id", slot_id),
        ("recipe_id", str(slot["recipe_id"])),
        ("seed", int(slot["seed"])),
        ("unit_id", str(unit["unit_id"])),
    ):
        if field in expected and expected[field] != value:
            raise RecoveryEvidenceError("selector_identity_mismatch")
    if expected.get("status") != "complete":
        raise RecoveryEvidenceError("selector_status_mismatch")
    record = _call("selector_record_failed", base_inputs.selector_record, unit, slot, summary)
    if record.get("status") != "complete":
        raise RecoveryEvidenceError("selector_status_mismatch")
    if _call(
        "selector_record_malformed", core._canon().canonical_json_bytes, dict(record)
    ) != _call("selector_record_malformed", core._canon().canonical_json_bytes, expected):
        raise RecoveryEvidenceError("selector_record_mismatch")
    return record


def _validate_digests(items: Sequence[Mapping[str, Any]]) -> None:
    for item in items:
        for record in item["result"].history:
            if not isinstance(record, Mapping):
                raise RecoveryEvidenceError("history_record_malformed")
            for field in _DIGEST_FIELDS:
                value = record.get(field)
                if not isinstance(value, str) or not core._is_hex64(value):
                    raise RecoveryEvidenceError("digest_invalid")


def _partial_cross_checks(items: Sequence[Mapping[str, Any]], unit: Mapping[str, Any]) -> None:
    _validate_digests(items)
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for item in items:
        grouped.setdefault(int(item["seed"]), []).append(item)
    for members in grouped.values():
        prefix = min(len(member["result"].history) for member in members)
        for field in _DIGEST_FIELDS:
            for index in range(prefix):
                values = {member["result"].history[index][field] for member in members}
                if len(values) != 1:
                    raise RecoveryEvidenceError("partial_shared_prefix_mismatch")
        initials = {member["result"].initial_backbone_digest for member in members}
        if len(initials) != 1 or any(
            not isinstance(value, str) or not core._is_hex64(value) for value in initials
        ):
            raise RecoveryEvidenceError("partial_initial_backbone_mismatch")
    support = unit.get("auxiliary_support") or {}
    if int(support.get("cross_instrument_master_pairs", 0)) == 0:
        for members in grouped.values():
            by_recipe = {str(item["slot"]["recipe_id"]): item for item in members}
            for left_id, right_id in _EQUIVALENT_RECIPE_PAIRS:
                left = by_recipe.get(left_id)
                right = by_recipe.get(right_id)
                if left is not None and right is not None:
                    _call(
                        "partial_equivalence_failed",
                        pilot._check_equivalent_results,
                        left["result"],
                        right["result"],
                    )


def load_verified_original_unit(
    *,
    recovery_bundle: Mapping[str, Any],
    unit: Mapping[str, Any],
    expected_selectors: Mapping[str, Mapping[str, Any]],
    torch: Any,
    device: str,
    deadline: float,
) -> dict[str, Any]:
    """Authenticate one original unit subtree and its completed fits in place."""

    _deadline(deadline)
    if device != "cuda":
        raise RecoveryEvidenceError("device_invalid")
    if not isinstance(expected_selectors, Mapping):
        raise RecoveryEvidenceError("expected_selectors_malformed")
    base = _base_bundle(recovery_bundle)
    ledger = base.get("ledger")
    contract = base.get("contract")
    if not isinstance(ledger, Mapping) or not isinstance(contract, Mapping):
        raise RecoveryEvidenceError("base_bundle_malformed")
    plan = recovery_bundle.get("plan")
    if not isinstance(plan, Mapping):
        raise RecoveryEvidenceError("recovery_plan_malformed")
    stage = _stage_root(recovery_bundle, base)
    unit_id, full = _resolve_unit(unit, ledger, plan)
    group = _unit_slots(ledger, unit_id)
    completed = _completed_slots(group, plan, full)
    unit_dir = stage / "units" / unit_id
    _call("unit_path_unreadable", inputs._reject_symlink_chain, unit_dir)
    expected_files = _expected_unit_files(recovery_bundle.get("original_inventory"), unit_id)
    _check_inventory(unit_dir, expected_files, deadline)
    pre_hashes = _hash_unit(unit_dir, expected_files, deadline)
    _bind_unit_inventory(
        unit_dir, unit_id, full, recovery_bundle["original_anchor"], expected_files, deadline
    )
    _deadline(deadline)
    _guard_resources(torch)
    unit_inputs = _call(
        "unit_inputs_failed",
        lambda: pilot.prepare_role_inputs({**base, "units": [unit]})[unit_id],
    )
    _deadline(deadline)
    items: list[dict[str, Any]] = []
    selector_records: list[dict[str, Any]] = []
    optimizer_updates = 0
    for slot in completed:
        _deadline(deadline)
        _guard_resources(torch)
        summary = _call("summary_unreadable", base_inputs._read_pilot_summary, unit_dir, unit, slot)
        result = _call(
            "result_unreadable", base_inputs.load_development_result, unit_dir, unit, slot
        )
        _deadline(deadline)
        _guard_resources(torch)
        _call("history_mismatch", base_inputs._check_history_matches, unit_dir, unit, slot, result)
        _call(
            "result_acceptance_failed",
            pilot.check_completed_result,
            result,
            unit_dir,
            unit,
            slot,
            contract,
            unit_inputs,
            torch,
            device,
        )
        _call("sparse_support_failed", pilot.check_sparse_support, unit, slot, contract, result)
        _deadline(deadline)
        _guard_resources(torch)
        record = _check_selector(expected_selectors, unit, slot, summary)
        optimizer_updates += _strict_int(result.optimizer_steps, "optimizer_steps_invalid")
        selector_records.append(record)
        items.append(
            {
                "slot": slot,
                "unit": unit,
                "unit_id": unit_id,
                "seed": int(slot["seed"]),
                "result": result,
            }
        )
    if full:
        _call("full_unit_cross_check_failed", pilot.check_shared_prefixes, items)
        _call(
            "full_unit_cross_check_failed",
            pilot.check_sparse_equivalences,
            items,
            [unit],
            contract,
        )
    else:
        _partial_cross_checks(items, unit)
    _deadline(deadline)
    _rehash_unit(unit_dir, expected_files, pre_hashes, deadline)
    _deadline(deadline)
    return {
        "items": items,
        "selector_records": selector_records,
        "unit_id": unit_id,
        "completed_slots": [str(slot["slot_id"]) for slot in completed],
        "optimizer_updates_exact": optimizer_updates,
        "complete_unit_cross_recipe_checks": bool(full),
        "deferred_until_replay": bool(not full),
        "fits_started": 0,
        "files_written": 0,
    }
