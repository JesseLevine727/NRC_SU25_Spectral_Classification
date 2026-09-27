"""P05 refit evidence: authorized logit loading, calibration persistence and
per-group digest/sparse checks.

Bounded helpers for the comprehensive refit stage. They load only authorized
private logits, persist one calibration plus audit without overwrites and
compare slim per-refit evidence. No held predictions, training or execution
authority live here.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_core_run as core

KNOWN_RECIPES = ("D0-M", "D1", "D2", "D3")
REQUIRED_RECIPES = ("D0-M", "D3")
GROUP_SIZES = (2, 3)
SPARSE_RECIPES = ("D2", "D3")
EQUIVALENT_PAIRS = (("D0-M", "D2"), ("D1", "D3"))
DIGEST_FIELDS = ("sampling_digest", "augmentation_digest", "pair_digest")
HEX_LENGTH = 64
HEX_ALPHABET = frozenset("0123456789abcdef")


class P05RefitEvidenceError(core.P05CoreError):
    """Stable, path-free evidence failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05RefitEvidenceError(code)


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return value


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HEX_LENGTH
        and all(char in HEX_ALPHABET for char in value)
    )


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _field(bundle: Mapping[str, Any], name: str, code: str) -> Any:
    value = bundle.get(name)
    _require(value is not None, code)
    return value


def make_logit_loader(bundle: Any) -> Any:
    """Return a callback that loads authorized private validation logits."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    ledger = _field(bundle, "ledger", "ledger_missing")
    _require(isinstance(ledger, Mapping), "ledger_malformed")
    units: dict[str, Mapping[str, Any]] = {}
    for raw in ledger.get("units", ()):
        _require(isinstance(raw, Mapping), "ledger_unit_malformed")
        unit_id = raw.get("unit_id")
        _require(isinstance(unit_id, str) and bool(unit_id), "ledger_unit_id_malformed")
        _require(unit_id not in units, "ledger_unit_duplicate")
        units[unit_id] = raw
    _require(bool(units), "ledger_units_empty")
    slots = {}
    for registered in ledger.get("slots", ()):
        _require(isinstance(registered, Mapping), "ledger_slot_malformed")
        identifier = registered.get("slot_id")
        _require(isinstance(identifier, str) and bool(identifier), "ledger_slot_id_malformed")
        _require(identifier not in slots, "ledger_slot_duplicate")
        slots[identifier] = dict(registered)
    _require(bool(slots), "ledger_slots_empty")
    artifact_root = Path(_field(bundle, "artifact_root", "artifact_root_missing"))
    permit_sha256 = _field(bundle, "permit_sha256", "permit_sha256_missing")
    _require(_hex64(permit_sha256), "permit_sha256_malformed")
    inputs = _module("atlas_sers.evaluation.p05_comprehensive_inputs")
    pilot = _module("atlas_sers.evaluation.p05_pilot")
    numpy = _module("numpy")
    pilot_slot_ids = set(inputs.pilot_slot_ids(bundle))
    pilot_root = Path(inputs._pilot_run_dir(artifact_root))
    develop_root = artifact_root / "p05comprehensive" / "runs" / permit_sha256 / "develop" / "units"

    def load_logits(slot: Any) -> Any:
        _require(isinstance(slot, Mapping), "slot_malformed")
        slot_id = slot.get("slot_id")
        _require(isinstance(slot_id, str) and bool(slot_id), "slot_id_malformed")
        _require(slot_id in slots and dict(slot) == slots[slot_id], "slot_ledger_mismatch")
        unit_id = slot.get("unit_id")
        _require(isinstance(unit_id, str) and bool(unit_id), "slot_unit_id_malformed")
        unit = units.get(unit_id)
        _require(unit is not None, "slot_unit_unknown")
        _require(str(unit.get("unit_id")) == unit_id, "slot_unit_identity_mismatch")
        for name in ("context_id", "selection_unit_id", "fitting_role_id", "validation_role_id"):
            _require(str(slot.get(name)) == str(unit.get(name)), f"slot_{name}_mismatch")
        base = pilot_root if slot_id in pilot_slot_ids else develop_root / unit_id
        execution = pilot.execution_id(unit, slot)
        _require(isinstance(execution, str) and bool(execution), "execution_id_malformed")
        path = base / "executions" / execution / "validation_logits.npz"
        core._reject_symlink_chain(path)
        return inputs._load_logits(numpy, path)

    return load_logits


def persist_calibration(unit_dir: Any, calibration: Any, audit: Any, spec: Any) -> dict[str, Any]:
    """Persist one calibration and audit, then verify the on-disk roundtrip."""

    unit_dir = Path(unit_dir)
    core._reject_symlink_chain(unit_dir)
    _require(unit_dir.is_dir(), "calibration_unit_dir_missing")
    _require(dataclasses.is_dataclass(calibration), "calibration_not_dataclass")
    _require(
        isinstance(audit, Mapping) and isinstance(spec, Mapping), "calibration_input_malformed"
    )
    _require(getattr(calibration, "optimizer_success", None) is True, "calibration_failed")
    temperature = getattr(calibration, "temperature", None)
    _require(
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and math.isfinite(float(temperature))
        and float(temperature) > 0.0,
        "calibration_temperature_invalid",
    )
    state = dataclasses.asdict(calibration)
    _require("state_sha256" not in state, "calibration_state_sha256_not_property")
    state_sha256 = getattr(calibration, "state_sha256", None)
    _require(_hex64(state_sha256), "calibration_state_sha256_missing")
    classes = tuple(str(value) for value in getattr(calibration, "class_vocabulary", ()))
    _require(bool(classes), "calibration_classes_missing")
    _require(
        classes == tuple(str(value) for value in spec.get("classes", ())),
        "calibration_classes_mismatch",
    )
    _require(
        str(audit.get("calibration_state_sha256")) == state_sha256,
        "calibration_audit_sha_mismatch",
    )
    for name in ("refit_id", "context_id", "recipe_id"):
        _require(str(audit.get(name)) == str(spec.get(name)), f"calibration_audit_{name}_mismatch")
    _require(audit.get("seed") == spec.get("seed"), "calibration_audit_seed_mismatch")
    _require(
        list(audit.get("calibration_slot_ids", ())) == list(spec.get("calibration_slot_ids", ())),
        "calibration_audit_slots_mismatch",
    )
    _require(audit.get("temperature") == temperature, "calibration_audit_temperature_mismatch")
    _require(audit.get("optimizer_success") is True, "calibration_audit_status_mismatch")
    objective = getattr(calibration, "optimizer_objective", None)
    _require(
        isinstance(objective, (int, float))
        and not isinstance(objective, bool)
        and math.isfinite(float(objective))
        and audit.get("optimizer_objective") == objective,
        "calibration_objective_invalid",
    )
    calibration_path = unit_dir / "calibration.json"
    audit_path = unit_dir / "calibration_audit.json"
    for path in (calibration_path, audit_path):
        core._reject_symlink_chain(path.parent)
        _require(not (path.exists() or path.is_symlink()), "calibration_output_exists")
    canon = core._canon()
    core._atomic_write(
        calibration_path,
        canon.canonical_json_bytes({"state": dict(state), "state_sha256": state_sha256}),
    )
    core._atomic_write(audit_path, canon.canonical_json_bytes(dict(audit)))
    reloaded = core._read_json(calibration_path, "calibration")
    _require(isinstance(reloaded, Mapping), "calibration_reload_malformed")
    _require(reloaded.get("state_sha256") == state_sha256, "calibration_stored_sha_mismatch")
    _require(
        core._read_json(audit_path, "calibration_audit") == dict(audit),
        "calibration_audit_reload_mismatch",
    )
    reloaded_state = reloaded.get("state")
    _require(isinstance(reloaded_state, Mapping), "calibration_state_reload_malformed")
    kwargs = dict(reloaded_state)
    raw_classes = kwargs.get("class_vocabulary")
    _require(
        isinstance(raw_classes, Sequence) and not isinstance(raw_classes, (str, bytes)),
        "calibration_classes_malformed",
    )
    kwargs["class_vocabulary"] = tuple(str(value) for value in raw_classes)
    restored = _module("atlas_sers.evaluation.classical").TemperatureCalibration(**kwargs)
    _require(restored.state_sha256 == state_sha256, "calibration_state_sha256_mismatch")
    _require(
        tuple(str(value) for value in restored.class_vocabulary) == classes,
        "calibration_reload_classes_mismatch",
    )
    return {
        "state_sha256": state_sha256,
        "calibration_sha256": canon.sha256_file(calibration_path),
        "audit_sha256": canon.sha256_file(audit_path),
    }


def summarize_result(spec: Any, result: Any) -> dict[str, Any]:
    """Return slim, tensor-free evidence for one completed refit result."""

    _require(isinstance(spec, Mapping), "spec_malformed")
    _module("atlas_sers.evaluation.p05_refit_io")._check_result_identity(spec, result)
    _require(getattr(result, "status", None) == "complete", "result_not_complete")
    history = getattr(result, "history", None)
    _require(
        isinstance(history, Sequence) and not isinstance(history, (str, bytes)),
        "result_history_malformed",
    )
    epochs = _integer(getattr(result, "epochs", None), "result_epochs_malformed")
    _require(len(history) == epochs, "result_history_length_mismatch")
    slim = []
    for record in history:
        _require(isinstance(record, Mapping), "result_history_record_malformed")
        slim.append({str(key): value for key, value in record.items() if _scalar(value)})
    initial_state = getattr(result, "initial_state_digest", None)
    initial_backbone = getattr(result, "initial_backbone_digest", None)
    _require(_hex64(initial_state), "result_initial_state_missing")
    _require(
        _hex64(initial_backbone),
        "result_initial_backbone_missing",
    )
    paired_support = getattr(result, "paired_support", None)
    _require(isinstance(paired_support, Mapping), "result_paired_support_malformed")
    _require(
        _hex64(getattr(result, "terminal_state_digest", None)), "result_terminal_state_missing"
    )
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in paired_support.values()
        ),
        "result_paired_support_malformed",
    )
    return {
        "refit_id": str(spec.get("refit_id")),
        "context_id": str(spec.get("context_id")),
        "fitting_role_id": str(spec.get("fitting_role_id")),
        "recipe_id": str(spec.get("recipe_id")),
        "seed": _integer(spec.get("seed"), "spec_seed_malformed"),
        "epochs": epochs,
        "initial_state_digest": initial_state,
        "initial_backbone_digest": initial_backbone,
        "terminal_state_digest": getattr(result, "terminal_state_digest", None),
        "paired_support": {str(key): int(value) for key, value in paired_support.items()},
        "history": slim,
    }


def check_recipe_group(summaries: Any, *, cross_instrument_pairs: int) -> dict[str, Any]:
    """Verify shared digest prefixes and sparse-pair evidence for one group."""

    _require(
        isinstance(cross_instrument_pairs, int)
        and not isinstance(cross_instrument_pairs, bool)
        and cross_instrument_pairs >= 0,
        "group_pair_count_malformed",
    )
    _require(
        isinstance(summaries, Sequence) and not isinstance(summaries, (str, bytes)),
        "group_malformed",
    )
    group = list(summaries)
    _require(len(group) in GROUP_SIZES, "group_size")
    recipes: dict[str, Mapping[str, Any]] = {}
    for item in group:
        _require(isinstance(item, Mapping), "group_item_malformed")
        recipe = str(item.get("recipe_id"))
        _require(recipe in KNOWN_RECIPES and recipe not in recipes, "group_recipe_invalid")
        recipes[recipe] = item
    for required in REQUIRED_RECIPES:
        _require(required in recipes, "group_required_recipe_missing")
    _require(len({str(item.get("context_id")) for item in group}) == 1, "group_context_mismatch")
    _require(len({item.get("seed") for item in group}) == 1, "group_seed_mismatch")
    _require(len({str(item.get("fitting_role_id")) for item in group}) == 1, "group_role_mismatch")
    _require(
        len({str(item.get("initial_backbone_digest")) for item in group}) == 1,
        "group_initial_backbone_mismatch",
    )
    epochs = [_integer(item.get("epochs"), "group_epochs_malformed") for item in group]
    _require(all(epoch > 0 for epoch in epochs), "group_epochs_malformed")
    for item in group:
        for name in ("initial_state_digest", "initial_backbone_digest", "terminal_state_digest"):
            _require(_hex64(item.get(name)), "group_state_digest_malformed")
        history = item.get("history")
        _require(
            isinstance(history, Sequence) and not isinstance(history, (str, bytes)),
            "group_history_malformed",
        )
        _require(
            len(history) == _integer(item.get("epochs"), "group_epochs_malformed"),
            "group_history_length_mismatch",
        )
    prefix = min(epochs)
    for name in DIGEST_FIELDS:
        series = [_digests(item, name) for item in group]
        reference = series[0][:prefix]
        for values in series[1:]:
            _require(values[:prefix] == reference, "group_digest_prefix_mismatch")
    if cross_instrument_pairs == 0:
        for recipe in SPARSE_RECIPES:
            item = recipes.get(recipe)
            if item is None:
                continue
            support = item.get("paired_support")
            _require(isinstance(support, Mapping), "group_paired_support_malformed")
            _require(support.get("enabled") == 1, "group_pair_enable_mismatch")
            for name in ("available_batches", "eligible_masters", "pairs"):
                _require(
                    _integer(support.get(name), "group_pair_support_malformed") == 0,
                    "group_sparse_pairs_present",
                )
    for left, right in EQUIVALENT_PAIRS:
        if cross_instrument_pairs != 0:
            break
        first, second = recipes.get(left), recipes.get(right)
        if first is None or second is None:
            continue
        if _integer(first.get("epochs"), "group_epochs_malformed") != _integer(
            second.get("epochs"), "group_epochs_malformed"
        ):
            continue
        _require(
            str(first.get("initial_state_digest")) == str(second.get("initial_state_digest"))
            and str(first.get("terminal_state_digest")) == str(second.get("terminal_state_digest")),
            "group_equivalent_state_mismatch",
        )
        _require(_numeric(first) == _numeric(second), "group_equivalent_history_mismatch")
    return {"recipes": sorted(recipes), "epochs_prefix": prefix}


def _digests(item: Mapping[str, Any], name: str) -> list[str]:
    values: list[str] = []
    for record in item.get("history", ()):
        _require(isinstance(record, Mapping), "group_history_record_malformed")
        value = record.get(name)
        _require(_hex64(value), "group_history_digest_malformed")
        values.append(str(value))
    return values


def _numeric(item: Mapping[str, Any]) -> tuple[tuple[tuple[str, Any], ...], ...]:
    rows = []
    for record in item.get("history", ()):
        _require(isinstance(record, Mapping), "group_history_record_malformed")
        rows.append(
            tuple(
                (str(key), record[key])
                for key in sorted(record)
                if not isinstance(record[key], bool) and isinstance(record[key], (int, float))
            )
        )
    return tuple(rows)


def cross_instrument_master_count(support: Any, spec: Any) -> int:
    """Count outer-fit masters with more than one instrument in a context."""

    _require(isinstance(spec, Mapping), "spec_malformed")
    context_id = str(spec.get("context_id"))
    fitting_role_id = str(spec.get("fitting_role_id"))
    roles = getattr(support, "roles", None)
    _require(roles is not None, "support_roles_missing")
    instruments: dict[str, set[str]] = {}
    for row in roles:
        _require(isinstance(row, Mapping), "support_role_malformed")
        if str(row.get("context_id")) != context_id or str(row.get("role_id")) != fitting_role_id:
            continue
        _require(row.get("role") == "outer_fit", "support_not_outer_fit")
        instruments.setdefault(str(row.get("master_sample_id")), set()).add(
            str(row.get("instrument"))
        )
    _require(bool(instruments), "support_outer_fit_empty")
    return sum(1 for values in instruments.values() if len(values) > 1)


__all__ = [
    "P05RefitEvidenceError",
    "check_recipe_group",
    "cross_instrument_master_count",
    "make_logit_loader",
    "persist_calibration",
    "summarize_result",
]
