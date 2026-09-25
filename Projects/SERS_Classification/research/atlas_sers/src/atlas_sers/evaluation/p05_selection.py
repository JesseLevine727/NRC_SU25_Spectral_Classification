"""Context-local P05 nested source-only selection and refit-epoch accounting.

This module implements the locked P05 G3 selection policy one outer context at
a time. It is a pure, standard-library-only accounting layer: it never reads a
file, never imports NumPy or torch, never fits a model, never reads held
outcomes and never aggregates across contexts.

Callers supply the registered slot plan emitted by
``atlas_sers.evaluation.p05_core_plan`` together with completed fit evidence.
Selection is fail-closed:

* Malformed slot plans and malformed evidence raise ``SelectionError``.
* Missing, failed and protocol-excluded records never raise on their own; they
  stay visible in the returned accounting, are counted in the required
  denominators and block the affected candidate.
* A candidate is only ever promoted from its own pseudo-domain evidence. The
  ``master_cv`` and inherited ``inner_master_cv`` modes always return the
  ``D0-M`` fallback and never present master-CV numbers as instrument
  validation.
* Exact ``Fraction`` arithmetic is used for every aggregation and margin so
  scientific thresholds are never weakened by binary floating point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from statistics import median
from typing import Any

__all__ = ["SelectionError", "select_context", "inherit_refit_epochs"]

SCHEMA_VERSION = "nato-sers-p05-selection-v1"
PROTOCOL_VERSION = "nato-sers-p05-core-20260925-v1"

RECIPE_IDS = ("D0-M", "D1", "D2", "D3")
CANDIDATE_RECIPE_IDS = ("D1", "D2", "D3")
FALLBACK_RECIPE_ID = "D0-M"
FIXED_CONTROL_RECIPE_ID = "D3"
SEEDS = (20260805, 20260817, 20260829)

PSEUDO_DOMAIN_MODE = "pseudo_domain"
MASTER_CV_MODE = "master_cv"
INNER_MASTER_CV_MODE = "inner_master_cv"
MASTER_ONLY_MODES = frozenset({MASTER_CV_MODE, INNER_MASTER_CV_MODE})
SELECTION_MODES = frozenset({PSEUDO_DOMAIN_MODE}) | MASTER_ONLY_MODES

INHERITED_SLOT_KIND = "inherited_selection_fit"
GUARD_SLOT_KIND = "guard_selection_fit"
SLOT_KINDS = frozenset({INHERITED_SLOT_KIND, GUARD_SLOT_KIND})
GUARD_UNIT_COUNT = 3

MINIMUM_EPOCH = 1
MAXIMUM_EPOCH = 200
REFIT_EPOCH_MINIMUM = 30
REFIT_EPOCH_MAXIMUM = 200
REQUIRED_CLASS_COUNT = 3

COMPLETE_STATUS = "complete"
FAILED_STATUS = "failed"
MISSING_STATUS = "missing"
EXCLUDED_STATUS = "excluded"
FAILURE_STATUSES = frozenset(
    {"failed", "numerical_failure", "resource_failure", "fit_failure", "data_failure"}
)
RESULT_STATUSES = FAILURE_STATUSES | {COMPLETE_STATUS, EXCLUDED_STATUS, MISSING_STATUS}

MINIMUM_MEAN_PSEUDO_BA_GAIN = Fraction("0.02")
MINIMUM_WORST_PSEUDO_BA_CHANGE = Fraction("-0.02")
MINIMUM_GUARD_BA_CHANGE = Fraction("-0.02")
MINIMUM_FRACTION_PSEUDO_DOMAINS_STRICTLY_IMPROVED = Fraction("0.6")
MAXIMUM_COLLAPSE_FRACTION = Fraction("0.05")

SLOT_REQUIRED_FIELDS = (
    "slot_id",
    "slot_kind",
    "context_id",
    "selection_unit_id",
    "guard_fold",
    "fitting_role_id",
    "validation_role_id",
    "recipe_id",
    "seed",
    "planned",
    "excluded_by_protocol",
    "exclusion_reason",
)

RESULT_IDENTITY_FIELDS = (
    "slot_id",
    "context_id",
    "selection_unit_id",
    "slot_kind",
    "fitting_role_id",
    "validation_role_id",
    "recipe_id",
    "seed",
    "status",
)

COMPLETE_METRIC_FIELDS = (
    "best_epoch",
    "best_validation_balanced_accuracy",
    "best_validation_nll",
    "best_validation_macro_f1",
    "best_validation_predicted_class_count",
)


class SelectionError(ValueError):
    """Raised when a slot plan or fit evidence is malformed or inconsistent."""


# --------------------------------------------------------------------------- #
# Primitive validation helpers
# --------------------------------------------------------------------------- #


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SelectionError(f"{name} must be a sequence of mappings.")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionError(f"{name} must be a mapping.")
    return value


def _require_field(mapping: Mapping[str, Any], field: str, name: str) -> Any:
    if field not in mapping:
        raise SelectionError(f"{name} is missing required field '{field}'.")
    return mapping[field]


def _require_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SelectionError(f"{name} must be a nonempty, unpadded string.")
    return value


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionError(f"{name} must be an integer.")
    return value


def _require_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SelectionError(f"{name} must be a boolean.")
    return value


def _require_metric(value: Any, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{name} must be a finite number.")
    if not math.isfinite(float(value)):
        raise SelectionError(f"{name} must be finite.")
    return Fraction(str(value))


def _mean_fraction(values: Sequence[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def _record(
    status: str,
    raw_status: str,
    reason: str | None,
    *,
    best_epoch: int | None = None,
    ba: Fraction | None = None,
    nll: Fraction | None = None,
    f1: Fraction | None = None,
    predicted_class_count: int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "raw_status": raw_status,
        "reason": reason,
        "best_epoch": best_epoch,
        "ba": ba,
        "nll": nll,
        "f1": f1,
        "predicted_class_count": predicted_class_count,
        "diagnostics": diagnostics,
    }


def _normalize_status(raw: str) -> str:
    if raw == COMPLETE_STATUS:
        return COMPLETE_STATUS
    if raw in FAILURE_STATUSES:
        return FAILED_STATUS
    if raw == EXCLUDED_STATUS:
        return EXCLUDED_STATUS
    if raw == MISSING_STATUS:
        return MISSING_STATUS
    raise SelectionError(f"unknown result status '{raw}'.")


# --------------------------------------------------------------------------- #
# Slot plan validation
# --------------------------------------------------------------------------- #


def _validate_slots(context_id: str, slots: Any) -> dict[str, dict[str, Any]]:
    sequence = _require_sequence(slots, "slots")
    index: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(sequence):
        name = f"slot[{position}]"
        slot = _require_mapping(raw, name)
        for field in SLOT_REQUIRED_FIELDS:
            _require_field(slot, field, name)
        slot_id = _require_identifier(slot["slot_id"], f"{name}.slot_id")
        if slot_id in index:
            raise SelectionError(f"duplicate slot_id '{slot_id}'.")
        slot_context = _require_identifier(slot["context_id"], f"{name}.context_id")
        if slot_context != context_id:
            raise SelectionError(
                f"slot '{slot_id}' belongs to foreign context '{slot_context}'."
            )
        slot_kind = slot["slot_kind"]
        if not isinstance(slot_kind, str) or slot_kind not in SLOT_KINDS:
            raise SelectionError(f"slot '{slot_id}' has unknown slot_kind '{slot_kind}'.")
        recipe_id = slot["recipe_id"]
        if not isinstance(recipe_id, str) or recipe_id not in RECIPE_IDS:
            raise SelectionError(
                f"slot '{slot_id}' has unregistered recipe_id '{recipe_id}'."
            )
        seed = _require_integer(slot["seed"], f"slot '{slot_id}' seed")
        if seed not in SEEDS:
            raise SelectionError(f"slot '{slot_id}' has unregistered seed '{seed}'.")
        unit_id = _require_identifier(
            slot["selection_unit_id"], f"slot '{slot_id}' selection_unit_id"
        )
        fitting_role_id = _require_identifier(
            slot["fitting_role_id"], f"slot '{slot_id}' fitting_role_id"
        )
        validation_role_id = _require_identifier(
            slot["validation_role_id"], f"slot '{slot_id}' validation_role_id"
        )
        if fitting_role_id == validation_role_id:
            raise SelectionError(
                f"slot '{slot_id}' must have distinct fitting and validation roles."
            )
        planned = _require_boolean(slot["planned"], f"slot '{slot_id}' planned")
        if planned is not True:
            raise SelectionError(f"slot '{slot_id}' is not a planned slot.")
        excluded = _require_boolean(
            slot["excluded_by_protocol"], f"slot '{slot_id}' excluded_by_protocol"
        )
        reason = slot["exclusion_reason"]
        if excluded:
            _require_identifier(reason, f"slot '{slot_id}' exclusion_reason")
        elif reason is not None:
            raise SelectionError(
                f"nonexcluded slot '{slot_id}' must carry a null exclusion_reason."
            )
        guard_fold = slot["guard_fold"]
        if slot_kind == GUARD_SLOT_KIND:
            guard_fold_value: int | None = _require_integer(
                guard_fold, f"slot '{slot_id}' guard_fold"
            )
            if guard_fold_value not in range(GUARD_UNIT_COUNT):
                raise SelectionError(
                    f"slot '{slot_id}' has out-of-range guard_fold '{guard_fold_value}'."
                )
        else:
            if guard_fold is not None:
                raise SelectionError(
                    f"inherited slot '{slot_id}' must not carry a guard_fold."
                )
            guard_fold_value = None
        index[slot_id] = {
            "slot_id": slot_id,
            "slot_kind": slot_kind,
            "context_id": slot_context,
            "selection_unit_id": unit_id,
            "guard_fold": guard_fold_value,
            "fitting_role_id": fitting_role_id,
            "validation_role_id": validation_role_id,
            "recipe_id": recipe_id,
            "seed": seed,
            "excluded_by_protocol": excluded,
            "exclusion_reason": reason,
        }
    if not index:
        raise SelectionError("at least one registered slot is required.")
    return index


def _group_units(
    slot_index: Mapping[str, Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    units: dict[str, list[dict[str, Any]]] = {}
    for slot in slot_index.values():
        units.setdefault(str(slot["selection_unit_id"]), []).append(dict(slot))
    expected = {(recipe_id, seed) for recipe_id in RECIPE_IDS for seed in SEEDS}
    for unit_id, unit_slots in units.items():
        kinds = {str(slot["slot_kind"]) for slot in unit_slots}
        if len(kinds) != 1:
            raise SelectionError(f"unit '{unit_id}' mixes slot kinds {sorted(kinds)}.")
        if len({str(slot["fitting_role_id"]) for slot in unit_slots}) != 1:
            raise SelectionError(f"unit '{unit_id}' has inconsistent fitting roles.")
        if len({str(slot["validation_role_id"]) for slot in unit_slots}) != 1:
            raise SelectionError(f"unit '{unit_id}' has inconsistent validation roles.")
        if len({bool(slot["excluded_by_protocol"]) for slot in unit_slots}) != 1:
            raise SelectionError(
                f"unit '{unit_id}' has inconsistent protocol-exclusion status."
            )
        if len({slot["exclusion_reason"] for slot in unit_slots}) != 1:
            raise SelectionError(
                f"unit '{unit_id}' has inconsistent protocol-exclusion reasons."
            )
        if kinds == {GUARD_SLOT_KIND}:
            if len({int(slot["guard_fold"]) for slot in unit_slots}) != 1:
                raise SelectionError(f"guard unit '{unit_id}' spans multiple guard folds.")
        identities = [(str(slot["recipe_id"]), int(slot["seed"])) for slot in unit_slots]
        if len(identities) != len(set(identities)):
            raise SelectionError(f"unit '{unit_id}' repeats a (recipe, seed) identity.")
        observed = set(identities)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise SelectionError(
                f"unit '{unit_id}' does not plan the exact recipe/seed ladder "
                f"(missing={missing}, extra={extra})."
            )
    return units


# --------------------------------------------------------------------------- #
# Result validation
# --------------------------------------------------------------------------- #


def _reject_scoring_evidence(
    result: Mapping[str, Any], normalized: str, slot_id: str
) -> None:
    """Reject non-success statuses that still carry scoring evidence."""

    present = sorted(
        field for field in COMPLETE_METRIC_FIELDS if result.get(field) is not None
    )
    if present:
        raise SelectionError(
            f"slot '{slot_id}' status '{normalized}' must not carry scoring "
            f"evidence {present}."
        )


def _first_reason(result: Mapping[str, Any], slot_id: str) -> str | None:
    """Return the canonical failure reason, preferring failure_reason."""

    for key in ("failure_reason", "reason_code"):
        value = result.get(key)
        if value is not None:
            return _require_identifier(value, f"slot '{slot_id}' {key}")
    return None


def _diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Retain any diagnostic metrics a failed run legitimately reported."""

    return {
        field: result[field]
        for field in COMPLETE_METRIC_FIELDS
        if result.get(field) is not None
    }


def _record_from_result(
    result: Mapping[str, Any], normalized: str, raw_status: str, slot_id: str
) -> dict[str, Any]:
    if normalized == COMPLETE_STATUS:
        for field in COMPLETE_METRIC_FIELDS:
            _require_field(result, field, f"slot '{slot_id}'")
        epoch = _require_integer(result["best_epoch"], f"slot '{slot_id}' best_epoch")
        if not (MINIMUM_EPOCH <= epoch <= MAXIMUM_EPOCH):
            raise SelectionError(f"slot '{slot_id}' best_epoch is out of range.")
        ba = _require_metric(
            result["best_validation_balanced_accuracy"],
            f"slot '{slot_id}' balanced accuracy",
        )
        if not (Fraction(0) <= ba <= Fraction(1)):
            raise SelectionError(
                f"slot '{slot_id}' balanced accuracy is outside [0, 1]."
            )
        nll = _require_metric(result["best_validation_nll"], f"slot '{slot_id}' nll")
        if nll < 0:
            raise SelectionError(f"slot '{slot_id}' nll is negative.")
        f1 = _require_metric(
            result["best_validation_macro_f1"], f"slot '{slot_id}' macro f1"
        )
        if not (Fraction(0) <= f1 <= Fraction(1)):
            raise SelectionError(f"slot '{slot_id}' macro f1 is outside [0, 1].")
        predicted = _require_integer(
            result["best_validation_predicted_class_count"],
            f"slot '{slot_id}' predicted class count",
        )
        if not (1 <= predicted <= REQUIRED_CLASS_COUNT):
            raise SelectionError(
                f"slot '{slot_id}' predicted class count is out of range."
            )
        return _record(
            COMPLETE_STATUS,
            raw_status,
            None,
            best_epoch=epoch,
            ba=ba,
            nll=nll,
            f1=f1,
            predicted_class_count=predicted,
        )
    if normalized == EXCLUDED_STATUS:
        _reject_scoring_evidence(result, normalized, slot_id)
        reason = _require_identifier(
            result.get("exclusion_reason"), f"slot '{slot_id}' exclusion_reason"
        )
        return _record(EXCLUDED_STATUS, raw_status, reason)
    if normalized == MISSING_STATUS:
        _reject_scoring_evidence(result, normalized, slot_id)
    return _record(
        normalized,
        raw_status,
        _first_reason(result, slot_id),
        diagnostics=_diagnostics(result) or None,
    )


def _validate_results(
    context_id: str,
    slot_index: Mapping[str, Mapping[str, Any]],
    results: Any,
) -> dict[str, dict[str, Any]]:
    sequence = _require_sequence(results, "results")
    by_slot: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(sequence):
        name = f"result[{position}]"
        result = _require_mapping(raw, name)
        for field in RESULT_IDENTITY_FIELDS:
            _require_field(result, field, name)
        slot_id = _require_identifier(result["slot_id"], f"{name}.slot_id")
        if slot_id in by_slot:
            raise SelectionError(f"duplicate result for slot '{slot_id}'.")
        slot = slot_index.get(slot_id)
        if slot is None:
            raise SelectionError(f"{name} references unknown slot '{slot_id}'.")
        result_context = _require_identifier(result["context_id"], f"{name}.context_id")
        result_unit = _require_identifier(
            result["selection_unit_id"], f"{name}.selection_unit_id"
        )
        result_kind = result["slot_kind"]
        if not isinstance(result_kind, str) or result_kind not in SLOT_KINDS:
            raise SelectionError(f"{name} has unknown slot_kind '{result_kind}'.")
        result_fitting = _require_identifier(
            result["fitting_role_id"], f"{name}.fitting_role_id"
        )
        result_validation = _require_identifier(
            result["validation_role_id"], f"{name}.validation_role_id"
        )
        result_recipe = result["recipe_id"]
        if not isinstance(result_recipe, str) or result_recipe not in RECIPE_IDS:
            raise SelectionError(f"{name} has unregistered recipe_id '{result_recipe}'.")
        result_seed = _require_integer(result["seed"], f"{name}.seed")
        if (
            result_context != slot["context_id"]
            or result_unit != slot["selection_unit_id"]
            or result_kind != slot["slot_kind"]
            or result_fitting != slot["fitting_role_id"]
            or result_validation != slot["validation_role_id"]
            or result_recipe != slot["recipe_id"]
            or result_seed != slot["seed"]
        ):
            raise SelectionError(
                f"{name} contradicts slot '{slot_id}' on its registered identity."
            )
        raw_status = result["status"]
        if not isinstance(raw_status, str) or raw_status not in RESULT_STATUSES:
            raise SelectionError(f"{name} has unknown status '{raw_status}'.")
        normalized = _normalize_status(raw_status)
        if slot["excluded_by_protocol"]:
            if normalized == COMPLETE_STATUS:
                raise SelectionError(
                    f"protocol-excluded slot '{slot_id}' supplied a complete score."
                )
            if normalized == EXCLUDED_STATUS:
                reason = _require_identifier(
                    result.get("exclusion_reason"), f"{name}.exclusion_reason"
                )
                if reason != slot["exclusion_reason"]:
                    raise SelectionError(
                        f"{name} exclusion reason contradicts the slot plan."
                    )
        elif normalized == EXCLUDED_STATUS:
            raise SelectionError(
                f"{name} marks nonexcluded slot '{slot_id}' as excluded."
            )
        by_slot[slot_id] = _record_from_result(result, normalized, raw_status, slot_id)
    return by_slot


def _effective_record(
    slot: Mapping[str, Any], result: Mapping[str, Any] | None
) -> dict[str, Any]:
    if slot["excluded_by_protocol"]:
        return _record(EXCLUDED_STATUS, "excluded", str(slot["exclusion_reason"]))
    if result is None:
        return _record(MISSING_STATUS, "missing", "missing_evidence")
    return dict(result)


# --------------------------------------------------------------------------- #
# Unit aggregation
# --------------------------------------------------------------------------- #


def _unit_evidence(
    unit_slots: Sequence[Mapping[str, Any]],
    result_by_slot: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    evidence: dict[str, dict[int, dict[str, Any]]] = {}
    for slot in unit_slots:
        recipe_id = str(slot["recipe_id"])
        seed = int(slot["seed"])
        evidence.setdefault(recipe_id, {})[seed] = _effective_record(
            slot, result_by_slot.get(str(slot["slot_id"]))
        )
    return evidence


def _context_evidence(
    units: Mapping[str, Sequence[Mapping[str, Any]]],
    required_units: Sequence[str],
    result_by_slot: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    return {
        unit_id: _unit_evidence(units[unit_id], result_by_slot)
        for unit_id in required_units
    }


def _aggregate_unit_recipe(
    evidence: Mapping[str, Mapping[int, Mapping[str, Any]]], recipe_id: str
) -> tuple[dict[str, Any] | None, list[str]]:
    records = [evidence[recipe_id][seed] for seed in SEEDS]
    statuses = [str(record["status"]) for record in records]
    if any(status != COMPLETE_STATUS for status in statuses):
        return None, statuses
    return (
        {
            "ba": _mean_fraction([record["ba"] for record in records]),
            "f1": _mean_fraction([record["f1"] for record in records]),
            "nll": _mean_fraction([record["nll"] for record in records]),
            "epochs": [int(record["best_epoch"]) for record in records],
        },
        statuses,
    )


def _context_counts(
    evidence: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    required_units: Sequence[str],
) -> dict[str, int]:
    counts = {
        COMPLETE_STATUS: 0,
        MISSING_STATUS: 0,
        FAILED_STATUS: 0,
        EXCLUDED_STATUS: 0,
    }
    for unit_id in required_units:
        for recipe_id in RECIPE_IDS:
            for seed in SEEDS:
                counts[str(evidence[unit_id][recipe_id][seed]["status"])] += 1
    return counts


def _baseline_summary(
    evidence: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    required_units: Sequence[str],
) -> tuple[
    dict[str, dict[str, Any] | None], dict[str, list[str]], list[str]
]:
    aggregates: dict[str, dict[str, Any] | None] = {}
    statuses_map: dict[str, list[str]] = {}
    reasons: list[str] = []
    for unit_id in required_units:
        aggregate, statuses = _aggregate_unit_recipe(
            evidence[unit_id], FALLBACK_RECIPE_ID
        )
        aggregates[unit_id] = aggregate
        statuses_map[unit_id] = statuses
        if aggregate is None:
            reasons.append(f"baseline_unit_{unit_id}_incomplete:" + "/".join(statuses))
    return aggregates, statuses_map, reasons


def _all_nonexcluded_complete(
    units: Mapping[str, Sequence[Mapping[str, Any]]],
    result_by_slot: Mapping[str, Mapping[str, Any]],
) -> bool:
    for unit_slots in units.values():
        for slot in unit_slots:
            if slot["excluded_by_protocol"]:
                continue
            record = _effective_record(slot, result_by_slot.get(str(slot["slot_id"])))
            if record["status"] != COMPLETE_STATUS:
                return False
    return True


def _recipe_inherited_complete(
    units: Mapping[str, Sequence[Mapping[str, Any]]],
    inherited: Sequence[str],
    recipe_id: str,
    result_by_slot: Mapping[str, Mapping[str, Any]],
) -> bool:
    if not inherited:
        return False
    for unit_id in inherited:
        for slot in units[unit_id]:
            if slot["recipe_id"] != recipe_id:
                continue
            record = _effective_record(slot, result_by_slot.get(str(slot["slot_id"])))
            if record["status"] != COMPLETE_STATUS:
                return False
    return True


# --------------------------------------------------------------------------- #
# Candidate evaluation
# --------------------------------------------------------------------------- #


def _threshold_result(
    observed: Fraction, threshold: Fraction, operator: str
) -> dict[str, Any]:
    if operator == ">=":
        passed = observed >= threshold
    elif operator == "<=":
        passed = observed <= threshold
    else:
        raise SelectionError(f"unknown comparison operator '{operator}'.")
    return {
        "observed": float(observed),
        "threshold": float(threshold),
        "operator": operator,
        "passed": passed,
    }


def _evaluate_candidate(
    *,
    recipe_id: str,
    inherited: Sequence[str],
    guards: Sequence[str],
    evidence: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    baseline: Mapping[str, dict[str, Any] | None],
    baseline_ok: bool,
    baseline_reasons: Sequence[str],
) -> dict[str, Any]:
    required_units = list(inherited) + list(guards)
    counts = {
        COMPLETE_STATUS: 0,
        MISSING_STATUS: 0,
        FAILED_STATUS: 0,
        EXCLUDED_STATUS: 0,
    }
    failure_details: list[dict[str, Any]] = []
    exclusion_details: list[dict[str, Any]] = []
    candidate_units: dict[str, dict[str, Any] | None] = {}
    for unit_id in required_units:
        aggregate, _ = _aggregate_unit_recipe(evidence[unit_id], recipe_id)
        candidate_units[unit_id] = aggregate
        for seed in SEEDS:
            record = evidence[unit_id][recipe_id][seed]
            counts[str(record["status"])] += 1
            if record["status"] == FAILED_STATUS:
                failure_details.append(
                    {
                        "selection_unit_id": unit_id,
                        "seed": seed,
                        "status": str(record["raw_status"]),
                        "reason": record["reason"],
                    }
                )
            elif record["status"] == EXCLUDED_STATUS:
                exclusion_details.append(
                    {
                        "selection_unit_id": unit_id,
                        "seed": seed,
                        "reason": record["reason"],
                    }
                )
    scheduled = len(required_units) * len(SEEDS)
    blocking: list[str] = []
    if not baseline_ok:
        blocking.extend(baseline_reasons)
    for unit_id in required_units:
        if candidate_units[unit_id] is None:
            blocking.append(f"candidate_unit_{unit_id}_incomplete")
    candidate: dict[str, Any] = {
        "recipe_id": recipe_id,
        "eligible": False,
        "passed": False,
        "metrics": None,
        "thresholds": None,
        "blocking_reasons": blocking,
        "counts": {
            "scheduled_seed_fits": scheduled,
            "complete": counts[COMPLETE_STATUS],
            "missing": counts[MISSING_STATUS],
            "failed": counts[FAILED_STATUS],
            "excluded": counts[EXCLUDED_STATUS],
        },
        "failure_details": failure_details,
        "exclusion_details": exclusion_details,
    }
    if blocking:
        return candidate
    candidate_inherited_ba = [candidate_units[unit_id]["ba"] for unit_id in inherited]
    baseline_inherited_ba = [baseline[unit_id]["ba"] for unit_id in inherited]
    gains = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(
            candidate_inherited_ba, baseline_inherited_ba, strict=True
        )
    ]
    mean_gain = _mean_fraction(gains)
    worst_change = min(candidate_inherited_ba) - min(baseline_inherited_ba)
    strictly_improved = Fraction(sum(1 for gain in gains if gain > 0), len(gains))
    guard_change = _mean_fraction(
        [
            candidate_units[unit_id]["ba"] - baseline[unit_id]["ba"]
            for unit_id in guards
        ]
    )
    mean_ba = _mean_fraction(candidate_inherited_ba)
    worst_ba = min(candidate_inherited_ba)
    mean_f1 = _mean_fraction([candidate_units[unit_id]["f1"] for unit_id in inherited])
    collapsed = 0
    for unit_id in required_units:
        for seed in SEEDS:
            record = evidence[unit_id][recipe_id][seed]
            if (
                record["status"] == COMPLETE_STATUS
                and int(record["predicted_class_count"]) < 2
            ):
                collapsed += 1
    collapse_fraction = Fraction(collapsed, scheduled)
    thresholds = {
        "mean_pseudo_domain_ba_gain": _threshold_result(
            mean_gain, MINIMUM_MEAN_PSEUDO_BA_GAIN, ">="
        ),
        "worst_pseudo_domain_ba_change": _threshold_result(
            worst_change, MINIMUM_WORST_PSEUDO_BA_CHANGE, ">="
        ),
        "mean_guard_ba_change": _threshold_result(
            guard_change, MINIMUM_GUARD_BA_CHANGE, ">="
        ),
        "fraction_pseudo_domains_strictly_improved": _threshold_result(
            strictly_improved,
            MINIMUM_FRACTION_PSEUDO_DOMAINS_STRICTLY_IMPROVED,
            ">=",
        ),
        "collapse_fraction": _threshold_result(
            collapse_fraction, MAXIMUM_COLLAPSE_FRACTION, "<="
        ),
    }
    candidate.update(
        {
            "eligible": True,
            "passed": all(result["passed"] for result in thresholds.values()),
            "metrics": {
                "mean_pseudo_domain_ba": mean_ba,
                "worst_pseudo_domain_ba": worst_ba,
                "mean_pseudo_domain_macro_f1": mean_f1,
                "mean_pseudo_domain_ba_gain": mean_gain,
                "worst_pseudo_domain_ba_change": worst_change,
                "mean_guard_ba_change": guard_change,
                "fraction_pseudo_domains_strictly_improved": strictly_improved,
                "collapse_fraction": collapse_fraction,
                "collapsed_seed_fit_count": collapsed,
                "scheduled_seed_fit_count": scheduled,
            },
            "thresholds": thresholds,
        }
    )
    return candidate


def _metrics_to_json(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: float(value) if isinstance(value, Fraction) else value
        for key, value in metrics.items()
    }


def _candidate_to_json(candidate: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(candidate)
    if candidate["metrics"] is not None:
        output["metrics"] = _metrics_to_json(candidate["metrics"])
    return output


def _thresholds_json() -> dict[str, float]:
    return {
        "minimum_mean_pseudo_ba_gain": float(MINIMUM_MEAN_PSEUDO_BA_GAIN),
        "minimum_worst_pseudo_ba_change": float(MINIMUM_WORST_PSEUDO_BA_CHANGE),
        "minimum_guard_ba_change": float(MINIMUM_GUARD_BA_CHANGE),
        "minimum_fraction_pseudo_domains_strictly_improved": float(
            MINIMUM_FRACTION_PSEUDO_DOMAINS_STRICTLY_IMPROVED
        ),
        "maximum_collapse_fraction": float(MAXIMUM_COLLAPSE_FRACTION),
    }


def _counts_json(
    required_units: Sequence[str], counts: Mapping[str, int]
) -> dict[str, int]:
    return {
        "inherited_unit_count": 0,
        "guard_unit_count": 0,
        "required_unit_count": len(required_units),
        "scheduled_fit_count": len(required_units) * len(RECIPE_IDS) * len(SEEDS),
        "scheduled_seed_fit_count_per_recipe": len(required_units) * len(SEEDS),
        "complete_fit_count": counts[COMPLETE_STATUS],
        "missing_fit_count": counts[MISSING_STATUS],
        "failed_fit_count": counts[FAILED_STATUS],
        "excluded_fit_count": counts[EXCLUDED_STATUS],
    }


def _baseline_json(
    baseline: Mapping[str, dict[str, Any] | None],
    statuses_map: Mapping[str, list[str]],
) -> dict[str, Any]:
    return {
        unit_id: {
            "complete": baseline[unit_id] is not None,
            "statuses": list(statuses_map[unit_id]),
        }
        for unit_id in baseline
    }


# --------------------------------------------------------------------------- #
# Context results
# --------------------------------------------------------------------------- #


def _pseudo_domain_result(
    context_id: str,
    units: Mapping[str, Sequence[Mapping[str, Any]]],
    inherited: Sequence[str],
    guards: Sequence[str],
    result_by_slot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required_units = list(inherited) + list(guards)
    evidence = _context_evidence(units, required_units, result_by_slot)
    counts = _context_counts(evidence, required_units)
    baseline, statuses_map, baseline_reasons = _baseline_summary(
        evidence, required_units
    )
    inherited_baseline_usable = all(baseline[unit_id] is not None for unit_id in inherited)
    guard_baseline_usable = all(baseline[unit_id] is not None for unit_id in guards)
    baseline_ok = inherited_baseline_usable and guard_baseline_usable
    candidates = {
        recipe_id: _evaluate_candidate(
            recipe_id=recipe_id,
            inherited=inherited,
            guards=guards,
            evidence=evidence,
            baseline=baseline,
            baseline_ok=baseline_ok,
            baseline_reasons=baseline_reasons,
        )
        for recipe_id in CANDIDATE_RECIPE_IDS
    }
    passing = [
        recipe_id
        for recipe_id in CANDIDATE_RECIPE_IDS
        if candidates[recipe_id]["passed"]
    ]
    passing.sort(
        key=lambda recipe_id: (
            -candidates[recipe_id]["metrics"]["mean_pseudo_domain_ba"],
            -candidates[recipe_id]["metrics"]["worst_pseudo_domain_ba"],
            -candidates[recipe_id]["metrics"]["mean_pseudo_domain_macro_f1"],
            CANDIDATE_RECIPE_IDS.index(recipe_id),
        )
    )
    if passing:
        selected_recipe_id = passing[0]
        selection_source = "candidate"
        selection_reason = "candidate_passed_all_g3_thresholds_and_ranked_first"
    else:
        selected_recipe_id = FALLBACK_RECIPE_ID
        selection_source = "fallback"
        selection_reason = "no_candidate_passed_g3_thresholds_fallback_D0-M"
    all_nonexcluded_complete = _all_nonexcluded_complete(units, result_by_slot)
    selected_complete = _recipe_inherited_complete(
        units, inherited, selected_recipe_id, result_by_slot
    )
    fixed_control_complete = _recipe_inherited_complete(
        units, inherited, FIXED_CONTROL_RECIPE_ID, result_by_slot
    )
    ready_for_refit = (
        all_nonexcluded_complete and selected_complete and fixed_control_complete
    )
    reasons = list(baseline_reasons)
    for recipe_id in CANDIDATE_RECIPE_IDS:
        reasons.extend(candidates[recipe_id]["blocking_reasons"])
    if not all_nonexcluded_complete:
        reasons.append("scheduled_evidence_incomplete")
    output_counts = _counts_json(required_units, counts)
    output_counts["inherited_unit_count"] = len(inherited)
    output_counts["guard_unit_count"] = len(guards)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "context_id": context_id,
        "selection_mode": PSEUDO_DOMAIN_MODE,
        "selected_recipe_id": selected_recipe_id,
        "selection_source": selection_source,
        "selection_reason": selection_reason,
        "unsupported_transfer_selection": False,
        "selected_is_fixed_mechanistic_control": (
            selected_recipe_id == FIXED_CONTROL_RECIPE_ID
        ),
        "fixed_mechanistic_control_recipe_id": FIXED_CONTROL_RECIPE_ID,
        "fallback_policy_recipe_id": FALLBACK_RECIPE_ID,
        "fallback_policy_usable": inherited_baseline_usable,
        "full_baseline_comparability": baseline_ok,
        "context_evidence_complete": all_nonexcluded_complete,
        "selected_recipe_evidence_complete": selected_complete,
        "fixed_control_evidence_complete": fixed_control_complete,
        "ready_for_refit": ready_for_refit,
        "thresholds": _thresholds_json(),
        "counts": output_counts,
        "baseline": _baseline_json(baseline, statuses_map),
        "baseline_blocking_reasons": list(baseline_reasons),
        "candidate_ranking": list(passing),
        "candidates": {
            recipe_id: _candidate_to_json(candidates[recipe_id])
            for recipe_id in CANDIDATE_RECIPE_IDS
        },
        "reasons": reasons,
    }


def _master_cv_result(
    context_id: str,
    selection_mode: str,
    units: Mapping[str, Sequence[Mapping[str, Any]]],
    inherited: Sequence[str],
    result_by_slot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required_units = list(inherited)
    evidence = _context_evidence(units, required_units, result_by_slot)
    counts = _context_counts(evidence, required_units)
    baseline, statuses_map, baseline_reasons = _baseline_summary(
        evidence, required_units
    )
    inherited_baseline_usable = all(baseline[unit_id] is not None for unit_id in inherited)
    d0m_complete = _recipe_inherited_complete(
        units, inherited, FALLBACK_RECIPE_ID, result_by_slot
    )
    d3_complete = _recipe_inherited_complete(
        units, inherited, FIXED_CONTROL_RECIPE_ID, result_by_slot
    )
    all_nonexcluded_complete = _all_nonexcluded_complete(units, result_by_slot)
    ready_for_refit = all_nonexcluded_complete and d0m_complete and d3_complete
    reasons = list(baseline_reasons)
    reasons.append(
        f"{selection_mode}_context_does_not_support_transfer_selection"
    )
    if not all_nonexcluded_complete:
        reasons.append("scheduled_evidence_incomplete")
    output_counts = _counts_json(required_units, counts)
    output_counts["inherited_unit_count"] = len(inherited)
    output_counts["guard_unit_count"] = 0
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "context_id": context_id,
        "selection_mode": selection_mode,
        "selected_recipe_id": FALLBACK_RECIPE_ID,
        "selection_source": "unsupported_transfer_selection",
        "selection_reason": (
            f"{selection_mode}_context_does_not_support_transfer_selection_"
            "fallback_D0-M"
        ),
        "unsupported_transfer_selection": True,
        "selected_is_fixed_mechanistic_control": False,
        "fixed_mechanistic_control_recipe_id": FIXED_CONTROL_RECIPE_ID,
        "fallback_policy_recipe_id": FALLBACK_RECIPE_ID,
        "fallback_policy_usable": inherited_baseline_usable,
        "full_baseline_comparability": inherited_baseline_usable,
        "context_evidence_complete": all_nonexcluded_complete,
        "selected_recipe_evidence_complete": d0m_complete,
        "fixed_control_evidence_complete": d3_complete,
        "ready_for_refit": ready_for_refit,
        "thresholds": _thresholds_json(),
        "counts": output_counts,
        "baseline": _baseline_json(baseline, statuses_map),
        "baseline_blocking_reasons": list(baseline_reasons),
        "candidate_ranking": [],
        "candidates": {},
        "reasons": reasons,
    }


def select_context(
    *,
    context_id: Any,
    selection_mode: Any,
    slots: Any,
    results: Any,
) -> dict[str, Any]:
    """Select exactly one outer context's recipe under the locked P05 G3 policy.

    Slots and results from any other context are rejected rather than filtered.
    ``master_cv`` and inherited ``inner_master_cv`` contexts always return the
    ``D0-M`` fallback and never present master-CV numbers as transfer evidence.
    """

    context = _require_identifier(context_id, "context_id")
    if not isinstance(selection_mode, str) or selection_mode not in SELECTION_MODES:
        raise SelectionError(f"unknown selection_mode '{selection_mode}'.")
    slot_index = _validate_slots(context, slots)
    result_by_slot = _validate_results(context, slot_index, results)
    units = _group_units(slot_index)
    kinds = {
        unit_id: str(unit_slots[0]["slot_kind"])
        for unit_id, unit_slots in units.items()
    }
    inherited = sorted(
        unit_id for unit_id, kind in kinds.items() if kind == INHERITED_SLOT_KIND
    )
    guards = sorted(
        unit_id for unit_id, kind in kinds.items() if kind == GUARD_SLOT_KIND
    )
    if not inherited:
        raise SelectionError(
            "at least one inherited selection unit is required for every mode."
        )
    if selection_mode in MASTER_ONLY_MODES:
        if guards:
            raise SelectionError(
                f"{selection_mode} contexts must not register guard units."
            )
        return _master_cv_result(
            context, selection_mode, units, inherited, result_by_slot
        )
    if len(guards) != GUARD_UNIT_COUNT:
        raise SelectionError(
            "pseudo-domain selection requires exactly three distinct guard units."
        )
    guard_folds = {int(units[unit_id][0]["guard_fold"]) for unit_id in guards}
    if guard_folds != set(range(GUARD_UNIT_COUNT)):
        raise SelectionError(
            "pseudo-domain selection requires the three registered guard folds."
        )
    return _pseudo_domain_result(context, units, inherited, guards, result_by_slot)


# --------------------------------------------------------------------------- #
# Inherited refit epochs
# --------------------------------------------------------------------------- #


def _infer_context(slots: Any) -> str:
    sequence = _require_sequence(slots, "slots")
    contexts: set[str] = set()
    for position, raw in enumerate(sequence):
        name = f"slot[{position}]"
        slot = _require_mapping(raw, name)
        _require_field(slot, "context_id", name)
        contexts.add(_require_identifier(slot["context_id"], f"{name}.context_id"))
    if len(contexts) != 1:
        raise SelectionError("slots must describe exactly one context.")
    return next(iter(contexts))


def inherit_refit_epochs(
    *,
    recipe_id: Any,
    slots: Any,
    results: Any,
) -> dict[int, int]:
    """Return per-seed refit epochs inherited from complete selection evidence.

    For each registered training seed, the selected recipe's best epochs are
    collected across the context's inherited (not guard) selection units, the
    median is taken, rounded with the repository Python convention, and clipped
    to ``[30, 200]``. Guard units, other recipes, other seeds and other contexts
    never influence the result. Calibration is out of scope.
    """

    if not isinstance(recipe_id, str) or recipe_id not in RECIPE_IDS:
        raise SelectionError(f"unknown recipe_id '{recipe_id}'.")
    context = _infer_context(slots)
    slot_index = _validate_slots(context, slots)
    result_by_slot = _validate_results(context, slot_index, results)
    units = _group_units(slot_index)
    inherited = sorted(
        unit_id
        for unit_id, unit_slots in units.items()
        if str(unit_slots[0]["slot_kind"]) == INHERITED_SLOT_KIND
    )
    if not inherited:
        raise SelectionError(
            "no inherited selection units are registered for the context."
        )
    selected_slots = [
        slot
        for slot in slot_index.values()
        if slot["slot_kind"] == INHERITED_SLOT_KIND and slot["recipe_id"] == recipe_id
    ]
    if not selected_slots:
        raise SelectionError(
            f"no inherited slots are scheduled for selected recipe '{recipe_id}'."
        )
    per_seed: dict[int, int] = {}
    for seed in SEEDS:
        epochs: list[int] = []
        for slot in selected_slots:
            if int(slot["seed"]) != seed:
                continue
            record = _effective_record(
                slot, result_by_slot.get(str(slot["slot_id"]))
            )
            if record["status"] != COMPLETE_STATUS:
                raise SelectionError(
                    f"selected slot '{slot['slot_id']}' is not complete "
                    f"(status '{record['status']}')."
                )
            epochs.append(int(record["best_epoch"]))
        if len(epochs) != len(inherited):
            raise SelectionError(
                f"seed '{seed}' does not cover every inherited selection unit."
            )
        rounded = int(round(median(epochs)))
        per_seed[seed] = max(
            REFIT_EPOCH_MINIMUM, min(REFIT_EPOCH_MAXIMUM, rounded)
        )
    return per_seed
