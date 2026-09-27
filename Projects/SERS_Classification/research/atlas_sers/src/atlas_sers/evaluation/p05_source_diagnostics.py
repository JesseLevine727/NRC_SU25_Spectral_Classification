"""Anonymous, aggregate-only source-side diagnostics for the P05 programme.

This module is pure and in-memory.  It never reads a file, touches torch, fits a
model, infers, selects a recipe or aggregates held outcomes.  Callers
authenticate the frozen refit plan, the complete source selector records, the
registered slot ledger and the anonymous public strategy contexts before
handing them in.

The module cross-checks that the frozen plan, the source records and the public
context table describe exactly the same registered slots, units, recipes, seeds
and contexts.  It then emits five descriptive tables built only from an
allowlist of public metadata (station, phase, domain, held instrument, model,
aggregation) and aggregate numbers (recipe, selection mode, slot kind, seed,
epochs, counts, means and collapse fractions).

Private identifiers such as ``context_id``, ``selection_unit_id``, ``slot_id``,
source paths, UID digests, raw labels or predictions and arbitrary blocking
reasons are never copied into the returned frames.  Context identity is replaced
by the same anonymous 1-based ``point_index`` used by
``atlas_sers.evaluation.p05_public_metrics`` (sorted private context ids).

Two source quantities are kept strictly separate from held quantities.  The
source balanced accuracy is the mean of the per-seed source-validation metric
over seeds and then over inherited selection units with equal unit weight.  The
held balanced accuracy is the public three-seed ensemble metric.  Every
``source_vs_held`` row therefore carries an explicit policy string and a
``source_transfer_validation_available`` flag that is false for master-CV
contexts. Availability does not establish successful generalization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p05_selection import (
    CANDIDATE_RECIPE_IDS,
    COMPLETE_METRIC_FIELDS,
    COMPLETE_STATUS,
    FALLBACK_RECIPE_ID,
    FIXED_CONTROL_RECIPE_ID,
    GUARD_SLOT_KIND,
    GUARD_UNIT_COUNT,
    INHERITED_SLOT_KIND,
    MASTER_ONLY_MODES,
    MAXIMUM_EPOCH,
    MINIMUM_EPOCH,
    PSEUDO_DOMAIN_MODE,
    RECIPE_IDS,
    REFIT_EPOCH_MAXIMUM,
    REFIT_EPOCH_MINIMUM,
    RESULT_IDENTITY_FIELDS,
    SEEDS,
    SELECTION_MODES,
    SLOT_KINDS,
    SLOT_REQUIRED_FIELDS,
)

__all__ = [
    "P05_SOURCE_DIAGNOSTIC_AGGREGATIONS",
    "P05_SOURCE_DIAGNOSTIC_MODELS",
    "P05_SOURCE_DIAGNOSTIC_PHASES",
    "P05_SOURCE_DIAGNOSTIC_STATIONS",
    "P05_SOURCE_DIAGNOSTIC_TABLE_NAMES",
    "SOURCE_VS_HELD_POLICY",
    "SourceDiagnosticsError",
    "build_source_diagnostics",
]

SCHEMA_VERSION = "nato-sers-p05-source-diagnostics-v1"
PROTOCOL_VERSION = "nato-sers-p05-core-20260925-v1"

D0M_MODEL_ID = "D0-M"
SELECTED_MODEL_ID = "P05-SELECTED"
D3_MODEL_ID = "D3"
P05_SOURCE_DIAGNOSTIC_MODELS = (D0M_MODEL_ID, SELECTED_MODEL_ID, D3_MODEL_ID)
P05_SOURCE_DIAGNOSTIC_AGGREGATIONS = ("M01", "M06")
P05_SOURCE_DIAGNOSTIC_STATIONS = ("cwa", "pills", "surfaces")
P05_SOURCE_DIAGNOSTIC_PHASES = ("development", "held_evaluation")

P05_SOURCE_DIAGNOSTIC_TABLE_NAMES = (
    "selection_counts",
    "fit_summary",
    "best_epoch_distribution",
    "refit_epochs",
    "source_vs_held",
)

SOURCE_VS_HELD_POLICY = (
    "source_mean_seed_unit_balanced_accuracy_is_the_mean_of_per_seed_source_"
    "validation_balanced_accuracy_across_seeds_then_equal_weight_over_inherited_"
    "selection_units_only;held_balanced_accuracy_is_the_public_three_seed_"
    "ensemble_metric;the_two_quantities_are_not_directly_comparable"
)

_PSEUDO_DOMAIN_EVIDENCE_POLICY = (
    "pseudo_domain_source_validation_supports_transfer_comparison"
)
_MASTER_CV_EVIDENCE_POLICY = (
    "master_cv_source_validation_does_not_support_instrument_generalization"
)
_ALIAS_COUNT_SEMANTICS = "strategy_alias_contributions_not_unique_executions"

_DECISION_REQUIRED_FIELDS = (
    "context_id",
    "selection_mode",
    "selected_recipe_id",
    "selection_source",
    "unsupported_transfer_selection",
    "station",
    "phase_gate",
    "context_evidence_complete",
    "selected_recipe_evidence_complete",
    "fixed_control_evidence_complete",
    "ready_for_refit",
)

_SELECTION_SOURCES = frozenset(
    {"candidate", "fallback", "unsupported_transfer_selection"}
)

_STRATEGY_REQUIRED_COLUMNS = (
    "point_index",
    "station",
    "phase",
    "domain",
    "held_instrument",
    "model_id",
    "aggregation_id",
    "balanced_accuracy",
)

_GROUP_SELECTION = (
    "station",
    "phase",
    "selection_mode",
    "selected_recipe",
    "selection_source",
    "unsupported_transfer_selection",
)
_GROUP_FIT = ("station", "phase", "selection_mode", "recipe", "slot_kind")
_GROUP_EPOCH = _GROUP_FIT + ("best_epoch",)
_GROUP_REFIT = ("station", "phase", "strategy", "recipe", "seed", "epochs")


class SourceDiagnosticsError(ValueError):
    """Raised with a stable, path-free reason code for invalid diagnostics input."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# --------------------------------------------------------------------------- #
# Primitive validation helpers
# --------------------------------------------------------------------------- #


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceDiagnosticsError(code)
    return value


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SourceDiagnosticsError(code)
    return value


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceDiagnosticsError(code)
    return value


def _optional_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise SourceDiagnosticsError(code)
    return value


def _integer(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceDiagnosticsError(code)
    return value


def _number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceDiagnosticsError(code)
    number = float(value)
    if not np.isfinite(number):
        raise SourceDiagnosticsError(code)
    return number


def _boolean(value: Any, code: str) -> bool:
    if not isinstance(value, bool):
        raise SourceDiagnosticsError(code)
    return value


def _pandas_integer(value: Any, code: str) -> int:
    if isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA:
        raise SourceDiagnosticsError(code)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return int(number)
    raise SourceDiagnosticsError(code)


def _pandas_number(value: Any, code: str) -> float:
    if isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA:
        raise SourceDiagnosticsError(code)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SourceDiagnosticsError(code) from None
    if not np.isfinite(number):
        raise SourceDiagnosticsError(code)
    return number


# --------------------------------------------------------------------------- #
# Frozen plan validation
# --------------------------------------------------------------------------- #


def _decision_index(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = _sequence(plan.get("decisions"), "plan_decisions_malformed")
    index: dict[str, dict[str, Any]] = {}
    for item in raw:
        record = _mapping(item, "plan_decision_malformed")
        for field in _DECISION_REQUIRED_FIELDS:
            if field not in record:
                raise SourceDiagnosticsError("plan_decision_field_missing")
        context_id = _text(record["context_id"], "plan_decision_context_malformed")
        if context_id in index:
            raise SourceDiagnosticsError("plan_decision_duplicate_context")
        mode = _text(record["selection_mode"], "plan_decision_mode_malformed")
        if mode not in SELECTION_MODES:
            raise SourceDiagnosticsError("plan_decision_mode_unknown")
        recipe = _text(record["selected_recipe_id"], "plan_decision_recipe_malformed")
        if recipe not in RECIPE_IDS:
            raise SourceDiagnosticsError("plan_decision_recipe_unknown")
        source = _text(record["selection_source"], "plan_decision_source_malformed")
        if source not in _SELECTION_SOURCES:
            raise SourceDiagnosticsError("plan_decision_source_unknown")
        unsupported = _boolean(
            record["unsupported_transfer_selection"],
            "plan_decision_flag_malformed",
        )
        station = _text(record["station"], "plan_decision_station_malformed")
        phase_gate = _text(record["phase_gate"], "plan_decision_phase_malformed")
        for field in (
            "context_evidence_complete",
            "selected_recipe_evidence_complete",
            "fixed_control_evidence_complete",
            "ready_for_refit",
        ):
            if _boolean(record[field], "plan_decision_flag_malformed") is not True:
                raise SourceDiagnosticsError("plan_decision_incomplete")
        if mode in MASTER_ONLY_MODES:
            if (
                not unsupported
                or source != "unsupported_transfer_selection"
                or recipe != FALLBACK_RECIPE_ID
            ):
                raise SourceDiagnosticsError("plan_decision_master_cv_inconsistent")
        else:
            if unsupported or source == "unsupported_transfer_selection":
                raise SourceDiagnosticsError("plan_decision_pseudo_domain_inconsistent")
            if source == "candidate" and recipe not in CANDIDATE_RECIPE_IDS:
                raise SourceDiagnosticsError("plan_decision_candidate_inconsistent")
            if source == "fallback" and recipe != FALLBACK_RECIPE_ID:
                raise SourceDiagnosticsError("plan_decision_fallback_inconsistent")
        index[context_id] = {
            "context_id": context_id,
            "selection_mode": mode,
            "selected_recipe_id": recipe,
            "selection_source": source,
            "unsupported_transfer_selection": unsupported,
            "station": station,
            "phase_gate": phase_gate,
        }
    if not index:
        raise SourceDiagnosticsError("plan_decisions_empty")
    return index


def _endpoint_index(
    plan: Mapping[str, Any], decision_index: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    raw = _sequence(plan.get("endpoints"), "plan_endpoints_malformed")
    index: dict[str, dict[str, Any]] = {}
    for item in raw:
        record = _mapping(item, "plan_endpoint_malformed")
        context_id = _text(record.get("context_id"), "plan_endpoint_context_malformed")
        if context_id in index:
            raise SourceDiagnosticsError("plan_endpoint_duplicate_context")
        held = _optional_text(
            record.get("held_instrument"),
            "plan_endpoint_held_instrument_malformed",
        )
        endpoint = {"held_instrument": held}
        for field in ("station", "phase_gate", "selection_mode", "domain"):
            endpoint[field] = _text(record.get(field), "plan_endpoint_metadata_malformed")
        decision = decision_index.get(context_id)
        if decision is None or any(
            endpoint[field] != decision[field]
            for field in ("station", "phase_gate", "selection_mode")
        ):
            raise SourceDiagnosticsError("plan_endpoint_decision_mismatch")
        index[context_id] = endpoint
    if set(index) != set(decision_index):
        raise SourceDiagnosticsError("plan_endpoint_context_mismatch")
    return index


def _refit_index(
    plan: Mapping[str, Any], decision_index: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    raw = _mapping(plan.get("unique_refits"), "plan_unique_refits_malformed")
    index: dict[str, dict[str, Any]] = {}
    for key, item in raw.items():
        record = _mapping(item, "plan_unique_refit_malformed")
        refit_id = _text(record.get("refit_id"), "plan_unique_refit_id_malformed")
        if refit_id != key:
            raise SourceDiagnosticsError("plan_unique_refit_id_mismatch")
        context_id = _text(
            record.get("context_id"), "plan_unique_refit_context_malformed"
        )
        if context_id not in decision_index:
            raise SourceDiagnosticsError("plan_unique_refit_unknown_context")
        recipe = _text(record.get("recipe_id"), "plan_unique_refit_recipe_malformed")
        if recipe not in RECIPE_IDS:
            raise SourceDiagnosticsError("plan_unique_refit_recipe_unknown")
        seed = _integer(record.get("seed"), "plan_unique_refit_seed_malformed")
        if seed not in SEEDS:
            raise SourceDiagnosticsError("plan_unique_refit_seed_unknown")
        epochs = _integer(record.get("epochs"), "plan_unique_refit_epochs_malformed")
        if not (REFIT_EPOCH_MINIMUM <= epochs <= REFIT_EPOCH_MAXIMUM):
            raise SourceDiagnosticsError("plan_unique_refit_epochs_out_of_range")
        index[refit_id] = {
            "refit_id": refit_id,
            "context_id": context_id,
            "recipe_id": recipe,
            "seed": seed,
            "epochs": epochs,
        }
    if not index:
        raise SourceDiagnosticsError("plan_unique_refits_empty")
    return index


def _alias_list(
    plan: Mapping[str, Any],
    refit_index: Mapping[str, Mapping[str, Any]],
    decision_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw = _sequence(plan.get("strategy_aliases"), "plan_strategy_aliases_malformed")
    aliases: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in raw:
        record = _mapping(item, "plan_strategy_alias_malformed")
        context_id = _text(
            record.get("context_id"), "plan_strategy_alias_context_malformed"
        )
        if context_id not in decision_index:
            raise SourceDiagnosticsError("plan_strategy_alias_unknown_context")
        strategy = _text(
            record.get("strategy"), "plan_strategy_alias_strategy_malformed"
        )
        if strategy not in P05_SOURCE_DIAGNOSTIC_MODELS:
            raise SourceDiagnosticsError("plan_strategy_alias_strategy_unknown")
        seed = _integer(record.get("seed"), "plan_strategy_alias_seed_malformed")
        if seed not in SEEDS:
            raise SourceDiagnosticsError("plan_strategy_alias_seed_unknown")
        refit_id = _text(
            record.get("refit_id"), "plan_strategy_alias_refit_malformed"
        )
        refit = refit_index.get(refit_id)
        if refit is None:
            raise SourceDiagnosticsError("plan_strategy_alias_unknown_refit")
        if refit["context_id"] != context_id or refit["seed"] != seed:
            raise SourceDiagnosticsError("plan_strategy_alias_refit_mismatch")
        if refit["recipe_id"] != _recipe_for_model(
            strategy, decision_index[context_id]["selected_recipe_id"]
        ):
            raise SourceDiagnosticsError("plan_strategy_alias_recipe_mismatch")
        key = (context_id, strategy, seed)
        if key in seen:
            raise SourceDiagnosticsError("plan_strategy_alias_duplicate")
        seen.add(key)
        aliases.append(
            {
                "context_id": context_id,
                "strategy": strategy,
                "seed": seed,
                "refit_id": refit_id,
            }
        )
    expected = (
        len(decision_index) * len(P05_SOURCE_DIAGNOSTIC_MODELS) * len(SEEDS)
    )
    if len(aliases) != expected:
        raise SourceDiagnosticsError("plan_strategy_alias_count_mismatch")
    if {alias["refit_id"] for alias in aliases} != set(refit_index):
        raise SourceDiagnosticsError("plan_unique_refit_unreferenced")
    return aliases


# --------------------------------------------------------------------------- #
# Registered slot ledger validation
# --------------------------------------------------------------------------- #


def _slot_index(
    slots: Any, decision_index: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    raw = _sequence(slots, "slots_malformed")
    index: dict[str, dict[str, Any]] = {}
    for item in raw:
        record = _mapping(item, "slot_malformed")
        for field in SLOT_REQUIRED_FIELDS:
            if field not in record:
                raise SourceDiagnosticsError("slot_field_missing")
        slot_id = _text(record["slot_id"], "slot_id_malformed")
        if slot_id in index:
            raise SourceDiagnosticsError("slot_duplicate")
        context_id = _text(record["context_id"], "slot_context_malformed")
        if context_id not in decision_index:
            raise SourceDiagnosticsError("slot_unknown_context")
        slot_kind = _text(record["slot_kind"], "slot_kind_malformed")
        if slot_kind not in SLOT_KINDS:
            raise SourceDiagnosticsError("slot_kind_unknown")
        recipe = _text(record["recipe_id"], "slot_recipe_malformed")
        if recipe not in RECIPE_IDS:
            raise SourceDiagnosticsError("slot_recipe_unknown")
        seed = _integer(record["seed"], "slot_seed_malformed")
        if seed not in SEEDS:
            raise SourceDiagnosticsError("slot_seed_unknown")
        unit_id = _text(record["selection_unit_id"], "slot_unit_malformed")
        fitting = _text(record["fitting_role_id"], "slot_fitting_role_malformed")
        validation = _text(
            record["validation_role_id"], "slot_validation_role_malformed"
        )
        if fitting == validation:
            raise SourceDiagnosticsError("slot_roles_not_distinct")
        planned = _boolean(record["planned"], "slot_planned_malformed")
        if planned is not True:
            raise SourceDiagnosticsError("slot_not_planned")
        excluded = _boolean(
            record["excluded_by_protocol"], "slot_excluded_malformed"
        )
        reason = record["exclusion_reason"]
        if excluded:
            _text(reason, "slot_exclusion_reason_malformed")
            raise SourceDiagnosticsError("source_slot_excluded")
        if reason is not None:
            raise SourceDiagnosticsError("slot_unexpected_exclusion_reason")
        guard_fold = record["guard_fold"]
        if slot_kind == GUARD_SLOT_KIND:
            fold = _integer(guard_fold, "slot_guard_fold_malformed")
            if fold not in range(GUARD_UNIT_COUNT):
                raise SourceDiagnosticsError("slot_guard_fold_out_of_range")
        elif guard_fold is not None:
            raise SourceDiagnosticsError("slot_unexpected_guard_fold")
        index[slot_id] = {
            "slot_id": slot_id,
            "context_id": context_id,
            "selection_unit_id": unit_id,
            "slot_kind": slot_kind,
            "recipe_id": recipe,
            "seed": seed,
            "fitting_role_id": fitting,
            "validation_role_id": validation,
            "guard_fold": guard_fold,
        }
    if not index:
        raise SourceDiagnosticsError("slots_empty")
    return index


def _unit_index(
    slot_index: Mapping[str, Mapping[str, Any]],
    decision_index: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    units: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for slot in slot_index.values():
        units.setdefault(
            (str(slot["context_id"]), str(slot["selection_unit_id"])), []
        ).append(dict(slot))
    expected = {(recipe, seed) for recipe in RECIPE_IDS for seed in SEEDS}
    inherited_units: dict[str, list[str]] = {}
    guard_units: dict[str, list[str]] = {}
    guard_folds: dict[str, set[int]] = {}
    for (context_id, unit_id), unit_slots in units.items():
        kinds = {str(slot["slot_kind"]) for slot in unit_slots}
        if len(kinds) != 1:
            raise SourceDiagnosticsError("unit_mixed_slot_kinds")
        kind = next(iter(kinds))
        for field in (
            "fitting_role_id",
            "validation_role_id",
            "guard_fold",
        ):
            if len({slot[field] for slot in unit_slots}) != 1:
                raise SourceDiagnosticsError("unit_inconsistent_metadata")
        identities = [(str(slot["recipe_id"]), int(slot["seed"])) for slot in unit_slots]
        if len(identities) != len(set(identities)):
            raise SourceDiagnosticsError("unit_duplicate_identity")
        if set(identities) != expected:
            raise SourceDiagnosticsError("unit_recipe_seed_ladder_incomplete")
        if kind == INHERITED_SLOT_KIND:
            inherited_units.setdefault(context_id, []).append(unit_id)
        else:
            guard_units.setdefault(context_id, []).append(unit_id)
            guard_folds.setdefault(context_id, set()).add(int(unit_slots[0]["guard_fold"]))
    for context_id, decision in decision_index.items():
        if not inherited_units.get(context_id):
            raise SourceDiagnosticsError("context_without_inherited_units")
        guards = guard_units.get(context_id, [])
        if decision["selection_mode"] in MASTER_ONLY_MODES:
            if guards:
                raise SourceDiagnosticsError("master_cv_guard_units_present")
        else:
            if len(guards) != GUARD_UNIT_COUNT:
                raise SourceDiagnosticsError("pseudo_domain_guard_unit_count")
            if guard_folds.get(context_id, set()) != set(range(GUARD_UNIT_COUNT)):
                raise SourceDiagnosticsError("pseudo_domain_guard_folds_incomplete")
    observed_contexts = {str(slot["context_id"]) for slot in slot_index.values()}
    if observed_contexts != set(decision_index):
        raise SourceDiagnosticsError("plan_context_coverage_mismatch")
    return units, inherited_units, guard_units


# --------------------------------------------------------------------------- #
# Source selector-record validation
# --------------------------------------------------------------------------- #


def _record_index(
    selector_records: Any, slot_index: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    raw = _sequence(selector_records, "selector_records_malformed")
    index: dict[str, dict[str, Any]] = {}
    for item in raw:
        record = _mapping(item, "selector_record_malformed")
        for field in RESULT_IDENTITY_FIELDS:
            if field not in record:
                raise SourceDiagnosticsError("selector_record_field_missing")
        slot_id = _text(record["slot_id"], "selector_record_slot_malformed")
        if slot_id in index:
            raise SourceDiagnosticsError("selector_record_duplicate_slot")
        slot = slot_index.get(slot_id)
        if slot is None:
            raise SourceDiagnosticsError("selector_record_unknown_slot")
        _integer(record["seed"], "selector_record_seed_malformed")
        for field in (
            "context_id",
            "selection_unit_id",
            "slot_kind",
            "fitting_role_id",
            "validation_role_id",
            "recipe_id",
            "seed",
        ):
            if record[field] != slot[field]:
                raise SourceDiagnosticsError("selector_record_identity_mismatch")
        status = _text(record["status"], "selector_record_status_malformed")
        if status != COMPLETE_STATUS:
            raise SourceDiagnosticsError("selector_record_incomplete")
        for field in COMPLETE_METRIC_FIELDS:
            if field not in record:
                raise SourceDiagnosticsError("selector_record_metric_missing")
        epoch = _integer(record["best_epoch"], "selector_record_best_epoch_malformed")
        if not (MINIMUM_EPOCH <= epoch <= MAXIMUM_EPOCH):
            raise SourceDiagnosticsError("selector_record_best_epoch_out_of_range")
        balanced_accuracy = _number(
            record["best_validation_balanced_accuracy"],
            "selector_record_balanced_accuracy_malformed",
        )
        if not 0.0 <= balanced_accuracy <= 1.0:
            raise SourceDiagnosticsError("selector_record_balanced_accuracy_out_of_range")
        nll = _number(record["best_validation_nll"], "selector_record_nll_malformed")
        if nll < 0.0:
            raise SourceDiagnosticsError("selector_record_nll_out_of_range")
        macro_f1 = _number(
            record["best_validation_macro_f1"], "selector_record_macro_f1_malformed"
        )
        if not 0.0 <= macro_f1 <= 1.0:
            raise SourceDiagnosticsError("selector_record_macro_f1_out_of_range")
        predicted = _integer(
            record["best_validation_predicted_class_count"],
            "selector_record_predicted_class_count_malformed",
        )
        if not 1 <= predicted <= 3:
            raise SourceDiagnosticsError(
                "selector_record_predicted_class_count_out_of_range"
            )
        index[slot_id] = {
            "slot_id": slot_id,
            "context_id": str(slot["context_id"]),
            "selection_unit_id": str(slot["selection_unit_id"]),
            "slot_kind": str(slot["slot_kind"]),
            "recipe_id": str(slot["recipe_id"]),
            "seed": int(slot["seed"]),
            "best_epoch": epoch,
            "ba": balanced_accuracy,
            "nll": nll,
            "f1": macro_f1,
            "predicted_class_count": predicted,
            "collapsed": predicted < 2,
        }
    if set(index) != set(slot_index):
        raise SourceDiagnosticsError("selector_record_coverage_mismatch")
    return index


# --------------------------------------------------------------------------- #
# Anonymous public strategy-context validation
# --------------------------------------------------------------------------- #


def _public_contexts(strategy_contexts: Any) -> pd.DataFrame:
    if not isinstance(strategy_contexts, pd.DataFrame):
        raise SourceDiagnosticsError("strategy_contexts_malformed")
    missing = [
        column
        for column in _STRATEGY_REQUIRED_COLUMNS
        if column not in strategy_contexts.columns
    ]
    if missing:
        raise SourceDiagnosticsError("strategy_contexts_columns_missing")
    frame = strategy_contexts[list(_STRATEGY_REQUIRED_COLUMNS)].copy()
    if frame.empty:
        raise SourceDiagnosticsError("strategy_contexts_empty")
    point_indexes = [
        _pandas_integer(value, "strategy_context_point_index_invalid")
        for value in frame["point_index"].tolist()
    ]
    if any(value <= 0 for value in point_indexes):
        raise SourceDiagnosticsError("strategy_context_point_index_invalid")
    frame["point_index"] = point_indexes
    for column, allowed, code in (
        ("station", P05_SOURCE_DIAGNOSTIC_STATIONS, "strategy_context_station_unknown"),
        ("phase", P05_SOURCE_DIAGNOSTIC_PHASES, "strategy_context_phase_unknown"),
        ("model_id", P05_SOURCE_DIAGNOSTIC_MODELS, "strategy_context_model_unknown"),
        (
            "aggregation_id",
            P05_SOURCE_DIAGNOSTIC_AGGREGATIONS,
            "strategy_context_aggregation_unknown",
        ),
    ):
        for value in frame[column].tolist():
            if not isinstance(value, str) or value not in allowed:
                raise SourceDiagnosticsError(code)
    for column, code in (
        ("domain", "strategy_context_domain_invalid"),
        ("held_instrument", "strategy_context_held_instrument_invalid"),
    ):
        for value in frame[column].tolist():
            if not isinstance(value, str) or value != value.strip():
                raise SourceDiagnosticsError(code)
            if column == "domain" and not value:
                raise SourceDiagnosticsError(code)
    balanced = []
    for value in frame["balanced_accuracy"].tolist():
        number = _pandas_number(value, "strategy_context_balanced_accuracy_invalid")
        if not 0.0 <= number <= 1.0:
            raise SourceDiagnosticsError("strategy_context_balanced_accuracy_invalid")
        balanced.append(number)
    frame["balanced_accuracy"] = balanced
    if frame.duplicated(["point_index", "model_id", "aggregation_id"]).any():
        raise SourceDiagnosticsError("strategy_context_duplicate_key")
    for _, cell in frame.groupby("point_index", sort=True, dropna=False):
        if set(zip(cell["model_id"], cell["aggregation_id"], strict=True)) != {
            (model, aggregation)
            for model in P05_SOURCE_DIAGNOSTIC_MODELS
            for aggregation in P05_SOURCE_DIAGNOSTIC_AGGREGATIONS
        }:
            raise SourceDiagnosticsError("strategy_context_endpoint_coverage")
        for column in ("station", "phase", "domain", "held_instrument"):
            if cell[column].nunique(dropna=False) != 1:
                raise SourceDiagnosticsError("strategy_context_metadata_conflict")
    return frame


def _point_index_map(
    decision_index: Mapping[str, Mapping[str, Any]], contexts: pd.DataFrame
) -> dict[int, str]:
    context_ids = sorted(decision_index)
    mapping = {
        index: context_id for index, context_id in enumerate(context_ids, start=1)
    }
    observed = sorted({int(value) for value in contexts["point_index"].tolist()})
    if observed != list(range(1, len(context_ids) + 1)):
        raise SourceDiagnosticsError("point_index_mapping_mismatch")
    return mapping


def _context_metadata(
    decision_index: Mapping[str, Mapping[str, Any]],
    endpoint_index: Mapping[str, Mapping[str, Any]],
    contexts: pd.DataFrame,
    mapping: Mapping[int, str],
) -> dict[str, dict[str, Any]]:
    public_meta: dict[int, dict[str, str]] = {}
    for point_index, cell in contexts.groupby("point_index", sort=True, dropna=False):
        public_meta[int(point_index)] = {
            "station": str(cell["station"].iloc[0]),
            "phase": str(cell["phase"].iloc[0]),
            "domain": str(cell["domain"].iloc[0]),
            "held_instrument": str(cell["held_instrument"].iloc[0]),
        }
    metadata: dict[str, dict[str, Any]] = {}
    for point_index, context_id in mapping.items():
        observed = public_meta.get(point_index)
        if observed is None:
            raise SourceDiagnosticsError("public_context_missing")
        decision = decision_index[context_id]
        if observed["station"] != decision["station"]:
            raise SourceDiagnosticsError("public_station_mismatch")
        if observed["phase"] != decision["phase_gate"]:
            raise SourceDiagnosticsError("public_phase_mismatch")
        if observed["domain"] != endpoint_index[context_id]["domain"]:
            raise SourceDiagnosticsError("public_domain_mismatch")
        if observed["held_instrument"] != endpoint_index[context_id]["held_instrument"]:
            raise SourceDiagnosticsError("public_held_instrument_mismatch")
        metadata[context_id] = {
            "station": observed["station"],
            "phase": observed["phase"],
            "domain": observed["domain"],
            "held_instrument": observed["held_instrument"],
            "selection_mode": decision["selection_mode"],
            "selected_recipe_id": decision["selected_recipe_id"],
            "selection_source": decision["selection_source"],
            "unsupported_transfer_selection": decision["unsupported_transfer_selection"],
        }
    return metadata


# --------------------------------------------------------------------------- #
# Source aggregation
# --------------------------------------------------------------------------- #


def _recipe_for_model(model_id: str, selected_recipe_id: str) -> str:
    if model_id == D0M_MODEL_ID:
        return FALLBACK_RECIPE_ID
    if model_id == SELECTED_MODEL_ID:
        return selected_recipe_id
    if model_id == D3_MODEL_ID:
        return FIXED_CONTROL_RECIPE_ID
    raise SourceDiagnosticsError("strategy_context_model_unknown")


def _source_evidence_policy(selection_mode: str) -> str:
    if selection_mode == PSEUDO_DOMAIN_MODE:
        return _PSEUDO_DOMAIN_EVIDENCE_POLICY
    return _MASTER_CV_EVIDENCE_POLICY


def _source_statistics(
    inherited_units: Mapping[str, Sequence[str]],
    guard_units: Mapping[str, Sequence[str]],
    record_index: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, int], str] = {}
    for slot_id, record in record_index.items():
        lookup[
            (
                str(record["context_id"]),
                str(record["selection_unit_id"]),
                str(record["recipe_id"]),
                int(record["seed"]),
            )
        ] = slot_id
    statistics: dict[tuple[str, str], dict[str, Any]] = {}
    for context_id, unit_ids in inherited_units.items():
        guard_count = len(guard_units.get(context_id, ()))
        ordered_units = sorted(unit_ids)
        for recipe in RECIPE_IDS:
            unit_balanced: list[float] = []
            unit_nll: list[float] = []
            for unit_id in ordered_units:
                seed_balanced: list[float] = []
                seed_nll: list[float] = []
                for seed in SEEDS:
                    slot_id = lookup.get((context_id, unit_id, recipe, seed))
                    if slot_id is None:
                        raise SourceDiagnosticsError("source_recipe_incomplete")
                    record = record_index[slot_id]
                    seed_balanced.append(float(record["ba"]))
                    seed_nll.append(float(record["nll"]))
                unit_balanced.append(sum(seed_balanced) / len(seed_balanced))
                unit_nll.append(sum(seed_nll) / len(seed_nll))
            statistics[(context_id, recipe)] = {
                "ba": sum(unit_balanced) / len(unit_balanced),
                "nll": sum(unit_nll) / len(unit_nll),
                "unit_count": len(ordered_units),
                "guard_count": guard_count,
            }
    return statistics


# --------------------------------------------------------------------------- #
# Public tables
# --------------------------------------------------------------------------- #


def _selection_counts(
    decision_index: Mapping[str, Mapping[str, Any]],
    context_metadata: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    counts: dict[tuple[Any, ...], int] = {}
    for context_id, metadata in context_metadata.items():
        decision = decision_index[context_id]
        key = (
            metadata["station"],
            metadata["phase"],
            decision["selection_mode"],
            decision["selected_recipe_id"],
            decision["selection_source"],
            bool(decision["unsupported_transfer_selection"]),
        )
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for key in sorted(counts):
        station, phase, selection_mode, selected_recipe, source, unsupported = key
        rows.append(
            {
                "station": station,
                "phase": phase,
                "selection_mode": selection_mode,
                "selected_recipe": selected_recipe,
                "selection_source": source,
                "unsupported_transfer_selection": unsupported,
                "context_count": counts[key],
            }
        )
    columns = [*_GROUP_SELECTION, "context_count"]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(list(_GROUP_SELECTION), kind="stable").reset_index(drop=True)


def _record_frame(
    slot_index: Mapping[str, Mapping[str, Any]],
    record_index: Mapping[str, Mapping[str, Any]],
    context_metadata: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for slot_id, record in record_index.items():
        slot = slot_index[slot_id]
        metadata = context_metadata[str(slot["context_id"])]
        rows.append(
            {
                "station": metadata["station"],
                "phase": metadata["phase"],
                "selection_mode": metadata["selection_mode"],
                "recipe": str(slot["recipe_id"]),
                "slot_kind": str(slot["slot_kind"]),
                "best_epoch": int(record["best_epoch"]),
                "ba": float(record["ba"]),
                "f1": float(record["f1"]),
                "nll": float(record["nll"]),
                "collapsed": bool(record["collapsed"]),
            }
        )
    columns = [
        *_GROUP_FIT,
        "best_epoch",
        "ba",
        "f1",
        "nll",
        "collapsed",
    ]
    return pd.DataFrame(rows, columns=columns)


def _fit_summary(record_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in record_frame.groupby(list(_GROUP_FIT), sort=True, dropna=False):
        record = dict(zip(_GROUP_FIT, keys, strict=True))
        fit_count = int(len(cell))
        collapsed = int(cell["collapsed"].sum())
        record["fit_count"] = fit_count
        record["collapsed_fit_count"] = collapsed
        record["collapse_fraction"] = float(collapsed / fit_count)
        record["best_epoch_min"] = int(cell["best_epoch"].min())
        record["best_epoch_median"] = float(cell["best_epoch"].median())
        record["best_epoch_max"] = int(cell["best_epoch"].max())
        record["mean_best_validation_balanced_accuracy"] = float(cell["ba"].mean())
        record["mean_best_validation_macro_f1"] = float(cell["f1"].mean())
        record["mean_best_validation_negative_log_likelihood"] = float(cell["nll"].mean())
        rows.append(record)
    columns = [
        *_GROUP_FIT,
        "fit_count",
        "collapsed_fit_count",
        "collapse_fraction",
        "best_epoch_min",
        "best_epoch_median",
        "best_epoch_max",
        "mean_best_validation_balanced_accuracy",
        "mean_best_validation_macro_f1",
        "mean_best_validation_negative_log_likelihood",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(list(_GROUP_FIT), kind="stable").reset_index(drop=True)


def _best_epoch_distribution(record_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in record_frame.groupby(list(_GROUP_EPOCH), sort=True, dropna=False):
        record = dict(zip(_GROUP_EPOCH, keys, strict=True))
        record["fit_count"] = int(len(cell))
        rows.append(record)
    columns = [*_GROUP_EPOCH, "fit_count"]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(list(_GROUP_EPOCH), kind="stable").reset_index(drop=True)


def _refit_epochs(
    aliases: Sequence[Mapping[str, Any]],
    refit_index: Mapping[str, Mapping[str, Any]],
    context_metadata: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for alias in aliases:
        refit = refit_index[str(alias["refit_id"])]
        metadata = context_metadata[str(alias["context_id"])]
        rows.append(
            {
                "station": metadata["station"],
                "phase": metadata["phase"],
                "strategy": str(alias["strategy"]),
                "recipe": str(refit["recipe_id"]),
                "seed": int(refit["seed"]),
                "epochs": int(refit["epochs"]),
                "refit_id": str(alias["refit_id"]),
                "context_id": str(alias["context_id"]),
            }
        )
    columns = [*_GROUP_REFIT, "refit_id", "context_id"]
    frame = pd.DataFrame(rows, columns=columns)
    out_rows = []
    for keys, cell in frame.groupby(list(_GROUP_REFIT), sort=True, dropna=False):
        record = dict(zip(_GROUP_REFIT, keys, strict=True))
        record["context_count"] = int(cell["context_id"].nunique())
        record["alias_contribution_count"] = int(len(cell))
        record["unique_refit_count"] = int(cell["refit_id"].nunique())
        record["count_semantics"] = _ALIAS_COUNT_SEMANTICS
        out_rows.append(record)
    out_columns = [
        *_GROUP_REFIT,
        "context_count",
        "alias_contribution_count",
        "unique_refit_count",
        "count_semantics",
    ]
    out = pd.DataFrame(out_rows, columns=out_columns)
    return out.sort_values(list(_GROUP_REFIT), kind="stable").reset_index(drop=True)


def _source_vs_held(
    contexts: pd.DataFrame,
    mapping: Mapping[int, str],
    decision_index: Mapping[str, Mapping[str, Any]],
    source_statistics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for row in contexts.itertuples(index=False):
        point_index = int(row.point_index)
        context_id = mapping[point_index]
        decision = decision_index[context_id]
        selection_mode = decision["selection_mode"]
        recipe = _recipe_for_model(str(row.model_id), decision["selected_recipe_id"])
        statistics = source_statistics.get((context_id, recipe))
        if statistics is None:
            raise SourceDiagnosticsError("source_recipe_incomplete")
        rows.append(
            {
                "point_index": point_index,
                "station": str(row.station),
                "phase": str(row.phase),
                "domain": str(row.domain),
                "held_instrument": str(row.held_instrument),
                "model_id": str(row.model_id),
                "aggregation_id": str(row.aggregation_id),
                "selection_mode": selection_mode,
                "recipe": recipe,
                "source_mean_seed_unit_balanced_accuracy": float(statistics["ba"]),
                "source_mean_seed_unit_negative_log_likelihood": float(
                    statistics["nll"]
                ),
                "source_unit_count": int(statistics["unit_count"]),
                "guard_unit_count": int(statistics["guard_count"]),
                "held_balanced_accuracy": float(row.balanced_accuracy),
                "source_evidence_policy": _source_evidence_policy(selection_mode),
                "source_transfer_validation_available": selection_mode
                == PSEUDO_DOMAIN_MODE,
                "source_vs_held_policy": SOURCE_VS_HELD_POLICY,
            }
        )
    columns = [
        "point_index",
        "station",
        "phase",
        "domain",
        "held_instrument",
        "model_id",
        "aggregation_id",
        "selection_mode",
        "recipe",
        "source_mean_seed_unit_balanced_accuracy",
        "source_mean_seed_unit_negative_log_likelihood",
        "source_unit_count",
        "guard_unit_count",
        "held_balanced_accuracy",
        "source_evidence_policy",
        "source_transfer_validation_available",
        "source_vs_held_policy",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(
        ["point_index", "model_id", "aggregation_id"], kind="stable"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def build_source_diagnostics(
    *,
    plan: Any,
    selector_records: Any,
    slots: Any,
    strategy_contexts: Any,
) -> dict[str, pd.DataFrame]:
    """Build anonymous, aggregate-only source diagnostics for the frozen plan.

    The caller must already have authenticated the frozen refit plan, the
    complete source selector records and ledger, and the public strategy
    contexts.  Every registered slot must be planned, protocol-included and
    covered by exactly one complete source record.  Any mismatch raises a
    stable, path-free :class:`SourceDiagnosticsError`; malformed rows are never
    silently dropped.  The function never re-runs ``select_context`` and never
    alters a decision, epoch or threshold.
    """

    plan_map = _mapping(plan, "plan_malformed")
    for key in ("decisions", "unique_refits", "strategy_aliases", "endpoints"):
        if key not in plan_map:
            raise SourceDiagnosticsError("plan_key_missing")
    decision_index = _decision_index(plan_map)
    slot_index = _slot_index(slots, decision_index)
    _units, inherited_units, guard_units = _unit_index(slot_index, decision_index)
    record_index = _record_index(selector_records, slot_index)
    endpoint_index = _endpoint_index(plan_map, decision_index)
    refit_index = _refit_index(plan_map, decision_index)
    aliases = _alias_list(plan_map, refit_index, decision_index)
    contexts = _public_contexts(strategy_contexts)
    mapping = _point_index_map(decision_index, contexts)
    context_metadata = _context_metadata(
        decision_index, endpoint_index, contexts, mapping
    )
    source_statistics = _source_statistics(
        inherited_units, guard_units, record_index
    )
    record_frame = _record_frame(slot_index, record_index, context_metadata)
    return {
        "selection_counts": _selection_counts(decision_index, context_metadata),
        "fit_summary": _fit_summary(record_frame),
        "best_epoch_distribution": _best_epoch_distribution(record_frame),
        "refit_epochs": _refit_epochs(aliases, refit_index, context_metadata),
        "source_vs_held": _source_vs_held(
            contexts, mapping, decision_index, source_statistics
        ),
    }
