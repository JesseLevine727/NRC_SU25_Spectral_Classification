"""Deterministic metadata-only P05 refit plan builder.

This module is a pure, standard-library-only accounting layer. It consumes an
authenticated development ledger, the authoritative support contexts and roles,
completed selection fit evidence and a permit digest. It selects every context
with the locked P05 policy, inherits refit epochs from inherited evidence only,
and emits content-addressed refit specifications, per-strategy aliases and
outer-test endpoints.

It never reads a file, array, checkpoint, held outcome, logit or held metric,
never fits a model and never authorizes execution. Refit identities bind the
context, the outer-fit role, the source UID set, the recipe, the seed, the
inherited epochs, the calibration slots and the permit digest, so identical
recipes alias and different recipes or contexts never do.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from atlas_sers.evaluation.p05_selection import (
    FALLBACK_RECIPE_ID,
    FIXED_CONTROL_RECIPE_ID,
    INHERITED_SLOT_KIND,
    SEEDS,
    SelectionError,
    inherit_refit_epochs,
    select_context,
)

__all__ = ["RefitPlanError", "build_refit_plan"]

SCHEMA_VERSION = "nato-sers-p05-refit-plan-v1"
PROTOCOL_VERSION = "nato-sers-p05-core-20260925-v1"

STRATEGIES = ("D0-M", "P05-SELECTED", "D3")
SELECTED_STRATEGY = "P05-SELECTED"

OUTER_FIT_ROLE = "outer_fit"
OUTER_TEST_ROLE = "outer_test"

HELD_INSTRUMENT_SENTINELS = frozenset({"", "not_applicable"})
MAXIMUM_STRATEGY_ALIAS_COUNT = 2880

REQUIRED_CONTEXT_FIELDS = (
    "context_id",
    "station",
    "held_instrument",
    "selection_mode",
    "phase_gate",
)
REQUIRED_ROLE_FIELDS = (
    "role_id",
    "context_id",
    "role",
    "observation_uid",
    "master_sample_id",
    "instrument",
    "target_analyte",
)


class RefitPlanError(ValueError):
    """Raised when the ledger, support records, evidence or permit are invalid."""


def _sha256_canonical(value: Any) -> str:
    """Return the governance-compatible canonical JSON SHA-256 of ``value``."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RefitPlanError(code)
    return value


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RefitPlanError(code)
    return value


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RefitPlanError(code)
    return value


def _optional_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise RefitPlanError(code)
    return value


def _integer(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RefitPlanError(code)
    return value


def _context_index(contexts: Any) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for raw in _sequence(contexts, "contexts_malformed"):
        record = _mapping(raw, "context_malformed")
        for field in REQUIRED_CONTEXT_FIELDS:
            if field not in record:
                raise RefitPlanError("context_field_missing")
        context_id = _text(record["context_id"], "context_id_malformed")
        if context_id in index:
            raise RefitPlanError("context_duplicate")
        index[context_id] = {
            **record,
            "context_id": context_id,
            "station": _text(record["station"], "context_station_malformed"),
            "held_instrument": _optional_text(
                record["held_instrument"], "context_held_instrument_malformed"
            ),
            "selection_mode": _text(record["selection_mode"], "context_selection_mode_malformed"),
            "phase_gate": _text(record["phase_gate"], "context_phase_gate_malformed"),
        }
    if not index:
        raise RefitPlanError("contexts_empty")
    return index


def _role_index(
    roles: Any,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, tuple[str, str]]]:
    by_role: dict[str, list[dict[str, str]]] = {}
    identity: dict[str, tuple[str, str]] = {}
    uid_meta: dict[str, dict[str, str]] = {}
    for raw in _sequence(roles, "roles_malformed"):
        record = _mapping(raw, "role_malformed")
        for field in REQUIRED_ROLE_FIELDS:
            if field not in record:
                raise RefitPlanError("role_field_missing")
        role_id = _text(record["role_id"], "role_id_malformed")
        context_id = _text(record["context_id"], "role_context_malformed")
        role = _text(record["role"], "role_name_malformed")
        uid = _text(record["observation_uid"], "role_uid_malformed")
        entry = {
            "role_id": role_id,
            "context_id": context_id,
            "role": role,
            "observation_uid": uid,
            "master_sample_id": _text(record["master_sample_id"], "role_master_malformed"),
            "instrument": _text(record["instrument"], "role_instrument_malformed"),
            "target_analyte": _text(record["target_analyte"], "role_target_malformed"),
        }
        previous = identity.get(role_id)
        if previous is None:
            identity[role_id] = (context_id, role)
        elif previous != (context_id, role):
            raise RefitPlanError("role_identity_inconsistent")
        by_role.setdefault(role_id, []).append(entry)
        observed = uid_meta.get(uid)
        if observed is None:
            uid_meta[uid] = entry
        elif (
            observed["master_sample_id"] != entry["master_sample_id"]
            or observed["instrument"] != entry["instrument"]
            or observed["target_analyte"] != entry["target_analyte"]
        ):
            raise RefitPlanError("role_uid_metadata_conflict")
    if not by_role:
        raise RefitPlanError("roles_empty")
    return by_role, identity


def _outer_roles(
    by_role: Mapping[str, Sequence[Mapping[str, str]]],
    identity: Mapping[str, tuple[str, str]],
    context_id: str,
    role_name: str,
) -> tuple[str, list[str], list[str], list[str], set[str]]:
    matches = [
        role_id
        for role_id, (ctx, role) in identity.items()
        if ctx == context_id and role == role_name
    ]
    if len(matches) != 1:
        raise RefitPlanError("outer_role_cardinality")
    role_id = matches[0]
    entries = by_role[role_id]
    uids = [entry["observation_uid"] for entry in entries]
    if len(set(uids)) != len(uids):
        raise RefitPlanError("outer_role_uid_duplicate")
    masters = sorted({entry["master_sample_id"] for entry in entries})
    classes = sorted({entry["target_analyte"] for entry in entries})
    instruments = {entry["instrument"] for entry in entries}
    return role_id, sorted(uids), masters, classes, instruments


def _strategy_recipe(strategy: str, selected_recipe: str) -> str:
    if strategy == SELECTED_STRATEGY:
        return selected_recipe
    if strategy == "D0-M":
        return FALLBACK_RECIPE_ID
    if strategy == "D3":
        return FIXED_CONTROL_RECIPE_ID
    raise RefitPlanError("unknown_strategy")


def build_refit_plan(
    *,
    ledger: Any,
    contexts: Any,
    roles: Any,
    results: Any,
    permit_sha256: Any,
) -> dict[str, Any]:
    """Build the content-addressed P05 refit plan for every outer context."""

    ledger = _mapping(ledger, "ledger_malformed")
    permit = _text(permit_sha256, "permit_sha256_malformed")
    if len(permit) != 64 or any(c not in "0123456789abcdef" for c in permit):
        raise RefitPlanError("permit_sha256_malformed")
    ledger_id = _text(ledger.get("ledger_id"), "ledger_id_malformed")
    slot_records = _sequence(ledger.get("slots"), "ledger_slots_malformed")
    unit_records = _sequence(ledger.get("units"), "ledger_units_malformed")
    context_index = _context_index(contexts)
    by_role, role_identity = _role_index(roles)
    result_records = _sequence(results, "results_malformed")

    slots_by_context: dict[str, list[dict[str, Any]]] = {}
    ledger_slot_ids: set[str] = set()
    required_result_ids: set[str] = set()
    for raw in slot_records:
        slot = dict(_mapping(raw, "ledger_slot_malformed"))
        context_id = _text(slot.get("context_id"), "ledger_slot_context_malformed")
        slot_id = _text(slot.get("slot_id"), "ledger_slot_id_malformed")
        if slot_id in ledger_slot_ids:
            raise RefitPlanError("ledger_slot_duplicate")
        ledger_slot_ids.add(slot_id)
        if slot.get("excluded_by_protocol") is not True:
            required_result_ids.add(slot_id)
        slots_by_context.setdefault(context_id, []).append(slot)

    units_by_context: dict[str, list[dict[str, Any]]] = {}
    for raw in unit_records:
        unit = dict(_mapping(raw, "ledger_unit_malformed"))
        context_id = _text(unit.get("context_id"), "ledger_unit_context_malformed")
        units_by_context.setdefault(context_id, []).append(unit)

    results_by_context: dict[str, list[dict[str, Any]]] = {}
    seen_result_ids: set[str] = set()
    for raw in result_records:
        result = dict(_mapping(raw, "result_malformed"))
        context_id = _text(result.get("context_id"), "result_context_malformed")
        slot_id = _text(result.get("slot_id"), "result_slot_malformed")
        if slot_id in seen_result_ids:
            raise RefitPlanError("result_duplicate")
        seen_result_ids.add(slot_id)
        results_by_context.setdefault(context_id, []).append(result)

    if set(slots_by_context) != set(context_index):
        raise RefitPlanError("ledger_context_mismatch")
    if set(units_by_context) != set(context_index):
        raise RefitPlanError("ledger_context_mismatch")
    if required_result_ids - seen_result_ids:
        raise RefitPlanError("result_coverage_incomplete")
    if seen_result_ids - ledger_slot_ids:
        raise RefitPlanError("result_unknown_slot")

    decisions: list[dict[str, Any]] = []
    unique_refits: dict[str, dict[str, Any]] = {}
    strategy_aliases: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []

    for context_id in sorted(context_index):
        context = context_index[context_id]
        context_slots = slots_by_context[context_id]
        context_results = results_by_context.get(context_id, [])
        try:
            decision = select_context(
                context_id=context_id,
                selection_mode=context["selection_mode"],
                slots=context_slots,
                results=context_results,
            )
        except SelectionError as error:
            raise RefitPlanError("select_context_failed") from error
        if not decision.get("ready_for_refit"):
            raise RefitPlanError("context_not_ready_for_refit")

        fit_role_id, fit_uids, fit_masters, fit_classes, fit_instruments = _outer_roles(
            by_role, role_identity, context_id, OUTER_FIT_ROLE
        )
        test_role_id, test_uids, test_masters, test_classes, _ = _outer_roles(
            by_role, role_identity, context_id, OUTER_TEST_ROLE
        )
        if len(fit_classes) != 3 or not test_classes or not set(test_classes) <= set(fit_classes):
            raise RefitPlanError("outer_role_class_support")
        for field, uids in (
            ("outer_fit_uid_sha256", fit_uids),
            ("outer_test_uid_sha256", test_uids),
        ):
            if field in context and context[field] != _sha256_canonical(uids):
                raise RefitPlanError("outer_role_uid_digest_mismatch")
        if set(fit_uids) & set(test_uids):
            raise RefitPlanError("source_test_uid_overlap")
        if set(fit_masters) & set(test_masters):
            raise RefitPlanError("source_test_master_overlap")
        held = context["held_instrument"]
        if held not in HELD_INSTRUMENT_SENTINELS and held in fit_instruments:
            raise RefitPlanError("held_instrument_in_source")

        fit_uid_set = set(fit_uids)
        for unit in units_by_context[context_id]:
            for key in ("fitting_uids", "validation_uids"):
                observed = unit.get(key)
                if isinstance(observed, (str, bytes)) or not isinstance(observed, Sequence):
                    raise RefitPlanError("ledger_unit_uids_malformed")
                if not {str(uid) for uid in observed} <= fit_uid_set:
                    raise RefitPlanError("unit_uid_outside_outer_fit")

        epochs_by_recipe: dict[str, dict[int, int]] = {}
        for recipe in (
            FALLBACK_RECIPE_ID,
            FIXED_CONTROL_RECIPE_ID,
            decision["selected_recipe_id"],
        ):
            try:
                epochs_by_recipe[recipe] = inherit_refit_epochs(
                    recipe_id=recipe, slots=context_slots, results=context_results
                )
            except SelectionError as error:
                raise RefitPlanError("inherit_refit_epochs_failed") from error

        calibration_by_recipe_seed: dict[tuple[str, int], list[str]] = {}
        for slot in context_slots:
            if slot.get("slot_kind") != INHERITED_SLOT_KIND:
                continue
            key = (
                str(slot.get("recipe_id")),
                _integer(slot.get("seed"), "ledger_slot_seed_malformed"),
            )
            calibration_by_recipe_seed.setdefault(key, []).append(
                _text(slot.get("slot_id"), "ledger_slot_id_malformed")
            )

        source_uid_hash = _sha256_canonical(sorted(fit_uids))

        for strategy in STRATEGIES:
            recipe = _strategy_recipe(strategy, decision["selected_recipe_id"])
            epochs_map = epochs_by_recipe[recipe]
            for seed in SEEDS:
                if seed not in epochs_map:
                    raise RefitPlanError("refit_epoch_seed_missing")
                epochs = _integer(epochs_map[seed], "refit_epoch_malformed")
                calibration_slot_ids = sorted(calibration_by_recipe_seed.get((recipe, seed), []))
                if not calibration_slot_ids:
                    raise RefitPlanError("calibration_slots_missing")
                spec = {
                    "context_id": context_id,
                    "fitting_role_id": fit_role_id,
                    "source_uid_set_sha256": source_uid_hash,
                    "recipe_id": recipe,
                    "seed": seed,
                    "epochs": epochs,
                    "calibration_slot_ids": calibration_slot_ids,
                    "permit_sha256": permit,
                }
                refit_id = _sha256_canonical(spec)
                unique_refits.setdefault(
                    refit_id,
                    {
                        **spec,
                        "refit_id": refit_id,
                        "context_id": context_id,
                        "fitting_role_id": fit_role_id,
                        "fitting_uids": list(fit_uids),
                        "classes": list(fit_classes),
                        "recipe_id": recipe,
                        "seed": seed,
                        "epochs": epochs,
                        "calibration_slot_ids": calibration_slot_ids,
                    },
                )
                strategy_aliases.append(
                    {
                        "context_id": context_id,
                        "strategy": strategy,
                        "seed": seed,
                        "refit_id": refit_id,
                    }
                )

        endpoints.append(
            {
                **context,
                "context_id": context_id,
                "outer_test_role_id": test_role_id,
                "test_uids": list(test_uids),
                "test_masters": list(test_masters),
                "test_classes": list(test_classes),
            }
        )
        decisions.append(
            {**decision, "station": context["station"], "phase_gate": context["phase_gate"]}
        )

    expected_aliases = len(STRATEGIES) * len(SEEDS) * len(context_index)
    if len(strategy_aliases) != expected_aliases:
        raise RefitPlanError("strategy_alias_count_mismatch")
    if len(strategy_aliases) > MAXIMUM_STRATEGY_ALIAS_COUNT:
        raise RefitPlanError("strategy_alias_count_exceeded")

    counts = {
        "context_count": len(context_index),
        "strategy_count": len(STRATEGIES),
        "seed_count": len(SEEDS),
        "strategy_alias_count": len(strategy_aliases),
        "unique_refit_count": len(unique_refits),
        "endpoint_count": len(endpoints),
        "expected_strategy_alias_count": expected_aliases,
        "maximum_strategy_alias_count": MAXIMUM_STRATEGY_ALIAS_COUNT,
    }
    content = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "ledger_id": ledger_id,
        "permit_sha256": permit,
        "decisions": decisions,
        "unique_refits": unique_refits,
        "strategy_aliases": strategy_aliases,
        "endpoints": endpoints,
        "counts": counts,
    }
    return {**content, "plan_id": _sha256_canonical(content)}
