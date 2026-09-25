"""Deterministic metadata-only P05 core registry and bounded smoke plan.

Planning only. This module authenticates the locked P05 core contract by its
canonical SHA-256, reuses the validated inherited source-support audit, and
enumerates the private no-fit smoke and later-core registries. It never reads
spectra, never fits a model, never imports NumPy or torch, and never writes a
file. Every new identifier is a canonical SHA-256 binding over the contract
digest, immutable input pins, and the frozen metadata selections.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from atlas_sers.evaluation.p05_support import (
    SupportInputs,
    build_support_report,
    sha256_value,
    uid_set_hash,
)

SCHEMA_VERSION = "nato-sers-p05-core-plan-v1"
PROTOCOL_VERSION = "nato-sers-p05-core-20260925-v1"
LOCKED_CONTRACT_SHA256 = "60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae"

CURRENT_SMOKE_EXECUTIONS = 34
GUARD_FOLD_COUNT = 3
REQUIRED_CLASSES = 3
LATER_RECIPE_COUNT = 4
GUARD_VERSION = "p05-source-master-cv-guard-v1"

DENSE_ROLE_STATIONS = (
    ("cwa_dense", "cwa"),
    ("pills_dense", "pills"),
    ("surfaces_dense", "surfaces"),
)
SPARSE_ROLE_LABEL = "surfaces_sparse"
SPARSE_STATION = "surfaces"

PIN_KEYS = (
    "p01_run_id",
    "p04plan_run_id",
    "manifest_sha256",
    "representation_sha256",
    "contexts_sha256",
    "roles_sha256",
)

UNKNOWN_SUBSTRATE_LABELS = frozenset(
    {"", "na", "n/a", "none", "unknown", "not_applicable", "unspecified"}
)

SUPPORT_COUNTER_KEYS = (
    "observation_count",
    "master_count",
    "instrument_count",
    "class_count",
    "master_counts_per_chemical",
    "same_chemical_pair_categories",
    "same_chemical_pair_count",
    "different_chemical_pair_count",
    "same_chemical_cross_substrate_pair_count",
    "positive_pair_digest",
    "masters_with_multiple_instruments",
    "anchors_without_same_chemical_peer",
    "chemicals_with_two_or_more_masters",
    "two_chemical_two_master_support",
    "all_chemicals_have_two_or_more_masters",
    "cells_with_two_or_more_spectra",
    "cells_with_two_or_more_masters",
)


class CorePlanError(ValueError):
    """Raised when the contract, pins, or metadata selections are inconsistent."""


# --------------------------------------------------------------------------- #
# Canonical identities
# --------------------------------------------------------------------------- #


def canonical_identity(prefix: str, version: str, fields: Mapping[str, Any]) -> str:
    """Hash a named identifier payload with the repository canonical convention."""

    payload: dict[str, Any] = {"version": str(version), "protocol_version": PROTOCOL_VERSION}
    payload.update(fields)
    return f"{prefix}-{sha256_value(payload)[:24]}"


def _pin_fields(input_pins: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(input_pins, Mapping):
        raise CorePlanError("input_pins must be a mapping.")
    missing = [key for key in PIN_KEYS if key not in input_pins]
    if missing:
        raise CorePlanError("input_pins is missing required pinned identifiers.")
    return {key: input_pins[key] for key in PIN_KEYS}


def core_role_identity(
    *,
    contract_sha256: str,
    role_label: str,
    parent_context_id: str,
    parent_role_id: str,
    role: str,
    selection_unit_id: str,
) -> str:
    return canonical_identity(
        "P05CORE",
        "p05-core-role-v1",
        {
            "contract_sha256": str(contract_sha256),
            "role_label": str(role_label),
            "parent_context_id": str(parent_context_id),
            "parent_role_id": str(parent_role_id),
            "role": str(role),
            "selection_unit_id": str(selection_unit_id),
        },
    )


def guard_role_identity(
    *, contract_sha256: str, context_id: str, guard_fold: int, role: str
) -> str:
    return canonical_identity(
        "P05GUARD",
        "p05-core-guard-role-v1",
        {
            "contract_sha256": str(contract_sha256),
            "context_id": str(context_id),
            "guard_fold": int(guard_fold),
            "role": str(role),
        },
    )


def guard_unit_identity(*, contract_sha256: str, context_id: str, guard_fold: int) -> str:
    return canonical_identity(
        "P05GUARDUNIT",
        "p05-core-guard-unit-v1",
        {
            "contract_sha256": str(contract_sha256),
            "context_id": str(context_id),
            "guard_fold": int(guard_fold),
        },
    )


def fit_identity(
    *,
    contract_sha256: str,
    input_pins: Mapping[str, Any],
    p05_role_id: str,
    recipe_id: str,
    seed: int,
) -> str:
    return canonical_identity(
        "P05FIT",
        "p05-core-fit-v1",
        {
            "contract_sha256": str(contract_sha256),
            "input_pins": _pin_fields(input_pins),
            "p05_role_id": str(p05_role_id),
            "recipe_id": str(recipe_id),
            "seed": int(seed),
        },
    )


def execution_identity(*, fit_id: str, execution_kind: str, replay_index: int | None = None) -> str:
    fields: dict[str, Any] = {"fit_id": str(fit_id), "execution_kind": str(execution_kind)}
    if replay_index is not None:
        fields["replay_index"] = int(replay_index)
    return canonical_identity("P05EXEC", "p05-core-execution-v1", fields)


def shared_rng_group(*, p05_role_id: str, seed: int) -> str:
    payload = {
        "version": "p05-core-rng-v1",
        "protocol_version": PROTOCOL_VERSION,
        "p05_role_id": str(p05_role_id),
        "seed": int(seed),
    }
    return f"P05RNG-{sha256_value(payload)[:16]}"


def core_pair_identity(
    *,
    contract_sha256: str,
    input_pins: Mapping[str, Any],
    p05_role_id: str,
    uid_a: str,
    uid_b: str,
) -> str:
    """Return the contract-bound unordered positive-pair identifier."""

    for value in (uid_a, uid_b):
        if not isinstance(value, str) or not value or value != value.strip():
            raise CorePlanError("A positive pair requires nonempty unpadded observation UIDs.")
    if uid_a == uid_b:
        raise CorePlanError("A positive pair requires two distinct observation UIDs.")
    return canonical_identity(
        "P05PAIR",
        "p05-core-pair-v1",
        {
            "contract_sha256": str(contract_sha256),
            "input_pins": _pin_fields(input_pins),
            "p05_role_id": str(p05_role_id),
            "sorted_uids": sorted((uid_a, uid_b)),
        },
    )


def slot_identity(
    *,
    contract_sha256: str,
    slot_kind: str,
    context_id: str,
    selection_unit_id: str,
    fitting_role_id: str,
    validation_role_id: str,
    guard_fold: int | None,
    recipe_id: str,
    seed: int,
) -> str:
    return canonical_identity(
        "P05SLOT",
        "p05-core-slot-v1",
        {
            "contract_sha256": str(contract_sha256),
            "slot_kind": str(slot_kind),
            "context_id": str(context_id),
            "selection_unit_id": str(selection_unit_id),
            "fitting_role_id": str(fitting_role_id),
            "validation_role_id": str(validation_role_id),
            "guard_fold": None if guard_fold is None else int(guard_fold),
            "recipe_id": str(recipe_id),
            "seed": int(seed),
        },
    )


# --------------------------------------------------------------------------- #
# Contract authentication and ledger arithmetic
# --------------------------------------------------------------------------- #


def check_smoke_schedule(smoke: Mapping[str, Any], sampler: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bounded 8x4 smoke arithmetic without executing anything."""

    seeds = [int(seed) for seed in smoke["seeds"]]
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise CorePlanError("The smoke schedule requires exactly two distinct training seeds.")
    replays = list(smoke["replays"])
    if int(smoke["primary_fit_count"]) + len(replays) != int(smoke["maximum_fit_executions"]):
        raise CorePlanError("Primary fits plus reserved replays disagree with the execution cap.")
    steps = (
        int(smoke["maximum_fit_executions"])
        * int(smoke["epochs"])
        * int(sampler["batch_draws_per_epoch"])
    )
    if steps != int(smoke["maximum_optimizer_steps"]):
        raise CorePlanError(
            "The smoke optimizer-step ceiling does not reconcile with its schedule."
        )
    labels = {str(label) for label in smoke["role_labels"]}
    for replay in replays:
        if str(replay["role_label"]) not in labels:
            raise CorePlanError("A reserved replay references an unregistered smoke role.")
        if int(replay["seed"]) not in seeds:
            raise CorePlanError("A reserved replay references an unregistered smoke seed.")
    return {
        "primary_fit_count": int(smoke["primary_fit_count"]),
        "replay_count": len(replays),
        "maximum_fit_executions": int(smoke["maximum_fit_executions"]),
        "maximum_optimizer_steps": steps,
    }


def check_later_ledger(later_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the finite later-core ledger arithmetic and zero-slot guards."""

    seeds = [int(seed) for seed in later_plan["seeds"]]
    if len(seeds) != 3:
        raise CorePlanError("The later-core ledger requires exactly three training seeds.")
    guard_slots = int(later_plan["extra_guard_unit_slots"])
    expected_guard = int(later_plan["pseudo_domain_contexts"]) * int(
        later_plan["extra_guard_units_per_pseudo_domain_context"]
    )
    if guard_slots != expected_guard:
        raise CorePlanError("Guard unit slots disagree with the pseudo-domain guard expansion.")
    inner = (
        (int(later_plan["inherited_inner_units"]) + guard_slots) * LATER_RECIPE_COUNT * len(seeds)
    )
    if inner != int(later_plan["inner_fit_slot_ceiling"]):
        raise CorePlanError("Inner fit slots do not reconcile with inherited and guard units.")
    later_neural = inner + int(later_plan["refit_slot_ceiling"])
    if later_neural != int(later_plan["later_neural_fit_ceiling"]):
        raise CorePlanError("Later neural fits do not reconcile with inner and refit slots.")
    including = later_neural + CURRENT_SMOKE_EXECUTIONS
    if including != int(later_plan["including_smoke_neural_execution_ceiling"]):
        raise CorePlanError("Including-smoke ceiling does not reconcile with later neural fits.")
    if int(later_plan["retry_slots"]) != 0:
        raise CorePlanError("This version authorizes no retry slots.")
    if int(later_plan["D4_slots"]) != 0 or int(later_plan["D5_slots"]) != 0:
        raise CorePlanError("This version authorizes no D4 or D5 slots.")
    if later_plan["execution_authorized"] is not False:
        raise CorePlanError("The later core plan must remain unauthorized for execution.")
    if later_plan["selection"] != "nested_source_only_per_outer_context":
        raise CorePlanError("Later selection must remain nested source-only per outer context.")
    return {
        "guard_unit_slots": guard_slots,
        "inner_fit_slot_ceiling": inner,
        "later_neural_fit_ceiling": later_neural,
        "including_smoke_neural_execution_ceiling": including,
        "retry_slots": 0,
        "d4_slots": 0,
        "d5_slots": 0,
    }


def validate_core_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the locked P05 core contract against its canonical digest.

    Any changed field, type, budget, replay, or unknown recipe changes the
    canonical digest and is rejected. This exact pin cannot be replaced by a
    caller-supplied value; a future contract change needs a code amendment.
    """

    if not isinstance(contract, dict):
        raise CorePlanError("The P05 core contract must be a JSON object.")
    observed = sha256_value(contract)
    if observed != LOCKED_CONTRACT_SHA256:
        raise CorePlanError("The core contract does not match the locked P05 specification.")
    model = contract["model"]
    if int(model["base_parameters"]) + int(model["projection_parameters"]) != int(
        model["projection_model_parameters"]
    ):
        raise CorePlanError("Model parameter accounting does not reconcile.")
    if int(model["projection_model_parameters"]) >= int(model["maximum_parameters_exclusive"]):
        raise CorePlanError("Model parameter count is not below the exclusive ceiling.")
    schedule = check_smoke_schedule(contract["smoke"], contract["sampler"])
    ledger = check_later_ledger(contract["later_core_plan"])
    return {
        "contract_sha256": LOCKED_CONTRACT_SHA256,
        "schema_version": str(contract["schema_version"]),
        "protocol_version": str(contract["protocol_version"]),
        "authorized_stage": str(contract["authorized_stage"]),
        "input_pins": dict(contract["input_pins"]),
        "population": dict(contract["population"]),
        "recipes": [dict(recipe) for recipe in contract["recipes"]],
        "smoke": dict(contract["smoke"]),
        "sampler": dict(contract["sampler"]),
        "objective": dict(contract["objective"]),
        "optimization": dict(contract["optimization"]),
        "model": dict(model),
        "later_core_plan": dict(contract["later_core_plan"]),
        "g3": dict(contract["g3"]),
        "publication": dict(contract["publication"]),
        "smoke_schedule": schedule,
        "ledger": ledger,
        "full_development_authorized": False,
        "outer_evaluation_authorized": False,
    }


# --------------------------------------------------------------------------- #
# Metadata-only helpers
# --------------------------------------------------------------------------- #


def known_substrate_family(value: Any) -> str | None:
    """Return a normalized known substrate family, or None for unknown labels."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if normalized in UNKNOWN_SUBSTRATE_LABELS:
        return None
    return normalized


def different_known_substrate_family(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return True only when both sides carry different known substrate families."""

    left_family = known_substrate_family(left.get("substrate", left.get("family")))
    right_family = known_substrate_family(right.get("substrate", right.get("family")))
    return left_family is not None and right_family is not None and left_family != right_family


def classify_positive_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> str | None:
    """Classify a same-chemical unordered pair, or None when ineligible.

    Denser master/instrument aggregates are never returned as unique pairs; the
    excluded same-master/same-instrument combination yields None.
    """

    if left["target"] != right["target"]:
        return None
    same_master = left["master"] == right["master"]
    same_instrument = left["instrument"] == right["instrument"]
    if same_master and same_instrument:
        return None
    if same_master:
        return "same_master_different_instrument"
    if same_instrument:
        return "different_master_same_instrument"
    return "different_master_different_instrument"


def assign_guard_folds(*, context_id: str, masters: list[str]) -> dict[str, int]:
    """Assign masters to guard folds by SHA-256 ordering and round-robin."""

    ordered = sorted(
        (str(master) for master in masters),
        key=lambda master: sha256_value([GUARD_VERSION, str(context_id), master]),
    )
    return {master: index % GUARD_FOLD_COUNT for index, master in enumerate(ordered)}


def _dense_candidates(entries: list[dict[str, Any]], station: str) -> list[dict[str, Any]]:
    candidates = []
    for entry in entries:
        if entry.get("role") != "selection_fit" or entry.get("phase_gate") != "development":
            continue
        if entry.get("station") != station:
            continue
        try:
            instruments = int(entry.get("instrument_count", 0))
        except (TypeError, ValueError):
            continue
        categories = entry.get("same_chemical_pair_categories")
        if instruments < 2 or not isinstance(categories, Mapping):
            continue
        try:
            cross = int(categories.get("same_master_different_instrument", 0))
        except (TypeError, ValueError):
            continue
        if cross <= 0:
            continue
        candidates.append(entry)
    return sorted(
        candidates,
        key=lambda entry: (
            str(entry.get("parent_role_id", "")),
            str(entry.get("selection_unit_id", "")),
            str(entry.get("audit_id", "")),
        ),
    )


def _sparse_candidates(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for entry in entries:
        if entry.get("role") != "selection_fit":
            continue
        if entry.get("station") != SPARSE_STATION:
            continue
        try:
            instruments = int(entry.get("instrument_count", 0))
            masters = int(entry.get("master_count", 0))
        except (TypeError, ValueError):
            continue
        if instruments != 1:
            continue
        candidates.append((masters, entry))
    return [
        entry
        for _, entry in sorted(
            candidates,
            key=lambda item: (
                item[0],
                str(item[1].get("parent_role_id", "")),
                str(item[1].get("selection_unit_id", "")),
                str(item[1].get("audit_id", "")),
            ),
        )
    ]


def select_smoke_roles(audit_entries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select the four registered smoke roles in declared role-label order."""

    entries = [dict(entry) for entry in audit_entries]
    selected: list[dict[str, Any]] = []
    for role_label, station in DENSE_ROLE_STATIONS:
        candidates = _dense_candidates(entries, station)
        if not candidates:
            raise CorePlanError(
                f"No supported development selection_fit role exists for '{role_label}'."
            )
        selected.append({"role_label": role_label, "audit": candidates[0]})
    sparse = _sparse_candidates(entries)
    if not sparse:
        raise CorePlanError(
            f"No supported surface single-instrument selection_fit role exists for "
            f"'{SPARSE_ROLE_LABEL}'."
        )
    selected.append({"role_label": SPARSE_ROLE_LABEL, "audit": sparse[0]})
    return selected


# --------------------------------------------------------------------------- #
# Guard folds
# --------------------------------------------------------------------------- #


def _guard_role_definition(
    *,
    role_id: str,
    role: str,
    masters: list[str],
    rows_by_master: Mapping[str, list[dict[str, Any]]],
    classes: list[str],
) -> dict[str, Any]:
    uids = sorted(str(row["uid"]) for master in masters for row in rows_by_master[master])
    return {
        "guard_role_id": role_id,
        "role": role,
        "classes": list(classes),
        "class_count": len(classes),
        "master_count": len(masters),
        "observation_count": len(uids),
        "masters": list(masters),
        "fitting_uids": uids,
        "uid_set_sha256": uid_set_hash(uids),
    }


def build_guard_fold_roles(
    *,
    contract_sha256: str,
    context_id: str,
    station: str,
    outer_fit_rows: list[Mapping[str, Any]],
    held_instrument: str,
    outer_test_masters: Any = (),
) -> list[dict[str, Any]]:
    """Build three source-master-CV guard units from outer_fit metadata only."""

    rows_by_master: dict[str, list[dict[str, Any]]] = {}
    class_of: dict[str, str] = {}
    for row in outer_fit_rows:
        master = str(row["master"])
        if str(row["instrument"]) == held_instrument:
            raise CorePlanError("Guard construction found the held instrument in outer_fit.")
        target = str(row["target"])
        previous = class_of.get(master)
        if previous is not None and previous != target:
            raise CorePlanError("A guard master maps to more than one chemical class.")
        class_of[master] = target
        rows_by_master.setdefault(master, []).append(dict(row))
    outer_test = {str(master) for master in outer_test_masters}
    if set(rows_by_master) & outer_test:
        raise CorePlanError("Guard construction found an outer-test master inside outer_fit.")
    fold_of: dict[str, int] = {}
    for chemical in sorted(set(class_of.values())):
        masters = [master for master, value in class_of.items() if value == chemical]
        fold_of.update(assign_guard_folds(context_id=context_id, masters=masters))
    all_masters = set(rows_by_master)
    units: list[dict[str, Any]] = []
    for fold in range(GUARD_FOLD_COUNT):
        fitting_masters = sorted(master for master in all_masters if fold_of[master] != fold)
        validation_masters = sorted(master for master in all_masters if fold_of[master] == fold)
        if set(fitting_masters) & set(validation_masters):
            raise CorePlanError("Guard fitting and validation masters overlap.")
        if not (set(fitting_masters) | set(validation_masters)) <= all_masters:
            raise CorePlanError("Guard masters exceed the outer-fit master set.")
        fitting_classes = sorted({class_of[master] for master in fitting_masters})
        validation_classes = sorted({class_of[master] for master in validation_masters})
        support_ok = (
            len(fitting_classes) == REQUIRED_CLASSES
            and len(validation_classes) == REQUIRED_CLASSES
            and bool(fitting_masters)
            and bool(validation_masters)
        )
        reason = None if support_ok else "guard_fold_lacks_three_classes_in_one_or_both_roles"
        units.append(
            {
                "guard_unit_id": guard_unit_identity(
                    contract_sha256=contract_sha256, context_id=context_id, guard_fold=fold
                ),
                "context_id": context_id,
                "station": station,
                "guard_fold": fold,
                "support_ok": support_ok,
                "exclusion_reason": reason,
                "fitting_role": _guard_role_definition(
                    role_id=guard_role_identity(
                        contract_sha256=contract_sha256,
                        context_id=context_id,
                        guard_fold=fold,
                        role="fit",
                    ),
                    role="fit",
                    masters=fitting_masters,
                    rows_by_master=rows_by_master,
                    classes=fitting_classes,
                ),
                "validation_role": _guard_role_definition(
                    role_id=guard_role_identity(
                        contract_sha256=contract_sha256,
                        context_id=context_id,
                        guard_fold=fold,
                        role="validation",
                    ),
                    role="validation",
                    masters=validation_masters,
                    rows_by_master=rows_by_master,
                    classes=validation_classes,
                ),
            }
        )
    return units


# --------------------------------------------------------------------------- #
# Smoke fits, pairs, slots, budget
# --------------------------------------------------------------------------- #


def expand_smoke_fits(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    input_pins: Mapping[str, Any],
    selected_roles: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand 32 primary fits plus reserved replays with stable identities."""

    recipes = contract["recipes"]
    seeds = [int(seed) for seed in contract["smoke"]["seeds"]]
    epochs = int(contract["smoke"]["epochs"])
    draws = int(contract["sampler"]["batch_draws_per_epoch"])
    steps = epochs * draws
    fits: list[dict[str, Any]] = []
    primary: dict[tuple[str, str, int], dict[str, Any]] = {}
    for role_label in contract["smoke"]["role_labels"]:
        role_label = str(role_label)
        p05_role_id = str(selected_roles[role_label]["p05_role_id"])
        for recipe in recipes:
            recipe_id = str(recipe["recipe_id"])
            for seed in seeds:
                fit_id = fit_identity(
                    contract_sha256=contract_sha256,
                    input_pins=input_pins,
                    p05_role_id=p05_role_id,
                    recipe_id=recipe_id,
                    seed=seed,
                )
                row = {
                    "fit_id": fit_id,
                    "execution_id": execution_identity(fit_id=fit_id, execution_kind="primary"),
                    "execution_kind": "primary",
                    "replay_of": None,
                    "role_label": role_label,
                    "p05_role_id": p05_role_id,
                    "recipe_id": recipe_id,
                    "lambda_supcon": float(recipe["lambda_supcon"]),
                    "lambda_pair": float(recipe["lambda_pair"]),
                    "projection": bool(recipe["projection"]),
                    "seed": seed,
                    "epochs": epochs,
                    "batches_per_epoch": draws,
                    "optimizer_steps": steps,
                    "shared_rng_group": shared_rng_group(p05_role_id=p05_role_id, seed=seed),
                }
                fits.append(row)
                primary[(role_label, recipe_id, seed)] = row
    for replay_index, replay in enumerate(contract["smoke"]["replays"]):
        key = (
            str(replay["role_label"]),
            str(replay["recipe_id"]),
            int(replay["seed"]),
        )
        base = primary.get(key)
        if base is None:
            raise CorePlanError("A reserved smoke replay does not match a registered primary fit.")
        execution_id = execution_identity(
            fit_id=str(base["fit_id"]), execution_kind="replay", replay_index=replay_index
        )
        if execution_id == base["execution_id"]:
            raise CorePlanError("Replay execution identity collided with its primary identity.")
        row = dict(base)
        row["execution_id"] = execution_id
        row["execution_kind"] = "replay"
        row["replay_of"] = base["execution_id"]
        fits.append(row)
    return fits


def enumerate_smoke_pairs(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    input_pins: Mapping[str, Any],
    selected_roles: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Enumerate every eligible unordered same-chemical positive UID pair."""

    pairs: list[dict[str, Any]] = []
    for role_label in contract["smoke"]["role_labels"]:
        role_label = str(role_label)
        entry = selected_roles[role_label]
        p05_role_id = str(entry["p05_role_id"])
        rows = list(entry["rows"])
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                category = classify_positive_pair(left, right)
                if category is None:
                    continue
                pairs.append(
                    {
                        "pair_id": core_pair_identity(
                            contract_sha256=contract_sha256,
                            input_pins=input_pins,
                            p05_role_id=p05_role_id,
                            uid_a=str(left["uid"]),
                            uid_b=str(right["uid"]),
                        ),
                        "role_label": role_label,
                        "p05_role_id": p05_role_id,
                        "observation_uid_a": left["uid"],
                        "observation_uid_b": right["uid"],
                        "master_sample_id_a": left["master"],
                        "master_sample_id_b": right["master"],
                        "instrument_a": left["instrument"],
                        "instrument_b": right["instrument"],
                        "target_analyte": left["target"],
                        "pair_category": category,
                        "same_master": left["master"] == right["master"],
                        "same_instrument": left["instrument"] == right["instrument"],
                        "different_known_substrate_family": different_known_substrate_family(
                            left, right
                        ),
                        "paired_consistency_eligible": (
                            category == "same_master_different_instrument"
                        ),
                    }
                )
    return pairs


def _slot_record(
    *,
    contract_sha256: str,
    spec: Mapping[str, Any],
    recipe_id: str,
    seed: int,
) -> dict[str, Any]:
    guard_fold = spec.get("guard_fold")
    return {
        "slot_id": slot_identity(
            contract_sha256=contract_sha256,
            slot_kind=str(spec["slot_kind"]),
            context_id=str(spec["context_id"]),
            selection_unit_id=str(spec["selection_unit_id"]),
            fitting_role_id=str(spec["fitting_role_id"]),
            validation_role_id=str(spec["validation_role_id"]),
            guard_fold=guard_fold,
            recipe_id=recipe_id,
            seed=seed,
        ),
        "slot_kind": str(spec["slot_kind"]),
        "context_id": str(spec["context_id"]),
        "selection_unit_id": str(spec["selection_unit_id"]),
        "guard_fold": guard_fold,
        "fitting_role_id": str(spec["fitting_role_id"]),
        "validation_role_id": str(spec["validation_role_id"]),
        "recipe_id": recipe_id,
        "seed": int(seed),
        "planned": True,
        "excluded_by_protocol": bool(spec.get("excluded_by_protocol", False)),
        "exclusion_reason": spec.get("exclusion_reason"),
    }


def expand_development_slots(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    specs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand later-core slot specifications across four recipes and three seeds."""

    recipes = [str(recipe["recipe_id"]) for recipe in contract["recipes"]]
    seeds = [int(seed) for seed in contract["later_core_plan"]["seeds"]]
    slots: list[dict[str, Any]] = []
    for spec in specs:
        for recipe_id in recipes:
            for seed in seeds:
                slots.append(
                    _slot_record(
                        contract_sha256=contract_sha256,
                        spec=spec,
                        recipe_id=recipe_id,
                        seed=seed,
                    )
                )
    return slots


def reconcile_budget(*, counts: Mapping[str, Any], later_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile computed metadata counts against the locked later-core ledger."""

    def _require(name: str, observed: int, expected: int) -> None:
        if int(observed) != int(expected):
            raise CorePlanError(
                "Budget field '" + name + "' does not reconcile with the locked ledger."
            )

    inherited = int(counts["inherited_inner_units"])
    pseudo = int(counts["pseudo_domain_contexts"])
    master_cv = int(counts["master_cv_fallback_t3_contexts"])
    outer = int(counts["outer_contexts"])
    guard_units = int(counts["guard_unit_slots"])
    inner_slots = int(counts["inner_fit_slots"])
    eligible = int(counts["inner_fit_slots_eligible"])
    excluded = int(counts["inner_fit_slots_excluded"])
    _require("inherited_inner_units", inherited, later_plan["inherited_inner_units"])
    _require("pseudo_domain_contexts", pseudo, later_plan["pseudo_domain_contexts"])
    _require(
        "master_cv_fallback_t3_contexts",
        master_cv,
        later_plan["master_cv_fallback_t3_contexts"],
    )
    _require("outer_contexts", outer, later_plan["outer_contexts"])
    _require("extra_guard_unit_slots", guard_units, later_plan["extra_guard_unit_slots"])
    _require(
        "guard_unit_expansion",
        pseudo * int(later_plan["extra_guard_units_per_pseudo_domain_context"]),
        guard_units,
    )
    expected_inner = (inherited + guard_units) * LATER_RECIPE_COUNT * 3
    _require("inner_fit_slot_ceiling", expected_inner, later_plan["inner_fit_slot_ceiling"])
    _require("inner_fit_slots", inner_slots, expected_inner)
    if eligible + excluded != inner_slots:
        raise CorePlanError("Eligible and excluded inner slots must sum to the inner total.")
    later_neural = int(later_plan["inner_fit_slot_ceiling"]) + int(later_plan["refit_slot_ceiling"])
    _require("later_neural_fit_ceiling", later_neural, later_plan["later_neural_fit_ceiling"])
    _require(
        "including_smoke_neural_execution_ceiling",
        later_neural + CURRENT_SMOKE_EXECUTIONS,
        later_plan["including_smoke_neural_execution_ceiling"],
    )
    _require("retry_slots", int(later_plan["retry_slots"]), 0)
    _require("D4_slots", int(later_plan["D4_slots"]), 0)
    _require("D5_slots", int(later_plan["D5_slots"]), 0)
    return {
        "outer_contexts": outer,
        "inherited_inner_units": inherited,
        "pseudo_domain_contexts": pseudo,
        "master_cv_fallback_t3_contexts": master_cv,
        "guard_unit_slots": guard_units,
        "inner_fit_slots": inner_slots,
        "inner_fit_slots_eligible": eligible,
        "inner_fit_slots_excluded": excluded,
        "upper_refit_slots": int(later_plan["refit_slot_ceiling"]),
        "scalar_calibrations": int(later_plan["scalar_calibration_slot_ceiling"]),
        "later_neural_fits": later_neural,
        "including_smoke_neural_executions": later_neural + CURRENT_SMOKE_EXECUTIONS,
        "retry_slots": 0,
        "d4_slots": 0,
        "d5_slots": 0,
        "authorization": "ceiling_only_not_execution_permission",
    }


# --------------------------------------------------------------------------- #
# Input indexing and verification
# --------------------------------------------------------------------------- #


def _index_inputs(
    inputs: SupportInputs,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, str]]],
    dict[str, tuple[str, str, str]],
]:
    manifest: dict[str, dict[str, str]] = {}
    for row in inputs.manifest:
        uid = row["observation_uid"]
        manifest[uid] = {
            "uid": uid,
            "master": row["master_sample_id"],
            "station": row["station"],
            "target": row["target_analyte"],
            "instrument": row["instrument"],
            "substrate": row["sensor_family"],
        }
    contexts = {row["context_id"]: dict(row) for row in inputs.contexts}
    role_rows: dict[str, list[dict[str, str]]] = {}
    role_units: dict[str, tuple[str, str, str]] = {}
    for row in inputs.roles:
        role_id = row["role_id"]
        role_rows.setdefault(role_id, []).append(manifest[row["observation_uid"]])
        role_units[role_id] = (row["context_id"], row["role"], row["selection_unit_id"])
    for role_id in role_rows:
        role_rows[role_id].sort(key=lambda record: record["uid"])
    return manifest, contexts, role_rows, role_units


def _verify_caller_pins(inputs: SupportInputs, pins: Mapping[str, Any]) -> None:
    for attribute in ("manifest_sha256", "contexts_sha256", "roles_sha256"):
        observed = getattr(inputs, attribute)
        if observed is not None and str(observed).lower() != str(pins[attribute]).lower():
            raise CorePlanError(
                "A caller-pinned input digest does not match the locked core contract pin."
            )


def _verify_population(report: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    counts = report["counts"]
    population = contract["population"]
    if int(counts.get("manifest_rows", -1)) != int(population["rows"]):
        raise CorePlanError("Manifest row count does not match the pinned population.")
    if int(counts.get("master_samples", -1)) != int(population["masters"]):
        raise CorePlanError("Master count does not match the pinned population.")
    if int(counts.get("contexts", -1)) != int(contract["later_core_plan"]["outer_contexts"]):
        raise CorePlanError("Context count does not match the locked outer-context count.")


def _verify_smoke_support(selected: list[dict[str, Any]], contract: Mapping[str, Any]) -> None:
    smoke = contract["smoke"]
    labels = [str(label) for label in smoke["role_labels"]]
    if [record["role_label"] for record in selected] != labels:
        raise CorePlanError("Selected smoke roles do not follow the declared role-label order.")
    expected_masters = [int(value) for value in smoke["expected_master_counts"]]
    expected_observations = [int(value) for value in smoke["expected_observation_counts"]]
    for index, record in enumerate(selected):
        audit = record["audit"]
        try:
            masters = int(audit.get("master_count", -1))
            observations = int(audit.get("observation_count", -1))
        except (TypeError, ValueError) as error:
            raise CorePlanError(
                "A selected smoke role carries a non-integer support counter."
            ) from error
        if masters != expected_masters[index]:
            raise CorePlanError(
                "A selected smoke role master count disagrees with the locked contract."
            )
        if observations != expected_observations[index]:
            raise CorePlanError(
                "A selected smoke role observation count disagrees with the locked contract."
            )


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #


def build_core_plan(
    inputs: SupportInputs,
    *,
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the pinned contract and inherited support, then build the plan.

    The returned dictionary is JSON-compatible and contains no spectra,
    outcomes, predictions, source paths, or training implementation. It never
    authorizes execution; a separate supervisor gate is required.
    """

    summary = validate_core_contract(contract)
    contract_sha256 = str(summary["contract_sha256"])
    pins = dict(summary["input_pins"])
    _verify_caller_pins(inputs, pins)
    report = build_support_report(inputs, readiness=readiness)
    _verify_population(report, contract)
    selected = select_smoke_roles(report["source_role_audit"])
    _verify_smoke_support(selected, contract)
    manifest, contexts, role_rows, role_units = _index_inputs(inputs)

    # ------------------------------------------------------------ smoke roles
    smoke_roles: list[dict[str, Any]] = []
    smoke_observations: list[dict[str, Any]] = []
    selected_roles: dict[str, dict[str, Any]] = {}
    for record in selected:
        role_label = str(record["role_label"])
        audit = record["audit"]
        parent_role_id = str(audit["parent_role_id"])
        p05_role_id = core_role_identity(
            contract_sha256=contract_sha256,
            role_label=role_label,
            parent_context_id=str(audit["parent_context_id"]),
            parent_role_id=parent_role_id,
            role=str(audit["role"]),
            selection_unit_id=str(audit["selection_unit_id"]),
        )
        rows = role_rows[parent_role_id]
        uids = sorted(str(row["uid"]) for row in rows)
        selected_roles[role_label] = {"p05_role_id": p05_role_id, "rows": rows, "audit": audit}
        smoke_roles.append(
            {
                "role_label": role_label,
                "role": str(audit["role"]),
                "parent_context_id": str(audit["parent_context_id"]),
                "parent_role_id": parent_role_id,
                "selection_unit_id": str(audit["selection_unit_id"]),
                "p05_role_id": p05_role_id,
                "station": str(audit["station"]),
                "domain": str(audit.get("domain", "")),
                "phase_gate": str(audit.get("phase_gate", "")),
                "support": {key: audit.get(key) for key in SUPPORT_COUNTER_KEYS},
                "uid_set_sha256": uid_set_hash(uids),
                "fitting_uids": uids,
            }
        )
        for row in rows:
            smoke_observations.append(
                {
                    "role_label": role_label,
                    "p05_role_id": p05_role_id,
                    "uid": row["uid"],
                    "master": row["master"],
                    "station": row["station"],
                    "target": row["target"],
                    "instrument": row["instrument"],
                    "substrate": row["substrate"],
                }
            )

    # ------------------------------------------------------------ smoke fits
    smoke_fits = expand_smoke_fits(
        contract=contract,
        contract_sha256=contract_sha256,
        input_pins=pins,
        selected_roles=selected_roles,
    )
    primary_count = sum(1 for row in smoke_fits if row["execution_kind"] == "primary")
    if primary_count != int(contract["smoke"]["primary_fit_count"]):
        raise CorePlanError("Registered primary fit count disagrees with the contract.")
    if len(smoke_fits) != int(contract["smoke"]["maximum_fit_executions"]):
        raise CorePlanError("Registered fit execution count disagrees with the contract.")
    total_steps = sum(int(row["optimizer_steps"]) for row in smoke_fits)
    if total_steps != int(contract["smoke"]["maximum_optimizer_steps"]):
        raise CorePlanError("Registered optimizer-step count disagrees with the contract.")

    # ------------------------------------------------------------ smoke pairs
    smoke_pairs = enumerate_smoke_pairs(
        contract=contract,
        contract_sha256=contract_sha256,
        input_pins=pins,
        selected_roles=selected_roles,
    )

    # ------------------------------------------------------------ guard roles
    pseudo_contexts = sorted(
        context_id
        for context_id, context in contexts.items()
        if str(context.get("phase_gate")) == "held_evaluation"
        and str(context.get("selection_mode")) == "pseudo_domain"
    )
    outer_fit_roles: dict[str, str] = {}
    outer_test_roles: dict[str, str] = {}
    for role_id, (context_id, role, _unit) in role_units.items():
        if role == "outer_fit":
            outer_fit_roles[context_id] = role_id
        elif role == "outer_test":
            outer_test_roles[context_id] = role_id
    guard_roles: list[dict[str, Any]] = []
    for context_id in pseudo_contexts:
        context = contexts[context_id]
        outer_fit_rows = role_rows.get(outer_fit_roles.get(context_id, ""), [])
        outer_test_rows = role_rows.get(outer_test_roles.get(context_id, ""), [])
        units = build_guard_fold_roles(
            contract_sha256=contract_sha256,
            context_id=context_id,
            station=str(context["station"]),
            outer_fit_rows=list(outer_fit_rows),
            held_instrument=str(context["held_instrument"]),
            outer_test_masters=sorted(str(row["master"]) for row in outer_test_rows),
        )
        guard_roles.extend(units)

    # ------------------------------------------------------- development slots
    inherited_specs: list[dict[str, Any]] = []
    units_roles: dict[tuple[str, str], dict[str, str]] = {}
    for role_id, (context_id, role, unit_id) in role_units.items():
        units_roles.setdefault((context_id, unit_id), {})[role] = role_id
    for (context_id, unit_id), roles in sorted(units_roles.items()):
        fitting = roles.get("selection_fit")
        validation = roles.get("selection_validation")
        if fitting is None or validation is None:
            continue
        inherited_specs.append(
            {
                "slot_kind": "inherited_selection_fit",
                "context_id": context_id,
                "selection_unit_id": unit_id,
                "fitting_role_id": fitting,
                "validation_role_id": validation,
                "guard_fold": None,
                "excluded_by_protocol": False,
                "exclusion_reason": None,
            }
        )
    guard_specs: list[dict[str, Any]] = []
    for unit in guard_roles:
        guard_specs.append(
            {
                "slot_kind": "guard_selection_fit",
                "context_id": unit["context_id"],
                "selection_unit_id": unit["guard_unit_id"],
                "fitting_role_id": unit["fitting_role"]["guard_role_id"],
                "validation_role_id": unit["validation_role"]["guard_role_id"],
                "guard_fold": unit["guard_fold"],
                "excluded_by_protocol": not unit["support_ok"],
                "exclusion_reason": unit["exclusion_reason"],
            }
        )
    development_slots = expand_development_slots(
        contract=contract,
        contract_sha256=contract_sha256,
        specs=inherited_specs + guard_specs,
    )
    excluded_slots = sum(1 for slot in development_slots if slot["excluded_by_protocol"])

    # ------------------------------------------------------------ budget
    master_cv_contexts = sum(
        1
        for context in contexts.values()
        if str(context.get("phase_gate")) == "held_evaluation"
        and str(context.get("selection_mode")) == "master_cv"
    )
    budget = reconcile_budget(
        counts={
            "outer_contexts": len(contexts),
            "inherited_inner_units": len(inherited_specs),
            "pseudo_domain_contexts": len(pseudo_contexts),
            "master_cv_fallback_t3_contexts": master_cv_contexts,
            "guard_unit_slots": len(guard_roles),
            "inner_fit_slots": len(development_slots),
            "inner_fit_slots_eligible": len(development_slots) - excluded_slots,
            "inner_fit_slots_excluded": excluded_slots,
        },
        later_plan=contract["later_core_plan"],
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "contract_sha256": contract_sha256,
        "authorized_stage": summary["authorized_stage"],
        "input_pins": pins,
        "population": dict(contract["population"]),
        "smoke_roles": smoke_roles,
        "smoke_observations": smoke_observations,
        "smoke_fits": smoke_fits,
        "smoke_pairs": smoke_pairs,
        "development_slots": development_slots,
        "guard_roles": guard_roles,
        "budget": budget,
        "validation": {
            "metadata_only": True,
            "no_fits_performed": True,
            "smoke_execution_requires_supervisor_gate": True,
            "full_development_authorized": False,
            "outer_evaluation_authorized": False,
            "contract_sha256": contract_sha256,
            "input_hashes": {
                "manifest_sha256": pins["manifest_sha256"],
                "contexts_sha256": pins["contexts_sha256"],
                "roles_sha256": pins["roles_sha256"],
                "representation_sha256": pins["representation_sha256"],
            },
            "smoke_role_count": len(smoke_roles),
            "smoke_observation_count": len(smoke_observations),
            "smoke_fit_count": len(smoke_fits),
            "smoke_pair_count": len(smoke_pairs),
            "guard_unit_count": len(guard_roles),
            "development_slot_count": len(development_slots),
            "development_slots_excluded": excluded_slots,
            "checks": {
                "contract_hash_verified": True,
                "ledger_reconciled": True,
                "population_verified": True,
                "smoke_counts_verified": True,
                "guard_folds_disjoint_and_contained": True,
                "guard_units_pseudo_domain_only": True,
            },
        },
    }
