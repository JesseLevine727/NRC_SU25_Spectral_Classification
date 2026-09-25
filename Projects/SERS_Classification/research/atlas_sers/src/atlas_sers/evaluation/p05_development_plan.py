"""Deterministic metadata-only P05 development ledger.

This module converts the immutable P05 core plan's development slots and
source-master guard memberships into an executable metadata ledger. It reads
only authenticated metadata: no spectra, intensities, predictions, checkpoints,
leases or training outcomes are read, produced or authorized.

Binding boundary: the builder hashes the supplied contract, requires the
supplied plan to carry the same contract digest and the same ``input_pins``,
and requires any caller-pinned support digests to match those pins. It then
re-derives every membership from the source metadata. It does not claim
independent authentication of the upstream registry; the read-only CLI always
re-authenticates the pinned original contract and compares the rebuilt plan
against the pinned plan identifier.

Source-only boundary: every fitting and validation observation must belong to
the context's single ``outer_fit`` role. Outer-test masters and the outer held
instrument are rejected in either role.

Legacy naming: guard role definitions expose their observation UIDs under the
historical ``fitting_uids`` key for BOTH the fitting and the validation role.
The key name is inherited naming, not meaning; each role's UID list is
role-local and is re-derived from the manifest before use.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from atlas_sers.evaluation.p05_core_plan import (
    PROTOCOL_VERSION,
    build_guard_fold_roles,
    canonical_identity,
    slot_identity,
)
from atlas_sers.evaluation.p05_support import (
    OUTER_UNIT_BY_ROLE,
    ROLE_NAMES,
    SELECTION_MODE_UNIT_PREFIX,
    SELECTION_MODES_BY_PHASE,
    sha256_value,
    uid_set_hash,
)

SCHEMA_VERSION = "nato-sers-p05-development-ledger-v1"
LEDGER_ID_VERSION = "p05-development-ledger-v1"
UNIT_ID_VERSION = "p05-development-unit-v1"

PINNED_CONTRACT_SHA256 = "60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae"
PINNED_PLAN_ID = "a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37"

INHERITED_SLOT_KIND = "inherited_selection_fit"
GUARD_SLOT_KIND = "guard_selection_fit"

REQUIRED_RECIPE_COUNT = 4
REQUIRED_SEED_COUNT = 3
REQUIRED_CLASSES = 3
MAX_SAMPLING_CAPACITY = 48
BYTES_PER_PARAMETER = 4

PSEUDO_UNIT_PREFIX = SELECTION_MODE_UNIT_PREFIX["pseudo_domain"]

RECONCILIATION_KEYS = (
    "inherited_inner_units",
    "extra_guard_unit_slots",
    "inner_fit_slot_ceiling",
    "outer_contexts",
    "pseudo_domain_contexts",
    "master_cv_fallback_t3_contexts",
)


class DevelopmentLedgerError(ValueError):
    """Stable, stage-specific failure with a path-free reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


# --------------------------------------------------------------------------- #
# Canonical identities
# --------------------------------------------------------------------------- #


def development_unit_identity(
    *,
    contract_sha256: str,
    plan_id: str,
    context_id: str,
    selection_unit_id: str,
    unit_kind: str,
) -> str:
    """Return the contract- and plan-bound canonical unit identifier."""

    return canonical_identity(
        "P05DEVUNIT",
        UNIT_ID_VERSION,
        {
            "contract_sha256": str(contract_sha256),
            "plan_id": str(plan_id),
            "context_id": str(context_id),
            "selection_unit_id": str(selection_unit_id),
            "unit_kind": str(unit_kind),
        },
    )


def development_ledger_identity(
    *,
    contract_sha256: str,
    plan_id: str,
    input_pins: Mapping[str, Any],
    units_sha256: str,
    slots_sha256: str,
) -> str:
    """Return the canonical ledger identifier binding every semantic record."""

    return canonical_identity(
        "P05DEV",
        LEDGER_ID_VERSION,
        {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": str(contract_sha256),
            "plan_id": str(plan_id),
            "input_pins": dict(input_pins),
            "units_sha256": str(units_sha256),
            "slots_sha256": str(slots_sha256),
        },
    )


# --------------------------------------------------------------------------- #
# Small typed helpers
# --------------------------------------------------------------------------- #


def _require_mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentLedgerError(code)
    return value


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DevelopmentLedgerError(code)
    return value


def _input_pins(contract: Mapping[str, Any]) -> dict[str, Any]:
    pins = contract.get("input_pins")
    if not isinstance(pins, Mapping):
        raise DevelopmentLedgerError("contract_input_pins_malformed")
    return dict(pins)


def _recipes(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = contract.get("recipes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DevelopmentLedgerError("contract_recipes_malformed")
    recipes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise DevelopmentLedgerError("contract_recipes_malformed")
        recipe_id = _text(entry.get("recipe_id"), "contract_recipe_id_malformed")
        if recipe_id in seen:
            raise DevelopmentLedgerError("contract_recipe_duplicate")
        seen.add(recipe_id)
        recipes.append(
            {
                "recipe_id": recipe_id,
                "projection": bool(entry.get("projection", False)),
            }
        )
    if len(recipes) != REQUIRED_RECIPE_COUNT:
        raise DevelopmentLedgerError("contract_recipe_count_mismatch")
    return recipes


def _seeds(contract: Mapping[str, Any]) -> list[int]:
    later = _require_mapping(
        contract.get("later_core_plan"), "contract_later_plan_malformed"
    )
    raw = later.get("seeds")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DevelopmentLedgerError("contract_seeds_malformed")
    seeds: list[int] = []
    for seed in raw:
        value = _integer(seed)
        if value is None:
            raise DevelopmentLedgerError("contract_seeds_malformed")
        seeds.append(value)
    if len(seeds) != REQUIRED_SEED_COUNT or len(set(seeds)) != REQUIRED_SEED_COUNT:
        raise DevelopmentLedgerError("contract_seed_count_mismatch")
    return sorted(seeds)


def _model(contract: Mapping[str, Any]) -> dict[str, int]:
    model = _require_mapping(contract.get("model"), "contract_model_malformed")
    base = _integer(model.get("base_parameters"))
    projection = _integer(model.get("projection_model_parameters"))
    if base is None or projection is None or base <= 0 or projection <= 0:
        raise DevelopmentLedgerError("contract_model_malformed")
    return {"base_parameters": base, "projection_model_parameters": projection}


def _optimizer_schedule(contract: Mapping[str, Any]) -> tuple[int, int]:
    """Return the per-fit scheduled optimizer-update floor and ceiling.

    The schedule is owned by the locked contract's ``later_core_plan``:
    ``minimum_epochs`` and ``maximum_epochs`` multiplied by
    ``sampler.batch_draws_per_epoch``. No alternate key spellings are accepted.
    """

    later = _require_mapping(
        contract.get("later_core_plan"), "contract_later_plan_malformed"
    )
    sampler = _require_mapping(contract.get("sampler"), "contract_sampler_malformed")
    minimum_epochs = _integer(later.get("minimum_epochs"))
    maximum_epochs = _integer(later.get("maximum_epochs"))
    draws = _integer(sampler.get("batch_draws_per_epoch"))
    if minimum_epochs is None or maximum_epochs is None or draws is None:
        raise DevelopmentLedgerError("contract_optimizer_schedule_malformed")
    if minimum_epochs <= 0 or maximum_epochs < minimum_epochs or draws <= 0:
        raise DevelopmentLedgerError("contract_optimizer_schedule_malformed")
    return minimum_epochs * draws, maximum_epochs * draws


# --------------------------------------------------------------------------- #
# Support indexing and pin binding
# --------------------------------------------------------------------------- #


def _check_support_pins(support: Any, pins: Mapping[str, Any]) -> None:
    for attribute, key in (
        ("manifest_sha256", "manifest_sha256"),
        ("contexts_sha256", "contexts_sha256"),
        ("roles_sha256", "roles_sha256"),
    ):
        if key not in pins:
            raise DevelopmentLedgerError("contract_input_pins_malformed")
        observed = getattr(support, attribute, None)
        if not isinstance(observed, str) or not observed:
            raise DevelopmentLedgerError("support_pin_missing")
        if observed.lower() != str(pins[key]).lower():
            raise DevelopmentLedgerError("support_pin_mismatch")


def _manifest_index(support: Any) -> dict[str, dict[str, str]]:
    rows = getattr(support, "manifest", None)
    if not rows:
        raise DevelopmentLedgerError("support_manifest_empty")
    index: dict[str, dict[str, str]] = {}
    master_station: dict[str, str] = {}
    master_target: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DevelopmentLedgerError("support_manifest_malformed")
        uid = _text(row.get("observation_uid"), "support_manifest_uid_malformed")
        master = _text(row.get("master_sample_id"), "support_manifest_master_malformed")
        station = _text(row.get("station"), "support_manifest_station_malformed")
        target = _text(row.get("target_analyte"), "support_manifest_target_malformed")
        instrument = _text(
            row.get("instrument"), "support_manifest_instrument_malformed"
        )
        family = row.get("sensor_family")
        if not isinstance(family, str) or family != family.strip():
            raise DevelopmentLedgerError("support_manifest_substrate_malformed")
        if uid in index:
            raise DevelopmentLedgerError("support_manifest_uid_duplicate")
        previous_station = master_station.get(master)
        if previous_station is not None and previous_station != station:
            raise DevelopmentLedgerError("support_master_station_conflict")
        previous_target = master_target.get(master)
        if previous_target is not None and previous_target != target:
            raise DevelopmentLedgerError("support_master_target_conflict")
        master_station[master] = station
        master_target[master] = target
        index[uid] = {
            "uid": uid,
            "master": master,
            "station": station,
            "target": target,
            "instrument": instrument,
            "substrate": family,
        }
    return index


def _context_index(support: Any) -> dict[str, dict[str, str]]:
    rows = getattr(support, "contexts", None)
    if not rows:
        raise DevelopmentLedgerError("support_contexts_empty")
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DevelopmentLedgerError("support_context_malformed")
        context_id = _text(row.get("context_id"), "support_context_malformed")
        if context_id in index:
            raise DevelopmentLedgerError("support_context_duplicate")
        station = _text(row.get("station"), "support_context_malformed")
        held_instrument = _text(
            row.get("held_instrument"), "support_context_malformed"
        )
        selection_mode = _text(row.get("selection_mode"), "support_context_malformed")
        phase_gate = _text(row.get("phase_gate"), "support_context_malformed")
        if phase_gate not in SELECTION_MODES_BY_PHASE:
            raise DevelopmentLedgerError("unknown_phase_gate")
        if selection_mode not in SELECTION_MODE_UNIT_PREFIX:
            raise DevelopmentLedgerError("unknown_selection_mode")
        if selection_mode not in SELECTION_MODES_BY_PHASE[phase_gate]:
            raise DevelopmentLedgerError("selection_mode_phase_conflict")
        index[context_id] = {
            "context_id": context_id,
            "station": station,
            "held_instrument": held_instrument,
            "selection_mode": selection_mode,
            "phase_gate": phase_gate,
        }
    return index


def _role_index(
    support: Any,
    manifest: Mapping[str, Mapping[str, str]],
    contexts: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, tuple[str, str, str]], dict[str, list[str]]]:
    rows = getattr(support, "roles", None)
    if not rows:
        raise DevelopmentLedgerError("support_roles_empty")
    meta: dict[str, tuple[str, str, str]] = {}
    uids_by_role: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DevelopmentLedgerError("support_role_malformed")
        context_id = _text(row.get("context_id"), "support_role_malformed")
        role_id = _text(row.get("role_id"), "support_role_malformed")
        role = _text(row.get("role"), "support_role_malformed")
        unit_id = _text(row.get("selection_unit_id"), "support_role_malformed")
        uid = _text(row.get("observation_uid"), "support_role_malformed")
        if role not in ROLE_NAMES:
            raise DevelopmentLedgerError("unknown_role")
        context = contexts.get(context_id)
        if context is None:
            raise DevelopmentLedgerError("unknown_context")
        source = manifest.get(uid)
        if source is None:
            raise DevelopmentLedgerError("role_uid_unknown")
        if (
            row.get("master_sample_id") != source["master"]
            or row.get("target_analyte") != source["target"]
            or row.get("instrument") != source["instrument"]
        ):
            raise DevelopmentLedgerError("role_metadata_mismatch")
        if source["station"] != context["station"]:
            raise DevelopmentLedgerError("role_station_mismatch")
        expected_outer_unit = OUTER_UNIT_BY_ROLE.get(role)
        if expected_outer_unit is not None:
            if unit_id != expected_outer_unit:
                raise DevelopmentLedgerError("outer_role_unit_mismatch")
        else:
            prefix = SELECTION_MODE_UNIT_PREFIX[context["selection_mode"]]
            if not unit_id.startswith(prefix) or len(unit_id) == len(prefix):
                raise DevelopmentLedgerError("selection_unit_prefix_mismatch")
        key = (context_id, role, unit_id)
        previous = meta.get(role_id)
        if previous is None:
            meta[role_id] = key
        elif previous != key:
            raise DevelopmentLedgerError("role_identity_inconsistent")
        uids_by_role.setdefault(role_id, []).append(uid)
    for role_id, uids in uids_by_role.items():
        if len(set(uids)) != len(uids):
            raise DevelopmentLedgerError("role_uid_duplicate")
        uids_by_role[role_id] = sorted(uids)
    return meta, uids_by_role


def _single_outer_roles(
    meta: Mapping[str, tuple[str, str, str]],
    contexts: Mapping[str, Any],
    role_name: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for role_id, (context_id, role, _unit) in meta.items():
        if role != role_name:
            continue
        if context_id in result:
            raise DevelopmentLedgerError("duplicate_outer_role")
        result[context_id] = role_id
    for context_id in contexts:
        if context_id not in result:
            raise DevelopmentLedgerError("missing_outer_role")
    return result


def _inherited_pairs(
    meta: Mapping[str, tuple[str, str, str]]
) -> set[tuple[str, str]]:
    roles_by_unit: dict[tuple[str, str], dict[str, str]] = {}
    for role_id, (context_id, role, unit_id) in meta.items():
        if role not in ("selection_fit", "selection_validation"):
            continue
        unit_roles = roles_by_unit.setdefault((context_id, unit_id), {})
        previous = unit_roles.get(role)
        if previous is not None and previous != role_id:
            raise DevelopmentLedgerError("duplicate_selection_role")
        unit_roles[role] = role_id
    pairs: set[tuple[str, str]] = set()
    for key, roles in roles_by_unit.items():
        if set(roles) != {"selection_fit", "selection_validation"}:
            raise DevelopmentLedgerError("selection_unit_incomplete")
        pairs.add(key)
    return pairs


def _station_classes(
    manifest: Mapping[str, Mapping[str, str]]
) -> dict[str, frozenset[str]]:
    classes: dict[str, set[str]] = {}
    for row in manifest.values():
        classes.setdefault(row["station"], set()).add(row["target"])
    return {station: frozenset(values) for station, values in classes.items()}


# --------------------------------------------------------------------------- #
# Guard reconstruction
# --------------------------------------------------------------------------- #


def _expected_guard_units(
    *,
    contract_sha256: str,
    contexts: Mapping[str, Mapping[str, str]],
    fit_uids: Mapping[str, set[str]],
    test_uids: Mapping[str, set[str]],
    manifest: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Rebuild the expected guard units from source metadata only."""

    units: dict[str, dict[str, Any]] = {}
    for context_id, context in contexts.items():
        if (
            context["phase_gate"] != "held_evaluation"
            or context["selection_mode"] != "pseudo_domain"
        ):
            continue
        outer_fit_rows = [manifest[uid] for uid in sorted(fit_uids[context_id])]
        outer_test_masters = sorted(
            {manifest[uid]["master"] for uid in test_uids[context_id]}
        )
        try:
            built = build_guard_fold_roles(
                contract_sha256=contract_sha256,
                context_id=context_id,
                station=context["station"],
                outer_fit_rows=outer_fit_rows,
                held_instrument=context["held_instrument"],
                outer_test_masters=outer_test_masters,
            )
        except Exception as error:  # noqa: BLE001 - normalized to a stable code
            raise DevelopmentLedgerError("guard_rebuild_failed") from error
        for unit in built:
            unit_id = str(unit["guard_unit_id"])
            if unit_id in units:
                raise DevelopmentLedgerError("duplicate_guard_unit")
            units[unit_id] = dict(unit)
    return units


def _plan_guard_units(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = plan.get("guard_roles")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DevelopmentLedgerError("plan_guard_roles_malformed")
    units: dict[str, dict[str, Any]] = {}
    for unit in raw:
        if not isinstance(unit, Mapping):
            raise DevelopmentLedgerError("plan_guard_unit_malformed")
        unit_id = _text(unit.get("guard_unit_id"), "plan_guard_unit_malformed")
        if unit_id in units:
            raise DevelopmentLedgerError("duplicate_guard_unit")
        units[unit_id] = dict(unit)
    return units


# --------------------------------------------------------------------------- #
# Auxiliary support
# --------------------------------------------------------------------------- #


def _auxiliary_support(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    """Classify fitting-only auxiliary support.

    ``same_chemical_positive_pairs`` counts unordered same-target pairs and
    excludes same-master/same-instrument stored repeats. ``cross_instrument_
    master_pairs`` counts same-master pairs measured on different instruments.
    Absence of either is metadata, not an exclusion.
    """

    by_target = Counter(row["target"] for row in rows)
    same_chemical_all = sum(
        count * (count - 1) // 2 for count in by_target.values()
    )
    by_cell = Counter((row["master"], row["instrument"]) for row in rows)
    same_master_same_instrument = sum(
        count * (count - 1) // 2 for count in by_cell.values()
    )
    per_master: dict[str, Counter[str]] = {}
    for row in rows:
        per_master.setdefault(row["master"], Counter())[row["instrument"]] += 1
    cross_instrument = 0
    for counts in per_master.values():
        total = sum(counts.values())
        pairs = total * (total - 1) // 2
        for count in counts.values():
            pairs -= count * (count - 1) // 2
        cross_instrument += pairs
    return {
        "same_chemical_positive_pairs": (
            same_chemical_all - same_master_same_instrument
        ),
        "cross_instrument_master_pairs": cross_instrument,
    }


# --------------------------------------------------------------------------- #
# Summary and reconciliation
# --------------------------------------------------------------------------- #


def _build_summary(
    *,
    contexts: Mapping[str, Mapping[str, str]],
    units: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
    recipes: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    model: Mapping[str, int],
    minimum_updates: int,
    maximum_updates: int,
) -> dict[str, Any]:
    modes = Counter(context["selection_mode"] for context in contexts.values())
    phases = Counter(context["phase_gate"] for context in contexts.values())
    eligible_slots = [slot for slot in slots if not slot["excluded_by_protocol"]]
    excluded_slots = [slot for slot in slots if slot["excluded_by_protocol"]]

    parameters_by_recipe = {
        recipe["recipe_id"]: (
            int(model["projection_model_parameters"])
            if recipe["projection"]
            else int(model["base_parameters"])
        )
        for recipe in recipes
    }
    eligible_by_recipe = Counter(str(slot["recipe_id"]) for slot in eligible_slots)
    total_checkpoint = sum(
        count * parameters_by_recipe[recipe_id] * BYTES_PER_PARAMETER
        for recipe_id, count in eligible_by_recipe.items()
    )
    maximum_checkpoint = max(
        (
            parameters_by_recipe[recipe_id] * BYTES_PER_PARAMETER
            for recipe_id in eligible_by_recipe
        ),
        default=0,
    )

    return {
        "context_count": len(contexts),
        "contexts_by_selection_mode": dict(sorted(modes.items())),
        "contexts_by_phase_gate": dict(sorted(phases.items())),
        "pseudo_domain_context_count": int(modes.get("pseudo_domain", 0)),
        "master_cv_context_count": int(modes.get("master_cv", 0)),
        "inner_master_cv_context_count": int(modes.get("inner_master_cv", 0)),
        "inherited_unit_count": sum(
            1 for unit in units if unit["unit_kind"] == "inherited"
        ),
        "guard_unit_count": sum(
            1 for unit in units if unit["unit_kind"] == "guard"
        ),
        "eligible_unit_count": sum(
            1 for unit in units if not unit["excluded_by_protocol"]
        ),
        "excluded_unit_count": sum(
            1 for unit in units if unit["excluded_by_protocol"]
        ),
        "slot_count": len(slots),
        "eligible_slot_count": len(eligible_slots),
        "excluded_slot_count": len(excluded_slots),
        "recipe_count": len(recipes),
        "seed_count": len(seeds),
        "maximum_masters_per_unit": max(
            (int(unit["master_count"]) for unit in units), default=0
        ),
        "maximum_fitting_masters": max(
            (int(unit["fitting_master_count"]) for unit in units), default=0
        ),
        "maximum_validation_masters": max(
            (int(unit["validation_master_count"]) for unit in units), default=0
        ),
        "maximum_rows_per_unit": max(
            (int(unit["row_count"]) for unit in units), default=0
        ),
        "maximum_sampling_capacity": max(
            (int(unit["sampling_capacity"]) for unit in units), default=0
        ),
        "sampling_capacity_ceiling": MAX_SAMPLING_CAPACITY,
        "minimum_optimizer_updates_per_fit": int(minimum_updates),
        "maximum_optimizer_updates_per_fit": int(maximum_updates),
        "minimum_scheduled_optimizer_updates": (
            len(eligible_slots) * int(minimum_updates)
        ),
        "maximum_scheduled_optimizer_updates": (
            len(eligible_slots) * int(maximum_updates)
        ),
        "checkpoint_tensor_bytes_per_fit_max": maximum_checkpoint,
        "checkpoint_tensor_bytes_total_eligible": total_checkpoint,
        "checkpoint_estimate_note": (
            "Tensor-only floor from exact parameter counts and float32 width; "
            "not total checkpoint disk allocation, optimizer state, or "
            "concurrency."
        ),
        "execution_authorized": False,
        "arrays_loaded": False,
        "fits_started": 0,
        "outer_evaluation_authorized": False,
    }


def _reconcile_contract_counts(
    summary: Mapping[str, Any], later_plan: Any
) -> None:
    later = _require_mapping(later_plan, "contract_later_plan_malformed")
    observed = {
        "inherited_inner_units": summary["inherited_unit_count"],
        "extra_guard_unit_slots": summary["guard_unit_count"],
        "inner_fit_slot_ceiling": summary["slot_count"],
        "outer_contexts": summary["context_count"],
        "pseudo_domain_contexts": summary["pseudo_domain_context_count"],
        "master_cv_fallback_t3_contexts": summary["master_cv_context_count"],
    }
    for key in RECONCILIATION_KEYS:
        if key not in later:
            raise DevelopmentLedgerError("contract_reconciliation_key_missing")
        if int(later[key]) != int(observed[key]):
            raise DevelopmentLedgerError("contract_count_reconciliation_failed")


# --------------------------------------------------------------------------- #
# Slot validation
# --------------------------------------------------------------------------- #


def _group_slots(plan: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    slots = plan.get("development_slots")
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise DevelopmentLedgerError("plan_slots_malformed")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise DevelopmentLedgerError("plan_slot_malformed")
        context_id = _text(slot.get("context_id"), "plan_slot_field_malformed")
        unit_id = _text(slot.get("selection_unit_id"), "plan_slot_field_malformed")
        groups.setdefault((context_id, unit_id), []).append(dict(slot))
    return groups


def _validate_slot(
    slot: Mapping[str, Any], contract_sha256: str
) -> dict[str, Any]:
    slot_id = _text(slot.get("slot_id"), "plan_slot_field_malformed")
    slot_kind = _text(slot.get("slot_kind"), "plan_slot_field_malformed")
    if slot_kind not in (INHERITED_SLOT_KIND, GUARD_SLOT_KIND):
        raise DevelopmentLedgerError("unknown_slot_kind")
    context_id = _text(slot.get("context_id"), "plan_slot_field_malformed")
    selection_unit_id = _text(
        slot.get("selection_unit_id"), "plan_slot_field_malformed"
    )
    fitting_role_id = _text(
        slot.get("fitting_role_id"), "plan_slot_field_malformed"
    )
    validation_role_id = _text(
        slot.get("validation_role_id"), "plan_slot_field_malformed"
    )
    recipe_id = _text(slot.get("recipe_id"), "plan_slot_field_malformed")
    seed = _integer(slot.get("seed"))
    if seed is None:
        raise DevelopmentLedgerError("plan_slot_seed_malformed")
    if slot.get("planned") is not True:
        raise DevelopmentLedgerError("plan_slot_planned_flag_malformed")
    excluded = slot.get("excluded_by_protocol")
    if not isinstance(excluded, bool):
        raise DevelopmentLedgerError("plan_slot_exclusion_flag_malformed")
    reason = slot.get("exclusion_reason")
    if reason is not None and not isinstance(reason, str):
        raise DevelopmentLedgerError("plan_slot_exclusion_reason_malformed")
    guard_fold = slot.get("guard_fold")
    if guard_fold is not None and _integer(guard_fold) is None:
        raise DevelopmentLedgerError("plan_slot_guard_fold_malformed")
    expected_id = slot_identity(
        contract_sha256=contract_sha256,
        slot_kind=slot_kind,
        context_id=context_id,
        selection_unit_id=selection_unit_id,
        fitting_role_id=fitting_role_id,
        validation_role_id=validation_role_id,
        guard_fold=guard_fold,
        recipe_id=recipe_id,
        seed=seed,
    )
    if slot_id != expected_id:
        raise DevelopmentLedgerError("slot_identity_mismatch")
    return {
        "slot_id": slot_id,
        "slot_kind": slot_kind,
        "context_id": context_id,
        "selection_unit_id": selection_unit_id,
        "fitting_role_id": fitting_role_id,
        "validation_role_id": validation_role_id,
        "recipe_id": recipe_id,
        "seed": seed,
        "guard_fold": guard_fold,
        "excluded_by_protocol": excluded,
        "exclusion_reason": reason,
    }


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #


def build_development_ledger(
    *,
    plan: Mapping[str, Any],
    support: Any,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the executable development metadata ledger.

    The returned dictionary is JSON-compatible and private: it carries canonical
    unit and slot identities, role-bound UID lists and digests, and the source
    boundary checks. It never contains spectra, predictions, file paths, or
    training outcomes, and it never authorizes execution.
    """

    plan = _require_mapping(plan, "plan_malformed")
    contract = _require_mapping(contract, "contract_malformed")
    contract_sha256 = sha256_value(dict(contract))
    if plan.get("contract_sha256") != contract_sha256:
        raise DevelopmentLedgerError("plan_contract_identity_mismatch")
    pins = _input_pins(contract)
    plan_pins = plan.get("input_pins")
    if not isinstance(plan_pins, Mapping) or dict(plan_pins) != pins:
        raise DevelopmentLedgerError("plan_pin_mismatch")
    _check_support_pins(support, pins)
    plan_id = sha256_value(dict(plan))

    recipes = _recipes(contract)
    recipe_ids = [recipe["recipe_id"] for recipe in recipes]
    seeds = _seeds(contract)
    model = _model(contract)
    minimum_updates, maximum_updates = _optimizer_schedule(contract)

    manifest = _manifest_index(support)
    contexts = _context_index(support)
    role_meta, uids_by_role = _role_index(support, manifest, contexts)

    fit_roles = _single_outer_roles(role_meta, contexts, "outer_fit")
    test_roles = _single_outer_roles(role_meta, contexts, "outer_test")
    fit_uids = {ctx: set(uids_by_role[rid]) for ctx, rid in fit_roles.items()}
    test_uids = {ctx: set(uids_by_role[rid]) for ctx, rid in test_roles.items()}
    fit_masters = {
        ctx: {manifest[uid]["master"] for uid in uids} for ctx, uids in fit_uids.items()
    }
    test_masters = {
        ctx: {manifest[uid]["master"] for uid in uids}
        for ctx, uids in test_uids.items()
    }
    for context_id in contexts:
        if fit_masters[context_id] & test_masters[context_id]:
            raise DevelopmentLedgerError("outer_master_overlap")

    station_classes = _station_classes(manifest)
    expected_pairs = _inherited_pairs(role_meta)

    expected_guard = _expected_guard_units(
        contract_sha256=contract_sha256,
        contexts=contexts,
        fit_uids=fit_uids,
        test_uids=test_uids,
        manifest=manifest,
    )
    supplied_guard = _plan_guard_units(plan)
    if set(supplied_guard) != set(expected_guard):
        raise DevelopmentLedgerError("guard_unit_coverage_mismatch")
    for unit_id, unit in supplied_guard.items():
        if sha256_value(unit) != sha256_value(expected_guard[unit_id]):
            raise DevelopmentLedgerError("guard_unit_mismatch")

    groups = _group_slots(plan)
    if not groups:
        raise DevelopmentLedgerError("plan_has_no_slots")
    plan_pairs: set[tuple[str, str]] = set()
    plan_guard_ids: set[str] = set()
    for (context_id, unit_id), group in groups.items():
        kinds = {slot.get("slot_kind") for slot in group}
        if kinds == {INHERITED_SLOT_KIND}:
            plan_pairs.add((context_id, unit_id))
        elif kinds == {GUARD_SLOT_KIND}:
            plan_guard_ids.add(unit_id)
        else:
            raise DevelopmentLedgerError("unit_slot_kind_inconsistent")
    if plan_pairs != expected_pairs:
        raise DevelopmentLedgerError("inherited_unit_coverage_mismatch")
    if plan_guard_ids != set(expected_guard):
        raise DevelopmentLedgerError("guard_unit_coverage_mismatch")

    seen_slot_ids: set[str] = set()
    seen_unit_ids: set[str] = set()
    units: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []

    for (context_id, selection_unit_id), group in sorted(groups.items()):
        normalized = [_validate_slot(slot, contract_sha256) for slot in group]
        normalized.sort(key=lambda slot: (slot["recipe_id"], slot["seed"]))

        kinds = {slot["slot_kind"] for slot in normalized}
        if kinds == {INHERITED_SLOT_KIND}:
            unit_kind = "inherited"
        elif kinds == {GUARD_SLOT_KIND}:
            unit_kind = "guard"
        else:
            raise DevelopmentLedgerError("unit_slot_kind_inconsistent")

        first = normalized[0]
        fitting_role_id = first["fitting_role_id"]
        validation_role_id = first["validation_role_id"]
        guard_fold = first["guard_fold"]
        excluded = first["excluded_by_protocol"]
        exclusion_reason = first["exclusion_reason"]
        for slot in normalized[1:]:
            if (
                slot["fitting_role_id"] != fitting_role_id
                or slot["validation_role_id"] != validation_role_id
                or slot["guard_fold"] != guard_fold
                or slot["excluded_by_protocol"] != excluded
                or slot["exclusion_reason"] != exclusion_reason
            ):
                raise DevelopmentLedgerError("unit_role_binding_inconsistent")

        observed_product = {(slot["recipe_id"], slot["seed"]) for slot in normalized}
        expected_product = {
            (recipe_id, seed) for recipe_id in recipe_ids for seed in seeds
        }
        if observed_product != expected_product or len(normalized) != len(
            expected_product
        ):
            raise DevelopmentLedgerError("unit_slot_product_incomplete")
        for slot in normalized:
            if slot["slot_id"] in seen_slot_ids:
                raise DevelopmentLedgerError("duplicate_slot")
            seen_slot_ids.add(slot["slot_id"])

        context = contexts.get(context_id)
        if context is None:
            raise DevelopmentLedgerError("unknown_context")

        if unit_kind == "inherited":
            if excluded:
                raise DevelopmentLedgerError("inherited_slot_excluded")
            if exclusion_reason is not None:
                raise DevelopmentLedgerError("inherited_slot_exclusion_reason")
            if guard_fold is not None:
                raise DevelopmentLedgerError("inherited_slot_guard_fold")
            if (context_id, selection_unit_id) not in expected_pairs:
                raise DevelopmentLedgerError("unknown_inherited_unit")
            if role_meta.get(fitting_role_id) != (
                context_id,
                "selection_fit",
                selection_unit_id,
            ):
                raise DevelopmentLedgerError("mislabeled_fitting_role")
            if role_meta.get(validation_role_id) != (
                context_id,
                "selection_validation",
                selection_unit_id,
            ):
                raise DevelopmentLedgerError("mislabeled_validation_role")
            fitting_uids = list(uids_by_role[fitting_role_id])
            validation_uids = list(uids_by_role[validation_role_id])
        else:
            expected_unit = expected_guard.get(selection_unit_id)
            if expected_unit is None:
                raise DevelopmentLedgerError("unknown_guard_unit")
            if context_id != expected_unit["context_id"]:
                raise DevelopmentLedgerError("guard_context_mismatch")
            expected_fit_role = str(
                expected_unit["fitting_role"]["guard_role_id"]
            )
            expected_validation_role = str(
                expected_unit["validation_role"]["guard_role_id"]
            )
            if (
                fitting_role_id != expected_fit_role
                or validation_role_id != expected_validation_role
            ):
                raise DevelopmentLedgerError("guard_role_binding_mismatch")
            if int(guard_fold) != int(expected_unit["guard_fold"]):
                raise DevelopmentLedgerError("guard_fold_mismatch")
            expected_support_ok = bool(expected_unit["support_ok"])
            expected_reason = expected_unit["exclusion_reason"]
            if excluded != (not expected_support_ok):
                raise DevelopmentLedgerError("guard_exclusion_mismatch")
            if exclusion_reason != expected_reason:
                raise DevelopmentLedgerError("guard_exclusion_reason_mismatch")
            fitting_uids = sorted(
                str(uid) for uid in expected_unit["fitting_role"]["fitting_uids"]
            )
            validation_uids = sorted(
                str(uid)
                for uid in expected_unit["validation_role"]["fitting_uids"]
            )

        if not set(fitting_uids) <= fit_uids[context_id] or not set(
            validation_uids
        ) <= fit_uids[context_id]:
            raise DevelopmentLedgerError("unit_uid_outside_outer_fit")
        fitting_rows = [manifest[uid] for uid in fitting_uids]
        validation_rows = [manifest[uid] for uid in validation_uids]
        fitting_masters = {row["master"] for row in fitting_rows}
        validation_masters = {row["master"] for row in validation_rows}
        if (
            not fitting_masters <= fit_masters[context_id]
            or not validation_masters <= fit_masters[context_id]
        ):
            raise DevelopmentLedgerError("unit_master_outside_outer_fit")
        if (fitting_masters | validation_masters) & test_masters[context_id]:
            raise DevelopmentLedgerError("outer_test_master_in_unit")
        if set(fitting_uids) & set(validation_uids):
            raise DevelopmentLedgerError("unit_uid_overlap")
        if fitting_masters & validation_masters:
            raise DevelopmentLedgerError("unit_master_overlap")
        if any(
            row["station"] != context["station"]
            for row in fitting_rows + validation_rows
        ):
            raise DevelopmentLedgerError("unit_station_mismatch")
        held_instrument = context["held_instrument"]
        if held_instrument != "not_applicable":
            instruments = {
                row["instrument"] for row in fitting_rows + validation_rows
            }
            if held_instrument in instruments:
                raise DevelopmentLedgerError("held_instrument_in_unit")

        required_classes = station_classes.get(context["station"])
        if required_classes is None or len(required_classes) != REQUIRED_CLASSES:
            raise DevelopmentLedgerError("station_class_count_mismatch")
        fitting_classes = {row["target"] for row in fitting_rows}
        validation_classes = {row["target"] for row in validation_rows}
        if not excluded:
            if not fitting_rows or not validation_rows:
                raise DevelopmentLedgerError("unit_role_empty")
            if (
                fitting_classes != required_classes
                or validation_classes != required_classes
            ):
                raise DevelopmentLedgerError("unit_missing_station_classes")

        if unit_kind == "inherited" and context["selection_mode"] == "pseudo_domain":
            if (
                not selection_unit_id.startswith(PSEUDO_UNIT_PREFIX)
                or len(selection_unit_id) == len(PSEUDO_UNIT_PREFIX)
            ):
                raise DevelopmentLedgerError("pseudo_unit_id_malformed")
            pseudo_instrument = selection_unit_id[len(PSEUDO_UNIT_PREFIX) :]
            if {row["instrument"] for row in validation_rows} != {pseudo_instrument}:
                raise DevelopmentLedgerError("pseudo_validation_instrument_mismatch")
            if pseudo_instrument in {
                row["instrument"] for row in fitting_rows
            }:
                raise DevelopmentLedgerError("pseudo_instrument_in_fitting")

        fitting_instruments_by_master: dict[str, set[str]] = {}
        for row in fitting_rows:
            fitting_instruments_by_master.setdefault(row["master"], set()).add(
                row["instrument"]
            )
        sampling_capacity = sum(
            min(2, len(instruments))
            for instruments in fitting_instruments_by_master.values()
        )
        if sampling_capacity > MAX_SAMPLING_CAPACITY:
            raise DevelopmentLedgerError("unit_sampling_capacity_exceeded")

        unit_id = development_unit_identity(
            contract_sha256=contract_sha256,
            plan_id=plan_id,
            context_id=context_id,
            selection_unit_id=selection_unit_id,
            unit_kind=unit_kind,
        )
        if unit_id in seen_unit_ids:
            raise DevelopmentLedgerError("duplicate_unit")
        seen_unit_ids.add(unit_id)

        all_masters = fitting_masters | validation_masters
        units.append(
            {
                "unit_id": unit_id,
                "unit_kind": unit_kind,
                "context_id": context_id,
                "selection_unit_id": selection_unit_id,
                "station": context["station"],
                "selection_mode": context["selection_mode"],
                "phase_gate": context["phase_gate"],
                "guard_fold": guard_fold,
                "fitting_role_id": fitting_role_id,
                "validation_role_id": validation_role_id,
                "fitting_uids": list(fitting_uids),
                "validation_uids": list(validation_uids),
                "fitting_uid_set_sha256": uid_set_hash(fitting_uids),
                "validation_uid_set_sha256": uid_set_hash(validation_uids),
                "fitting_masters": sorted(fitting_masters),
                "validation_masters": sorted(validation_masters),
                "fitting_master_count": len(fitting_masters),
                "validation_master_count": len(validation_masters),
                "master_count": len(all_masters),
                "row_count": len(fitting_uids) + len(validation_uids),
                "fitting_classes": sorted(fitting_classes),
                "validation_classes": sorted(validation_classes),
                "fitting_class_count": len(fitting_classes),
                "validation_class_count": len(validation_classes),
                "fitting_instruments": sorted(
                    {row["instrument"] for row in fitting_rows}
                ),
                "validation_instruments": sorted(
                    {row["instrument"] for row in validation_rows}
                ),
                "source_instruments": sorted(
                    {manifest[uid]["instrument"] for uid in fit_uids[context_id]}
                ),
                "sampling_capacity": sampling_capacity,
                "excluded_by_protocol": excluded,
                "exclusion_reason": exclusion_reason,
                "auxiliary_support": _auxiliary_support(fitting_rows),
            }
        )

        for slot in normalized:
            slots.append(
                {
                    "slot_id": slot["slot_id"],
                    "slot_kind": slot["slot_kind"],
                    "unit_id": unit_id,
                    "unit_kind": unit_kind,
                    "context_id": context_id,
                    "selection_unit_id": selection_unit_id,
                    "guard_fold": guard_fold,
                    "fitting_role_id": fitting_role_id,
                    "validation_role_id": validation_role_id,
                    "recipe_id": slot["recipe_id"],
                    "seed": slot["seed"],
                    "planned": True,
                    "excluded_by_protocol": excluded,
                    "exclusion_reason": exclusion_reason,
                }
            )

    units.sort(key=lambda unit: (unit["context_id"], unit["selection_unit_id"]))
    slots.sort(
        key=lambda slot: (
            slot["context_id"],
            slot["selection_unit_id"],
            slot["recipe_id"],
            slot["seed"],
        )
    )

    if max(
        (int(unit["sampling_capacity"]) for unit in units), default=0
    ) > MAX_SAMPLING_CAPACITY:
        raise DevelopmentLedgerError("maximum_sampling_capacity_exceeded")

    summary = _build_summary(
        contexts=contexts,
        units=units,
        slots=slots,
        recipes=recipes,
        seeds=seeds,
        model=model,
        minimum_updates=minimum_updates,
        maximum_updates=maximum_updates,
    )
    _reconcile_contract_counts(summary, contract.get("later_core_plan"))

    ledger_id = development_ledger_identity(
        contract_sha256=contract_sha256,
        plan_id=plan_id,
        input_pins=pins,
        units_sha256=sha256_value(units),
        slots_sha256=sha256_value(slots),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "contract_sha256": contract_sha256,
        "plan_id": plan_id,
        "ledger_id": ledger_id,
        "input_pins": pins,
        "units": units,
        "slots": slots,
        "summary": summary,
        "validation": {
            "metadata_only": True,
            "arrays_loaded": False,
            "fits_started": 0,
            "execution_authorized": False,
            "outer_evaluation_authorized": False,
            "plan_id": plan_id,
            "contract_sha256": contract_sha256,
            "ledger_id": ledger_id,
            "checks": {
                "contract_reconciled": True,
                "plan_identity_bound": True,
                "support_pins_bound": True,
                "units_referential_integrity": True,
                "slots_referential_integrity": True,
                "source_only_boundaries": True,
                "guard_registry_rebuilt": True,
                "canonical_order": True,
            },
        },
        "execution_authorized": False,
        "arrays_loaded": False,
        "fits_started": 0,
        "outer_evaluation_authorized": False,
    }
