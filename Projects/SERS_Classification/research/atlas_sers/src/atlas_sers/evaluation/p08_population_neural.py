"""P08-T172 filtered-population neural support audit (metadata only).

This module is a pure in-memory metadata adapter. It verifies that a supplied
``PopulationPlanTables`` object is internally consistent with the
caller-supplied table hashes and plan digest, then derives a deterministic
neural-support audit.

Integrity scope: the table hashes and plan digest provided by the caller
establish internal consistency only. They are not independent authentication
of a forged plan. Only the locked P05 core contract carries an immutable
validation pin (``LOCKED_CONTRACT_SHA256``), which this module checks through
``validate_core_contract``. Callers must independently authenticate the
upstream P08 population metadata against accepted evidence before trusting
any result here.

The exact classical-support mask (``metadata_ready`` and classical
``calibration_supported``) is deliberately NOT inherited as a neural exclusion
mask: every context remains visible with explicit reasons. Guard units are
rebuilt against population-bound P08 context IDs and are not asserted to
equal any historical P05 guard identity.

It never fits a model, computes a logit, reads a spectrum, writes a file, or
authorizes execution.
"""
from __future__ import annotations

import numbers
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p05_core_plan import (
    CorePlanError,
    build_guard_fold_roles,
    validate_core_contract,
)
from atlas_sers.evaluation.p08_population_plan import PopulationPlanTables
from atlas_sers.governance.canonical import sha256_value

SCHEMA_VERSION = "nato-sers-p08-population-neural-support-v1"
REQUIRED_CLASSES = 3
GUARD_FOLD_COUNT = 3
MASTER_CV_UNITS = 3
PSEUDO_MIN_UNITS = 2

_SELECTION_UNSUPPORTED = "unsupported"
_SELECTION_NOT_APPLICABLE = "not_applicable"
_SELECTION_MODES = frozenset(
    {
        "pseudo_domain",
        "master_cv",
        _SELECTION_UNSUPPORTED,
        _SELECTION_NOT_APPLICABLE,
    }
)
_ELIGIBLE_MODES = frozenset({"pseudo_domain", "master_cv"})
_TABLE_ATTRS = (
    ("master_splits", "master_splits"),
    ("domain_registry", "domain_registry"),
    ("t3_partitions", "t3_partitions"),
    ("context_registry", "context_registry"),
    ("inner_selection_registry", "inner_selection_registry"),
    ("inner_master_split_registry", "inner_master_split_registry"),
    ("unit_registry", "unit_registry"),
    ("role_registry", "role_registry"),
)
_MAX_INSTRUMENTS_PER_MASTER = 2


# --------------------------------------------------------------------------- #
# Small validators and canonical hashing (identical to the plan builder)
# --------------------------------------------------------------------------- #


def _hash_table(frame: pd.DataFrame) -> str:
    return sha256_value(
        {
            "columns": [str(column) for column in frame.columns],
            "records": frame.to_dict(orient="records"),
        }
    )


def _hash_set(values: list[str]) -> str:
    return sha256_value(sorted(set(values)))


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _as_bool(value: Any, name: str) -> bool:
    if not _is_bool(value):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, not a boolean")
    return int(value)


def _as_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if value != value.strip():
        raise ValueError(f"{name} must be a trimmed string")
    if not allow_empty and value == "":
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _records(frame: pd.DataFrame, name: str) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    return frame.to_dict(orient="records")


def _check_provenance(
    row: dict[str, Any],
    name: str,
    population_id: str,
    population_sha256: str,
) -> None:
    if row.get("population_id") != population_id:
        raise ValueError(f"{name}.population_id disagrees with the plan report")
    if row.get("population_sha256") != population_sha256:
        raise ValueError(f"{name}.population_sha256 disagrees with the plan report")


def _capacity(rows: list[dict[str, Any]]) -> int:
    instruments: dict[str, set[str]] = {}
    for entry in rows:
        instruments.setdefault(entry["master"], set()).add(entry["instrument"])
    return sum(
        min(_MAX_INSTRUMENTS_PER_MASTER, len(found))
        for found in instruments.values()
    )


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #


def build_population_neural_support(
    *,
    population_plan: PopulationPlanTables,
    core_contract: dict,
) -> dict[str, Any]:
    """Build the deterministic neural-support audit for one filtered population."""

    if not isinstance(population_plan, PopulationPlanTables):
        raise ValueError("population_plan must be a PopulationPlanTables instance")
    if not isinstance(core_contract, dict):
        raise ValueError("core_contract must be a mapping")

    try:
        core_summary = validate_core_contract(core_contract)
    except CorePlanError as error:
        raise ValueError(f"core contract rejected: {error}") from error
    core_contract_sha256 = str(core_summary["contract_sha256"])
    sampler = core_summary["sampler"]
    batch_ceiling = _as_int(sampler["batch_size_ceiling"], "sampler.batch_size_ceiling")
    if batch_ceiling < 1:
        raise ValueError("sampler.batch_size_ceiling must be positive")

    report = population_plan.validation_report
    if not isinstance(report, dict):
        raise ValueError("validation_report must be a mapping")
    if report.get("execution_authorized") is not False:
        raise ValueError("validation_report.execution_authorized must be False")
    operations = _as_int(
        report.get("scientific_operations"), "scientific_operations"
    )
    if operations != 0:
        raise ValueError("validation_report.scientific_operations must be the integer 0")

    table_hashes = report.get("table_hashes")
    if not isinstance(table_hashes, dict):
        raise ValueError("validation_report.table_hashes must be a mapping")
    if set(table_hashes) != {name for name, _ in _TABLE_ATTRS}:
        raise ValueError("validation_report.table_hashes keys are inconsistent")
    for name, attribute in _TABLE_ATTRS:
        if _hash_table(getattr(population_plan, attribute)) != table_hashes[name]:
            raise ValueError(
                f"population plan table hash is internally inconsistent: {name}"
            )

    plan_payload = {
        "population_id": report.get("population_id"),
        "population_sha256": report.get("population_sha256"),
        "metadata_canonical_sha256": report.get("metadata_canonical_sha256"),
        "split_contract_sha256": report.get("split_contract_sha256"),
        "p02_contract_sha256": report.get("p02_contract_sha256"),
        "table_hashes": table_hashes,
    }
    if sha256_value(plan_payload) != report.get("plan_sha256"):
        raise ValueError("population plan digest is internally inconsistent")
    population_id = _as_str(report.get("population_id"), "population_id")
    population_sha256 = _as_str(
        report.get("population_sha256"), "population_sha256"
    )
    plan_sha256 = _as_str(report.get("plan_sha256"), "population_plan_sha256")

    # ------------------------------------------------------------ index tables
    contexts: dict[str, dict[str, Any]] = {}
    for row in _records(population_plan.context_registry, "context_registry"):
        _check_provenance(row, "context_registry", population_id, population_sha256)
        cid = _as_str(row.get("context_id"), "context_registry.context_id")
        if cid in contexts:
            raise ValueError(f"duplicate context_id in context_registry: {cid}")
        mode = _as_str(row.get("selection_mode"), "context_registry.selection_mode")
        if mode not in _SELECTION_MODES:
            raise ValueError(f"unknown selection_mode {mode!r} for context {cid}")
        domain_eligible = _as_bool(
            row.get("domain_eligible"), "context_registry.domain_eligible"
        )
        if domain_eligible and mode == "not_applicable":
            raise ValueError(
                f"eligible context {cid} cannot have selection_mode 'not_applicable'"
            )
        if not domain_eligible and mode != "not_applicable":
            raise ValueError(
                f"ineligible context {cid} must have selection_mode 'not_applicable'"
            )
        contexts[cid] = {
            "context_id": cid,
            "domain": _as_str(row.get("domain"), "context_registry.domain"),
            "station": _as_str(row.get("station"), "context_registry.station"),
            "held_instrument": _as_str(
                row.get("held_instrument"), "context_registry.held_instrument"
            ),
            "outer_repeat": _as_int(row.get("outer_repeat"), "outer_repeat"),
            "outer_seed": _as_int(row.get("outer_seed"), "outer_seed"),
            "outer_fold": _as_int(row.get("outer_fold"), "outer_fold"),
            "selection_mode": mode,
            "selection_supported": _as_bool(
                row.get("selection_supported"), "context_registry.selection_supported"
            ),
            "domain_eligible": _as_bool(
                row.get("domain_eligible"), "context_registry.domain_eligible"
            ),
            "source_observation_set_sha256": _as_str(
                row.get("source_observation_set_sha256"),
                "source_observation_set_sha256",
            ),
            "test_observation_set_sha256": _as_str(
                row.get("test_observation_set_sha256"),
                "test_observation_set_sha256",
            ),
            "source_master_set_sha256": _as_str(
                row.get("source_master_set_sha256"),
                "source_master_set_sha256",
            ),
            "test_master_set_sha256": _as_str(
                row.get("test_master_set_sha256"),
                "test_master_set_sha256",
            ),
            "source_rows": _as_int(row.get("source_rows"), "source_rows"),
            "source_masters": _as_int(row.get("source_masters"), "source_masters"),
            "source_classes": _as_int(row.get("source_classes"), "source_classes"),
            "test_rows": _as_int(row.get("test_rows"), "test_rows"),
            "test_masters": _as_int(row.get("test_masters"), "test_masters"),
            "test_classes": _as_int(row.get("test_classes"), "test_classes"),
        }

    selection_units: dict[str, list[dict[str, Any]]] = {}
    calibration_unit_ids: dict[str, set[str]] = {}
    seen_units: set[tuple[str, str, str]] = set()
    for row in _records(population_plan.unit_registry, "unit_registry"):
        _check_provenance(row, "unit_registry", population_id, population_sha256)
        cid = _as_str(row.get("context_id"), "unit_registry.context_id")
        if cid not in contexts:
            raise ValueError(f"unit_registry references unknown context: {cid}")
        purpose = _as_str(row.get("purpose"), "unit_registry.purpose")
        if purpose not in ("selection", "calibration"):
            raise ValueError(f"unknown unit purpose: {purpose!r}")
        unit_id = _as_str(row.get("unit_id"), "unit_registry.unit_id")
        key = (cid, purpose, unit_id)
        if key in seen_units:
            raise ValueError(f"duplicate unit registry row: {key!r}")
        seen_units.add(key)
        unit = {
            "context_id": cid,
            "unit_id": unit_id,
            "support": _as_bool(row.get("support"), "unit_registry.support"),
            "reason_code": _as_str(
                row.get("reason_code"), "unit_registry.reason_code", allow_empty=True
            ),
            "fit_rows": _as_int(row.get("fit_rows"), "fit_rows"),
            "fit_masters": _as_int(row.get("fit_masters"), "fit_masters"),
            "fit_classes": _as_int(row.get("fit_classes"), "fit_classes"),
            "validation_rows": _as_int(row.get("validation_rows"), "validation_rows"),
            "validation_masters": _as_int(
                row.get("validation_masters"), "validation_masters"
            ),
            "validation_classes": _as_int(
                row.get("validation_classes"), "validation_classes"
            ),
            "fit_observation_set_sha256": _as_str(
                row.get("fit_observation_set_sha256"), "fit_observation_set_sha256"
            ),
            "validation_observation_set_sha256": _as_str(
                row.get("validation_observation_set_sha256"),
                "validation_observation_set_sha256",
            ),
            "master_disjoint": _as_bool(
                row.get("master_disjoint"), "unit_registry.master_disjoint"
            ),
            "held_instrument_absent": _as_bool(
                row.get("held_instrument_absent"),
                "unit_registry.held_instrument_absent",
            ),
        }
        if purpose == "selection":
            if unit["support"] is not True:
                raise ValueError(
                    f"registered selection unit is unsupported: {cid}:{unit_id}"
                )
            selection_units.setdefault(cid, []).append(unit)
        else:
            calibration_unit_ids.setdefault(cid, set()).add(unit_id)

    outer_roles: dict[str, dict[str, list[dict[str, Any]]]] = {}
    selection_roles: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    calibration_role_units: set[tuple[str, str]] = set()
    seen_roles: set[tuple[str, str, str, str, str]] = set()
    uid_identity: dict[str, tuple[str, str, str, str]] = {}
    master_identity: dict[str, tuple[str, str]] = {}
    for row in _records(population_plan.role_registry, "role_registry"):
        _check_provenance(row, "role_registry", population_id, population_sha256)
        cid = _as_str(row.get("context_id"), "role_registry.context_id")
        if cid not in contexts:
            raise ValueError(f"role_registry references unknown context: {cid}")
        station = contexts[cid]["station"]
        purpose = _as_str(row.get("purpose"), "role_registry.purpose")
        role = _as_str(row.get("role"), "role_registry.role")
        unit_id = _as_str(row.get("unit_id"), "role_registry.unit_id")
        uid = _as_str(row.get("observation_uid"), "role_registry.observation_uid")
        master = _as_str(row.get("master_sample_id"), "role_registry.master_sample_id")
        target = _as_str(row.get("target_analyte"), "role_registry.target_analyte")
        instrument = _as_str(row.get("instrument"), "role_registry.instrument")
        identity = (cid, purpose, unit_id, role, uid)
        if identity in seen_roles:
            raise ValueError(f"duplicate role registry identity: {identity!r}")
        seen_roles.add(identity)
        uid_key = (master, station, instrument, target)
        previous_uid = uid_identity.get(uid)
        if previous_uid is not None and previous_uid != uid_key:
            raise ValueError(
                f"observation_uid {uid!r} has conflicting station or metadata"
            )
        uid_identity[uid] = uid_key
        master_key = (target, station)
        previous_master = master_identity.get(master)
        if previous_master is not None and previous_master != master_key:
            raise ValueError(
                f"master_sample_id {master!r} maps to multiple stations or classes"
            )
        master_identity[master] = master_key
        entry = {
            "uid": uid,
            "master": master,
            "instrument": instrument,
            "target": target,
        }
        if purpose == "outer":
            if role not in ("source", "test"):
                raise ValueError(f"unknown outer role: {role!r}")
            if unit_id != "outer":
                raise ValueError(f"outer role has unexpected unit_id: {unit_id!r}")
            outer_roles.setdefault(cid, {}).setdefault(role, []).append(entry)
        elif purpose == "selection":
            if role not in ("fit", "validation"):
                raise ValueError(f"unknown selection role: {role!r}")
            selection_roles.setdefault((cid, unit_id), {}).setdefault(role, []).append(
                entry
            )
        elif purpose == "calibration":
            if role not in ("fit", "validation"):
                raise ValueError(f"unknown calibration role: {role!r}")
            calibration_role_units.add((cid, unit_id))
        else:
            raise ValueError(f"unknown role purpose: {purpose!r}")

    for cid, units in selection_units.items():
        for unit in units:
            roles = selection_roles.get((cid, unit["unit_id"]))
            if not roles or "fit" not in roles or "validation" not in roles:
                raise ValueError(
                    f"selection unit {cid}:{unit['unit_id']} lacks role rows"
                )
    for cid, unit_id in selection_roles:
        if not any(
            unit["unit_id"] == unit_id for unit in selection_units.get(cid, [])
        ):
            raise ValueError(f"selection role references unknown unit: {cid}:{unit_id}")
    for cid, unit_id in calibration_role_units:
        if unit_id not in calibration_unit_ids.get(cid, set()):
            raise ValueError(
                f"calibration role references unknown unit: {cid}:{unit_id}"
            )

    # ------------------------------------------------------- per-context audit
    context_results: list[dict[str, Any]] = []
    guard_units: list[dict[str, Any]] = []
    guard_capacity: dict[tuple[str, int], int] = {}
    unit_fit_capacity: dict[tuple[str, str], int] = {}
    all_capacities: list[int] = []
    fallback_counts: dict[str, int] = {}
    counters = {
        "eligible": 0,
        "ineligible": 0,
        "unsupported": 0,
        "not_applicable": 0,
        "pseudo_domain": 0,
        "master_cv": 0,
        "calibration_supported": 0,
        "ordinary_supported": 0,
        "g3": 0,
        "fallback": 0,
        "inherited_units": 0,
        "supported_inherited_units": 0,
        "guards": 0,
        "supported_guards": 0,
    }

    for cid in sorted(contexts):
        ctx = contexts[cid]
        mode = ctx["selection_mode"]
        outer = outer_roles.get(cid, {})
        source_rows = outer.get("source", [])
        test_rows = outer.get("test", [])
        units = selection_units.get(cid, [])

        source_uids = [entry["uid"] for entry in source_rows]
        source_masters = [entry["master"] for entry in source_rows]
        source_classes = {entry["target"] for entry in source_rows}
        test_uids = [entry["uid"] for entry in test_rows]
        test_masters = [entry["master"] for entry in test_rows]
        test_classes = {entry["target"] for entry in test_rows}

        # Hash/count/disjointness disagreements are corrupt input, not an
        # unavailable plan. They raise rather than becoming exclusion reasons.
        if _hash_set(source_uids) != ctx["source_observation_set_sha256"]:
            raise ValueError(f"context {cid} source observation hash disagrees")
        if _hash_set(source_masters) != ctx["source_master_set_sha256"]:
            raise ValueError(f"context {cid} source master hash disagrees")
        if len(source_uids) != ctx["source_rows"]:
            raise ValueError(f"context {cid} source row count disagrees")
        if len(set(source_masters)) != ctx["source_masters"]:
            raise ValueError(f"context {cid} source master count disagrees")
        if len(source_classes) != ctx["source_classes"]:
            raise ValueError(f"context {cid} source class count disagrees")
        if _hash_set(test_uids) != ctx["test_observation_set_sha256"]:
            raise ValueError(f"context {cid} test observation hash disagrees")
        if _hash_set(test_masters) != ctx["test_master_set_sha256"]:
            raise ValueError(f"context {cid} test master hash disagrees")
        if len(test_uids) != ctx["test_rows"]:
            raise ValueError(f"context {cid} test row count disagrees")
        if len(set(test_masters)) != ctx["test_masters"]:
            raise ValueError(f"context {cid} test master count disagrees")
        if len(test_classes) != ctx["test_classes"]:
            raise ValueError(f"context {cid} test class count disagrees")
        if set(source_uids) & set(test_uids):
            raise ValueError(f"context {cid} source and test observations overlap")
        if set(source_masters) & set(test_masters):
            raise ValueError(f"context {cid} source and test masters overlap")
        if ctx["held_instrument"] in {entry["instrument"] for entry in source_rows}:
            raise ValueError(f"context {cid} source contains the held instrument")
        if any(entry["instrument"] != ctx["held_instrument"] for entry in test_rows):
            raise ValueError(f"context {cid} test contains a non-held instrument")

        source_has_three_classes = len(source_classes) == REQUIRED_CLASSES
        test_subset = test_classes <= source_classes
        outer_source_valid = bool(source_rows and source_has_three_classes)
        outer_test_valid = bool(test_rows and test_subset)

        valid_units: list[dict[str, Any]] = []
        for unit in units:
            unit_id = unit["unit_id"]
            roles = selection_roles[(cid, unit_id)]
            fit_rows = roles["fit"]
            val_rows = roles["validation"]
            fit_uids = [entry["uid"] for entry in fit_rows]
            val_uids = [entry["uid"] for entry in val_rows]
            fit_masters = {entry["master"] for entry in fit_rows}
            val_masters = {entry["master"] for entry in val_rows}
            fit_classes = {entry["target"] for entry in fit_rows}
            val_classes = {entry["target"] for entry in val_rows}
            if (
                len(fit_uids) != unit["fit_rows"]
                or len(fit_masters) != unit["fit_masters"]
                or len(fit_classes) != unit["fit_classes"]
                or _hash_set(fit_uids) != unit["fit_observation_set_sha256"]
            ):
                raise ValueError(
                    f"selection unit {cid}:{unit_id} fit metadata disagrees"
                )
            if (
                len(val_uids) != unit["validation_rows"]
                or len(val_masters) != unit["validation_masters"]
                or len(val_classes) != unit["validation_classes"]
                or _hash_set(val_uids) != unit["validation_observation_set_sha256"]
            ):
                raise ValueError(
                    f"selection unit {cid}:{unit_id} validation metadata disagrees"
                )
            overlap = bool(fit_masters & val_masters)
            if unit["master_disjoint"] == overlap:
                raise ValueError(
                    f"selection unit {cid}:{unit_id} master_disjoint flag disagrees"
                )
            held_present = (
                ctx["held_instrument"] in {entry["instrument"] for entry in fit_rows}
                or ctx["held_instrument"] in {entry["instrument"] for entry in val_rows}
            )
            if unit["held_instrument_absent"] == held_present:
                raise ValueError(
                    f"selection unit {cid}:{unit_id} held_instrument flag disagrees"
                )
            if overlap:
                raise ValueError(
                    f"selection unit {cid}:{unit_id} fit and validation masters overlap"
                )
            if held_present:
                raise ValueError(
                    f"selection unit {cid}:{unit_id} held instrument is present"
                )
            if (
                not set(fit_uids) <= set(source_uids)
                or not set(val_uids) <= set(source_uids)
            ):
                raise ValueError(
                    f"selection unit {cid}:{unit_id} leaves its outer source"
                )
            if (
                len(fit_classes) != REQUIRED_CLASSES
                or len(val_classes) != REQUIRED_CLASSES
            ):
                raise ValueError(
                    f"selection unit {cid}:{unit_id} lacks three chemical classes"
                )
            if not fit_classes <= source_classes or not val_classes <= source_classes:
                raise ValueError(
                    f"selection unit {cid}:{unit_id} disagrees with source classes"
                )
            unit_fit_capacity[(cid, unit_id)] = _capacity(fit_rows)
            valid_units.append(unit)

        if ctx["domain_eligible"]:
            if mode == "master_cv" and len(units) != MASTER_CV_UNITS:
                raise ValueError(
                    f"master_cv context {cid} must have {MASTER_CV_UNITS} units"
                )
            if mode == "pseudo_domain" and len(units) < PSEUDO_MIN_UNITS:
                raise ValueError(
                    f"pseudo_domain context {cid} needs at least "
                    f"{PSEUDO_MIN_UNITS} units"
                )
            if mode == _SELECTION_UNSUPPORTED and units:
                raise ValueError(
                    f"unsupported context {cid} must have no selection units"
                )
        elif units:
            raise ValueError(
                f"ineligible context {cid} must have no executable selection units"
            )

        derived_selection_supported = bool(
            ctx["domain_eligible"] and mode in _ELIGIBLE_MODES and valid_units
        )
        if ctx["selection_supported"] != derived_selection_supported:
            raise ValueError(
                f"context {cid} selection_supported flag disagrees with mode/units"
            )

        # ------------------------------------------------------------- guards
        guards_for_context: list[dict[str, Any]] = []
        if ctx["domain_eligible"] and mode == "pseudo_domain":
            outer_fit_rows = [
                {
                    "uid": entry["uid"],
                    "master": entry["master"],
                    "instrument": entry["instrument"],
                    "target": entry["target"],
                }
                for entry in source_rows
            ]
            try:
                built = build_guard_fold_roles(
                    contract_sha256=core_contract_sha256,
                    context_id=cid,
                    station=ctx["station"],
                    outer_fit_rows=outer_fit_rows,
                    held_instrument=ctx["held_instrument"],
                    outer_test_masters=sorted(
                        {entry["master"] for entry in test_rows}
                    ),
                )
            except CorePlanError as error:
                raise ValueError(
                    f"guard construction failed for {cid}: {error}"
                ) from error
            if len(built) != GUARD_FOLD_COUNT:
                raise ValueError(
                    f"guard construction for {cid} did not return "
                    f"{GUARD_FOLD_COUNT} folds"
                )
            folds: set[int] = set()
            for unit in built:
                fold = _as_int(unit["guard_fold"], "guard_fold")
                if fold in folds:
                    raise ValueError(f"duplicate guard fold {fold} for context {cid}")
                folds.add(fold)
                fitting_uids = set(unit["fitting_role"]["fitting_uids"])
                guard_capacity[(cid, fold)] = _capacity(
                    [entry for entry in outer_fit_rows if entry["uid"] in fitting_uids]
                )
                guard_entry = dict(unit)
                guard_entry["population_id"] = population_id
                guard_entry["population_sha256"] = population_sha256
                guards_for_context.append(guard_entry)
            guard_units.extend(guards_for_context)

        # --------------------------------------------------------- capacities
        inherited_capacities = [
            unit_fit_capacity[(cid, unit["unit_id"])] for unit in valid_units
        ]
        max_inherited = max(inherited_capacities) if inherited_capacities else None
        inherited_capacity_ok = bool(inherited_capacities) and all(
            value <= batch_ceiling for value in inherited_capacities
        )
        outer_capacity = _capacity(source_rows) if source_rows else None
        outer_capacity_ok = (
            outer_capacity is not None and outer_capacity <= batch_ceiling
        )
        supported_guards = [
            unit
            for unit in guards_for_context
            if _as_bool(unit["support_ok"], "guard support_ok")
        ]
        guard_total = len(guards_for_context)
        guard_supported = len(supported_guards)
        guard_capacities = [
            guard_capacity[(cid, _as_int(unit["guard_fold"], "guard_fold"))]
            for unit in supported_guards
        ]
        max_guard = max(guard_capacities) if guard_capacities else None
        guard_capacity_ok = bool(guard_capacities) and all(
            value <= batch_ceiling for value in guard_capacities
        )
        all_guards_supported = (
            guard_total == GUARD_FOLD_COUNT and guard_supported == GUARD_FOLD_COUNT
        )
        all_capacities.extend(inherited_capacities)
        all_capacities.extend(guard_capacities)
        if outer_capacity is not None:
            all_capacities.append(outer_capacity)

        # --------------------------------------------- source calibration audit
        inherited_validation_rows: list[dict[str, Any]] = []
        for unit in valid_units:
            inherited_validation_rows.extend(
                selection_roles[(cid, unit["unit_id"])]["validation"]
            )
        inherited_validation_classes = {
            entry["target"] for entry in inherited_validation_rows
        }
        inherited_validation_present = bool(
            valid_units and inherited_validation_rows
        )
        all_three_class_validation = bool(
            inherited_validation_present
            and len(inherited_validation_classes) == REQUIRED_CLASSES
        )

        ordinary_prereqs = bool(
            ctx["domain_eligible"]
            and mode in _ELIGIBLE_MODES
            and valid_units
            and outer_source_valid
            and outer_test_valid
            and inherited_validation_present
            and outer_capacity_ok
            and inherited_capacity_ok
        )
        ordinary_supported = ordinary_prereqs
        calibration_supported = bool(
            ordinary_prereqs and all_three_class_validation
        )
        g3_structurally_comparable = bool(
            ordinary_supported
            and mode == "pseudo_domain"
            and all_guards_supported
            and guard_capacity_ok
        )

        fallback_required = False
        fallback_cause = None
        if ordinary_supported and not g3_structurally_comparable:
            if mode == "master_cv":
                fallback_required = True
                fallback_cause = "master_cv_structural_fallback"
            elif mode == "pseudo_domain":
                fallback_required = True
                if not all_guards_supported:
                    fallback_cause = "guard_support_structural_fallback"
                elif not guard_capacity_ok:
                    fallback_cause = "guard_capacity_structural_fallback"
                else:
                    fallback_cause = "guard_structural_fallback"
        if fallback_cause is not None:
            fallback_counts[fallback_cause] = (
                fallback_counts.get(fallback_cause, 0) + 1
            )

        # ------------------------------------------------------------- reasons
        reasons: list[str] = []
        if not ctx["domain_eligible"]:
            reasons.append("ineligible_domain")
        if mode == _SELECTION_UNSUPPORTED:
            reasons.append("selection_unsupported")
        if mode == _SELECTION_NOT_APPLICABLE:
            reasons.append("selection_not_applicable")
        if not source_rows:
            reasons.append("empty_outer_source")
        elif not source_has_three_classes:
            reasons.append("outer_source_missing_three_classes")
        if not test_rows:
            reasons.append("empty_outer_test")
        elif not test_subset:
            reasons.append("outer_test_labels_not_subset")
        if not valid_units:
            reasons.append("no_supported_selection_units")
        if not inherited_validation_present:
            reasons.append("no_inherited_validation_coverage")
        elif not all_three_class_validation:
            reasons.append("inherited_validation_missing_three_classes")
        if source_rows and not outer_capacity_ok:
            reasons.append("outer_capacity_exceeded")
        if valid_units and not inherited_capacity_ok:
            reasons.append("inherited_capacity_exceeded")
        if mode == "pseudo_domain":
            if not all_guards_supported:
                reasons.append("guard_support_incomplete")
            if supported_guards and not guard_capacity_ok:
                reasons.append("guard_capacity_exceeded")

        calibration_uids = [entry["uid"] for entry in inherited_validation_rows]
        calibration_masters = [
            entry["master"] for entry in inherited_validation_rows
        ]
        context_results.append(
            {
                "context_id": cid,
                "domain": ctx["domain"],
                "station": ctx["station"],
                "held_instrument": ctx["held_instrument"],
                "outer_repeat": ctx["outer_repeat"],
                "outer_seed": ctx["outer_seed"],
                "outer_fold": ctx["outer_fold"],
                "selection_mode": mode,
                "domain_eligible": ctx["domain_eligible"],
                "outer_source_valid": outer_source_valid,
                "outer_test_valid": outer_test_valid,
                "inherited_selection_unit_ids": sorted(
                    unit["unit_id"] for unit in units
                ),
                "inherited_selection_unit_count": len(units),
                "supported_inherited_selection_unit_ids": sorted(
                    unit["unit_id"] for unit in valid_units
                ),
                "neural_source_calibration_metadata_supported": (
                    calibration_supported
                ),
                "neural_source_calibration": {
                    "unit_ids": sorted(unit["unit_id"] for unit in valid_units),
                    "appearance_count": len(inherited_validation_rows),
                    "distinct_validation_uid_count": len(set(calibration_uids)),
                    "distinct_validation_master_count": len(
                        set(calibration_masters)
                    ),
                    "class_count": len(inherited_validation_classes),
                    "validation_observation_set_sha256": _hash_set(calibration_uids),
                    "validation_master_set_sha256": _hash_set(calibration_masters),
                },
                "guard_unit_count": guard_total,
                "guard_supported_count": guard_supported,
                "g3_structurally_comparable": g3_structurally_comparable,
                "structural_D0M_fallback_required": fallback_required,
                "structural_D0M_fallback_cause": fallback_cause,
                "unavailable_reasons": sorted(set(reasons)),
                "outer_batch_capacity": outer_capacity,
                "max_inherited_batch_capacity": max_inherited,
                "max_guard_batch_capacity": max_guard,
                "ordinary_neural_metadata_supported": ordinary_supported,
            }
        )

        # ------------------------------------------------------------ counters
        counters["eligible" if ctx["domain_eligible"] else "ineligible"] += 1
        counters["unsupported"] += 1 if mode == _SELECTION_UNSUPPORTED else 0
        counters["not_applicable"] += 1 if mode == _SELECTION_NOT_APPLICABLE else 0
        counters["pseudo_domain"] += 1 if mode == "pseudo_domain" else 0
        counters["master_cv"] += 1 if mode == "master_cv" else 0
        counters["calibration_supported"] += 1 if calibration_supported else 0
        counters["ordinary_supported"] += 1 if ordinary_supported else 0
        counters["g3"] += 1 if g3_structurally_comparable else 0
        counters["fallback"] += 1 if fallback_required else 0
        counters["inherited_units"] += len(units)
        counters["supported_inherited_units"] += len(valid_units)
        counters["guards"] += guard_total
        counters["supported_guards"] += guard_supported

    guard_units.sort(
        key=lambda unit: (str(unit["context_id"]), int(unit["guard_fold"]))
    )

    counts = {
        "context_count": len(context_results),
        "eligible_context_count": counters["eligible"],
        "ineligible_context_count": counters["ineligible"],
        "selection_unsupported_context_count": counters["unsupported"],
        "selection_not_applicable_context_count": counters["not_applicable"],
        "pseudo_domain_context_count": counters["pseudo_domain"],
        "master_cv_context_count": counters["master_cv"],
        "neural_source_calibration_supported_context_count": (
            counters["calibration_supported"]
        ),
        "ordinary_neural_supported_context_count": counters["ordinary_supported"],
        "g3_structurally_comparable_context_count": counters["g3"],
        "structural_D0M_fallback_context_count": counters["fallback"],
        "structural_D0M_fallbacks_by_cause": dict(sorted(fallback_counts.items())),
        "inherited_selection_unit_count": counters["inherited_units"],
        "supported_inherited_selection_unit_count": counters[
            "supported_inherited_units"
        ],
        "guard_unit_count": counters["guards"],
        "guard_supported_count": counters["supported_guards"],
        "sampler_batch_size_ceiling": batch_ceiling,
        "maximum_observed_batch_capacity": max(all_capacities) if all_capacities else 0,
        "all_observed_capacities_within_ceiling": bool(all_capacities)
        and all(value <= batch_ceiling for value in all_capacities),
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "population_id": population_id,
        "population_plan_sha256": plan_sha256,
        "core_contract_sha256": core_contract_sha256,
        "execution_authorized": False,
        "scientific_operations": 0,
        "numerical_readiness_verified": False,
        "exact_model_operation_ledger_complete": False,
        "selected_recipe_id": None,
        "recipe_selected": False,
        "contexts": context_results,
        "guard_units": guard_units,
        "counts": counts,
        "denied_operations": [
            "model_fitting",
            "recipe_selection",
            "metric_computation",
            "preprocessing_fit",
            "prediction",
            "logit_computation",
            "benchmark_completion_claim",
        ],
    }
    result["result_sha256"] = sha256_value(result)
    return result
