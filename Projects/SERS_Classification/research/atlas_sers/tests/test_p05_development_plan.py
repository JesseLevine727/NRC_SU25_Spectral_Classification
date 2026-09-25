"""Tests for the metadata-only P05 development ledger and its read-only CLI."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from atlas_sers.evaluation.p05_core_plan import (
    build_guard_fold_roles,
    slot_identity,
)
from atlas_sers.evaluation.p05_development_plan import (
    GUARD_SLOT_KIND,
    INHERITED_SLOT_KIND,
    MAX_SAMPLING_CAPACITY,
    DevelopmentLedgerError,
    _auxiliary_support,
    _optimizer_schedule,
    build_development_ledger,
)
from atlas_sers.evaluation.p05_support import SupportInputs, sha256_value

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = _REPO_ROOT / "plan" / "contracts" / "p05_core_contract.json"

_PINS = {
    "p01_run_id": "P01-fixture",
    "p04plan_run_id": "P04PLAN-fixture",
    "manifest_sha256": "a" * 64,
    "representation_sha256": "d" * 64,
    "contexts_sha256": "b" * 64,
    "roles_sha256": "c" * 64,
}

_CONTEXTS = (
    {
        "context_id": "ctx_dev",
        "station": "cwa",
        "held_instrument": "not_applicable",
        "selection_mode": "inner_master_cv",
        "phase_gate": "development",
    },
    {
        "context_id": "ctx_pseudo",
        "station": "cwa",
        "held_instrument": "INS-X",
        "selection_mode": "pseudo_domain",
        "phase_gate": "held_evaluation",
    },
    {
        "context_id": "ctx_mcv",
        "station": "cwa",
        "held_instrument": "INS-Y",
        "selection_mode": "master_cv",
        "phase_gate": "held_evaluation",
    },
)

_BASE_MANIFEST = (
    ("p-MA1", "MA1", "A", "INS-P"),
    ("p-MB1", "MB1", "B", "INS-P"),
    ("p-MC1", "MC1", "C", "INS-P"),
    ("p-MA2", "MA2", "A", "INS-Q"),
    ("p-MB2", "MB2", "B", "INS-Q"),
    ("p-MC2", "MC2", "C", "INS-Q"),
    ("p-MA3", "MA3", "A", "INS-R"),
    ("p-MB3", "MB3", "B", "INS-R"),
    ("p-MC3", "MC3", "C", "INS-R"),
    ("p-MT1", "MT1", "A", "INS-X"),
    ("d-DA1", "DA1", "A", "INS-Q"),
    ("d-DB1", "DB1", "B", "INS-Q"),
    ("d-DC1", "DC1", "C", "INS-Q"),
    ("d-DA2", "DA2", "A", "INS-R"),
    ("d-DB2", "DB2", "B", "INS-R"),
    ("d-DC2", "DC2", "C", "INS-R"),
    ("d-DT1", "DT1", "A", "INS-R"),
    ("m-EA1", "EA1", "A", "INS-Q"),
    ("m-EB1", "EB1", "B", "INS-Q"),
    ("m-EC1", "EC1", "C", "INS-Q"),
    ("m-EA2", "EA2", "A", "INS-R"),
    ("m-EB2", "EB2", "B", "INS-R"),
    ("m-EC2", "EC2", "C", "INS-R"),
    ("m-ET1", "ET1", "A", "INS-Y"),
)

_BASE_ROLES = (
    (
        "ctx_dev",
        "r-dev-fit",
        "selection_fit",
        "outer_fold_as_inner:u1",
        ("d-DA1", "d-DB1", "d-DC1"),
    ),
    (
        "ctx_dev",
        "r-dev-val",
        "selection_validation",
        "outer_fold_as_inner:u1",
        ("d-DA2", "d-DB2", "d-DC2"),
    ),
    (
        "ctx_dev",
        "r-dev-of",
        "outer_fit",
        "outer_fit",
        ("d-DA1", "d-DB1", "d-DC1", "d-DA2", "d-DB2", "d-DC2"),
    ),
    ("ctx_dev", "r-dev-ot", "outer_test", "outer_test", ("d-DT1",)),
    (
        "ctx_pseudo",
        "r-ps-fit",
        "selection_fit",
        "pseudo:INS-P",
        ("p-MA2", "p-MB2", "p-MC2"),
    ),
    (
        "ctx_pseudo",
        "r-ps-val",
        "selection_validation",
        "pseudo:INS-P",
        ("p-MA1", "p-MB1", "p-MC1"),
    ),
    (
        "ctx_pseudo",
        "r-ps-of",
        "outer_fit",
        "outer_fit",
        (
            "p-MA1",
            "p-MB1",
            "p-MC1",
            "p-MA2",
            "p-MB2",
            "p-MC2",
            "p-MA3",
            "p-MB3",
            "p-MC3",
        ),
    ),
    ("ctx_pseudo", "r-ps-ot", "outer_test", "outer_test", ("p-MT1",)),
    (
        "ctx_mcv",
        "r-mc-fit",
        "selection_fit",
        "master_cv:u1",
        ("m-EA1", "m-EB1", "m-EC1"),
    ),
    (
        "ctx_mcv",
        "r-mc-val",
        "selection_validation",
        "master_cv:u1",
        ("m-EA2", "m-EB2", "m-EC2"),
    ),
    (
        "ctx_mcv",
        "r-mc-of",
        "outer_fit",
        "outer_fit",
        ("m-EA1", "m-EB1", "m-EC1", "m-EA2", "m-EB2", "m-EC2"),
    ),
    ("ctx_mcv", "r-mc-ot", "outer_test", "outer_test", ("m-ET1",)),
)

_BASE_UNITS = (
    {
        "context_id": "ctx_dev",
        "selection_unit_id": "outer_fold_as_inner:u1",
        "fitting_role_id": "r-dev-fit",
        "validation_role_id": "r-dev-val",
    },
    {
        "context_id": "ctx_pseudo",
        "selection_unit_id": "pseudo:INS-P",
        "fitting_role_id": "r-ps-fit",
        "validation_role_id": "r-ps-val",
    },
    {
        "context_id": "ctx_mcv",
        "selection_unit_id": "master_cv:u1",
        "fitting_role_id": "r-mc-fit",
        "validation_role_id": "r-mc-val",
    },
)

_GUARD_CONTEXTS = ("ctx_pseudo",)


# --------------------------------------------------------------------------- #
# Fixture construction
# --------------------------------------------------------------------------- #


def _load_public_contract() -> dict:
    return json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


def _fixture_contract() -> dict:
    contract = copy.deepcopy(_load_public_contract())
    contract["input_pins"] = dict(_PINS)
    later = contract["later_core_plan"]
    guard_slots = 3 * len(_GUARD_CONTEXTS)
    later["inherited_inner_units"] = len(_BASE_UNITS)
    later["extra_guard_unit_slots"] = guard_slots
    later["inner_fit_slot_ceiling"] = (len(_BASE_UNITS) + guard_slots) * 4 * 3
    later["outer_contexts"] = len(_CONTEXTS)
    later["pseudo_domain_contexts"] = 1
    later["master_cv_fallback_t3_contexts"] = 1
    return contract


def _build_support(manifest_entries, roles_entries) -> SupportInputs:
    lookup = {entry[0]: entry for entry in manifest_entries}
    manifest_rows = tuple(
        {
            "observation_uid": uid,
            "master_sample_id": master,
            "station": "cwa",
            "target_analyte": target,
            "instrument": instrument,
            "sensor_family": "sub",
        }
        for (uid, master, target, instrument) in manifest_entries
    )
    context_rows = tuple(dict(row) for row in _CONTEXTS)
    role_rows = []
    for context_id, role_id, role, unit_id, uids in roles_entries:
        for uid in uids:
            _uid, master, target, instrument = lookup[uid]
            role_rows.append(
                {
                    "context_id": context_id,
                    "role_id": role_id,
                    "role": role,
                    "selection_unit_id": unit_id,
                    "observation_uid": uid,
                    "master_sample_id": master,
                    "target_analyte": target,
                    "instrument": instrument,
                }
            )
    return SupportInputs(
        manifest=manifest_rows,
        contexts=context_rows,
        roles=tuple(role_rows),
        manifest_sha256=_PINS["manifest_sha256"],
        contexts_sha256=_PINS["contexts_sha256"],
        roles_sha256=_PINS["roles_sha256"],
    )


def _manifest_view(support) -> dict:
    return {
        row["observation_uid"]: {
            "uid": row["observation_uid"],
            "master": row["master_sample_id"],
            "station": row["station"],
            "target": row["target_analyte"],
            "instrument": row["instrument"],
            "substrate": row["sensor_family"],
        }
        for row in support.manifest
    }


def _build_plan(contract, support, units_spec, guard_contexts) -> dict:
    contract_sha256 = sha256_value(contract)
    recipe_ids = [recipe["recipe_id"] for recipe in contract["recipes"]]
    seeds = list(contract["later_core_plan"]["seeds"])
    manifest = _manifest_view(support)
    contexts = {row["context_id"]: row for row in support.contexts}
    roles = list(support.roles)

    def role_uids(role_id):
        return sorted(
            row["observation_uid"] for row in roles if row["role_id"] == role_id
        )

    def outer_role(context_id, role):
        for row in roles:
            if row["context_id"] == context_id and row["role"] == role:
                return row["role_id"]
        raise AssertionError("fixture is missing an outer role")

    slots = []
    for unit in units_spec:
        for recipe_id in recipe_ids:
            for seed in seeds:
                slots.append(
                    {
                        "slot_id": slot_identity(
                            contract_sha256=contract_sha256,
                            slot_kind=INHERITED_SLOT_KIND,
                            context_id=unit["context_id"],
                            selection_unit_id=unit["selection_unit_id"],
                            fitting_role_id=unit["fitting_role_id"],
                            validation_role_id=unit["validation_role_id"],
                            guard_fold=None,
                            recipe_id=recipe_id,
                            seed=seed,
                        ),
                        "slot_kind": INHERITED_SLOT_KIND,
                        "context_id": unit["context_id"],
                        "selection_unit_id": unit["selection_unit_id"],
                        "fitting_role_id": unit["fitting_role_id"],
                        "validation_role_id": unit["validation_role_id"],
                        "guard_fold": None,
                        "recipe_id": recipe_id,
                        "seed": seed,
                        "planned": True,
                        "excluded_by_protocol": False,
                        "exclusion_reason": None,
                    }
                )

    guard_roles = []
    for context_id in guard_contexts:
        context = contexts[context_id]
        fit_role_id = outer_role(context_id, "outer_fit")
        test_role_id = outer_role(context_id, "outer_test")
        outer_fit_rows = [manifest[uid] for uid in role_uids(fit_role_id)]
        outer_test_masters = sorted(
            {manifest[uid]["master"] for uid in role_uids(test_role_id)}
        )
        guard_roles.extend(
            build_guard_fold_roles(
                contract_sha256=contract_sha256,
                context_id=context_id,
                station=context["station"],
                outer_fit_rows=outer_fit_rows,
                held_instrument=context["held_instrument"],
                outer_test_masters=outer_test_masters,
            )
        )

    for unit in guard_roles:
        for recipe_id in recipe_ids:
            for seed in seeds:
                slots.append(
                    {
                        "slot_id": slot_identity(
                            contract_sha256=contract_sha256,
                            slot_kind=GUARD_SLOT_KIND,
                            context_id=unit["context_id"],
                            selection_unit_id=unit["guard_unit_id"],
                            fitting_role_id=unit["fitting_role"]["guard_role_id"],
                            validation_role_id=unit["validation_role"][
                                "guard_role_id"
                            ],
                            guard_fold=unit["guard_fold"],
                            recipe_id=recipe_id,
                            seed=seed,
                        ),
                        "slot_kind": GUARD_SLOT_KIND,
                        "context_id": unit["context_id"],
                        "selection_unit_id": unit["guard_unit_id"],
                        "fitting_role_id": unit["fitting_role"]["guard_role_id"],
                        "validation_role_id": unit["validation_role"][
                            "guard_role_id"
                        ],
                        "guard_fold": unit["guard_fold"],
                        "recipe_id": recipe_id,
                        "seed": seed,
                        "planned": True,
                        "excluded_by_protocol": not unit["support_ok"],
                        "exclusion_reason": unit["exclusion_reason"],
                    }
                )

    return {
        "schema_version": "nato-sers-p05-core-plan-v1",
        "protocol_version": contract["protocol_version"],
        "contract_sha256": contract_sha256,
        "input_pins": dict(contract["input_pins"]),
        "development_slots": slots,
        "guard_roles": guard_roles,
    }


def _fixture(
    *,
    manifest=None,
    roles=None,
    units=None,
    guard_contexts=None,
    contract=None,
):
    contract = contract if contract is not None else _fixture_contract()
    support = _build_support(manifest or _BASE_MANIFEST, roles or _BASE_ROLES)
    plan = _build_plan(
        contract, support, units or _BASE_UNITS, guard_contexts or _GUARD_CONTEXTS
    )
    return contract, support, plan


def _build_ledger(**kwargs) -> dict:
    contract, support, plan = _fixture(**kwargs)
    return build_development_ledger(plan=plan, support=support, contract=contract)


def _expect_failure(contract, support, plan, reason_code) -> None:
    with pytest.raises(DevelopmentLedgerError) as exc:
        build_development_ledger(plan=plan, support=support, contract=contract)
    assert exc.value.reason_code == reason_code


def _expect_fixture_failure(manifest, roles, reason_code) -> None:
    contract, support, plan = _fixture(manifest=manifest, roles=roles)
    _expect_failure(contract, support, plan, reason_code)


# --------------------------------------------------------------------------- #
# Optimizer schedule regression against the real public contract
# --------------------------------------------------------------------------- #


def test_optimizer_schedule_regression_on_real_contract():
    assert _optimizer_schedule(_load_public_contract()) == (120, 800)


def test_optimizer_schedule_reads_later_core_plan_not_optimization():
    contract = _load_public_contract()
    del contract["later_core_plan"]["minimum_epochs"]
    del contract["later_core_plan"]["maximum_epochs"]
    contract["optimization"]["minimum_epochs"] = 30
    contract["optimization"]["maximum_epochs"] = 200
    with pytest.raises(DevelopmentLedgerError) as exc:
        _optimizer_schedule(contract)
    assert exc.value.reason_code == "contract_optimizer_schedule_malformed"


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_base_fixture_counts_arithmetic_and_flags():
    contract, support, plan = _fixture()
    ledger = build_development_ledger(plan=plan, support=support, contract=contract)
    summary = ledger["summary"]
    assert ledger["plan_id"] == sha256_value(plan)
    assert ledger["contract_sha256"] == sha256_value(contract)
    assert summary["context_count"] == 3
    assert summary["inherited_unit_count"] == 3
    assert summary["guard_unit_count"] == 3
    assert summary["slot_count"] == 72
    assert summary["eligible_slot_count"] == 72
    assert summary["excluded_slot_count"] == 0
    assert summary["pseudo_domain_context_count"] == 1
    assert summary["master_cv_context_count"] == 1
    assert summary["inner_master_cv_context_count"] == 1
    assert summary["minimum_optimizer_updates_per_fit"] == 120
    assert summary["maximum_optimizer_updates_per_fit"] == 800
    assert summary["minimum_scheduled_optimizer_updates"] == 72 * 120
    assert summary["maximum_scheduled_optimizer_updates"] == 72 * 800
    expected_total = (208691 + 212851 + 208691 + 212851) * 18 * 4
    assert summary["checkpoint_tensor_bytes_total_eligible"] == expected_total
    assert summary["checkpoint_tensor_bytes_per_fit_max"] == 212851 * 4
    assert ledger["execution_authorized"] is False
    assert ledger["arrays_loaded"] is False
    assert ledger["fits_started"] == 0
    assert ledger["outer_evaluation_authorized"] is False
    assert ledger["summary"]["execution_authorized"] is False
    assert ledger["validation"]["execution_authorized"] is False
    assert ledger["validation"]["arrays_loaded"] is False


def test_slot_unit_referential_integrity():
    ledger = _build_ledger()
    unit_ids = {unit["unit_id"] for unit in ledger["units"]}
    assert len(unit_ids) == len(ledger["units"])
    counts = Counter(slot["unit_id"] for slot in ledger["slots"])
    assert set(counts) == unit_ids
    assert set(counts.values()) == {12}
    assert all(slot["planned"] is True for slot in ledger["slots"])


def test_sampling_capacity_counts_fitting_masters_only():
    ledger = _build_ledger()
    guard_units = [unit for unit in ledger["units"] if unit["unit_kind"] == "guard"]
    assert len(guard_units) == 3
    for unit in guard_units:
        assert unit["fitting_master_count"] == 6
        assert unit["validation_master_count"] == 3
        assert unit["master_count"] == 9
        assert unit["sampling_capacity"] == 6
        assert unit["sampling_capacity"] != 2 * unit["master_count"]
    assert ledger["summary"]["maximum_sampling_capacity"] == 6
    assert MAX_SAMPLING_CAPACITY == 48


def test_auxiliary_support_excludes_same_master_same_instrument_repeats():
    repeats = [
        {"master": "M1", "instrument": "I1", "target": "A"},
        {"master": "M1", "instrument": "I1", "target": "A"},
        {"master": "M2", "instrument": "I1", "target": "A"},
    ]
    assert _auxiliary_support(repeats) == {
        "same_chemical_positive_pairs": 2,
        "cross_instrument_master_pairs": 0,
    }
    cross = [
        {"master": "M1", "instrument": "I1", "target": "A"},
        {"master": "M1", "instrument": "I2", "target": "A"},
    ]
    assert _auxiliary_support(cross) == {
        "same_chemical_positive_pairs": 1,
        "cross_instrument_master_pairs": 1,
    }


def test_auxiliary_support_is_fitting_only():
    contract, support, plan = _fixture()
    ledger = build_development_ledger(plan=plan, support=support, contract=contract)
    unit = next(
        item for item in ledger["units"] if item["selection_unit_id"] == "pseudo:INS-P"
    )
    fitting_rows = [
        {
            "master": row["master_sample_id"],
            "instrument": row["instrument"],
            "target": row["target_analyte"],
        }
        for row in support.roles
        if row["role_id"] == "r-ps-fit"
    ]
    validation_rows = [
        {
            "master": row["master_sample_id"],
            "instrument": row["instrument"],
            "target": row["target_analyte"],
        }
        for row in support.roles
        if row["role_id"] == "r-ps-val"
    ]
    assert unit["auxiliary_support"] == _auxiliary_support(fitting_rows)
    assert unit["auxiliary_support"]["same_chemical_positive_pairs"] == 0
    assert (
        _auxiliary_support(fitting_rows + validation_rows)[
            "same_chemical_positive_pairs"
        ]
        == 3
    )


def test_shuffle_support_rows_is_invariant():
    contract, support, plan = _fixture()
    baseline = build_development_ledger(plan=plan, support=support, contract=contract)
    rng = random.Random(7)
    manifest = list(support.manifest)
    contexts = list(support.contexts)
    roles = list(support.roles)
    rng.shuffle(manifest)
    rng.shuffle(contexts)
    rng.shuffle(roles)
    shuffled = SupportInputs(
        manifest=tuple(manifest),
        contexts=tuple(contexts),
        roles=tuple(roles),
        manifest_sha256=support.manifest_sha256,
        contexts_sha256=support.contexts_sha256,
        roles_sha256=support.roles_sha256,
    )
    shuffled_plan = _build_plan(contract, shuffled, _BASE_UNITS, _GUARD_CONTEXTS)
    shuffled_ledger = build_development_ledger(
        plan=shuffled_plan, support=shuffled, contract=contract
    )
    assert shuffled_plan == plan
    assert shuffled_ledger == baseline


def test_builder_does_not_mutate_inputs():
    contract, support, plan = _fixture()
    contract_before = copy.deepcopy(contract)
    support_before = copy.deepcopy(support)
    plan_before = copy.deepcopy(plan)
    build_development_ledger(plan=plan, support=support, contract=contract)
    assert contract == contract_before
    assert support == support_before
    assert plan == plan_before


# --------------------------------------------------------------------------- #
# Binding and identity failures
# --------------------------------------------------------------------------- #


def test_plan_contract_identity_mismatch():
    contract, support, plan = _fixture()
    plan["contract_sha256"] = "0" * 64
    _expect_failure(contract, support, plan, "plan_contract_identity_mismatch")


def test_plan_pin_mismatch():
    contract, support, plan = _fixture()
    plan["input_pins"] = dict(plan["input_pins"])
    plan["input_pins"]["manifest_sha256"] = "e" * 64
    _expect_failure(contract, support, plan, "plan_pin_mismatch")


def test_support_pin_missing_is_rejected():
    contract, support, plan = _fixture()
    support = dataclasses.replace(support, manifest_sha256=None)
    _expect_failure(contract, support, plan, "support_pin_missing")


def test_support_pin_mismatch_is_rejected():
    contract, support, plan = _fixture()
    support = dataclasses.replace(support, roles_sha256="e" * 64)
    _expect_failure(contract, support, plan, "support_pin_mismatch")


# --------------------------------------------------------------------------- #
# Role and unit membership failures
# --------------------------------------------------------------------------- #


def test_unknown_role_name_rejected():
    contract, support, plan = _fixture()
    row = dict(support.roles[0])
    row["role"] = "bogus"
    row["role_id"] = "r-bogus"
    support = dataclasses.replace(support, roles=support.roles + (row,))
    _expect_failure(contract, support, plan, "unknown_role")


def test_foreign_context_role_rejected():
    contract, support, plan = _fixture()
    row = dict(support.roles[0])
    row["context_id"] = "nope"
    row["role_id"] = "r-foreign"
    support = dataclasses.replace(support, roles=support.roles + (row,))
    _expect_failure(contract, support, plan, "unknown_context")


def test_mislabeled_selection_role_unit_rejected():
    contract, support, plan = _fixture()
    row = {
        "context_id": "ctx_dev",
        "role_id": "r-bad-unit",
        "role": "selection_fit",
        "selection_unit_id": "wrong:u1",
        "observation_uid": "d-DA1",
        "master_sample_id": "DA1",
        "target_analyte": "A",
        "instrument": "INS-Q",
    }
    support = dataclasses.replace(support, roles=support.roles + (row,))
    _expect_failure(contract, support, plan, "selection_unit_prefix_mismatch")


def test_unknown_observation_uid_rejected():
    contract, support, plan = _fixture()
    row = dict(support.roles[0])
    row["role_id"] = "r-unknown-uid"
    row["observation_uid"] = "missing-uid"
    support = dataclasses.replace(support, roles=support.roles + (row,))
    _expect_failure(contract, support, plan, "role_uid_unknown")


def test_role_metadata_mismatch_rejected():
    contract, support, plan = _fixture()
    rows = list(support.roles)
    rows[0] = dict(rows[0])
    rows[0]["master_sample_id"] = "ZZZ"
    support = dataclasses.replace(support, roles=tuple(rows))
    _expect_failure(contract, support, plan, "role_metadata_mismatch")


def test_duplicate_selection_role_identity_rejected():
    contract, support, plan = _fixture()
    extra = []
    for uid, master, target, instrument in (
        ("d-DA1", "DA1", "A", "INS-Q"),
        ("d-DB1", "DB1", "B", "INS-Q"),
        ("d-DC1", "DC1", "C", "INS-Q"),
    ):
        extra.append(
            {
                "context_id": "ctx_dev",
                "role_id": "r-dev-fit-dup",
                "role": "selection_fit",
                "selection_unit_id": "outer_fold_as_inner:u1",
                "observation_uid": uid,
                "master_sample_id": master,
                "target_analyte": target,
                "instrument": instrument,
            }
        )
    support = dataclasses.replace(support, roles=support.roles + tuple(extra))
    _expect_failure(contract, support, plan, "duplicate_selection_role")


def test_unit_uid_overlap_rejected():
    contract, support, plan = _fixture()
    row = {
        "context_id": "ctx_dev",
        "role_id": "r-dev-val",
        "role": "selection_validation",
        "selection_unit_id": "outer_fold_as_inner:u1",
        "observation_uid": "d-DA1",
        "master_sample_id": "DA1",
        "target_analyte": "A",
        "instrument": "INS-Q",
    }
    support = dataclasses.replace(support, roles=support.roles + (row,))
    _expect_failure(contract, support, plan, "unit_uid_overlap")


def test_unit_master_overlap_rejected():
    manifest = _BASE_MANIFEST + (("d-DA1b", "DA1", "A", "INS-R"),)
    roles = []
    for entry in _BASE_ROLES:
        context_id, role_id, role, unit_id, uids = entry
        if role_id in ("r-dev-of", "r-dev-val"):
            roles.append((context_id, role_id, role, unit_id, uids + ("d-DA1b",)))
        else:
            roles.append(entry)
    _expect_fixture_failure(manifest, tuple(roles), "unit_master_overlap")


def test_held_instrument_leakage_rejected():
    manifest = _BASE_MANIFEST + (("m-EZ1", "EZ1", "A", "INS-Y"),)
    roles = []
    for entry in _BASE_ROLES:
        context_id, role_id, role, unit_id, uids = entry
        if role_id in ("r-mc-of", "r-mc-fit"):
            roles.append((context_id, role_id, role, unit_id, uids + ("m-EZ1",)))
        else:
            roles.append(entry)
    _expect_fixture_failure(manifest, tuple(roles), "held_instrument_in_unit")


def test_pseudo_instrument_leakage_rejected():
    manifest = _BASE_MANIFEST + (("p-MZ1", "MZ1", "A", "INS-P"),)
    roles = []
    for entry in _BASE_ROLES:
        context_id, role_id, role, unit_id, uids = entry
        if role_id in ("r-ps-of", "r-ps-fit"):
            roles.append((context_id, role_id, role, unit_id, uids + ("p-MZ1",)))
        else:
            roles.append(entry)
    _expect_fixture_failure(manifest, tuple(roles), "pseudo_instrument_in_fitting")


def test_missing_station_classes_rejected():
    roles = []
    for entry in _BASE_ROLES:
        context_id, role_id, role, unit_id, uids = entry
        if role_id == "r-dev-fit":
            roles.append(
                (
                    context_id,
                    role_id,
                    role,
                    unit_id,
                    tuple(uid for uid in uids if uid != "d-DC1"),
                )
            )
        else:
            roles.append(entry)
    _expect_fixture_failure(
        _BASE_MANIFEST, tuple(roles), "unit_missing_station_classes"
    )


def test_sampling_capacity_ceiling_enforced():
    manifest = list(_BASE_MANIFEST)
    extra = []
    for index in range(25):
        target = ("A", "B", "C")[index % 3]
        master = f"FX{index}"
        uid_a = f"fx{index}a"
        uid_b = f"fx{index}b"
        manifest.append((uid_a, master, target, "IX0"))
        manifest.append((uid_b, master, target, "IX1"))
        extra.extend([uid_a, uid_b])
    roles = []
    for entry in _BASE_ROLES:
        context_id, role_id, role, unit_id, uids = entry
        if role_id in ("r-dev-of", "r-dev-fit"):
            roles.append((context_id, role_id, role, unit_id, uids + tuple(extra)))
        else:
            roles.append(entry)
    _expect_fixture_failure(
        tuple(manifest), tuple(roles), "unit_sampling_capacity_exceeded"
    )


# --------------------------------------------------------------------------- #
# Guard and slot consistency failures
# --------------------------------------------------------------------------- #


def test_guard_uid_digest_corruption_rejected():
    contract, support, plan = _fixture()
    plan["guard_roles"][0]["fitting_role"]["uid_set_sha256"] = "0" * 64
    _expect_failure(contract, support, plan, "guard_unit_mismatch")


def test_guard_unit_coverage_mismatch_rejected():
    contract, support, plan = _fixture()
    plan["guard_roles"][0]["guard_unit_id"] = "P05GUARDUNIT-deadbeefdeadbeef"
    _expect_failure(contract, support, plan, "guard_unit_coverage_mismatch")


def test_unknown_inherited_unit_coverage_rejected():
    contract, support, plan = _fixture()
    for recipe in contract["recipes"]:
        for seed in contract["later_core_plan"]["seeds"]:
            plan["development_slots"].append(
                {
                    "slot_id": slot_identity(
                        contract_sha256=plan["contract_sha256"],
                        slot_kind=INHERITED_SLOT_KIND,
                        context_id="ctx_dev",
                        selection_unit_id="outer_fold_as_inner:u2",
                        fitting_role_id="r-dev-fit",
                        validation_role_id="r-dev-val",
                        guard_fold=None,
                        recipe_id=recipe["recipe_id"],
                        seed=seed,
                    ),
                    "slot_kind": INHERITED_SLOT_KIND,
                    "context_id": "ctx_dev",
                    "selection_unit_id": "outer_fold_as_inner:u2",
                    "fitting_role_id": "r-dev-fit",
                    "validation_role_id": "r-dev-val",
                    "guard_fold": None,
                    "recipe_id": recipe["recipe_id"],
                    "seed": seed,
                    "planned": True,
                    "excluded_by_protocol": False,
                    "exclusion_reason": None,
                }
            )
    _expect_failure(contract, support, plan, "inherited_unit_coverage_mismatch")


def test_incomplete_slot_product_rejected():
    contract, support, plan = _fixture()
    plan["development_slots"].pop()
    _expect_failure(contract, support, plan, "unit_slot_product_incomplete")


def test_inherited_slot_cannot_be_excluded():
    contract, support, plan = _fixture()
    for slot in plan["development_slots"]:
        if (
            slot["slot_kind"] == INHERITED_SLOT_KIND
            and slot["selection_unit_id"] == "outer_fold_as_inner:u1"
        ):
            slot["excluded_by_protocol"] = True
    _expect_failure(contract, support, plan, "inherited_slot_excluded")


def test_guard_exclusion_mismatch_rejected():
    contract, support, plan = _fixture()
    target_id = plan["guard_roles"][0]["guard_unit_id"]
    for slot in plan["development_slots"]:
        if (
            slot["slot_kind"] == GUARD_SLOT_KIND
            and slot["selection_unit_id"] == target_id
        ):
            slot["excluded_by_protocol"] = True
            slot["exclusion_reason"] = "forced"
    _expect_failure(contract, support, plan, "guard_exclusion_mismatch")


def test_malformed_exclusion_flag_rejected():
    contract, support, plan = _fixture()
    plan["development_slots"][0]["excluded_by_protocol"] = "yes"
    _expect_failure(contract, support, plan, "plan_slot_exclusion_flag_malformed")


# --------------------------------------------------------------------------- #
# Read-only CLI
# --------------------------------------------------------------------------- #


def _load_cli_module():
    path = _REPO_ROOT / "scripts" / "inspect_p05_development.py"
    spec = importlib.util.spec_from_file_location(
        "inspect_p05_development_cli", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_invalid_arguments_is_sanitized(capsys):
    module = _load_cli_module()
    code = module.main([])
    assert code == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {
        "status": "fail",
        "command": "inspect",
        "reason_code": "invalid_arguments",
    }


def test_cli_unavailable_roots_is_sanitized(capsys, tmp_path):
    module = _load_cli_module()
    missing = tmp_path / "does-not-exist"
    code = module.main(
        [
            "--project-root",
            str(missing),
            "--artifact-root",
            str(missing / "artifacts"),
            "--contract",
            str(missing / "contract.json"),
        ]
    )
    assert code == 1
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["status"] == "fail"
    assert payload["command"] == "inspect"
    assert payload["reason_code"]
    assert str(tmp_path) not in out
