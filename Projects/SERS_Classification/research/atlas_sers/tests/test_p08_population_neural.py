"""Synthetic metadata-only tests for the P08 neural support adapter.

Every plan used here is produced by the public ``build_population_plan`` from
invented but structurally valid manifests.  The tests exercise
``build_population_neural_support``; nothing fits a model, reads a spectrum,
computes a score or authorizes execution.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from atlas_sers.evaluation.p05_core_plan import (
    build_guard_fold_roles,
    validate_core_contract,
)
from atlas_sers.evaluation.p08_population_neural import (
    SCHEMA_VERSION,
    build_population_neural_support,
)
from atlas_sers.evaluation.p08_population_plan import (
    build_population_plan,
)
from atlas_sers.governance.canonical import sha256_value

CORE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "plan"
    / "contracts"
    / "p05_core_contract.json"
)
CLASSES = ("c0", "c1", "c2")
POPULATION_ID = "TOY"
POPULATION_SHA = sha256_value({"toy": 1})
_CEILING = 48

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


# --------------------------------------------------------------------------- #
# Canonical helpers (deliberately local, not imported from the adapter)
# --------------------------------------------------------------------------- #


def _hash_table(frame: pd.DataFrame) -> str:
    return sha256_value(
        {
            "columns": [str(column) for column in frame.columns],
            "records": frame.to_dict(orient="records"),
        }
    )


def _hash_set(values) -> str:
    return sha256_value(sorted(set(values)))


def reseal(plan):
    """Recanonicalise an invented plan after mutating one of its tables."""

    report = dict(plan.validation_report)
    hashes = {
        name: _hash_table(getattr(plan, attribute))
        for name, attribute in _TABLE_ATTRS
    }
    report["table_hashes"] = hashes
    report["plan_sha256"] = sha256_value(
        {
            "population_id": report["population_id"],
            "population_sha256": report["population_sha256"],
            "metadata_canonical_sha256": report["metadata_canonical_sha256"],
            "split_contract_sha256": report["split_contract_sha256"],
            "p02_contract_sha256": report["p02_contract_sha256"],
            "table_hashes": hashes,
        }
    )
    return dataclasses.replace(plan, validation_report=report)


def mutate_table(plan, attribute, mutate):
    frame = mutate(getattr(plan, attribute).copy())
    return reseal(
        dataclasses.replace(copy.deepcopy(plan), **{attribute: frame})
    )


# --------------------------------------------------------------------------- #
# Invented manifests
# --------------------------------------------------------------------------- #


def _rows(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=(
            "observation_uid",
            "master_sample_id",
            "station",
            "instrument",
            "target_analyte",
        ),
    )


def manifest_pattern_a(masters_per_class: int = 9) -> pd.DataFrame:
    """Every master measured on one held and two source instruments."""

    rows = []
    for class_index, label in enumerate(CLASSES):
        for master_index in range(masters_per_class):
            master = str(1000 + class_index * 100 + master_index)
            for instrument in ("H-1", "S-1", "S-2"):
                rows.append(
                    {
                        "observation_uid": f"U-{master}-{instrument}",
                        "master_sample_id": master,
                        "station": "ST1",
                        "instrument": instrument,
                        "target_analyte": label,
                    }
                )
    return _rows(rows)


def manifest_pattern_b(masters_per_class: int = 9) -> pd.DataFrame:
    """Every master held on H-1 plus one class-independent source instrument."""

    rows = []
    for class_index, label in enumerate(CLASSES):
        for master_index in range(masters_per_class):
            master = f"MB{class_index:02d}{master_index:04d}"
            rows.append(
                {
                    "observation_uid": f"U-{master}-H",
                    "master_sample_id": master,
                    "station": "ST1",
                    "instrument": "H-1",
                    "target_analyte": label,
                }
            )
            rows.append(
                {
                    "observation_uid": f"U-{master}-S",
                    "master_sample_id": master,
                    "station": "ST1",
                    "instrument": f"S-{master_index % 3 + 1}",
                    "target_analyte": label,
                }
            )
    return _rows(rows)


def manifest_pattern_c(masters_per_class: int = 12) -> pd.DataFrame:
    """Only the first three masters per class carry a source instrument."""

    rows = []
    for class_index, label in enumerate(CLASSES):
        for master_index in range(masters_per_class):
            master = f"M{class_index}-{master_index}"
            rows.append(
                {
                    "observation_uid": f"U-{master}-H",
                    "master_sample_id": master,
                    "station": "ST1",
                    "instrument": "H-1",
                    "target_analyte": label,
                }
            )
            if master_index < 3:
                rows.append(
                    {
                        "observation_uid": f"U-{master}-S",
                        "master_sample_id": master,
                        "station": "ST1",
                        "instrument": f"S-{master_index % 3 + 1}",
                        "target_analyte": label,
                    }
                )
    return _rows(rows)


# --------------------------------------------------------------------------- #
# Contract / plan builders
# --------------------------------------------------------------------------- #


def _load_core_contract() -> dict:
    with CORE_CONTRACT_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _split_contract(
    *,
    domains=("ST1:H-1",),
    test_classes: int = 3,
    pooled_min: int = 3,
    outer_folds: int = 4,
    seeds=(17,),
) -> dict:
    return {
        "canonical_algorithm": (
            "StratifiedGroupKFold with shuffle true and repeat seed"
        ),
        "stratification_label": "target_analyte",
        "group_label": "master_sample_id",
        "outer_repeat_seeds": list(seeds),
        "outer_folds_per_station": outer_folds,
        "primary_domain_eligibility": {
            "requirements": {
                "test_classes": test_classes,
                "pooled_test_masters_minimum": pooled_min,
            },
            "domains": list(domains),
        },
    }


def _build_plan(
    manifest,
    split_contract=None,
    population_id: str = POPULATION_ID,
    population_sha256: str = POPULATION_SHA,
):
    return build_population_plan(
        manifest=manifest,
        population_id=population_id,
        population_sha256=population_sha256,
        split_contract=(
            _split_contract() if split_contract is None else split_contract
        ),
        p02_contract={"inner_master_folds": 3},
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def core_contract() -> dict:
    return _load_core_contract()


@pytest.fixture(scope="module")
def plan_a():
    return _build_plan(manifest_pattern_a())


@pytest.fixture(scope="module")
def plan_b():
    return _build_plan(manifest_pattern_b())


@pytest.fixture(scope="module")
def plan_c():
    return _build_plan(manifest_pattern_c())


@pytest.fixture(scope="module")
def plan_d():
    return _build_plan(manifest_pattern_a(12))


@pytest.fixture(scope="module")
def plan_e():
    return _build_plan(
        manifest_pattern_a(), split_contract=_split_contract(pooled_min=10_000)
    )


@pytest.fixture(scope="module")
def neural_a(plan_a, core_contract):
    return build_population_neural_support(
        population_plan=plan_a, core_contract=core_contract
    )


@pytest.fixture(scope="module")
def neural_b(plan_b, core_contract):
    return build_population_neural_support(
        population_plan=plan_b, core_contract=core_contract
    )


@pytest.fixture(scope="module")
def neural_c(plan_c, core_contract):
    return build_population_neural_support(
        population_plan=plan_c, core_contract=core_contract
    )


@pytest.fixture(scope="module")
def neural_d(plan_d, core_contract):
    return build_population_neural_support(
        population_plan=plan_d, core_contract=core_contract
    )


@pytest.fixture(scope="module")
def neural_e(plan_e, core_contract):
    return build_population_neural_support(
        population_plan=plan_e, core_contract=core_contract
    )


# --------------------------------------------------------------------------- #
# Structural / authorization invariants
# --------------------------------------------------------------------------- #


def test_schema_version(neural_a):
    assert neural_a["schema_version"] == SCHEMA_VERSION


def test_result_sha256_excludes_itself(neural_a):
    payload = dict(neural_a)
    recorded = payload.pop("result_sha256")
    assert recorded == sha256_value(payload)


def test_no_execution_authorization(neural_a):
    assert neural_a["execution_authorized"] is False
    assert neural_a["scientific_operations"] == 0
    assert neural_a["numerical_readiness_verified"] is False
    assert neural_a["exact_model_operation_ledger_complete"] is False


def test_contexts_are_preserved(plan_a, neural_a):
    planned = set(plan_a.context_registry["context_id"])
    reported = {context["context_id"] for context in neural_a["contexts"]}
    assert reported == planned
    assert len(neural_a["contexts"]) == len(planned)


def test_counts_are_internally_consistent(neural_b):
    counts = neural_b["counts"]
    assert counts["context_count"] == len(neural_b["contexts"])
    assert counts["guard_unit_count"] == len(neural_b["guard_units"])
    assert (
        counts["eligible_context_count"] + counts["ineligible_context_count"]
        == counts["context_count"]
    )
    assert (
        counts["supported_inherited_selection_unit_count"]
        <= counts["inherited_selection_unit_count"]
    )


def test_outputs_are_ordered(neural_b):
    ids = [context["context_id"] for context in neural_b["contexts"]]
    assert ids == sorted(ids)
    keys = [
        (str(unit["context_id"]), int(unit["guard_fold"]))
        for unit in neural_b["guard_units"]
    ]
    assert keys == sorted(keys)


def test_deterministic(plan_a, core_contract):
    first = build_population_neural_support(
        population_plan=plan_a, core_contract=core_contract
    )
    second = build_population_neural_support(
        population_plan=plan_a, core_contract=core_contract
    )
    assert first == second


def test_inputs_are_not_mutated(plan_a, core_contract):
    plan_before = copy.deepcopy(plan_a)
    core_before = copy.deepcopy(core_contract)
    build_population_neural_support(
        population_plan=plan_a, core_contract=core_contract
    )
    for _name, attribute in _TABLE_ATTRS:
        assert getattr(plan_a, attribute).equals(getattr(plan_before, attribute))
    assert plan_a.validation_report == plan_before.validation_report
    assert core_contract == core_before


# --------------------------------------------------------------------------- #
# Pattern A: master-CV fallback, compact capacity
# --------------------------------------------------------------------------- #


def test_pattern_a_falls_back_to_master_cv(neural_a):
    contexts = neural_a["contexts"]
    assert contexts
    assert {context["selection_mode"] for context in contexts} == {"master_cv"}
    for context in contexts:
        assert context["ordinary_neural_metadata_supported"] is True
        assert context["neural_source_calibration_metadata_supported"] is True
        assert context["g3_structurally_comparable"] is False
        assert context["structural_D0M_fallback_required"] is True
        assert (
            context["structural_D0M_fallback_cause"]
            == "master_cv_structural_fallback"
        )


def test_pattern_a_capacity_below_ceiling(neural_a):
    ceiling = neural_a["counts"]["sampler_batch_size_ceiling"]
    assert ceiling == _CEILING
    for context in neural_a["contexts"]:
        assert context["outer_batch_capacity"] <= ceiling
        assert context["max_inherited_batch_capacity"] <= ceiling


def test_numeric_looking_master_ids_remain_strings(plan_a, neural_a):
    masters = set(plan_a.role_registry["master_sample_id"])
    assert masters and all(isinstance(master, str) for master in masters)
    context = neural_a["contexts"][0]
    roles = plan_a.role_registry
    mask = (
        (roles["context_id"] == context["context_id"])
        & (roles["purpose"] == "outer")
        & (roles["role"] == "source")
    )
    report = plan_a.context_registry.set_index("context_id").loc[
        context["context_id"]
    ]
    assert context["outer_source_valid"] is True
    assert report["source_master_set_sha256"] == _hash_set(
        roles.loc[mask, "master_sample_id"].tolist()
    )


# --------------------------------------------------------------------------- #
# Pattern B: dense pseudo-domain selection with supported guards
# --------------------------------------------------------------------------- #


def test_pattern_b_is_pseudo_domain(neural_b):
    contexts = neural_b["contexts"]
    assert contexts
    assert {context["selection_mode"] for context in contexts} == {
        "pseudo_domain"
    }
    for context in contexts:
        assert context["ordinary_neural_metadata_supported"] is True
        assert context["neural_source_calibration_metadata_supported"] is True
        assert context["structural_D0M_fallback_required"] is False


def test_pattern_b_guards_are_supported(neural_b):
    for context in neural_b["contexts"]:
        assert context["guard_unit_count"] == 3
        assert context["guard_supported_count"] == 3
        assert context["g3_structurally_comparable"] is True


def test_pattern_b_guard_counts(neural_b):
    counts = neural_b["counts"]
    assert counts["pseudo_domain_context_count"] == counts["context_count"]
    assert counts["guard_unit_count"] == 3 * counts["context_count"]
    assert counts["guard_supported_count"] == counts["guard_unit_count"]


def test_guard_context_ids_are_current_population_contexts(plan_b, neural_b):
    planned = set(plan_b.context_registry["context_id"])
    pseudo = {
        context["context_id"]
        for context in neural_b["contexts"]
        if context["selection_mode"] == "pseudo_domain"
    }
    guard_ids = {str(unit["context_id"]) for unit in neural_b["guard_units"]}
    assert guard_ids == pseudo
    assert guard_ids <= planned
    assert all(cid.startswith("P08POPCTX-") for cid in guard_ids)


def test_guard_units_match_inherited_builder(plan_b, neural_b, core_contract):
    digest = validate_core_contract(core_contract)["contract_sha256"]
    meta = {context["context_id"]: context for context in neural_b["contexts"]}
    roles = plan_b.role_registry.to_dict(orient="records")
    guards_by_context: dict[str, list[dict]] = {}
    for unit in neural_b["guard_units"]:
        guards_by_context.setdefault(str(unit["context_id"]), []).append(unit)
    assert guards_by_context
    for context_id, units in guards_by_context.items():
        context = meta[context_id]
        source = [
            {
                "uid": row["observation_uid"],
                "master": row["master_sample_id"],
                "instrument": row["instrument"],
                "target": row["target_analyte"],
            }
            for row in roles
            if row["context_id"] == context_id
            and row["purpose"] == "outer"
            and row["role"] == "source"
        ]
        test_masters = sorted(
            {
                row["master_sample_id"]
                for row in roles
                if row["context_id"] == context_id
                and row["purpose"] == "outer"
                and row["role"] == "test"
            }
        )
        expected = build_guard_fold_roles(
            contract_sha256=digest,
            context_id=context_id,
            station=context["station"],
            outer_fit_rows=source,
            held_instrument=context["held_instrument"],
            outer_test_masters=test_masters,
        )
        actual = {int(unit["guard_fold"]): unit for unit in units}
        assert set(actual) == {int(unit["guard_fold"]) for unit in expected}
        for unit in expected:
            got = actual[int(unit["guard_fold"])]
            for key, value in unit.items():
                assert got[key] == value
            assert got["population_id"] == POPULATION_ID
            assert got["population_sha256"] == POPULATION_SHA


# --------------------------------------------------------------------------- #
# Pattern C: sparse-guard fold with supported selection but incomplete guards
# --------------------------------------------------------------------------- #


def test_pattern_c_sparse_guard_not_g3(neural_c):
    pseudo = [
        context
        for context in neural_c["contexts"]
        if context["selection_mode"] == "pseudo_domain"
    ]
    assert len(pseudo) == 1
    context = pseudo[0]
    assert context["ordinary_neural_metadata_supported"] is True
    assert context["neural_source_calibration_metadata_supported"] is True
    assert context["g3_structurally_comparable"] is False
    assert context["structural_D0M_fallback_required"] is True
    assert (
        context["structural_D0M_fallback_cause"]
        == "guard_support_structural_fallback"
    )
    assert context["guard_unit_count"] == 3
    assert 0 < context["guard_supported_count"] < 3


def test_pattern_c_other_folds_unsupported(neural_c):
    others = [
        context
        for context in neural_c["contexts"]
        if context["selection_mode"] != "pseudo_domain"
    ]
    assert others
    for context in others:
        assert context["selection_mode"] in {"unsupported", "not_applicable"}
        assert context["ordinary_neural_metadata_supported"] is False
        assert context["neural_source_calibration_metadata_supported"] is False


def test_neural_support_ignores_classical_metadata_mask(plan_c, neural_c):
    pseudo = [
        context
        for context in neural_c["contexts"]
        if context["selection_mode"] == "pseudo_domain"
    ]
    assert pseudo
    classical = plan_c.context_registry.set_index("context_id")
    for context in pseudo:
        row = classical.loc[context["context_id"]]
        assert bool(row["metadata_ready"]) is False
        assert context["ordinary_neural_metadata_supported"] is True
        assert context["neural_source_calibration_metadata_supported"] is True


# --------------------------------------------------------------------------- #
# Pattern D: capacity above the sampler ceiling
# --------------------------------------------------------------------------- #


def test_pattern_d_over_capacity_contexts_rejected(neural_d):
    over = [
        context
        for context in neural_d["contexts"]
        if context["outer_batch_capacity"]
        and context["outer_batch_capacity"] > _CEILING
    ]
    assert over
    for context in over:
        assert context["ordinary_neural_metadata_supported"] is False
        assert context["neural_source_calibration_metadata_supported"] is False
        assert "outer_capacity_exceeded" in context["unavailable_reasons"]


def test_pattern_d_does_not_mutate_core_contract(plan_d, core_contract):
    before = copy.deepcopy(core_contract)
    result = build_population_neural_support(
        population_plan=plan_d, core_contract=core_contract
    )
    assert core_contract == before
    assert result["core_contract_sha256"] == validate_core_contract(
        core_contract
    )["contract_sha256"]


# --------------------------------------------------------------------------- #
# Pattern E: pooled-ineligible domain retains contexts but has no units
# --------------------------------------------------------------------------- #


def test_pattern_e_ineligible_domain_has_no_units(plan_e, neural_e):
    assert neural_e["contexts"]
    for context in neural_e["contexts"]:
        assert context["domain_eligible"] is False
        assert context["selection_mode"] == "not_applicable"
        assert context["ordinary_neural_metadata_supported"] is False
        assert context["neural_source_calibration_metadata_supported"] is False
    assert neural_e["counts"]["inherited_selection_unit_count"] == 0
    assert neural_e["counts"]["guard_unit_count"] == 0
    assert len(neural_e["contexts"]) == len(plan_e.context_registry)


# --------------------------------------------------------------------------- #
# Inherited units, calibration source and disjointness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", ["neural_a", "neural_b"])
def test_all_inherited_units_supported(request, fixture_name):
    result = request.getfixturevalue(fixture_name)
    counts = result["counts"]
    assert (
        counts["supported_inherited_selection_unit_count"]
        == counts["inherited_selection_unit_count"]
    )
    for context in result["contexts"]:
        assert context["inherited_selection_unit_count"] == len(
            context["inherited_selection_unit_ids"]
        )
        assert (
            context["supported_inherited_selection_unit_ids"]
            == context["inherited_selection_unit_ids"]
        )


def test_calibration_uses_selection_validation_rows_only(plan_b, neural_b):
    roles = plan_b.role_registry
    for context in neural_b["contexts"]:
        context_id = context["context_id"]
        validation = roles[
            (roles["context_id"] == context_id)
            & (roles["purpose"] == "selection")
            & (roles["role"] == "validation")
        ]
        outer_source = roles[
            (roles["context_id"] == context_id)
            & (roles["purpose"] == "outer")
            & (roles["role"] == "source")
        ]
        outer_test = roles[
            (roles["context_id"] == context_id)
            & (roles["purpose"] == "outer")
            & (roles["role"] == "test")
        ]
        calibration = context["neural_source_calibration"]
        assert calibration["appearance_count"] == len(validation)
        assert (
            calibration["distinct_validation_uid_count"]
            == validation["observation_uid"].nunique()
        )
        assert (
            calibration["distinct_validation_master_count"]
            == validation["master_sample_id"].nunique()
        )
        assert (
            calibration["appearance_count"]
            >= calibration["distinct_validation_uid_count"]
        )
        assert set(validation["observation_uid"]).isdisjoint(
            set(outer_test["observation_uid"])
        )
        assert set(validation["observation_uid"]) <= set(
            outer_source["observation_uid"]
        )
        assert context["held_instrument"] not in set(validation["instrument"])


def test_selection_roles_exclude_held_and_overlap(plan_b):
    selection = plan_b.role_registry[plan_b.role_registry["purpose"] == "selection"]
    assert not selection.empty
    for (_context_id, _unit_id), group in selection.groupby(
        ["context_id", "unit_id"], sort=False
    ):
        fit = set(group.loc[group["role"] == "fit", "master_sample_id"])
        validation = set(
            group.loc[group["role"] == "validation", "master_sample_id"]
        )
        assert fit.isdisjoint(validation)
        assert "H-1" not in set(group["instrument"])


# --------------------------------------------------------------------------- #
# Corruption must be rejected with ValueError
# --------------------------------------------------------------------------- #


def test_table_digest_mismatch_rejected(plan_a, core_contract):
    broken = copy.deepcopy(plan_a)
    frame = broken.domain_registry.copy()
    frame.loc[frame.index[0], "observed_rows"] = (
        int(frame.loc[frame.index[0], "observed_rows"]) + 1
    )
    broken = dataclasses.replace(broken, domain_registry=frame)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_plan_digest_mismatch_rejected(plan_a, core_contract):
    report = dict(plan_a.validation_report)
    report["plan_sha256"] = "0" * 64
    broken = dataclasses.replace(
        copy.deepcopy(plan_a), validation_report=report
    )
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def _bump_ceiling_to_zero(contract):
    contract["sampler"]["batch_size_ceiling"] = 0


def _bump_ceiling_to_bool(contract):
    contract["sampler"]["batch_size_ceiling"] = True


def _tamper_protocol(contract):
    contract["protocol_version"] = "tampered"


def _shrink_parameter_budget(contract):
    contract["model"]["maximum_parameters_exclusive"] = 1


def _drop_optimization(contract):
    contract.pop("optimization", None)


@pytest.mark.parametrize(
    "mutator",
    [
        _bump_ceiling_to_zero,
        _bump_ceiling_to_bool,
        _tamper_protocol,
        _shrink_parameter_budget,
        _drop_optimization,
    ],
)
def test_core_contract_mutations_rejected(plan_a, core_contract, mutator):
    mutated = copy.deepcopy(core_contract)
    mutator(mutated)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=plan_a, core_contract=mutated
        )


@pytest.mark.parametrize("value", [True, False, 1, -1, 2, 1.5, "0"])
def test_report_scientific_operations_must_be_zero(plan_a, core_contract, value):
    report = dict(plan_a.validation_report)
    report["scientific_operations"] = value
    broken = reseal(
        dataclasses.replace(copy.deepcopy(plan_a), validation_report=report)
    )
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_report_execution_authorized_true_rejected(plan_a, core_contract):
    report = dict(plan_a.validation_report)
    report["execution_authorized"] = True
    broken = reseal(
        dataclasses.replace(copy.deepcopy(plan_a), validation_report=report)
    )
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


@pytest.mark.parametrize(
    "attribute", ["context_registry", "unit_registry", "role_registry"]
)
def test_duplicate_registry_rows_rejected(plan_b, core_contract, attribute):
    def mutate(frame):
        return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    broken = mutate_table(plan_b, attribute, mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


@pytest.mark.parametrize(
    "attribute,column",
    [
        ("unit_registry", "context_id"),
        ("role_registry", "context_id"),
    ],
)
def test_unknown_context_rejected(plan_b, core_contract, attribute, column):
    def mutate(frame):
        frame = frame.copy()
        frame.loc[frame.index[0], column] = "P08POPCTX-UNKNOWN"
        return frame

    broken = mutate_table(plan_b, attribute, mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_selection_role_unknown_unit_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        mask = frame["purpose"] == "selection"
        frame.loc[frame.index[mask][0], "unit_id"] = "ghost-unit"
        return frame

    broken = mutate_table(plan_b, "role_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_malformed_selection_mode_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        frame.loc[frame.index[0], "selection_mode"] = "mystery"
        return frame

    broken = mutate_table(plan_b, "context_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_malformed_boolean_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        frame["domain_eligible"] = frame["domain_eligible"].astype(object)
        frame.loc[frame.index[0], "domain_eligible"] = 1
        return frame

    broken = mutate_table(plan_b, "context_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_blank_context_id_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        frame.loc[frame.index[0], "context_id"] = "   "
        return frame

    broken = mutate_table(plan_b, "context_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("population_id", "OTHER"),
        ("population_sha256", "0" * 64),
    ],
)
def test_context_population_provenance_mismatch_rejected(
    plan_b, core_contract, column, value
):
    def mutate(frame):
        frame = frame.copy()
        frame.loc[frame.index[0], column] = value
        return frame

    broken = mutate_table(plan_b, "context_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_context_station_domain_mismatch_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        frame.loc[frame.index[0], "station"] = "ST2"
        return frame

    broken = mutate_table(plan_b, "context_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


@pytest.mark.parametrize(
    "column",
    [
        "source_rows",
        "source_masters",
        "source_classes",
        "test_rows",
        "test_masters",
        "test_classes",
    ],
)
def test_context_count_mismatch_rejected(plan_b, core_contract, column):
    def mutate(frame):
        frame = frame.copy()
        frame.loc[frame.index[0], column] = (
            int(frame.loc[frame.index[0], column]) + 1
        )
        return frame

    broken = mutate_table(plan_b, "context_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


@pytest.mark.parametrize(
    "column",
    [
        "source_observation_set_sha256",
        "source_master_set_sha256",
        "test_observation_set_sha256",
        "test_master_set_sha256",
    ],
)
def test_context_set_hash_mismatch_rejected(plan_b, core_contract, column):
    def mutate(frame):
        frame = frame.copy()
        frame.loc[frame.index[0], column] = "0" * 64
        return frame

    broken = mutate_table(plan_b, "context_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_held_instrument_in_selection_fit_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        mask = (frame["purpose"] == "selection") & (frame["role"] == "fit")
        frame.loc[frame.index[mask][0], "instrument"] = "H-1"
        return frame

    broken = mutate_table(plan_b, "role_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_selection_unit_master_overlap_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        mask = (frame["purpose"] == "selection") & (frame["role"] == "validation")
        row = frame.loc[frame.index[mask][0]].copy()
        row["role"] = "fit"
        return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)

    broken = mutate_table(plan_b, "role_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_conflicting_uid_identity_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        mask = frame["purpose"] == "selection"
        frame.loc[frame.index[mask][0], "instrument"] = "S-99"
        return frame

    broken = mutate_table(plan_b, "role_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_conflicting_master_class_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        mask = (frame["purpose"] == "outer") & (frame["role"] == "test")
        frame.loc[frame.index[mask][0], "target_analyte"] = "cX"
        return frame

    broken = mutate_table(plan_b, "role_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_false_support_unit_rejected(plan_b, core_contract):
    def mutate(frame):
        frame = frame.copy()
        mask = frame["purpose"] == "selection"
        frame.loc[frame.index[mask][0], "support"] = False
        return frame

    broken = mutate_table(plan_b, "unit_registry", mutate)
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=broken, core_contract=core_contract
        )


def test_supported_unit_missing_class_rejected(plan_b, core_contract):
    plan = copy.deepcopy(plan_b)
    units = plan.unit_registry.copy()
    roles = plan.role_registry.copy()
    unit_index = units.index[units["purpose"] == "selection"][0]
    context_id = str(units.loc[unit_index, "context_id"])
    unit_id = str(units.loc[unit_index, "unit_id"])
    validation_mask = (
        (roles["context_id"] == context_id)
        & (roles["purpose"] == "selection")
        & (roles["unit_id"] == unit_id)
        & (roles["role"] == "validation")
    )
    target_class = roles.loc[roles.index[validation_mask][0], "target_analyte"]
    roles = roles.loc[
        ~(validation_mask & (roles["target_analyte"] == target_class))
    ].reset_index(drop=True)
    remaining = roles[
        (roles["context_id"] == context_id)
        & (roles["purpose"] == "selection")
        & (roles["unit_id"] == unit_id)
        & (roles["role"] == "validation")
    ]
    units.loc[unit_index, "validation_rows"] = int(len(remaining))
    units.loc[unit_index, "validation_masters"] = int(
        remaining["master_sample_id"].nunique()
    )
    units.loc[unit_index, "validation_classes"] = int(
        remaining["target_analyte"].nunique()
    )
    units.loc[unit_index, "validation_observation_set_sha256"] = _hash_set(
        remaining["observation_uid"].tolist()
    )
    plan = reseal(
        dataclasses.replace(plan, unit_registry=units, role_registry=roles)
    )
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan=plan, core_contract=core_contract
        )


def test_empty_held_test_fold_not_ordinary_supported(plan_a, core_contract):
    plan = copy.deepcopy(plan_a)
    roles = plan.role_registry.copy()
    contexts = plan.context_registry.copy()
    context_id = str(contexts["context_id"].iloc[0])
    test_mask = (
        (roles["context_id"] == context_id)
        & (roles["purpose"] == "outer")
        & (roles["role"] == "test")
    )
    roles = roles.loc[~test_mask].reset_index(drop=True)
    row_mask = contexts["context_id"] == context_id
    empty_hash = _hash_set([])
    contexts.loc[row_mask, "test_rows"] = 0
    contexts.loc[row_mask, "test_masters"] = 0
    contexts.loc[row_mask, "test_classes"] = 0
    contexts.loc[row_mask, "test_observation_set_sha256"] = empty_hash
    contexts.loc[row_mask, "test_master_set_sha256"] = empty_hash
    plan = reseal(
        dataclasses.replace(
            plan, role_registry=roles, context_registry=contexts
        )
    )
    result = build_population_neural_support(
        population_plan=plan, core_contract=core_contract
    )
    context = next(
        entry for entry in result["contexts"] if entry["context_id"] == context_id
    )
    assert context["outer_test_valid"] is False
    assert context["ordinary_neural_metadata_supported"] is False
    assert "empty_outer_test" in context["unavailable_reasons"]


def test_population_plan_type_required(core_contract):
    with pytest.raises(ValueError):
        build_population_neural_support(
            population_plan={"not": "a plan"}, core_contract=core_contract
        )


def test_population_plan_overlap_declared_rejected(plan_b, core_contract):
    plan = copy.deepcopy(plan_b)
    roles = plan.role_registry
    units = plan.unit_registry
    source = roles[
        (roles["purpose"] == "selection") & (roles["role"] == "validation")
    ].iloc[0]
    context_id = source["context_id"]
    unit_id = source["unit_id"]
    fit_row = source.copy()
    fit_row["role"] = "fit"
    roles = pd.concat([roles, pd.DataFrame([fit_row])], ignore_index=True)
    fit_mask = (
        (roles["context_id"] == context_id)
        & (roles["unit_id"] == unit_id)
        & (roles["purpose"] == "selection")
        & (roles["role"] == "fit")
    )
    fit = roles[fit_mask]
    unit_mask = (
        (units["context_id"] == context_id)
        & (units["unit_id"] == unit_id)
        & (units["purpose"] == "selection")
    )
    units.loc[unit_mask, "fit_rows"] = len(fit)
    units.loc[unit_mask, "fit_masters"] = fit["master_sample_id"].nunique()
    units.loc[unit_mask, "fit_classes"] = fit["target_analyte"].nunique()
    units.loc[unit_mask, "fit_observation_set_sha256"] = _hash_set(
        fit["observation_uid"]
    )
    units.loc[unit_mask, "master_disjoint"] = False
    plan = dataclasses.replace(plan, role_registry=roles, unit_registry=units)
    plan = reseal(plan)
    with pytest.raises(ValueError, match="fit and validation masters overlap"):
        build_population_neural_support(
            population_plan=plan, core_contract=core_contract
        )


def test_population_plan_pseudo_domain_context_rejected(plan_e, core_contract):
    plan = copy.deepcopy(plan_e)
    plan.context_registry["selection_mode"] = "pseudo_domain"
    plan = reseal(plan)
    with pytest.raises(ValueError, match="ineligible context"):
        build_population_neural_support(
            population_plan=plan, core_contract=core_contract
        )


def test_population_plan_not_applicable_context_rejected(plan_b, core_contract):
    plan = copy.deepcopy(plan_b)
    plan.context_registry["selection_mode"] = "not_applicable"
    plan = reseal(plan)
    with pytest.raises(ValueError, match="eligible context"):
        build_population_neural_support(
            population_plan=plan, core_contract=core_contract
        )


def test_population_plan_class_sparse_held_context_supported(plan_a, core_contract):
    plan = copy.deepcopy(plan_a)
    roles = plan.role_registry
    contexts = plan.context_registry
    context_id = contexts.iloc[0]["context_id"]
    test_mask = (
        (roles["context_id"] == context_id)
        & (roles["purpose"] == "outer")
        & (roles["role"] == "test")
    )
    held = roles[test_mask]
    assert not held.empty
    first_class = held.iloc[0]["target_analyte"]
    keep_mask = test_mask & (roles["target_analyte"] == first_class)
    held_kept = roles[keep_mask]
    assert not held_kept.empty
    assert held_kept["target_analyte"].nunique() == 1
    roles = roles[~test_mask | keep_mask].copy()
    context_mask = contexts["context_id"] == context_id
    contexts.loc[context_mask, "test_rows"] = len(held_kept)
    contexts.loc[context_mask, "test_masters"] = held_kept["master_sample_id"].nunique()
    contexts.loc[context_mask, "test_classes"] = held_kept["target_analyte"].nunique()
    contexts.loc[context_mask, "test_observation_set_sha256"] = _hash_set(
        held_kept["observation_uid"]
    )
    contexts.loc[context_mask, "test_master_set_sha256"] = _hash_set(
        held_kept["master_sample_id"]
    )
    plan = dataclasses.replace(
        plan, role_registry=roles, context_registry=contexts
    )
    plan = reseal(plan)
    result = build_population_neural_support(
        population_plan=plan, core_contract=core_contract
    )
    row = next(r for r in result["contexts"] if r["context_id"] == context_id)
    assert row["ordinary_neural_metadata_supported"]
    assert row["neural_source_calibration_metadata_supported"]
    assert row["outer_test_valid"]


def test_population_plan_shuffled_manifest_matches(neural_b, core_contract):
    plan = _build_plan(manifest_pattern_b().sample(frac=1, random_state=3))
    assert (
        build_population_neural_support(
            population_plan=plan, core_contract=core_contract
        )
        == neural_b
    )
