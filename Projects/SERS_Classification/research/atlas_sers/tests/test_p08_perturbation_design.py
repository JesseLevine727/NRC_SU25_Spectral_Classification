"""Contract tests for the P08 perturbation case manifest enumerator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from atlas_sers.evaluation.p08_perturbation_design import build_perturbation_case_manifest
from atlas_sers.governance.canonical import sha256_value

PACKAGE = Path(__file__).resolve().parents[1]
DESIGN_PATH = PACKAGE / "plan" / "contracts" / "p08_perturbation_design.json"
SCHEMA_VERSION = "nato-sers-p08-perturbation-cases-v1"
CLEAN_CASE_ID = "P08-STRESS-CLEAN"
FAMILY_ORDER = ("shift", "slope", "quadratic", "gaussian", "impulse", "clipping")
FAMILY_COUNTS = {
    "shift": 11,
    "slope": 9,
    "quadratic": 5,
    "gaussian": 41,
    "impulse": 31,
    "clipping": 4,
}
LEVELS = {
    "shift": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    "slope": [-0.1, -0.075, -0.05, -0.025, 0.025, 0.05, 0.075, 0.1],
    "quadratic": [0.025, 0.05, 0.075, 0.1],
    "gaussian": [0.5, 0.75, 0.9, 0.95],
    "impulse": [1, 3, 5],
    "clipping": [0.999, 0.995, 0.99],
}
STOCHASTIC = {"gaussian", "impulse"}
SOURCE_BINDINGS = {
    "primary_statistics_contract_sha256": PACKAGE
    / "plan"
    / "contracts"
    / "p08_statistics_contract.json",
    "P01_contract_sha256": PACKAGE / "plan" / "contracts" / "p01_governance_contract.json",
    "P01_transform_source_sha256": PACKAGE
    / "src"
    / "atlas_sers"
    / "preprocessing"
    / "representations.py",
    "native_QC_source_sha256": PACKAGE / "src" / "atlas_sers" / "data" / "native.py",
}
CASE_KEYS = {"case_id", "family", "value", "replicate_index"}


@pytest.fixture()
def design():
    return json.loads(DESIGN_PATH.read_text())


@pytest.fixture()
def manifest(design):
    return build_perturbation_case_manifest(design=design)


def _derived_id(family, value, replicate_index):
    return "P08-STRESS-" + sha256_value(
        {"family": family, "value": value, "replicate_index": replicate_index}
    )


def _rejects(design, mutate):
    candidate = copy.deepcopy(design)
    mutate(candidate)
    with pytest.raises(ValueError):
        build_perturbation_case_manifest(design=candidate)


def test_flags_schema_hashes_and_source_evidence(manifest, design):
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["execution_authorized"] is False
    assert manifest["scientific_operations"] == 0
    assert manifest["numerical_perturbations_computed"] is False
    assert manifest["job_ledger_complete"] is False
    assert manifest["input_provenance_independently_verified"] is False
    assert manifest["design_sha256"] == sha256_value(design)
    assert manifest["case_definition_sha256"] == sha256_value(design["case_definition"])
    body = {key: value for key, value in manifest.items() if key != "report_sha256"}
    assert manifest["report_sha256"] == sha256_value(body)
    for key, path in SOURCE_BINDINGS.items():
        assert (
            design["immutable_source_evidence"][key]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_case_inventory_summary_and_zero_alias(manifest):
    cases = manifest["cases"]
    assert isinstance(cases, list) and len(cases) == 96
    assert all(set(case) == CASE_KEYS for case in cases)
    assert cases[0] == {
        "case_id": CLEAN_CASE_ID,
        "family": "clean",
        "value": 0,
        "replicate_index": None,
    }
    assert manifest["summary"] == {
        "unique_cases_with_shared_zero": 96,
        "nonzero_cases": 95,
        "family_memberships_including_zero_aliases": 101,
        "family_case_counts": dict(FAMILY_COUNTS),
    }
    family_cases = manifest["family_cases"]
    assert list(family_cases) == list(FAMILY_ORDER)
    for family, identifiers in family_cases.items():
        assert isinstance(identifiers, list)
        assert len(identifiers) == FAMILY_COUNTS[family]
        assert identifiers.count(CLEAN_CASE_ID) == 1
        expected_index = 5 if family == "shift" else 4 if family == "slope" else 0
        assert identifiers[expected_index] == CLEAN_CASE_ID


def test_ids_are_derived_exclusive_and_collision_free(manifest):
    identifiers = [case["case_id"] for case in manifest["cases"]]
    assert len(identifiers) == len(set(identifiers)) == 96
    for case in manifest["cases"][1:]:
        assert len(case["case_id"].removeprefix("P08-STRESS-")) == 64
        assert case["case_id"] == _derived_id(
            case["family"], case["value"], case["replicate_index"]
        )
    seen = set()
    for entries in manifest["family_cases"].values():
        nonzero = [item for item in entries if item != CLEAN_CASE_ID]
        assert seen.isdisjoint(nonzero)
        seen.update(nonzero)
    assert len(seen) == 95


def test_stochastic_replicates_and_encounter_order(manifest):
    cases = manifest["cases"]
    expected = [("clean", 0, None)]
    for family in FAMILY_ORDER:
        reps = range(10) if family in STOCHASTIC else (None,)
        for value in LEVELS[family]:
            for rep in reps:
                expected.append((family, value, rep))
    observed = [(case["family"], case["value"], case["replicate_index"]) for case in cases]
    assert observed == expected
    grouped = {}
    for case in cases[1:]:
        grouped.setdefault(case["family"], {}).setdefault(case["value"], []).append(
            case["replicate_index"]
        )
    for family in STOCHASTIC:
        for reps in grouped[family].values():
            assert sorted(reps) == list(range(10))
    for family in set(FAMILY_ORDER) - STOCHASTIC:
        assert all(rep is None for reps in grouped[family].values() for rep in reps)


def test_deterministic_and_output_not_aliasing(design):
    snapshot = copy.deepcopy(design)
    first = build_perturbation_case_manifest(design=design)
    second = build_perturbation_case_manifest(design=design)
    assert design == snapshot
    assert first == second
    first["cases"][0]["case_id"] = "tampered"
    first["family_cases"]["shift"].append("tampered")
    first["summary"]["nonzero_cases"] = -1
    assert second["cases"][0]["case_id"] == CLEAN_CASE_ID
    assert "tampered" not in second["family_cases"]["shift"]
    assert second["summary"]["nonzero_cases"] == 95


def test_rejects_authority_schema_and_extra_keys(design, manifest):
    _rejects(design, lambda d: d.update(execution_authorized=True))
    _rejects(design, lambda d: d.update(authorized_scientific_operations=True))
    _rejects(design, lambda d: d.update(authorized_scientific_operations=1))
    _rejects(design, lambda d: d.update(numerical_execution_accepted=True))
    _rejects(design, lambda d: d.update(resource_proposal_approved=True))
    _rejects(design, lambda d: d.update(schema_version="bad"))
    candidate = copy.deepcopy(design)
    candidate["unexpected_extra"] = {1, 2, 3}
    with pytest.raises(ValueError):
        build_perturbation_case_manifest(design=candidate)
    enriched = copy.deepcopy(design)
    enriched["unexpected_extra"] = "x"
    enriched_manifest = build_perturbation_case_manifest(design=enriched)
    assert enriched_manifest["cases"] == manifest["cases"]
    assert enriched_manifest["design_sha256"] != manifest["design_sha256"]


def test_rejects_case_definition_parameter_mutations(design):
    for bad in (0, True, 11):
        _rejects(
            design,
            lambda d, bad=bad: d["case_definition"].update(stochastic_replicates=bad),
        )
    _rejects(design, lambda d: d["case_definition"].update(impulse_height_range_fraction=0.5))
    _rejects(design, lambda d: d["case_definition"].update(impulse_minimum_index_separation=2))
    _rejects(design, lambda d: d["case_definition"].update(unexpected_extra="x"))


def test_rejects_family_grid_mutations(design):
    def drop_family(d):
        d["case_definition"]["family_order"].remove("quadratic")

    def wrong_slope(d):
        d["case_definition"]["slope_range_fraction"] = [-0.2, -0.1, 0.1, 0.2]

    def shift_zero(d):
        d["case_definition"]["shift_cm1"][5] = False

    def bad_counts(d):
        d["expected_case_counts"]["nonzero_cases"] = 94

    for mutate in (drop_family, wrong_slope, shift_zero, bad_counts):
        _rejects(design, mutate)


def test_published_case_inventory_matches_design(design):
    path = PACKAGE / "results" / "p08_readiness" / "perturbation_case_inventory.json"
    saved = json.loads(path.read_text())
    assert saved == build_perturbation_case_manifest(design=design)
    assert saved["execution_authorized"] is False
    assert saved["numerical_perturbations_computed"] is False
    assert saved["job_ledger_complete"] is False


def test_owner_scope_approvals_remain_planning_only(design):
    assert design["selected_universal_panel"] == [
        "C-RBF-SVM",
        "C-RANDOM-FOREST",
        "C-EXTRA-TREES",
        "D0-M",
        "P05-SELECTED",
    ]
    assert design["selected_qc_stress_mode"] == "fixed_clean_route_sensitivity"
    assert design["scope_approvals"] == ["P08-A10", "P08-A11"]
    assert design["scope_approval_date"] == "2026-10-07"
    assert design["scope_approval_is_execution_permission"] is False
    assert design["native_grid_gate_reaction_experiment_included"] is False
    assert design["pending_decisions"] == []
    assert design["execution_authorized"] is False
    assert type(design["authorized_scientific_operations"]) is int
    assert design["authorized_scientific_operations"] == 0

    readiness_path = PACKAGE / "plan" / "contracts" / "p08_readiness_contract.json"
    readiness = json.loads(readiness_path.read_text())

    owner_decisions = {record["decision_id"]: record for record in readiness["owner_decisions"]}
    assert owner_decisions["P08-A10"]["status"] == "approved"
    assert (
        owner_decisions["P08-A10"]["answer"] == "Use all five universal-panel methods (Recommended)"
    )
    assert owner_decisions["P08-A11"]["status"] == "approved"
    assert (
        owner_decisions["P08-A11"]["answer"]
        == "Plan a clearly labelled fixed-route QC sensitivity (Recommended)"
    )
    assert owner_decisions["P08-A12"]["status"] == "approved"
    assert (
        owner_decisions["P08-A12"]["answer"]
        == "Include Extra Trees in both sensitivities"
    )

    perturbation_design = readiness["perturbation_design"]
    assert perturbation_design["robustness_panel_scope_resolved"] is True
    assert perturbation_design["qc_stress_scope_resolved"] is True
    assert perturbation_design["execution_authorized"] is False
    assert (
        perturbation_design["case_inventory"]
        == "results/p08_readiness/perturbation_case_inventory.json"
    )

    assert readiness["pending_later_branch_choices"] == []
