"""Exact negative and provenance tests for the P08 population resource proposal."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from atlas_sers.evaluation.p08_population_resources import (
    build_population_resource_proposal,
)

PACKAGE = Path(__file__).resolve().parents[1]
READINESS = PACKAGE / "results/p08_readiness"

INPUT_FILES = {
    "slot_audit": "population_slot_ledger_audit.json",
    "historical_classical": "historical_timing_basis.json",
    "historical_neural": "population_historical_neural_cost.json",
}

S = ("slot_audit", "catalogs", 0, "summary")
C = ("historical_classical", "classical", 0)
N = ("historical_neural", "neural", 0)
DUPLICATES = [
    ("historical_classical", "classical", 1),
    ("historical_neural", "neural", 1),
    ("slot_audit", "catalogs", 1),
]

CASES = [
    (("slot_audit", "execution_authorized"), True),
    (("slot_audit", "scientific_operations"), True),
    (("slot_audit", "scientific_operations"), 1),
    (("slot_audit", "scientific_fit_count"), True),
    (("slot_audit", "scientific_fit_count"), 1),
    (("slot_audit", "arrays_loaded"), True),
    (("slot_audit", "outcomes_loaded"), True),
    (("slot_audit", "numerical_readiness_verified"), True),
    (("historical_classical", "scientific_operations_started"), 1),
    (("historical_classical", "measured_p08_timing"), True),
    (("historical_neural", "execution_authorized"), True),
    (("historical_neural", "new_scientific_operations"), True),
    (("historical_neural", "measured_P08_costs"), True),
    (("historical_neural", "outcome_fields_used"), True),
    (S + ("authorized_scientific_operations",), 1),
    (S + ("bounds_are_nonadditive",), False),
    (S + ("per_model", "bounds_are_nonadditive"), False),
    (S + ("lower", "total_jobs"), 0),
    (S + ("upper", "total_jobs"), 0),
    (S + ("lower", "model_fit_slots"), 0),
    (S + ("upper", "model_fit_slots"), 0),
    (S + ("lower", "scalar_calibrations"), 0),
    (S + ("upper", "scalar_calibrations"), 0),
    (S + ("catalog_stage_counts", "source_fit"), 0),
    (S + ("lower", "stage_counts", "held_prediction"), 0),
    (S + ("upper", "stage_counts", "source_fit"), 0),
    (S + ("per_model", "lower", "C-RBF-SVM", "total_jobs"), 0),
    (S + ("per_model", "lower", "C-RBF-SVM", "model_fit_slots"), 0),
    (S + ("per_model", "lower", "C-RBF-SVM", "scalar_calibrations"), 0),
    (C + ("seconds",), float("nan")),
    (C + ("seconds",), float("inf")),
    (C + ("seconds",), -1),
    (C + ("seconds",), True),
    (C + ("seconds",), 10**400),
    (C + ("seconds",), 1e308),
    (C + ("seconds_recorded",), 0),
    (C + ("size_recorded",), 0),
    (C + ("cache_reuse_verified",), 0),
    (N + ("elapsed_seconds_mean",), float("nan")),
    (N + ("elapsed_seconds_mean",), float("inf")),
    (N + ("elapsed_seconds_mean",), -1),
    (N + ("elapsed_seconds_mean",), True),
    (N + ("elapsed_seconds_mean",), 100),
    (N + ("retained_checkpoint_bytes_mean",), -1),
    (N + ("retained_checkpoint_bytes_mean",), 0),
    (N + ("retained_checkpoint_bytes_sum",), 0),
    (N + ("retained_checkpoint_bytes_max",), True),
    (N + ("reused_pilot_fits",), -1),
    (N + ("reused_pilot_fits",), True),
    (N + ("reused_pilot_fits",), 781),
    (N + ("reused_pilot_fits",), 1),
]


@pytest.fixture()
def inputs():
    return {k: json.loads((READINESS / v).read_text()) for k, v in INPUT_FILES.items()}


def _set_path(root, path, value):
    target = root
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize("path,value", CASES)
def test_rejects_unsafe_or_malformed_inputs(inputs, path, value):
    _set_path(inputs, path, value)
    with pytest.raises(ValueError):
        build_population_resource_proposal(**inputs)


@pytest.mark.parametrize("path", DUPLICATES)
def test_rejects_duplicate_entries(inputs, path):
    target = inputs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = copy.deepcopy(target[0])
    assert target[0] == target[path[-1]]
    assert isinstance(target[0], dict)
    assert isinstance(target[path[-1]], dict)
    with pytest.raises(ValueError):
        build_population_resource_proposal(**inputs)


def test_rejects_unknown_stage(inputs):
    _set_path(inputs, S + ("lower", "stage_counts", "unknown_stage"), 1)
    with pytest.raises(ValueError):
        build_population_resource_proposal(**inputs)


@pytest.mark.parametrize("bad", [{1, 2}, object()])
def test_rejects_non_json_extra(inputs, bad):
    inputs["slot_audit"]["provenance_extra"] = bad
    with pytest.raises(ValueError):
        build_population_resource_proposal(**inputs)


def test_rejects_missing_mandatory_flag(inputs):
    del inputs["slot_audit"]["numerical_readiness_verified"]
    with pytest.raises(ValueError):
        build_population_resource_proposal(**inputs)


def test_audit_provenance(inputs):
    audit = json.loads((READINESS / "population_resource_proposal_audit.json").read_text())
    digests = {
        key: hashlib.sha256((READINESS / name).read_bytes()).hexdigest()
        for key, name in INPUT_FILES.items()
    }
    assert audit["input_file_sha256"] == digests
    assert audit["proposal"] == build_population_resource_proposal(**inputs)
    source = PACKAGE / "src/atlas_sers/evaluation/p08_population_resources.py"
    assert audit["calculator_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit["execution_authorized"] is False
    assert audit["resource_proposal_approved"] is False


def test_contract_gates_resource_proposal():
    contract = json.loads((PACKAGE / "plan/contracts/p08_readiness_contract.json").read_text())
    block = contract["population_membership_prescreen"]
    assert block["population_resource_proposal_complete"] is True
    assert block["population_resource_proposal_approved"] is False
    assert block["execution_authorized"] is False
    assert (PACKAGE / block["resource_audit"]).is_file()
