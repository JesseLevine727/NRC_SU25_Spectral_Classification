"""P08 range-inference declaration contract.

Metadata-only validation of the reviewed public range-inference registry,
readiness contract and ledger audit.  These tests fit no model, compute no
weight, draw no resample and imply no execution authority; they check exact
declared keys, counts and SHA256 pins only.  The original statistical
protocol is verified byte-for-byte unchanged via its pinned hash.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

RANGE_INFERENCE = PACKAGE / "plan/contracts/p08_range_inference.json"
READINESS_CONTRACT = PACKAGE / "plan/contracts/p08_readiness_contract.json"
STATISTICAL_PROTOCOL = PACKAGE / "plan/P08_STATISTICAL_PROTOCOL.md"
RANGE_INPUT_AUDIT = PACKAGE / "results/p08_readiness/range_input_audit.json"
RANGE_LEDGER_AUDIT = PACKAGE / "results/p08_readiness/range_ledger_audit.json"

MODEL_PANEL = ["C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "P05-SELECTED"]
ENDPOINTS = ["M01", "M06"]
NEURAL_MODELS = ["D0-M", "P05-SELECTED"]
CLASSICAL_MODELS = ["C-RBF-SVM", "C-RANDOM-FOREST"]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def range_registry():
    return _load(RANGE_INFERENCE)


@pytest.fixture(scope="module")
def readiness():
    return _load(READINESS_CONTRACT)


@pytest.fixture(scope="module")
def ledger():
    return _load(RANGE_LEDGER_AUDIT)


def test_registry_pins_protocol_and_audit_files(range_registry):
    assert (
        hashlib.sha256(STATISTICAL_PROTOCOL.read_bytes()).hexdigest()
        == (range_registry["inherited_protocol_sha256"])
    )
    assert (
        hashlib.sha256(RANGE_INPUT_AUDIT.read_bytes()).hexdigest()
        == (range_registry["input_audit_sha256"])
    )
    assert (
        hashlib.sha256(RANGE_LEDGER_AUDIT.read_bytes()).hexdigest()
        == (range_registry["ledger_audit_sha256"])
    )


def test_ledger_pins_range_plan(range_registry, ledger):
    assert ledger["plan_sha256"] == range_registry["range_plan_sha256"]


def test_model_panel_and_endpoints(range_registry):
    assert range_registry["model_panel"] == MODEL_PANEL
    assert range_registry["endpoints"] == ENDPOINTS
    assert "C-EXTRA-TREES" not in range_registry["model_panel"]
    assert "D0-M" in range_registry["model_panel"]
    assert "P05-SELECTED" in range_registry["model_panel"]


def test_effects_are_complete_cartesian(range_registry):
    effects = range_registry["effects"]
    assert len(effects) == 8
    assert [effect["contrast_id"] for effect in effects] == [
        f"R-E{index:02d}" for index in range(1, 9)
    ]
    assert {(effect["method"], effect["endpoint"]) for effect in effects} == {
        (model, endpoint) for model in MODEL_PANEL for endpoint in ENDPOINTS
    }


def test_interactions_are_complete_cartesian(range_registry):
    interactions = range_registry["interactions"]
    assert len(interactions) == 8
    assert [entry["contrast_id"] for entry in interactions] == [
        f"R-I{index:02d}" for index in range(1, 9)
    ]
    assert {(entry["neural"], entry["classical"], entry["endpoint"]) for entry in interactions} == {
        (neural, classical, endpoint)
        for neural in NEURAL_MODELS
        for classical in CLASSICAL_MODELS
        for endpoint in ENDPOINTS
    }


def test_contrast_directions(range_registry):
    assert range_registry["effect_direction"] == "comparison_minus_reference"
    assert range_registry["interaction_direction"] == (
        "neural_range_effect_minus_classical_range_effect"
    )


def test_weighted_uncertainty_declaration(range_registry):
    weighted = range_registry["weighted_uncertainty"]
    assert weighted["draws"] == 10000
    assert weighted["generator"] == "PCG64"
    assert weighted["master_seed"] == 2026093001
    assert weighted["instrument_seed"] == 2026093002
    assert weighted["distribution"] == "Exponential_mean_1"
    assert weighted["share_existing_global_P08_weight_arrays"] is True
    assert weighted["maximum_batch_size"] == 128
    assert weighted["unit_weight_absolute_tolerance"] == 1e-12
    assert weighted["quantiles"] == [0.025, 0.975]
    assert weighted["quantile_method"] == "linear"
    assert weighted["includes_retraining_uncertainty"] is False
    assert weighted["bca_available"] is False
    assert weighted["intervals_simultaneous"] is False
    assert weighted["master_array_shape"] == [10000, 69]
    assert weighted["instrument_array_shape"] == [10000, 10]
    assert weighted["array_dtype"] == "float64"
    assert weighted["array_axes"] == ["draw", "lexicographically_sorted_identity"]


def test_hierarchical_feasibility_declaration(range_registry):
    hierarchy = range_registry["hierarchical_feasibility"]
    assert hierarchy["draws"] == 10000
    assert hierarchy["generator"] == "PCG64"
    assert hierarchy["seed"] == 2026093003
    assert hierarchy["empty_original_context_class_rule"] == "undefined_draw_no_drop_no_retry"
    assert hierarchy["unconditional_interval_requires_all_draws_defined"] is True
    assert hierarchy["reset_per_contrast_endpoint"] is True


def test_multiplicity_and_sign_sensitivities(range_registry):
    multiplicity = range_registry["multiplicity"]
    assert multiplicity["procedure"] == "Holm"
    assert multiplicity["families"] == {"range_effects": 8, "range_model_interactions": 8}
    assert multiplicity["separate_sign_sensitivities"] == ["domain", "instrument_identity"]
    assert multiplicity["unavailable_adjustment_bookkeeping_p"] == 1
    assert multiplicity["unavailable_estimate_and_raw_p"] is None
    assert multiplicity["remove_structural_duplicates"] is False
    assert multiplicity["primary_family_sizes_changed"] is False
    signs = range_registry["sign_sensitivities"]
    assert signs["full_support_domain_assignments"] == 8192
    assert signs["full_support_instrument_assignments"] == 1024
    assert signs["enumeration"] == "exhaustive_including_observed"
    assert signs["equality_tolerance"] == 1e-12
    assert signs["randomized_treatment_interpretation"] is False


def test_context_counts_agree_with_readiness(range_registry, readiness):
    assert range_registry["contexts"] == 260
    assert range_registry["domains"] == 13
    assert range_registry["physical_masters"] == 69
    assert range_registry["instruments"] == 10
    assert range_registry["spectra"] == 598
    assert range_registry["held_evaluation_spectra"] == 557
    assert range_registry["spectra_outside_primary_held_evaluation"] == 41
    nested = readiness["range_sensitivity_readiness"]
    assert nested["model_panel"] == range_registry["model_panel"]
    assert nested["context_count"] == range_registry["contexts"]
    assert nested["plan_sha256"] == range_registry["range_plan_sha256"]
    assert nested["numerical_inference_lock_complete"] is True
    assert nested["inference_implementation_accepted"] is False
    assert nested["execution_authorized"] is False
    assert nested["extra_trees_included"] is False


def test_missingness_declaration(range_registry):
    missingness = range_registry["missingness"]
    assert missingness["full_headline_requires_all_registered_contexts_and_rows"] is True
    assert missingness["interaction_requires_common_four_cell_support"] is True
    assert missingness["minimal_fallback_for_failed_range_cell"] is False
    assert missingness["average_incomplete_seeds"] is False
    assert missingness["repair_historical_C_SELECTED_missing_endpoints"] is False


def test_execution_and_acceptance_flags_are_closed(range_registry):
    assert range_registry["execution_authorized"] is False
    assert range_registry["authorized_model_fits"] == 0
    assert range_registry["authorized_new_predictions"] == 0
    assert range_registry["authorized_resampling_draws"] == 0
    assert range_registry["inference_implementation_accepted"] is False
    assert range_registry["range_runtime_accepted"] is False
    assert range_registry["resource_proposal_approved"] is False


def test_ledger_counts_match_registry(range_registry, ledger):
    summary = ledger["summary"]
    assert summary["context_count"] == range_registry["contexts"]
    assert summary["model_fit_slots"] == 62981
    assert summary["scalar_calibrations"] == 1417
    assert summary["jobs_count"] == 131199
    assert summary["authorized_fit_slots"] == 0
    assert ledger["neural_strategy_aliases_verified"] == 520
    assert ledger["extra_trees_included"] is False
    assert ledger["execution_authorized"] is False
    assert ledger["scientific_operations_started"] == 0
    assert ledger["models"] == MODEL_PANEL
